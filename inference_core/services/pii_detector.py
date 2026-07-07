"""PII detection abstraction (locale-neutral).

Mirrors :mod:`inference_core.services.embedding_service`: one service with
pluggable backends selected by the ``PII_BACKEND`` setting.

- ``regex``  — generic patterns + product-injected locale detectors
               (:func:`inference_core.observability.sentry_scrubbing.register_extra_detectors`).
               No model, no extra dependency. **Default.**
- ``gliner-local`` — in-process GLiNER multilingual PII model, loaded lazily
               per process (never at import). Heavy (~0.5 GB + torch) — enable
               only on a host/worker that can afford it. Needs the optional
               ``pii`` dependency group (``gliner``).
- ``remote`` — best-effort HTTP call to an external PII service (fail-soft).
- ``fake``   — no model; generic regex only (tests).

Every egress sink (the Sentry ``before_send`` scrubber, ``redact_pii_text``,
``sanitize_error``, the Langfuse mask) routes through
``get_pii_detector().redact_fast()``. ``redact_fast`` is **egress-safe**: it is
hard-time-boxed (``PII_EGRESS_TIMEOUT_MS``) and always falls back to the
pure-regex redactor, so a slow, absent, or OOM ML backend can never block,
drop, or crash an error event.

Deployment note: keep ``PII_BACKEND=regex`` on latency/RAM-sensitive processes
(the API and the shared threads worker). Run ``gliner-local`` in the dedicated
``--profile local-pii`` prefork worker, or use ``remote``. See
``docs/PRIVACY_CORE_NEUTRALITY_GLINER_PLAN.md``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from inference_core.core.config import Settings, get_settings
from inference_core.observability import sentry_scrubbing as _scrub

logger = logging.getLogger(__name__)

# Neutral, multilingual PII entity labels for the GLiNER/remote backends.
_DEFAULT_PII_LABELS = [
    "person",
    "email address",
    "phone number",
    "address",
    "iban",
    "credit card number",
    "national id",
    "passport number",
    "date of birth",
    "organization",
]


@dataclass(frozen=True)
class PiiSpan:
    """A detected PII span (0-based half-open ``[start, end)`` offsets)."""

    start: int
    end: int
    label: str
    text: str
    score: float = 1.0


def _regex_spans(text: str) -> List[PiiSpan]:
    """Structured spans from the generic + injected regex detectors."""
    return [
        PiiSpan(start, end, label, matched)
        for (start, end, label, matched) in _scrub.detect_spans(text)
    ]


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class BasePiiBackend(ABC):
    """Protocol for PII backends."""

    @abstractmethod
    def detect(self, text: str) -> List[PiiSpan]:
        """Return structured PII spans."""
        ...

    @abstractmethod
    def redact(self, text: str) -> str:
        """Return ``text`` with detected PII replaced. May block (batch use)."""
        ...

    async def adetect(self, text: str) -> List[PiiSpan]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.detect, text)

    async def aredact(self, text: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.redact, text)

    def redact_fast(self, text: str) -> str:
        """Egress-safe redaction — MUST NOT raise, SHOULD be time-bounded.

        Default equals :meth:`redact`; ML backends override to add a hard
        timeout + regex fallback.
        """
        return self.redact(text)


# ---------------------------------------------------------------------------
# regex backend (default) — generic patterns + injected locale detectors
# ---------------------------------------------------------------------------


class RegexPiiBackend(BasePiiBackend):
    """No model. Delegates to the neutral scrubber (generic + injected packs)."""

    def __init__(self, settings: Optional[Settings] = None) -> None:  # noqa: D401
        # No state — the injected detectors live in the sentry_scrubbing registry.
        pass

    def detect(self, text: str) -> List[PiiSpan]:
        return _regex_spans(text)

    def redact(self, text: str) -> str:
        return _scrub._redact_str(text)

    def redact_fast(self, text: str) -> str:
        return _scrub._redact_str(text)


# ---------------------------------------------------------------------------
# gliner-local backend — in-process multilingual PII model (opt-in, heavy)
# ---------------------------------------------------------------------------


class LocalGlinerBackend(BasePiiBackend):
    """In-process GLiNER PII model, loaded lazily and cached per process.

    OOM/latency sensitive — enable only where affordable (a dedicated prefork
    worker, or a host with headroom). ``redact_fast`` is hard-time-boxed and
    falls back to pure regex, so even here the synchronous egress path is safe.
    """

    _lock = threading.Lock()
    _model = None
    _model_name_loaded: Optional[str] = None
    _load_failed = False
    # >1 worker so a nested egress call has a free slot (deadlock mitigation).
    _executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="pii-egress"
    )
    _egress_guard = threading.local()

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.pii_local_model
        self._labels = list(_DEFAULT_PII_LABELS)
        self._egress_timeout = max(int(settings.pii_egress_timeout_ms), 1) / 1000.0

    @classmethod
    def _get_model(cls, model_name: str):
        if cls._model is not None and cls._model_name_loaded == model_name:
            return cls._model
        if cls._load_failed:
            return None
        # Timed acquire: a load-time re-entrancy (an error captured mid-load)
        # can never deadlock the process — it just falls back to regex.
        if not cls._lock.acquire(timeout=1.0):
            return None
        try:
            if cls._model is not None and cls._model_name_loaded == model_name:
                return cls._model
            if cls._load_failed:
                return None
            from gliner import GLiNER

            logger.info("Loading GLiNER PII model: %s", model_name)
            cls._model = GLiNER.from_pretrained(model_name)
            cls._model_name_loaded = model_name
            return cls._model
        except Exception:
            cls._load_failed = True
            logger.error(
                "Failed to load GLiNER model %s; PII detection falls back to "
                "regex for this process.",
                model_name,
                exc_info=True,
            )
            return None
        finally:
            cls._lock.release()

    @classmethod
    def reset(cls) -> None:
        """Reset the cached model/latch. Testing only."""
        if cls._lock.acquire(timeout=1.0):
            try:
                cls._model = None
                cls._model_name_loaded = None
                cls._load_failed = False
            finally:
                cls._lock.release()

    def detect(self, text: str) -> List[PiiSpan]:
        model = self._get_model(self._model_name)
        if model is None:
            return _regex_spans(text)
        try:
            ents = model.predict_entities(text, self._labels)
        except Exception:
            logger.warning("GLiNER predict_entities failed; regex spans", exc_info=True)
            return _regex_spans(text)
        return [
            PiiSpan(
                int(e["start"]),
                int(e["end"]),
                e.get("label", "PII"),
                e.get("text", text[int(e["start"]) : int(e["end"])]),
                float(e.get("score", 1.0)),
            )
            for e in ents
        ]

    def redact(self, text: str) -> str:
        model = self._get_model(self._model_name)
        out = text
        if model is not None:
            try:
                ents = model.predict_entities(text, self._labels)
                for e in sorted(ents, key=lambda x: int(x["start"]), reverse=True):
                    out = (
                        out[: int(e["start"])] + _scrub._REDACTED + out[int(e["end"]) :]
                    )
            except Exception:
                logger.warning("GLiNER redact failed; regex layer only", exc_info=True)
        # Always run the generic + injected regex layer too (belt & braces).
        return _scrub._redact_str(out)

    def redact_fast(self, text: str) -> str:
        # Same-thread re-entrancy guard (an error captured mid-redact routes
        # back through the scrubber) → immediate pure-regex.
        if getattr(self._egress_guard, "active", False) or self._load_failed:
            return _scrub._redact_str(text)
        self._egress_guard.active = True
        try:
            future = self._executor.submit(self.redact, text)
            return future.result(timeout=self._egress_timeout)
        except Exception:
            # Timeout / cold load / any error → pure regex now. A timed-out
            # load keeps running in the pool and warms the cache for next time.
            return _scrub._redact_str(text)
        finally:
            self._egress_guard.active = False


# ---------------------------------------------------------------------------
# remote backend — external PII service (best-effort, fail-soft)
# ---------------------------------------------------------------------------


class RemotePiiBackend(BasePiiBackend):
    """Best-effort external PII service.

    Contract: ``POST {pii_remote_url}`` json ``{"text": ...}`` →
    ``{"spans": [{start,end,label,text,score}], "redacted": "..."}``. Any
    failure (no URL, no ``httpx``, network/HTTP error, bad body) falls back to
    the generic regex layer — which still redacts generic + injected PII.
    """

    def __init__(self, settings: Settings) -> None:
        self._url = settings.pii_remote_url
        self._timeout = int(settings.pii_remote_timeout)

    def _call(self, text: str) -> Optional[dict]:
        if not self._url:
            return None
        try:
            import httpx

            resp = httpx.post(self._url, json={"text": text}, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.warning("Remote PII call failed; regex fallback", exc_info=True)
            return None

    def detect(self, text: str) -> List[PiiSpan]:
        data = self._call(text)
        if not data or not isinstance(data.get("spans"), list):
            return _regex_spans(text)
        return [
            PiiSpan(
                int(s["start"]),
                int(s["end"]),
                s.get("label", "PII"),
                s.get("text", ""),
                float(s.get("score", 1.0)),
            )
            for s in data["spans"]
        ]

    def redact(self, text: str) -> str:
        data = self._call(text)
        if data and isinstance(data.get("redacted"), str):
            return _scrub._redact_str(data["redacted"])
        return _scrub._redact_str(text)

    def redact_fast(self, text: str) -> str:
        # The synchronous egress hot path must not make a per-event network
        # call; remote detection is for deliberate detect()/redact() use.
        return _scrub._redact_str(text)


# ---------------------------------------------------------------------------
# fake backend — no model, generic regex only (tests)
# ---------------------------------------------------------------------------


class FakePiiBackend(BasePiiBackend):
    """Deterministic, dependency-free backend for tests (no model load)."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        pass

    def detect(self, text: str) -> List[PiiSpan]:
        return _regex_spans(text)

    def redact(self, text: str) -> str:
        return _scrub._redact_str(text)

    def redact_fast(self, text: str) -> str:
        return _scrub._redact_str(text)


# ---------------------------------------------------------------------------
# Unified service
# ---------------------------------------------------------------------------


class PiiDetector:
    """Unified PII detector. Backend chosen by ``PII_BACKEND`` (or injected)."""

    _BACKENDS = {
        "regex": RegexPiiBackend,
        "gliner-local": LocalGlinerBackend,
        "remote": RemotePiiBackend,
        "fake": FakePiiBackend,
    }

    def __init__(self, backend: Optional[BasePiiBackend] = None) -> None:
        if backend is not None:
            self._backend = backend
        else:
            settings = get_settings()
            backend_cls = self._BACKENDS.get(settings.pii_backend)
            if backend_cls is None:
                raise ValueError(f"Unknown PII_BACKEND: {settings.pii_backend}")
            self._backend = backend_cls(settings)

    def detect(self, text: str) -> List[PiiSpan]:
        return self._backend.detect(text)

    def redact(self, text: str) -> str:
        return self._backend.redact(text)

    def redact_fast(self, text: str) -> str:
        try:
            return self._backend.redact_fast(text)
        except Exception:  # last-resort guard: egress must never raise
            return _scrub._redact_str(text)

    async def adetect(self, text: str) -> List[PiiSpan]:
        return await self._backend.adetect(text)

    async def aredact(self, text: str) -> str:
        return await self._backend.aredact(text)

    @property
    def backend_name(self) -> str:
        return type(self._backend).__name__


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_pii_detector: Optional[PiiDetector] = None


def get_pii_detector() -> PiiDetector:
    """Get or create the global :class:`PiiDetector` singleton."""
    global _pii_detector
    if _pii_detector is None:
        _pii_detector = PiiDetector()
    return _pii_detector


def clear_pii_detector_cache() -> None:
    """Reset the singleton (for testing)."""
    global _pii_detector
    _pii_detector = None
