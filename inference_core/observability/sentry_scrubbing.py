"""PII scrubbing for Sentry telemetry.

Sentry is an EXTERNAL SaaS sink — every event that leaves the box must be
stripped of user content and personal identifiers first. This module provides
the ``before_send`` / ``before_breadcrumb`` hooks plus a single
:func:`privacy_sentry_options` helper consumed by both init sites
(``main_factory.py`` for FastAPI and ``app/celery_main.py`` for Celery), so
their privacy posture cannot drift apart.

What gets scrubbed:

* string values matching e-mail addresses, bearer tokens, API-key shapes and
  long digit runs (phone/card-like) — replaced with ``***REDACTED***``;
* dict entries whose key is inherently sensitive (passwords, tokens, cookies,
  authorization headers, message/prompt content fields) — replaced wholesale;
* ``request`` payloads (data/cookies/query_string) and sensitive headers;
* ``user`` context — only the internal ``id`` survives (no ip/email/username);
* exception messages and log entry params (they routinely embed user input).

Everything here is fail-soft: a scrubbing error must never lose an event or
crash the caller, so hooks fall back to a minimal safe subset on failure.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Pattern

logger = logging.getLogger(__name__)

_REDACTED = "***REDACTED***"

# Conservative, LOCALE-NEUTRAL PII patterns applied to every string value
# before egress. inference-core ships only language-agnostic detectors here;
# locale-specific formats (national phone grouping, ID/bank/address shapes) are
# injected by the product via ``register_extra_detectors`` (see below) and are
# also consulted by the Langfuse mask through the shared ``redact_pii_text``.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_BEARER_RE = re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+")
_SECRET_RE = re.compile(r"(?i)(sk-|pk-|api[_-]?key[\"'=:\s]+)[A-Za-z0-9._\-]{8,}")
_LONG_DIGITS_RE = re.compile(r"\b\d{11,}\b")  # phone / card-like sequences
# Credential-bearing URL query params (OAuth callbacks, signed links).
_URL_SECRET_RE = re.compile(
    r"(?i)([?&](?:token|access_token|refresh_token|id_token|code|state|key|"
    r"signature|sig|client_secret)=)[^&\s\"']+"
)
# Bank accounts: generic IBAN (``XX00 0000 …``, any country). Locale-specific
# bare account numbers (e.g. Polish NRB), national phone grouping and address
# formats are NOT hardcoded here — a product injects them via
# ``register_extra_detectors`` so inference-core stays language-neutral.
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?\d{4}){4,7}(?:\s?\d{1,4})?\b")

# Dict keys whose values are always replaced wholesale, regardless of content.
_SENSITIVE_KEYS = {
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "cookie",
    "cookies",
    "set-cookie",
    "x-api-key",
    "encrypted_credentials",
    # user-content fields that must never ride along in telemetry
    "message",
    "message_content",
    "content",
    "body",
    "body_text",
    "body_html",
    "prompt",
    "email",
    "user_email",
    "subject",
}

# Request headers allowed through untouched; everything else is dropped.
_ALLOWED_HEADERS = {
    "content-type",
    "content-length",
    "accept",
    "accept-language",
    "user-agent",
    "host",
    "x-request-id",
}

_MAX_DEPTH = 8


# ---------------------------------------------------------------------------
# Product/locale detector injection seam
# ---------------------------------------------------------------------------
# The neutral core ships only locale-agnostic patterns. A product (e.g. a
# Polish deployment) injects its own detectors via ``register_extra_detectors``;
# they are consulted by ``_redact_str`` (and thus every egress sink) and by
# ``detect_spans``. This keeps inference-core free of any language coupling.


@dataclass(frozen=True)
class Detector:
    """A product/locale-injected redactor.

    Provide either ``pattern`` (a compiled regex — supports span extraction)
    or ``redactor`` (an opaque ``str -> str`` callable — redaction only).
    """

    name: str
    pattern: Optional[Pattern[str]] = None
    redactor: Optional[Callable[[str], str]] = None
    label: str = "PII"
    replacement: str = _REDACTED


_EXTRA_DETECTORS: "Dict[str, Detector]" = {}


def _coerce_detector(det: Any) -> Optional[Detector]:
    if isinstance(det, Detector):
        return det
    if isinstance(det, re.Pattern):
        return Detector(name=f"pattern:{det.pattern}", pattern=det)
    if callable(det):
        name = f"callable:{getattr(det, '__name__', id(det))}"
        return Detector(name=name, redactor=det)
    logger.warning("Ignoring unsupported extra PII detector: %r", det)
    return None


def register_extra_detectors(detectors: Iterable[Any]) -> None:
    """Register product/locale PII detectors. Idempotent by ``Detector.name``.

    Accepts :class:`Detector` instances, bare compiled ``re.Pattern`` objects,
    or ``str -> str`` callables (auto-wrapped). Safe to call from multiple
    startup sites (FastAPI lifespan + every Celery process).
    """
    for det in detectors:
        norm = _coerce_detector(det)
        if norm is not None:
            _EXTRA_DETECTORS[norm.name] = norm


def clear_extra_detectors() -> None:
    """Remove all injected detectors (test hook)."""
    _EXTRA_DETECTORS.clear()


def iter_extra_detectors() -> "List[Detector]":
    """Return the currently registered injected detectors."""
    return list(_EXTRA_DETECTORS.values())


def _apply_extra_detectors(text: str) -> str:
    for det in _EXTRA_DETECTORS.values():
        try:
            if det.pattern is not None:
                text = det.pattern.sub(det.replacement, text)
            elif det.redactor is not None:
                text = det.redactor(text)
        except Exception:  # a bad injected detector must never break egress
            continue
    return text


# Generic, locale-neutral patterns usable for structured span extraction
# (label, in application order). Callable-only injected detectors yield no spans.
_GENERIC_SPAN_PATTERNS = [
    (_EMAIL_RE, "email"),
    (_SECRET_RE, "secret"),
    (_BEARER_RE, "secret"),
    (_URL_SECRET_RE, "secret"),
    (_IBAN_RE, "iban"),
    (_LONG_DIGITS_RE, "long_digits"),
]


def detect_spans(text: str) -> "List[tuple]":
    """Return ``(start, end, label, matched_text)`` spans from generic +
    injected regex detectors. Used by the regex/fake PiiDetector backends.
    Never raises."""
    spans: List[Any] = []
    try:
        for pat, label in _GENERIC_SPAN_PATTERNS:
            for m in pat.finditer(text):
                spans.append((m.start(), m.end(), label, m.group(0)))
        for det in _EXTRA_DETECTORS.values():
            if det.pattern is not None:
                for m in det.pattern.finditer(text):
                    spans.append((m.start(), m.end(), det.label, m.group(0)))
    except Exception:  # pragma: no cover - detection must never raise
        return spans
    return spans


def _redact_str(text: str) -> str:
    try:
        text = _BEARER_RE.sub("bearer " + _REDACTED, text)
        text = _SECRET_RE.sub(_REDACTED, text)
        text = _URL_SECRET_RE.sub(r"\1" + _REDACTED, text)
        text = _EMAIL_RE.sub(_REDACTED, text)
        text = _IBAN_RE.sub(_REDACTED, text)
        # Product/locale-injected detectors (national phone/ID/address formats)
        # run between the specific generic patterns and the greedy long-digit
        # sweep — the ordering locale packs are written against.
        text = _apply_extra_detectors(text)
        text = _LONG_DIGITS_RE.sub(_REDACTED, text)
    except Exception:  # pragma: no cover - redaction must never raise
        return _REDACTED
    return text


def redact_pii_text(text: str) -> str:
    """Public helper: redact PII/secrets from a single string.

    Shared beyond Sentry — e.g. sanitizing exception text before it is
    logged or persisted into ``*_error`` columns, and the Langfuse mask.

    Routes through the configured :class:`~inference_core.services.pii_detector.PiiDetector`
    backend so an ML detector (when ``PII_BACKEND`` selects one) applies here
    too. Always fail-soft: any backend error falls back to the pure-regex
    :func:`_redact_str`, so egress never depends on the ML backend being
    healthy and this function never raises.
    """
    try:
        from inference_core.services.pii_detector import get_pii_detector

        return get_pii_detector().redact_fast(text)
    except Exception:  # pragma: no cover - fail-soft to pure regex
        return _redact_str(text)


def _redact(value: Any, depth: int = 0) -> Any:
    """Recursively redact PII from arbitrary event payloads."""
    if depth > _MAX_DEPTH:
        return _REDACTED
    if isinstance(value, str):
        return redact_pii_text(value)
    if isinstance(value, dict):
        out: Dict[Any, Any] = {}
        for key, val in value.items():
            if isinstance(key, str) and key.lower() in _SENSITIVE_KEYS:
                out[key] = _REDACTED
            else:
                out[key] = _redact(val, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        redacted = [_redact(item, depth + 1) for item in value]
        return type(value)(redacted) if isinstance(value, tuple) else redacted
    return value


def _scrub_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only routing metadata from the HTTP request context."""
    scrubbed: Dict[str, Any] = {}
    for key in ("url", "method"):
        if key in request:
            scrubbed[key] = _redact(request[key], _MAX_DEPTH - 1)
    headers = request.get("headers")
    if isinstance(headers, dict):
        scrubbed["headers"] = {
            k: v for k, v in headers.items() if k.lower() in _ALLOWED_HEADERS
        }
    # data / cookies / query_string / env are dropped entirely.
    return scrubbed


def _scrub_user(user: Dict[str, Any]) -> Dict[str, Any]:
    """Internal id only — no ip_address / email / username."""
    return {"id": user["id"]} if "id" in user else {}


def scrub_event(event: Dict[str, Any], hint: Optional[Dict[str, Any]] = None):
    """``before_send`` hook: strip PII from an outgoing Sentry event."""
    try:
        if "request" in event and isinstance(event["request"], dict):
            event["request"] = _scrub_request(event["request"])
        if "user" in event and isinstance(event["user"], dict):
            event["user"] = _scrub_user(event["user"])

        # Exception messages routinely embed user input (str(exc) chains).
        exception = event.get("exception")
        if isinstance(exception, dict):
            for entry in exception.get("values") or []:
                if isinstance(entry, dict) and isinstance(entry.get("value"), str):
                    entry["value"] = redact_pii_text(entry["value"])
                # Belt & braces: local variables should already be disabled
                # via include_local_variables=False.
                frames = (entry.get("stacktrace") or {}).get("frames")
                if isinstance(frames, list):
                    for frame in frames:
                        if isinstance(frame, dict) and "vars" in frame:
                            frame.pop("vars", None)

        # logger.xxx("...", params) messages captured as events.
        logentry = event.get("logentry")
        if isinstance(logentry, dict):
            if isinstance(logentry.get("message"), str):
                logentry["message"] = redact_pii_text(logentry["message"])
            if "params" in logentry:
                logentry["params"] = _redact(logentry["params"])

        if isinstance(event.get("message"), str):
            event["message"] = redact_pii_text(event["message"])

        for section in ("extra", "contexts", "tags"):
            if section in event and isinstance(event[section], dict):
                event[section] = _redact(event[section])

        breadcrumbs = event.get("breadcrumbs")
        if isinstance(breadcrumbs, dict) and isinstance(breadcrumbs.get("values"), list):
            breadcrumbs["values"] = [
                scrub_breadcrumb(crumb) or crumb for crumb in breadcrumbs["values"]
            ]
    except Exception:  # pragma: no cover - never lose the event over scrubbing
        logger.debug("Sentry event scrubbing failed; sending minimal event", exc_info=True)
        return {
            key: event[key]
            for key in ("event_id", "timestamp", "platform", "level", "release", "environment")
            if key in event
        }
    return event


def scrub_breadcrumb(crumb: Dict[str, Any], hint: Optional[Dict[str, Any]] = None):
    """``before_breadcrumb`` hook: strip PII from a single breadcrumb."""
    try:
        if isinstance(crumb.get("message"), str):
            crumb["message"] = redact_pii_text(crumb["message"])
        if isinstance(crumb.get("data"), dict):
            crumb["data"] = _redact(crumb["data"])
    except Exception:  # pragma: no cover
        return None  # drop the breadcrumb rather than leak it
    return crumb


def privacy_sentry_options() -> Dict[str, Any]:
    """Privacy-hardened kwargs shared by every ``sentry_sdk.init`` call site.

    Central so the FastAPI and Celery inits cannot drift apart:

    - ``send_default_pii=False`` — no IPs, cookies, request bodies, usernames;
    - ``include_local_variables=False`` — stack-frame locals (which can hold
      raw message/email content) never leave the process;
    - ``before_send`` / ``before_breadcrumb`` — content scrubbing above;
    - ``max_request_body_size='never'`` — request bodies are never attached.
    """
    return {
        "send_default_pii": False,
        "include_local_variables": False,
        "max_request_body_size": "never",
        "before_send": scrub_event,
        "before_breadcrumb": scrub_breadcrumb,
    }
