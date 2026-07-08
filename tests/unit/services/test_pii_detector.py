"""Tests for the PiiDetector abstraction and its backends.

Covers the default (regex) backend, the fake backend (no model), singleton
semantics, unknown-backend rejection, and the egress safety guarantees
(never-raise + hard-timeout fallback) that keep the synchronous Sentry/Langfuse
path safe when an ML backend is selected. The real GLiNER model is never loaded
here (it would download ~0.5 GB); the ML code paths are exercised via the
fallback/timeout seams.
"""

import time

import pytest

from inference_core.observability.sentry_scrubbing import (
    clear_extra_detectors,
    register_extra_detectors,
)
from inference_core.services import pii_detector as pd
from inference_core.services.pii_detector import (
    BasePiiBackend,
    FakePiiBackend,
    LocalGlinerBackend,
    PiiDetector,
    RegexPiiBackend,
    clear_pii_detector_cache,
    get_pii_detector,
)


@pytest.fixture(autouse=True)
def _reset():
    clear_extra_detectors()
    clear_pii_detector_cache()
    LocalGlinerBackend.reset()
    yield
    clear_extra_detectors()
    clear_pii_detector_cache()
    LocalGlinerBackend.reset()


def test_default_backend_is_regex():
    # PII_BACKEND defaults to 'regex' (and the testing profile pins it).
    assert isinstance(get_pii_detector()._backend, RegexPiiBackend)


def test_regex_backend_redacts_generic():
    detector = PiiDetector(backend=RegexPiiBackend())
    assert "a@b.com" not in detector.redact("mail a@b.com")


def test_regex_backend_detect_returns_email_span():
    detector = PiiDetector(backend=RegexPiiBackend())
    spans = detector.detect("mail a@b.com now")
    assert any(s.label == "email" for s in spans)


def test_regex_backend_honours_injected_detectors():
    import re

    register_extra_detectors([re.compile(r"PESEL\d+")])
    detector = PiiDetector(backend=RegexPiiBackend())
    assert "PESEL123" not in detector.redact("id PESEL123")


def test_fake_backend_no_model_but_redacts_generic():
    detector = PiiDetector(backend=FakePiiBackend())
    assert "a@b.com" not in detector.redact_fast("mail a@b.com")


def test_singleton_is_cached_and_resettable():
    first = get_pii_detector()
    assert get_pii_detector() is first
    clear_pii_detector_cache()
    assert get_pii_detector() is not first


def test_unknown_backend_raises(monkeypatch):
    class _S:
        pii_backend = "does-not-exist"

    monkeypatch.setattr(pd, "get_settings", lambda: _S())
    with pytest.raises(ValueError):
        PiiDetector()


def test_redact_fast_never_raises_on_backend_error():
    class Boom(BasePiiBackend):
        def detect(self, text):
            raise RuntimeError("x")

        def redact(self, text):
            raise RuntimeError("x")

        def redact_fast(self, text):
            raise RuntimeError("x")

    detector = PiiDetector(backend=Boom())
    # The unified guard must swallow the error and fall back to pure regex.
    assert "a@b.com" not in detector.redact_fast("mail a@b.com")


def test_gliner_redact_fast_times_out_and_falls_back(monkeypatch):
    class _S:
        pii_local_model = "unused"
        pii_egress_timeout_ms = 20

    backend = LocalGlinerBackend(_S())
    # Simulate a slow model redaction; redact_fast must time-box it.
    monkeypatch.setattr(backend, "redact", lambda text: (time.sleep(0.5), "LEAK")[1])
    out = backend.redact_fast("mail a@b.com")
    assert "LEAK" not in out
    assert "a@b.com" not in out  # regex fallback still redacts


def test_gliner_missing_model_falls_back_to_regex(monkeypatch):
    class _S:
        pii_local_model = "unused"
        pii_egress_timeout_ms = 50

    backend = LocalGlinerBackend(_S())
    # No model available (gliner not installed / load fails) -> regex layer only.
    monkeypatch.setattr(
        LocalGlinerBackend, "_get_model", classmethod(lambda cls, n: None)
    )
    assert "a@b.com" not in backend.redact("mail a@b.com")
    assert backend.detect("mail a@b.com")  # falls back to regex spans (non-empty)
