"""Tests for the product/locale PII detector injection seam.

The neutral core exposes ``register_extra_detectors`` so a product can inject
its own (e.g. locale-specific) redactors without any language coupling landing
in ``inference-core``. These verify the registry contract and that injected
detectors are honoured by the shared ``redact_pii_text`` egress path.
"""

import re

import pytest

from inference_core.observability.sentry_scrubbing import (
    Detector,
    clear_extra_detectors,
    iter_extra_detectors,
    redact_pii_text,
    register_extra_detectors,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_extra_detectors()
    yield
    clear_extra_detectors()


def test_register_regex_detector_applies():
    register_extra_detectors([re.compile(r"SECRET-\d{4}")])
    assert "SECRET-1234" not in redact_pii_text("code SECRET-1234 here")


def test_register_bare_pattern_and_callable():
    register_extra_detectors(
        [
            re.compile(r"ID\d+"),
            lambda t: t.replace("Kowalski", "***REDACTED***"),
        ]
    )
    out = redact_pii_text("Jan Kowalski ID42")
    assert "Kowalski" not in out
    assert "ID42" not in out


def test_detector_object_uses_custom_replacement():
    register_extra_detectors(
        [Detector("pesel", pattern=re.compile(r"\bPESEL\b"), replacement="[X]")]
    )
    assert redact_pii_text("PESEL") == "[X]"


def test_registration_is_idempotent_by_name():
    det = Detector("dup", pattern=re.compile(r"x"))
    register_extra_detectors([det, det])
    register_extra_detectors([det])
    assert sum(1 for d in iter_extra_detectors() if d.name == "dup") == 1


def test_clear_removes_all():
    register_extra_detectors([re.compile(r"x")])
    clear_extra_detectors()
    assert iter_extra_detectors() == []


def test_unsupported_detector_is_ignored():
    register_extra_detectors([12345])  # not a Detector / Pattern / callable
    assert iter_extra_detectors() == []


def test_faulty_detector_never_breaks_egress():
    def boom(_text):
        raise RuntimeError("nope")

    register_extra_detectors([boom])
    # A raising injected detector is swallowed; generic redaction still runs.
    assert "a@b.com" not in redact_pii_text("mail a@b.com boom")


def test_injected_detector_flows_through_scrub_event():
    from inference_core.observability.sentry_scrubbing import scrub_event

    register_extra_detectors([re.compile(r"TOPSECRET")])
    scrubbed = scrub_event({"event_id": "e1", "message": "leak TOPSECRET now"})
    assert "TOPSECRET" not in repr(scrubbed)
