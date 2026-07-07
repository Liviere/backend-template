"""Tests for the Sentry PII scrubbing hooks (privacy Faza 0)."""

from inference_core.observability.sentry_scrubbing import (
    privacy_sentry_options,
    scrub_breadcrumb,
    scrub_event,
)


def _sample_event() -> dict:
    """A realistic error event carrying every category of PII we scrub."""
    return {
        "event_id": "abc123",
        "level": "error",
        "message": "Failed to email jan.kowalski@example.com about order 48602123456789",
        "request": {
            "url": "https://api.example.com/api/v1/chat",
            "method": "POST",
            "query_string": "token=sk-live-abcdefgh12345678",
            "data": {"message": "Mój numer telefonu to 48602123456"},
            "cookies": "session=abcd",
            "headers": {
                "Authorization": "Bearer eyJhbGciOi.secret.token",
                "Content-Type": "application/json",
                "User-Agent": "pytest",
                "Cookie": "sid=123",
            },
            "env": {"REMOTE_ADDR": "10.0.0.7"},
        },
        "user": {
            "id": "u-42",
            "email": "jan.kowalski@example.com",
            "ip_address": "203.0.113.9",
            "username": "jkowalski",
        },
        "exception": {
            "values": [
                {
                    "type": "SMTPError",
                    "value": "550 rejected for jan.kowalski@example.com (key sk-live-abcdefgh12345678)",
                    "stacktrace": {
                        "frames": [
                            {
                                "filename": "mailer.py",
                                "vars": {"body": "treść prywatnego maila"},
                            }
                        ]
                    },
                }
            ]
        },
        "logentry": {
            "message": "sending to %s",
            "params": ["jan.kowalski@example.com"],
        },
        "extra": {
            "email_data": {"subject": "Poufne", "body_text": "sekret"},
            "task_id": "t-1",
        },
        "breadcrumbs": {
            "values": [
                {
                    "message": "IMAP login for jan.kowalski@example.com",
                    "data": {"password": "hunter2", "host": "imap.example.com"},
                }
            ]
        },
    }


class TestScrubEvent:
    def test_no_pii_survives_realistic_event(self):
        scrubbed = scrub_event(_sample_event())
        flat = repr(scrubbed)
        assert "jan.kowalski@example.com" not in flat
        assert "sk-live-abcdefgh12345678" not in flat
        assert "48602123456" not in flat
        assert "hunter2" not in flat
        assert "treść prywatnego maila" not in flat
        assert "sekret" not in flat
        assert "203.0.113.9" not in flat

    def test_request_reduced_to_routing_metadata(self):
        scrubbed = scrub_event(_sample_event())
        request = scrubbed["request"]
        assert request["url"] == "https://api.example.com/api/v1/chat"
        assert request["method"] == "POST"
        assert "data" not in request
        assert "cookies" not in request
        assert "query_string" not in request
        assert "env" not in request
        assert set(request["headers"]) == {"Content-Type", "User-Agent"}

    def test_user_keeps_internal_id_only(self):
        scrubbed = scrub_event(_sample_event())
        assert scrubbed["user"] == {"id": "u-42"}

    def test_exception_value_redacted_and_vars_dropped(self):
        scrubbed = scrub_event(_sample_event())
        entry = scrubbed["exception"]["values"][0]
        assert "jan.kowalski" not in entry["value"]
        assert entry["type"] == "SMTPError"  # error class kept for triage
        assert "vars" not in entry["stacktrace"]["frames"][0]
        assert entry["stacktrace"]["frames"][0]["filename"] == "mailer.py"

    def test_operational_extras_survive(self):
        scrubbed = scrub_event(_sample_event())
        assert scrubbed["extra"]["task_id"] == "t-1"
        assert scrubbed["event_id"] == "abc123"

    def test_scrub_failure_falls_back_to_minimal_event(self, monkeypatch):
        import inference_core.observability.sentry_scrubbing as mod

        monkeypatch.setattr(
            mod, "_scrub_request", lambda request: 1 / 0, raising=True
        )
        event = _sample_event()
        scrubbed = scrub_event(event)
        # Minimal safe subset — never the unscrubbed original.
        assert "request" not in scrubbed
        assert "exception" not in scrubbed
        assert scrubbed["event_id"] == "abc123"


class TestGenericPatterns:
    """Locale-NEUTRAL string redaction shipped by core (no language coupling)."""

    def _redact(self, text: str) -> str:
        from inference_core.observability.sentry_scrubbing import redact_pii_text

        return redact_pii_text(text)

    def test_email(self):
        assert "jan@example.com" not in self._redact("write to jan@example.com")

    def test_generic_iban(self):
        # Country-prefixed IBAN (any country) is generic and stays in core.
        out = self._redact("IBAN: PL61 1090 1014 0000 0712 1981 2874")
        assert "1090 1014" not in out

    def test_long_digit_runs(self):
        assert "48602123456789" not in self._redact("order 48602123456789 failed")

    def test_dates_and_amounts_survive(self):
        # Guard against over-redaction: timestamps and short money amounts
        # must NOT be treated as PII.
        text = "2026-07-05 04:31:00 charged 12.34 USD, retries=3"
        assert self._redact(text) == text

    def test_task_ids_survive(self):
        text = "task 8f14e45f-ceea-467f-a8d9-000000000000 finished"
        out = self._redact(text)
        assert "8f14e45f" in out


class TestLocaleNeutrality:
    """Core must NOT hardcode locale PII — a product injects it via the seam."""

    def setup_method(self):
        from inference_core.observability.sentry_scrubbing import clear_extra_detectors
        from inference_core.services.pii_detector import clear_pii_detector_cache

        clear_extra_detectors()
        clear_pii_detector_cache()

    def teardown_method(self):
        from inference_core.observability.sentry_scrubbing import clear_extra_detectors

        clear_extra_detectors()

    def _redact(self, text: str) -> str:
        from inference_core.observability.sentry_scrubbing import redact_pii_text

        return redact_pii_text(text)

    def test_polish_phone_survives_without_injection(self):
        # A bare Polish grouped phone is NOT generic PII — core leaves it alone.
        assert "602-123-456" in self._redact("tel. 602-123-456")

    def test_polish_street_survives_without_injection(self):
        assert "Marszałkowska" in self._redact("ul. Marszałkowska 12/34")

    def test_injected_detector_redacts(self):
        import re

        from inference_core.observability.sentry_scrubbing import (
            register_extra_detectors,
        )

        register_extra_detectors(
            [re.compile(r"(?<![\w.\-])\d{3}[\s\-]\d{3}[\s\-]\d{3}(?![\w\-])")]
        )
        assert "602-123-456" not in self._redact("tel. 602-123-456")


class TestScrubBreadcrumb:
    def test_message_and_data_redacted(self):
        crumb = scrub_breadcrumb(
            {
                "message": "poll ok for jan.kowalski@example.com",
                "data": {"token": "abc", "count": 3},
            }
        )
        assert "jan.kowalski" not in crumb["message"]
        assert crumb["data"]["token"] == "***REDACTED***"
        assert crumb["data"]["count"] == 3


class TestPrivacyOptions:
    def test_hardened_defaults(self):
        opts = privacy_sentry_options()
        assert opts["send_default_pii"] is False
        assert opts["include_local_variables"] is False
        assert opts["max_request_body_size"] == "never"
        assert opts["before_send"] is scrub_event
        assert opts["before_breadcrumb"] is scrub_breadcrumb

    def test_options_accepted_by_sentry_sdk(self):
        # Guard against option-name drift across sentry-sdk upgrades.
        from sentry_sdk.consts import DEFAULT_OPTIONS

        for key in privacy_sentry_options():
            assert key in DEFAULT_OPTIONS, f"unknown sentry_sdk option: {key}"
