"""Tests for the purpose-keyed content cipher (privacy Faza 2).

Covers the security-critical guarantees this module carries:

* encrypt -> decrypt round-trips and the ``enc.v1.<purpose>.<key_id>.<fernet>``
  token shape;
* the purpose fallback chain (``memory``/``checkpoints`` -> ``content``) and,
  crucially, that a token written during the *fallback window* stays decryptable
  after a dedicated key is introduced later (``_resolve_all_keys`` unions the
  whole chain);
* key rotation (prepend a fresh primary; old ciphertext still decrypts);
* the field/JSON helpers' dual-read + fail-soft semantics (plaintext passes
  through, undecryptable text -> returned as-is, undecryptable JSON -> ``None``);
* the ``assert_decryptable_json`` write-clobber guard (Finding #5), including the
  decrypt-ok-but-not-JSON branch;
* wire-compat with a bare app-written Fernet token (``gAAAA…``).

Every test manages the encryption env vars explicitly and resets the module key
cache, so the suite is hermetic regardless of the ambient environment.
"""

import pytest

from inference_core.services import content_cipher as cc

_ENV_VARS = (
    "CONTENT_ENCRYPTION_KEY",
    "MEMORY_ENCRYPTION_KEY",
    "CHECKPOINT_ENCRYPTION_KEY",
    "ENCRYPT_INSTANCES_WRITES",
    "ENCRYPT_MEMORY_WRITES",
    "ENCRYPT_QDRANT_WRITES",
    "ENCRYPT_CHECKPOINTS_WRITES",
)


@pytest.fixture(autouse=True)
def _clean_cipher_env(monkeypatch):
    """Start every test from a known-empty keyring + cache."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    cc.reset_content_cipher()
    yield
    cc.reset_content_cipher()


def _key() -> str:
    """A fresh, valid Fernet key as the module expects it in the env ring."""
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode()


def _key_id_of(token: str) -> str:
    # enc.v1.<purpose>.<key_id>.<fernet…>  (the fernet body has no '.')
    return token.split(".", 4)[3]


# ---------------------------------------------------------------------------
# Token recognition
# ---------------------------------------------------------------------------


def test_is_cipher_token_recognises_versioned_and_raw():
    assert cc.is_cipher_token("enc.v1.content.abcd1234.gAAAAxyz")
    assert cc.is_cipher_token("gAAAAraw_fernet_token")
    assert not cc.is_cipher_token("plain text")
    assert not cc.is_cipher_token("")


# ---------------------------------------------------------------------------
# Round-trip + key resolution
# ---------------------------------------------------------------------------


def test_round_trip_content(monkeypatch):
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", _key())
    cc.reset_content_cipher()
    token = cc.encrypt_text("hello world")
    assert token.startswith("enc.v1.content.")
    assert cc.is_cipher_token(token)
    assert token != "hello world"
    assert cc.decrypt_text(token) == "hello world"


def test_encrypt_without_key_raises():
    # 'content' has no fallback purpose — no key means no encryption possible.
    with pytest.raises(cc.ContentCipherError):
        cc.encrypt_text("x")


def test_decrypt_without_key_raises():
    with pytest.raises(cc.ContentCipherError):
        cc.decrypt_text("enc.v1.content.abcd1234.gAAAAxyz")


def test_memory_purpose_falls_back_to_content(monkeypatch):
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", _key())
    cc.reset_content_cipher()
    token = cc.encrypt_text("secret", purpose="memory")
    # Effective purpose is 'content' because no MEMORY key is configured.
    assert token.startswith("enc.v1.content.")
    assert cc.decrypt_text(token, purpose="memory") == "secret"


def test_dedicated_memory_key_is_used_when_set(monkeypatch):
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", _key())
    monkeypatch.setenv("MEMORY_ENCRYPTION_KEY", _key())
    cc.reset_content_cipher()
    token = cc.encrypt_text("m", purpose="memory")
    assert token.startswith("enc.v1.memory.")
    assert cc.decrypt_text(token, purpose="memory") == "m"


def test_fallback_window_token_decrypts_after_dedicated_key_added(monkeypatch):
    """The transition guarantee: a memory token written while only the content
    key existed must still decrypt once a dedicated MEMORY key is introduced."""
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", _key())
    cc.reset_content_cipher()
    token = cc.encrypt_text("legacy-memory", purpose="memory")
    assert token.startswith("enc.v1.content.")  # written under the fallback

    # A dedicated memory key is added later.
    monkeypatch.setenv("MEMORY_ENCRYPTION_KEY", _key())
    cc.reset_content_cipher()

    # _resolve_all_keys unions memory ring + content ring, so it still decrypts.
    assert cc.decrypt_text(token, purpose="memory") == "legacy-memory"


def test_key_rotation_old_and_new_both_decrypt(monkeypatch):
    old = _key()
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", old)
    cc.reset_content_cipher()
    token_old = cc.encrypt_text("data")

    # Rotate: prepend a fresh primary, keep the old key for decrypt.
    new = _key()
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", f"{new},{old}")
    cc.reset_content_cipher()
    token_new = cc.encrypt_text("data")

    # New writes use the new primary (distinct key id from the old token).
    assert _key_id_of(token_old) != _key_id_of(token_new)
    # Both decrypt under the rotated ring.
    assert cc.decrypt_text(token_old) == "data"
    assert cc.decrypt_text(token_new) == "data"


def test_decrypt_with_wrong_key_raises(monkeypatch):
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", _key())
    cc.reset_content_cipher()
    token = cc.encrypt_text("x")
    # Replace the ring with an unrelated key only.
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", _key())
    cc.reset_content_cipher()
    with pytest.raises(cc.ContentCipherError):
        cc.decrypt_text(token)


def test_decrypt_accepts_bare_app_fernet_token(monkeypatch):
    """Wire-compat: the app layer may store a bare Fernet token (no enc.v1
    wrapper). It must dual-read here."""
    from cryptography.fernet import Fernet

    key = _key()
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", key)
    cc.reset_content_cipher()
    raw = Fernet(key.encode()).encrypt(b"app-written").decode()
    assert raw.startswith("gAAAA")
    assert cc.is_cipher_token(raw)
    assert cc.decrypt_text(raw) == "app-written"
    assert cc.dec_field(raw) == "app-written"


# ---------------------------------------------------------------------------
# enc_field / dec_field
# ---------------------------------------------------------------------------


def test_enc_field_disabled_passthrough(monkeypatch):
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", _key())
    cc.reset_content_cipher()
    assert cc.enc_field("plain", enabled=False) == "plain"


def test_enc_field_none_and_empty_passthrough():
    assert cc.enc_field(None, enabled=True) is None
    assert cc.enc_field("", enabled=True) == ""


def test_enc_field_encrypts_when_enabled(monkeypatch):
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", _key())
    cc.reset_content_cipher()
    out = cc.enc_field("plain", enabled=True)
    assert cc.is_cipher_token(out)
    assert cc.dec_field(out) == "plain"


def test_enc_field_does_not_double_encrypt(monkeypatch):
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", _key())
    cc.reset_content_cipher()
    once = cc.enc_field("plain", enabled=True)
    twice = cc.enc_field(once, enabled=True)
    assert once == twice  # already a token -> returned unchanged


def test_dec_field_passthrough_non_token_and_non_str():
    assert cc.dec_field("just text") == "just text"
    assert cc.dec_field(None) is None
    assert cc.dec_field(123) == 123


def test_dec_field_undecryptable_token_fails_soft(monkeypatch):
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", _key())
    cc.reset_content_cipher()
    bogus = "enc.v1.content.deadbeef.gAAAAthis-is-not-valid"
    # Fails soft: returns the token unchanged rather than raising or losing it.
    assert cc.dec_field(bogus) == bogus


# ---------------------------------------------------------------------------
# enc_json / dec_json
# ---------------------------------------------------------------------------


def test_enc_dec_json_round_trip(monkeypatch):
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", _key())
    cc.reset_content_cipher()
    obj = {"a": 1, "b": ["x", "y"], "n": None}
    token = cc.enc_json(obj, enabled=True)
    assert cc.is_cipher_token(token)
    assert cc.dec_json(token) == obj


def test_enc_json_disabled_passthrough():
    obj = {"a": 1}
    assert cc.enc_json(obj, enabled=False) is obj


def test_enc_json_none_passthrough():
    assert cc.enc_json(None, enabled=True) is None


def test_dec_json_passthrough_non_token():
    assert cc.dec_json({"a": 1}) == {"a": 1}
    assert cc.dec_json(None) is None


def test_dec_json_undecryptable_returns_none(monkeypatch):
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", _key())
    cc.reset_content_cipher()
    bogus = "enc.v1.content.deadbeef.gAAAAnope"
    # JSON reads fail soft to None (the setters guard against clobbering — below).
    assert cc.dec_json(bogus) is None


# ---------------------------------------------------------------------------
# assert_decryptable_json — write-clobber guard (Finding #5)
# ---------------------------------------------------------------------------


def test_assert_decryptable_json_noop_for_none_and_plaintext():
    cc.assert_decryptable_json(None)
    cc.assert_decryptable_json({"a": 1})
    cc.assert_decryptable_json("plain string")


def test_assert_decryptable_json_ok_for_valid_token(monkeypatch):
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", _key())
    cc.reset_content_cipher()
    token = cc.enc_json({"a": 1}, enabled=True)
    cc.assert_decryptable_json(token)  # decrypts + parses -> no raise


def test_assert_decryptable_json_raises_on_undecryptable(monkeypatch):
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", _key())
    cc.reset_content_cipher()
    bogus = "enc.v1.content.deadbeef.gAAAAnope"
    with pytest.raises(cc.UndecryptableToken):
        cc.assert_decryptable_json(bogus)


def test_assert_decryptable_json_raises_when_decrypts_but_not_json(monkeypatch):
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", _key())
    cc.reset_content_cipher()
    # A valid content token whose plaintext is NOT valid JSON.
    token = cc.encrypt_text("this is not json")
    with pytest.raises(cc.UndecryptableToken):
        cc.assert_decryptable_json(token)


# ---------------------------------------------------------------------------
# Flag helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "val,expected",
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("", False),
    ],
)
def test_env_flag_parsing(monkeypatch, val, expected):
    monkeypatch.setenv("ENCRYPT_INSTANCES_WRITES", val)
    assert cc.instances_enc_enabled() is expected


def test_flag_helpers_default_false():
    assert cc.instances_enc_enabled() is False
    assert cc.memory_enc_enabled() is False
    assert cc.qdrant_enc_enabled() is False
