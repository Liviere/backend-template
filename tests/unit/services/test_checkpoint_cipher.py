"""Tests for the LangGraph checkpoint cipher (privacy Faza 2, Etap 7).

The two guarantees that matter most here:

* **Fail-closed construction** — a malformed key or a missing OSS serializer
  must raise, never silently degrade to a plaintext serde (which would write
  checkpoints in the clear while the operator believes encryption is on, and
  strand existing ``msgpack+fernet-v1`` blobs). Only the genuine *no-key* case
  returns ``None``.
* **Flip safety** — the gated serializer always decrypts (legacy plaintext rows
  and encrypted tokens alike) but encrypts new writes only behind
  ``ENCRYPT_CHECKPOINTS_WRITES``. Turning the flag off must never strand data.

Tests that build the serializer ``importorskip`` the OSS
``langgraph.checkpoint.serde.encrypted`` module so the suite is robust across
langgraph versions; the ``FernetKeyringCipher`` tests need only ``cryptography``.
"""

import pytest

from inference_core.services import checkpoint_cipher as ckc
from inference_core.services import content_cipher as cc
from inference_core.services.content_cipher import ContentCipherError

_ENV_VARS = (
    "CONTENT_ENCRYPTION_KEY",
    "MEMORY_ENCRYPTION_KEY",
    "CHECKPOINT_ENCRYPTION_KEY",
    "ENCRYPT_CHECKPOINTS_WRITES",
)


@pytest.fixture(autouse=True)
def _clean_cipher_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    cc.reset_content_cipher()
    yield
    cc.reset_content_cipher()


def _key() -> str:
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode()


# ---------------------------------------------------------------------------
# FernetKeyringCipher (no langgraph dependency)
# ---------------------------------------------------------------------------


def test_cipher_raises_without_any_key():
    with pytest.raises(ContentCipherError):
        ckc.FernetKeyringCipher()


def test_cipher_round_trip_with_dedicated_key(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_ENCRYPTION_KEY", _key())
    cipher = ckc.FernetKeyringCipher()
    name, ct = cipher.encrypt(b"payload-bytes")
    assert name == "fernet-v1"
    assert ct != b"payload-bytes"
    assert cipher.decrypt(name, ct) == b"payload-bytes"


def test_cipher_falls_back_to_content_key(monkeypatch):
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", _key())
    cipher = ckc.FernetKeyringCipher()  # uses the content ring via fallback
    name, ct = cipher.encrypt(b"x")
    assert cipher.decrypt(name, ct) == b"x"


def test_decrypt_rejects_unknown_ciphername(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_ENCRYPTION_KEY", _key())
    cipher = ckc.FernetKeyringCipher()
    with pytest.raises(ValueError):
        cipher.decrypt("aes-256-gcm", b"whatever")


def test_content_blob_decrypts_after_checkpoint_key_added(monkeypatch):
    """A blob written under the content fallback keeps loading once a dedicated
    CHECKPOINT_ENCRYPTION_KEY is added later (MultiFernet over the whole ring)."""
    monkeypatch.setenv("CONTENT_ENCRYPTION_KEY", _key())
    cipher_before = ckc.FernetKeyringCipher()
    name, ct = cipher_before.encrypt(b"legacy")

    monkeypatch.setenv("CHECKPOINT_ENCRYPTION_KEY", _key())
    cipher_after = ckc.FernetKeyringCipher()  # ring = checkpoint + content
    assert cipher_after.decrypt(name, ct) == b"legacy"


# ---------------------------------------------------------------------------
# build_checkpoint_serializer — fail-closed + no-key None
# ---------------------------------------------------------------------------


def test_build_returns_none_without_key():
    pytest.importorskip("langgraph.checkpoint.serde.encrypted")
    # The only intended silent fallback: no key at all -> default serde is used.
    assert ckc.build_checkpoint_serializer() is None


def test_build_fails_closed_on_malformed_key(monkeypatch):
    pytest.importorskip("langgraph.checkpoint.serde.encrypted")
    monkeypatch.setenv("CHECKPOINT_ENCRYPTION_KEY", "not-a-valid-fernet-key")
    # A malformed key raises (Fernet -> ValueError) and MUST propagate, rather
    # than being swallowed like the no-key ContentCipherError path.
    with pytest.raises(Exception) as excinfo:
        ckc.build_checkpoint_serializer()
    assert not isinstance(excinfo.value, ContentCipherError)


def test_build_returns_gated_serializer_with_key(monkeypatch):
    pytest.importorskip("langgraph.checkpoint.serde.encrypted")
    monkeypatch.setenv("CHECKPOINT_ENCRYPTION_KEY", _key())
    ser = ckc.build_checkpoint_serializer()
    assert ser is not None
    assert hasattr(ser, "dumps_typed") and hasattr(ser, "loads_typed")


# ---------------------------------------------------------------------------
# build_checkpoint_serializer — flip safety
# ---------------------------------------------------------------------------


def test_gated_serializer_flip_safety(monkeypatch):
    pytest.importorskip("langgraph.checkpoint.serde.encrypted")
    monkeypatch.setenv("CHECKPOINT_ENCRYPTION_KEY", _key())
    ser = ckc.build_checkpoint_serializer()
    assert ser is not None
    obj = {"messages": ["SENTINELSECRET"], "n": 1}

    # Flag OFF -> plaintext write (no '+' in the type tag), round-trips.
    monkeypatch.delenv("ENCRYPT_CHECKPOINTS_WRITES", raising=False)
    t_plain, b_plain = ser.dumps_typed(obj)
    assert "+" not in t_plain
    assert ser.loads_typed((t_plain, b_plain)) == obj

    # Flag ON -> encrypted write (type tag carries fernet-v1), round-trips.
    monkeypatch.setenv("ENCRYPT_CHECKPOINTS_WRITES", "true")
    t_enc, b_enc = ser.dumps_typed(obj)
    assert "fernet-v1" in t_enc
    assert b"SENTINELSECRET" not in b_enc  # payload is actually encrypted
    assert ser.loads_typed((t_enc, b_enc)) == obj

    # Turning the flag back OFF must NOT strand the encrypted blob (dual-read).
    monkeypatch.delenv("ENCRYPT_CHECKPOINTS_WRITES", raising=False)
    assert ser.loads_typed((t_enc, b_enc)) == obj
