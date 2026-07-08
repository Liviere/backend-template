"""Content-field cipher for core-owned models (privacy Faza 2).

Core code cannot import from ``app.*``, so this module re-implements the
minimal, wire-compatible subset of the app keyring
(``app/services/encryption.py``): versioned tokens
``enc.v1.<purpose>.<key_id>.<fernet>`` encrypted with the FIRST key of the
purpose's env keyring and decryptable with ANY key in the ring (rotation).
Tokens written here decrypt in the app layer and vice versa because the wire
format is identical; in-repo round-trip / rotation / fallback behaviour is
covered by ``tests/unit/services/test_content_cipher.py`` (cross-layer decrypt
against the real app keyring lives in the downstream app's own test suite —
core cannot import ``app.*``).

Purposes and env vars (mirror of the app registry):

    content     → CONTENT_ENCRYPTION_KEY   (no fallback)
    memory      → MEMORY_ENCRYPTION_KEY    (falls back to content)
    checkpoints → CHECKPOINT_ENCRYPTION_KEY (falls back to content)

Dual-read semantics match the app layer: plaintext passes through, tokens
decrypt, an undecryptable token fails soft (text → returned as-is, JSON →
``None``).
"""

import hashlib
import json
import logging
import os
from typing import Any, List, Optional

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

_TOKEN_PREFIX = "enc"
_TOKEN_VERSION = "v1"

#: purpose → (env var, fallback purpose)
_PURPOSES = {
    "content": ("CONTENT_ENCRYPTION_KEY", None),
    "memory": ("MEMORY_ENCRYPTION_KEY", "content"),
    "checkpoints": ("CHECKPOINT_ENCRYPTION_KEY", "content"),
}

_key_cache: dict[str, List[bytes]] = {}


class ContentCipherError(Exception):
    """Raised when encryption is requested but no key is configured."""


def reset_content_cipher() -> None:
    """Clear the key cache (tests / key rotation)."""
    _key_cache.clear()


def _key_id(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()[:8]


def _resolve_keys(purpose: str) -> tuple[str, List[bytes]]:
    """Return ``(effective_purpose, keys)`` following the fallback chain.

    Used by the ENCRYPT path: the effective purpose's FIRST key is the one
    new tokens are written with.
    """
    seen = set()
    current = purpose
    while current and current not in seen:
        seen.add(current)
        env_var, fallback = _PURPOSES[current]
        cached = _key_cache.get(current)
        if cached is not None:
            return current, cached
        raw = os.environ.get(env_var, "").strip()
        if raw:
            keys = [part.strip().encode() for part in raw.split(",") if part.strip()]
            _key_cache[current] = keys
            return current, keys
        current = fallback
    raise ContentCipherError(
        f"No encryption key configured for purpose {purpose!r} "
        f"(set {_PURPOSES[purpose][0]} or its fallback)"
    )


def _resolve_all_keys(purpose: str) -> List[bytes]:
    """All decrypt-candidate keys across the whole fallback chain (deduped).

    Used by the DECRYPT path. Unlike :func:`_resolve_keys` (which stops at the
    first purpose that has a key), this walks the ENTIRE chain and unions every
    ring — the dedicated purpose's keys first, then the fallback's. That is what
    keeps tokens written during the fallback window decryptable after a
    dedicated ``MEMORY_ENCRYPTION_KEY``/``CHECKPOINT_ENCRYPTION_KEY`` is added
    later: the content ring stays in the decrypt set instead of being dropped
    the moment the dedicated var is set.
    """
    out: List[bytes] = []
    seen: set[str] = set()
    current: Optional[str] = purpose
    while current and current not in seen:
        seen.add(current)
        env_var, fallback = _PURPOSES[current]
        raw = os.environ.get(env_var, "").strip()
        if raw:
            for part in raw.split(","):
                key = part.strip().encode()
                if key and key not in out:
                    out.append(key)
        current = fallback
    return out


def is_cipher_token(value: str) -> bool:
    """Heuristic mirror of the app-side ``is_encrypted_token``."""
    return value.startswith(f"{_TOKEN_PREFIX}.{_TOKEN_VERSION}.") or value.startswith(
        "gAAAA"
    )


def encrypt_text(value: str, *, purpose: str = "content") -> str:
    effective, keys = _resolve_keys(purpose)
    primary = keys[0]
    token = Fernet(primary).encrypt(value.encode()).decode()
    return f"{_TOKEN_PREFIX}.{_TOKEN_VERSION}.{effective}.{_key_id(primary)}.{token}"


def decrypt_text(token: str, *, purpose: str = "content") -> str:
    payload = token
    if token.startswith(f"{_TOKEN_PREFIX}.{_TOKEN_VERSION}."):
        parts = token.split(".", 4)
        if len(parts) == 5:
            payload = parts[4]
    keys = _resolve_all_keys(purpose)
    if not keys:
        raise ContentCipherError(
            f"No encryption key configured for purpose {purpose!r} "
            f"(set {_PURPOSES[purpose][0]} or its fallback)"
        )
    last_exc: Optional[Exception] = None
    for key in keys:
        try:
            return Fernet(key).decrypt(payload.encode()).decode()
        except Exception as exc:  # InvalidToken, ValueError
            last_exc = exc
    raise ContentCipherError(f"Decryption failed: {last_exc}")


# ---------------------------------------------------------------------------
# Field helpers (same contract as app.services.content_crypto)
# ---------------------------------------------------------------------------


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def instances_enc_enabled() -> bool:
    """user_agent_instances prompt/description/config/skills writes."""
    return _env_flag("ENCRYPT_INSTANCES_WRITES")


def memory_enc_enabled() -> bool:
    """LangGraph store value.content / value.topic writes."""
    return _env_flag("ENCRYPT_MEMORY_WRITES")


def qdrant_enc_enabled() -> bool:
    """Qdrant ``_text`` payload (+ artifact content metadata) writes."""
    return _env_flag("ENCRYPT_QDRANT_WRITES")


def enc_field(value: Optional[str], *, purpose: str = "content",
              enabled: bool) -> Optional[str]:
    if value is None or value == "" or not enabled:
        return value
    if is_cipher_token(value):
        return value
    return encrypt_text(value, purpose=purpose)


def dec_field(value: Optional[str], *, purpose: str = "content") -> Optional[str]:
    if not isinstance(value, str) or not is_cipher_token(value):
        return value
    try:
        return decrypt_text(value, purpose=purpose)
    except Exception as exc:
        logger.warning(
            "content_cipher.dec_field: undecryptable %s token (len=%d): %s",
            purpose, len(value), type(exc).__name__,
        )
        return value


def enc_json(obj: Any, *, purpose: str = "content", enabled: bool) -> Any:
    if obj is None or not enabled:
        return obj
    if isinstance(obj, str) and is_cipher_token(obj):
        return obj
    return encrypt_text(
        json.dumps(obj, ensure_ascii=False, default=str), purpose=purpose
    )


def dec_json(value: Any, *, purpose: str = "content") -> Any:
    if not isinstance(value, str) or not is_cipher_token(value):
        return value
    try:
        return json.loads(decrypt_text(value, purpose=purpose))
    except Exception as exc:
        logger.warning(
            "content_cipher.dec_json: undecryptable %s token (len=%d): %s",
            purpose, len(value), type(exc).__name__,
        )
        return None


# ---------------------------------------------------------------------------
# Write-clobber guard (privacy Faza 2, Finding #5) — twin of the app codec's
# app.services.content_crypto.assert_decryptable_json.
# ---------------------------------------------------------------------------


class UndecryptableToken(Exception):
    """A JSONB write was blocked to avoid clobbering recoverable ciphertext.

    ``dec_json`` fails soft to ``None``, so a read-modify-write loader that
    reads an *undecryptable* token rebuilds a partial value and reassigns,
    destroying still-recoverable ciphertext (Finding #5). The JSONB setters
    call :func:`assert_decryptable_json` first and raise this instead, aborting
    the transaction so the ciphertext survives. Mirror of the app-side class.
    """


def assert_decryptable_json(existing: Any, *, purpose: str = "content") -> None:
    """Guard a JSONB write against overwriting an undecryptable token.

    No-op unless ``existing`` (the currently stored raw column value) is a
    present token that fails to decrypt+parse right now — the exact condition
    under which :func:`dec_json` would return ``None`` and a read-modify-write
    cycle would clobber the ciphertext. Flag-independent (a plaintext overwrite
    destroys ciphertext just as a re-encrypt would).
    """
    if not isinstance(existing, str) or not is_cipher_token(existing):
        return
    try:
        json.loads(decrypt_text(existing, purpose=purpose))
    except Exception as exc:
        raise UndecryptableToken(purpose) from exc
