"""LangGraph checkpoint encryption (privacy Faza 2, Etap 7).

Wires the OSS ``EncryptedSerializer`` (langgraph-checkpoint) with a cipher
built on the already-installed ``cryptography`` package and the shared
purpose keyring (``CHECKPOINT_ENCRYPTION_KEY``, falling back to
``CONTENT_ENCRYPTION_KEY`` — see content_cipher). pycryptodome is NOT
installed, so ``EncryptedSerializer.from_pycryptodome_aes`` is unusable.

Flip safety: :func:`build_checkpoint_serializer` returns a serializer that
is ALWAYS able to decrypt (``loads_typed`` handles both ``msgpack`` legacy
plaintext rows — no ``+`` in the type tag — and ``msgpack+fernet-v1``
tokens), while ``dumps_typed`` encrypts only when
``ENCRYPT_CHECKPOINTS_WRITES`` is on. Turning the flag OFF therefore never
strands data: new writes are plaintext, old encrypted blobs keep loading.
"""

import logging
from typing import Any

from cryptography.fernet import Fernet, MultiFernet

logger = logging.getLogger(__name__)

_CIPHERNAME = "fernet-v1"  # must not contain '+', the serde type separator


class FernetKeyringCipher:
    """CipherProtocol implementation over the purpose keyring.

    ``encrypt`` uses the FIRST key of the checkpoints fallback chain (the
    dedicated ``CHECKPOINT_ENCRYPTION_KEY`` primary when set, else the content
    key). ``decrypt`` tries the WHOLE chain via MultiFernet (checkpoints ring +
    content ring), so both rotation (prepend a fresh key, redeploy) and the
    add-a-dedicated-key-later transition keep old blobs decrypting.
    """

    def __init__(self) -> None:
        from inference_core.services.content_cipher import (
            ContentCipherError,
            _resolve_all_keys,
        )

        # Decrypt with the WHOLE fallback chain (checkpoints ring + content
        # ring), so blobs written under the content fallback keep loading after
        # a dedicated CHECKPOINT_ENCRYPTION_KEY is added later. Encrypt uses the
        # first key — the dedicated checkpoints primary when set, else content.
        keys = _resolve_all_keys("checkpoints")
        if not keys:
            raise ContentCipherError(
                "No encryption key configured for purpose 'checkpoints' "
                "(set CHECKPOINT_ENCRYPTION_KEY or CONTENT_ENCRYPTION_KEY)"
            )
        self._ring = MultiFernet([Fernet(k) for k in keys])

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        return (_CIPHERNAME, self._ring.encrypt(plaintext))

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        if ciphername != _CIPHERNAME:
            raise ValueError(f"Unsupported checkpoint cipher: {ciphername!r}")
        return self._ring.decrypt(ciphertext)


def build_checkpoint_serializer():
    """Return the flip-safe serializer, or ``None`` when no key is configured.

    ``None`` (caller then constructs the saver with the default serde) is the
    ONLY intended silent fallback — the no-key case, where a deployment has no
    encryption key at all and the serializer is not yet needed.

    Any OTHER construction failure propagates deliberately. A blanket
    ``except`` here would fail OPEN: a malformed key (``Fernet(k)`` raises
    ``ValueError``, not ``ContentCipherError``) or a missing OSS serializer
    would silently degrade to a plaintext serde — writing checkpoints in the
    clear while the operator believes ``ENCRYPT_CHECKPOINTS_WRITES`` is active,
    AND stranding existing ``msgpack+fernet-v1`` blobs (the default serde
    raises ``NotImplementedError`` on their type tag). Failing loudly at
    checkpointer construction is the correct fail-CLOSED behaviour.
    """
    from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

    from inference_core.services.content_cipher import ContentCipherError, _env_flag

    try:
        cipher = FernetKeyringCipher()
    except ContentCipherError:
        logger.info(
            "No CHECKPOINT/CONTENT_ENCRYPTION_KEY configured — "
            "checkpointer runs without an encrypting serializer"
        )
        return None

    class _GatedEncryptedSerializer(EncryptedSerializer):
        """Encrypt-on-write only when the flag is on; always decrypt."""

        def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
            if not _env_flag("ENCRYPT_CHECKPOINTS_WRITES"):
                return self.serde.dumps_typed(obj)
            return super().dumps_typed(obj)

    return _GatedEncryptedSerializer(cipher)
