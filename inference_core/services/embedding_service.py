"""Embedding Service Abstraction.

Unified interface for generating text embeddings with two backends:

- **local**: Delegates to a dedicated Celery prefork worker running
  SentenceTransformer. The API process never loads the model.
- **remote**: Uses LangChain embedding classes (OpenAI, Gemini, DeepInfra, Ollama)
  configured in the ``embeddings:`` section of ``llm_config.yaml``.
- **fake**: Generates deterministic in-process vectors for no-cost tests.

The backend is selected by the ``EMBEDDING_BACKEND`` environment variable
(``local``, ``remote``, or ``fake``).

Consumers: ``AgentService._init_embeddings()``, ``QdrantProvider``,
``/api/v1/embeddings/generate`` endpoint.
"""

import asyncio
import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from inference_core.core.config import Settings, get_settings
from inference_core.llm.config import ModelProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class BaseEmbeddingBackend(ABC):
    """Protocol for embedding backends."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings synchronously. Safe to call from sync context."""
        ...

    @abstractmethod
    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings asynchronously."""
        ...

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding vector dimensionality for this backend."""
        ...


# ---------------------------------------------------------------------------
# Local backend — Celery prefork worker
# ---------------------------------------------------------------------------


class LocalCeleryBackend(BaseEmbeddingBackend):
    """Sends embedding work to a Celery prefork worker via the ``embeddings`` queue.

    The API process never imports or loads SentenceTransformer.
    """

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.embedding_local_model
        self._timeout = settings.embedding_local_timeout
        self._fallback_dim = settings.vector_dim
        self._cached_dim: Optional[int] = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        from inference_core.celery.celery_main import celery_app

        result = celery_app.send_task(
            "embeddings.generate",
            kwargs={"texts": texts, "model_name": self._model_name},
            queue="embeddings",
        )
        # disable_sync_subtasks=False: the default worker uses thread-pool
        # concurrency, not prefork, so waiting on a result from a different
        # queue (embeddings) is safe and will not deadlock.
        return result.get(timeout=self._timeout, disable_sync_subtasks=False)

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_texts, texts)

    def get_dimension(self) -> int:
        if self._cached_dim is not None:
            return self._cached_dim
        try:
            probe = self.embed_texts(["dimension probe"])
            self._cached_dim = len(probe[0])
        except Exception:
            logger.warning(
                "Dimension probe failed, using configured vector_dim=%d",
                self._fallback_dim,
            )
            self._cached_dim = self._fallback_dim
        return self._cached_dim


# ---------------------------------------------------------------------------
# Remote backend — LangChain embedding providers
# ---------------------------------------------------------------------------


class RemoteLangChainBackend(BaseEmbeddingBackend):
    """Uses LangChain embedding classes to call external API providers."""

    def __init__(self, settings: Settings) -> None:
        from inference_core.llm.config import get_llm_config

        llm_config = get_llm_config()
        embed_config = llm_config.get_embedding_config("default")
        if embed_config is None:
            raise ValueError(
                "EMBEDDING_BACKEND=remote but no 'embeddings.default' section "
                "found in llm_config.yaml"
            )

        self._embeddings = self._create_langchain_embeddings(
            embed_config, llm_config.providers
        )
        self._dim: Optional[int] = embed_config.dimensions

    @staticmethod
    def _create_langchain_embeddings(
        embed_config: Any, providers: dict[str, Any]
    ) -> Any:
        """Build the provider-specific embeddings client from shared config.

        The remote embeddings path mirrors chat model provider resolution so the
        same provider enum and top-level YAML defaults apply in both places.
        """
        provider = RemoteLangChainBackend._coerce_provider(embed_config.provider)
        model = embed_config.model

        if provider == ModelProvider.OPENAI:
            from langchain_openai import OpenAIEmbeddings

            provider_cfg = RemoteLangChainBackend._get_provider_config(
                providers, provider
            )
            api_key_env = RemoteLangChainBackend._resolve_api_key_env(
                embed_config,
                provider_cfg,
                default_env="OPENAI_API_KEY",
            )
            kwargs: dict[str, Any] = {
                "model": model,
                "api_key": os.getenv(api_key_env),
            }
            if embed_config.dimensions:
                kwargs["dimensions"] = embed_config.dimensions
            if embed_config.base_url:
                kwargs["base_url"] = embed_config.base_url
            return OpenAIEmbeddings(**kwargs)

        if provider == ModelProvider.GEMINI:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            provider_cfg = RemoteLangChainBackend._get_provider_config(
                providers, provider
            )
            api_key_env = RemoteLangChainBackend._resolve_api_key_env(
                embed_config,
                provider_cfg,
                default_env="GOOGLE_API_KEY",
            )
            return GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=os.getenv(api_key_env),
            )

        if provider == ModelProvider.OLLAMA:
            from langchain_ollama import OllamaEmbeddings

            provider_cfg = RemoteLangChainBackend._get_provider_config(
                providers, provider
            )
            base_url = embed_config.base_url or provider_cfg.get(
                "base_url", "http://localhost:11434"
            )
            return OllamaEmbeddings(model=model, base_url=base_url)

        if provider == ModelProvider.DEEPINFRA:
            from langchain_community.embeddings.deepinfra import DeepInfraEmbeddings

            provider_cfg = RemoteLangChainBackend._get_provider_config(
                providers, provider
            )
            api_key_env = RemoteLangChainBackend._resolve_api_key_env(
                embed_config,
                provider_cfg,
                default_env="DEEPINFRA_API_TOKEN",
            )
            return DeepInfraEmbeddings(
                model_id=model,
                deepinfra_api_token=os.getenv(api_key_env),
            )

        if provider == ModelProvider.FIREWORKS:
            from langchain_fireworks import FireworksEmbeddings

            provider_cfg = RemoteLangChainBackend._get_provider_config(
                providers, provider
            )
            api_key_env = RemoteLangChainBackend._resolve_api_key_env(
                embed_config,
                provider_cfg,
                default_env="FIREWORKS_API_KEY",
            )
            return FireworksEmbeddings(
                model=model,
                fireworks_api_key=os.getenv(api_key_env),
            )

        raise ValueError(f"Unsupported embedding provider: {provider}")

    @staticmethod
    def _coerce_provider(provider: Any) -> ModelProvider:
        """Normalize provider values so tests and config use one code path.

        Runtime config should already deliver ``ModelProvider`` values, but
        coercion here keeps mocks and legacy string-based callers compatible.
        """
        if isinstance(provider, ModelProvider):
            return provider
        try:
            return ModelProvider(str(provider))
        except ValueError as exc:
            raise ValueError(f"Unsupported embedding provider: {provider}") from exc

    @staticmethod
    def _get_provider_config(
        providers: dict[str, Any], provider: ModelProvider
    ) -> dict[str, Any]:
        """Read provider defaults without coupling to the raw storage type.

        The YAML loader stores providers as plain dicts today, but centralizing
        access here makes the embedding factory resilient to future refactors.
        """
        provider_cfg = providers.get(provider.value, {})
        if hasattr(provider_cfg, "model_dump"):
            return provider_cfg.model_dump()
        return dict(provider_cfg)

    @staticmethod
    def _resolve_api_key_env(
        embed_config: Any,
        provider_cfg: dict[str, Any],
        default_env: str,
    ) -> str:
        """Resolve credentials the same way chat model configs do.

        Embedding configs can override the env var explicitly, otherwise they
        inherit the provider-level default from ``providers:`` in YAML.
        """
        return embed_config.api_key_env or provider_cfg.get("api_key_env", default_env)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self._embeddings.aembed_documents(texts)

    def get_dimension(self) -> int:
        if self._dim is not None:
            return self._dim
        probe = self.embed_texts(["dimension probe"])
        self._dim = len(probe[0])
        return self._dim


# ---------------------------------------------------------------------------
# Fake backend — deterministic no-cost embeddings
# ---------------------------------------------------------------------------


class FakeDeterministicBackend(BaseEmbeddingBackend):
    """Generate stable in-process vectors for no-cost test environments.

    WHY: E2E, performance, and security tests need vector and memory flows to
    behave like production without requiring local model workers or remote
    embedding APIs.
    """

    def __init__(self, settings: Settings) -> None:
        self._dim = settings.vector_dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    def get_dimension(self) -> int:
        return self._dim

    def _embed_one(self, text: str) -> list[float]:
        """Create a deterministic unit-length vector from text content.

        WHY: Hash-derived vectors are repeatable across processes, which keeps
        vector-store assertions stable while avoiding heavyweight dependencies.
        """
        values: list[float] = []
        counter = 0
        while len(values) < self._dim:
            digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            for offset in range(0, len(digest), 4):
                if len(values) >= self._dim:
                    break
                raw = int.from_bytes(digest[offset : offset + 4], "big")
                values.append((raw / 2**32) * 2.0 - 1.0)
            counter += 1

        norm = sum(value * value for value in values) ** 0.5
        if norm == 0:
            return values
        return [value / norm for value in values]


# ---------------------------------------------------------------------------
# Unified service
# ---------------------------------------------------------------------------


class EmbeddingService:
    """Unified embedding service consumed by AgentService, QdrantProvider,
    and the embeddings API endpoint.

    Provides both sync and async interfaces, plus a sync ``embed_fn`` callable
    compatible with LangGraph store ``index={"embed": ..., "dims": ...}``.
    """

    def __init__(self, backend: Optional[BaseEmbeddingBackend] = None) -> None:
        if backend is not None:
            self._backend = backend
        else:
            settings = get_settings()
            if settings.embedding_backend == "local":
                self._backend = LocalCeleryBackend(settings)
            elif settings.embedding_backend == "remote":
                self._backend = RemoteLangChainBackend(settings)
            elif settings.embedding_backend == "fake":
                self._backend = FakeDeterministicBackend(settings)
            else:
                raise ValueError(
                    f"Unknown EMBEDDING_BACKEND: {settings.embedding_backend}"
                )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Synchronous batch embedding."""
        return self._backend.embed_texts(texts)

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        """Async batch embedding."""
        return await self._backend.aembed_texts(texts)

    def get_embed_fn(self) -> callable:
        """Return a sync callable ``(list[str]) -> list[list[float]]``.

        This is the function signature LangGraph stores expect in their
        ``index={"embed": ..., "dims": ...}`` parameter.

        Decrypt-before-embed (privacy Faza 2): the store extracts indexed
        fields from the STORED value, which may be an ``enc.v1.…`` token
        when memory encryption is on — embedding ciphertext would destroy
        semantic search. The wrapper dual-reads every text (tokens →
        plaintext, plaintext → passthrough, never raises), which keeps BOTH
        the put()-side document indexing and the search()-side query
        embedding correct: queries are plaintext and pass through untouched.
        """
        from inference_core.services.content_cipher import dec_field

        raw_embed = self._backend.embed_texts

        def _decrypting_embed(texts: list[str]) -> list[list[float]]:
            return raw_embed(
                [dec_field(t, purpose="memory") if isinstance(t, str) else t
                 for t in texts]
            )

        return _decrypting_embed

    def get_dimension(self) -> int:
        """Return the embedding vector dimensionality."""
        return self._backend.get_dimension()

    def embed_query(self, text: str) -> list[float]:
        """Single-text embedding (convenience for search queries)."""
        return self._backend.embed_texts([text])[0]

    async def aembed_query(self, text: str) -> list[float]:
        """Async single-text embedding."""
        result = await self._backend.aembed_texts([text])
        return result[0]


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global EmbeddingService singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def clear_embedding_service_cache() -> None:
    """Reset the singleton (for testing)."""
    global _embedding_service
    _embedding_service = None
