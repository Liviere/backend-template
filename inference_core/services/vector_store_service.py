"""
Vector Store Service

High-level service for vector store operations.
Provides business logic layer over vector store providers.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.retrievers import BaseRetriever

from ..observability.metrics import (
    record_vector_ingestion,
    record_vector_search,
    update_vector_collection_stats,
)
from ..vectorstores.base import (
    BaseVectorStoreProvider,
    CollectionStats,
    VectorStoreDocument,
)
from ..vectorstores.factory import get_vector_store_provider

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Service layer for vector store operations.

    Provides high-level methods for document ingestion, search, and management.
    Handles caching, validation, and error handling.
    """

    def __init__(self):
        self._provider: Optional[BaseVectorStoreProvider] = None
        self._retrievers_cache: Dict[str, BaseRetriever] = {}

    @property
    def provider(self) -> Optional[BaseVectorStoreProvider]:
        """Return the already initialized provider (no implicit lazy init).

        Tests expect that absence of a configured provider results in explicit
        RuntimeError in public methods instead of silently creating one.
        Use `initialize_provider()` if you want to create it explicitly.
        """
        return self._provider

    def initialize_provider(
        self, force: bool = False
    ) -> Optional[BaseVectorStoreProvider]:
        """Explicitly initialize the provider using factory.

        Args:
            force: If True re-create provider even if it already exists.
        Returns:
            Provider instance (may be None if factory returns None)
        """
        if self._provider is None or force:
            self._provider = get_vector_store_provider()
        return self._provider

    @property
    def is_available(self) -> bool:
        """Check if vector store is available (no side effects)."""
        return self._provider is not None

    async def ensure_collection(self, collection: Optional[str] = None) -> bool:
        """
        Ensure a collection exists.

        Args:
            collection: Collection name (uses default if None)

        Returns:
            True if collection was created, False if it already existed

        Raises:
            RuntimeError: If vector store is not available
        """
        if not self.is_available:
            raise RuntimeError("Vector store is not available")

        collection = collection or self.provider.get_default_collection()
        return await self.provider.ensure_collection(collection)

    async def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
        collection: Optional[str] = None,
    ) -> List[str]:
        """
        Add texts to vector store.

        Args:
            texts: Sequence of texts to add
            metadatas: Optional metadata for each text
            ids: Optional IDs for each text
            collection: Collection name (uses default if None)

        Returns:
            List of document IDs that were added

        Raises:
            RuntimeError: If vector store is not available
            ValueError: If input validation fails
        """
        if not self.is_available:
            raise RuntimeError("Vector store is not available")

        if not texts:
            raise ValueError("No texts provided")

        # Validate batch size
        max_batch_size = self.provider.config.get("max_batch_size", 1000)
        if len(texts) > max_batch_size:
            raise ValueError(
                f"Batch size {len(texts)} exceeds maximum {max_batch_size}"
            )

        # Sanitize metadata
        if metadatas:
            metadatas = [self._sanitize_metadata(meta) for meta in metadatas]

        collection = collection or self.provider.get_default_collection()

        # Ensure collection exists
        await self.ensure_collection(collection)

        # Add texts with metrics
        start_time = time.time()
        try:
            doc_ids = await self.provider.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids,
                collection=collection,
            )

            # Record metrics
            duration = time.time() - start_time
            backend = (
                getattr(self.provider, "__class__", type(self.provider))
                .__name__.replace("Provider", "")
                .lower()
            )
            record_vector_ingestion(backend, collection, len(texts), duration)

            # Update collection stats
            try:
                stats = await self.get_collection_stats(collection)
                update_vector_collection_stats(backend, collection, stats.count)
            except Exception:
                # Don't fail if we can't update stats
                pass

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Vector ingestion failed after {duration:.2f}s: {e}")
            raise

        logger.info(
            f"Added {len(texts)} texts to collection '{collection}' in {duration:.2f}s"
        )
        return doc_ids

    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[VectorStoreDocument]:
        """
        Search for similar documents.

        Args:
            query: Query text
            k: Number of results to return
            collection: Collection name (uses default if None)
            filters: Optional filters to apply
            **kwargs: Additional search parameters

        Returns:
            List of similar documents with scores

        Raises:
            RuntimeError: If vector store is not available
            ValueError: If input validation fails
        """
        if not self.is_available:
            raise RuntimeError("Vector store is not available")

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if k <= 0:
            raise ValueError("k must be positive")

        collection = collection or self.provider.get_default_collection()

        # Sanitize filters
        if filters:
            filters = self._sanitize_metadata(filters)

        # Perform similarity search with metrics
        start_time = time.time()
        try:
            results = await self.provider.similarity_search(
                query=query,
                k=k,
                collection=collection,
                filters=filters,
                **kwargs,
            )

            # Record metrics
            duration = time.time() - start_time
            backend = (
                getattr(self.provider, "__class__", type(self.provider))
                .__name__.replace("Provider", "")
                .lower()
            )
            record_vector_search(backend, collection, duration, success=True)

        except Exception as e:
            duration = time.time() - start_time
            backend = (
                getattr(self.provider, "__class__", type(self.provider))
                .__name__.replace("Provider", "")
                .lower()
            )
            record_vector_search(backend, collection, duration, success=False)
            logger.error(f"Vector search failed after {duration:.2f}s: {e}")
            raise

        logger.debug(
            f"Found {len(results)} similar documents for query in collection '{collection}' in {duration:.2f}s"
        )
        return results

    async def get_retriever(
        self,
        collection: Optional[str] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> BaseRetriever:
        """
        Get a LangChain retriever instance.

        WARNING — field-encryption bypass: the underlying provider retriever
        does NOT dual-read (decrypt) the ``_text`` payload the way
        ``similarity_search`` does. With ``ENCRYPT_QDRANT_WRITES`` on, documents
        it yields would contain ciphertext. This method has NO production
        callers today; do not use it for retrieval over an encrypted collection
        without adding decryption. Prefer ``VectorStoreService.search`` /
        provider ``similarity_search``.

        Args:
            collection: Collection name (uses default if None)
            search_kwargs: Additional search parameters

        Returns:
            LangChain BaseRetriever instance

        Raises:
            RuntimeError: If vector store is not available
        """
        if not self.is_available:
            raise RuntimeError("Vector store is not available")

        collection = collection or self.provider.get_default_collection()
        cache_key = f"{collection}:{hash(str(sorted((search_kwargs or {}).items())))}"

        # Check cache first
        if cache_key in self._retrievers_cache:
            return self._retrievers_cache[cache_key]

        # Create new retriever
        retriever = await self.provider.as_retriever(
            collection=collection,
            search_kwargs=search_kwargs,
        )

        # Cache it
        self._retrievers_cache[cache_key] = retriever
        logger.debug(f"Created retriever for collection '{collection}'")

        return retriever

    async def get_collection_stats(
        self, collection: Optional[str] = None
    ) -> CollectionStats:
        """
        Get collection statistics.

        Args:
            collection: Collection name (uses default if None)

        Returns:
            Collection statistics

        Raises:
            RuntimeError: If vector store is not available
        """
        if not self.is_available:
            raise RuntimeError("Vector store is not available")

        collection = collection or self.provider.get_default_collection()
        return await self.provider.collection_stats(collection)

    async def delete_collection(self, collection: str) -> bool:
        """
        Delete a collection.

        Args:
            collection: Collection name

        Returns:
            True if collection was deleted, False if it didn't exist

        Raises:
            RuntimeError: If vector store is not available
        """
        if not self.is_available:
            raise RuntimeError("Vector store is not available")

        # Clear related cache entries
        cache_keys_to_remove = [
            key
            for key in self._retrievers_cache.keys()
            if key.startswith(f"{collection}:")
        ]
        for key in cache_keys_to_remove:
            del self._retrievers_cache[key]

        result = await self.provider.delete_collection(collection)

        if result:
            logger.info(f"Deleted collection '{collection}'")
        else:
            logger.warning(f"Collection '{collection}' did not exist")

        return result

    async def health_check(self) -> Dict[str, Any]:
        """
        Check vector store health.

        Returns:
            Health status information
        """
        if not self.is_available:
            return {
                "status": "disabled",
                "backend": None,
                "message": "Vector store is not configured or available",
            }

        return await self.provider.health_check()

    async def list_documents(
        self,
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0,
        order_by: Optional[str] = None,
        order: str = "desc",
        include_scores: bool = False,
    ) -> tuple[List[VectorStoreDocument], int]:
        """
        List documents by metadata filters without a text query.

        Args:
            collection: Collection name (uses default if None)
            filters: Optional metadata filters to apply
            limit: Maximum number of results to return (1-1000)
            offset: Number of results to skip (for pagination)
            order_by: Field to order by (e.g., 'created_at', provider-specific)
            order: Sort order ('asc' or 'desc')
            include_scores: Whether to include scores (may not be supported)

        Returns:
            Tuple of (documents, total_count) where total_count is the total matching items

        Raises:
            RuntimeError: If vector store is not available
            ValueError: If input validation fails
        """
        if not self.is_available:
            raise RuntimeError("Vector store is not available")

        # Validate limit and offset
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        if offset < 0:
            raise ValueError("offset must be >= 0")

        # Validate order
        if order not in ("asc", "desc"):
            raise ValueError("order must be 'asc' or 'desc'")

        collection = collection or self.provider.get_default_collection()

        # Sanitize filters
        if filters:
            filters = self._sanitize_metadata(filters)

        # Perform listing with metrics
        start_time = time.time()
        try:
            documents, total_count = await self.provider.list_documents(
                collection=collection,
                filters=filters,
                limit=limit,
                offset=offset,
                order_by=order_by,
                order=order,
                include_scores=include_scores,
            )

            # Record metrics (reuse search metric with success=True)
            duration = time.time() - start_time
            backend = (
                getattr(self.provider, "__class__", type(self.provider))
                .__name__.replace("Provider", "")
                .lower()
            )
            record_vector_search(backend, collection, duration, success=True)

        except Exception as e:
            duration = time.time() - start_time
            backend = (
                getattr(self.provider, "__class__", type(self.provider))
                .__name__.replace("Provider", "")
                .lower()
            )
            record_vector_search(backend, collection, duration, success=False)
            logger.error(f"Vector listing failed after {duration:.2f}s: {e}")
            raise

        logger.debug(
            f"Listed {len(documents)} documents (total: {total_count}) from collection '{collection}' in {duration:.2f}s"
        )
        return (documents, total_count)

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata to ensure it's safe for storage.

        Args:
            metadata: Raw metadata dict

        Returns:
            Sanitized metadata dict
        """
        sanitized = {}

        for key, value in metadata.items():
            # Skip private/internal keys
            if key.startswith("_"):
                continue

            # Ensure key is string and reasonable length
            if not isinstance(key, str) or len(key) > 100:
                continue

            # Handle different value types
            if isinstance(value, (str, int, float, bool)):
                # Truncate long strings
                if isinstance(value, str) and len(value) > 1000:
                    value = value[:1000]
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                # Only allow simple lists of basic types
                if all(isinstance(item, (str, int, float, bool)) for item in value):
                    sanitized[key] = list(value)
            elif isinstance(value, dict):
                # Recursively sanitize nested dicts (one level only)
                nested = {}
                for nested_key, nested_value in value.items():
                    if isinstance(nested_key, str) and isinstance(
                        nested_value, (str, int, float, bool)
                    ):
                        nested[nested_key] = nested_value
                if nested:
                    sanitized[key] = nested

        return sanitized

    def clear_cache(self):
        """Clear internal caches"""
        self._retrievers_cache.clear()
        logger.debug("Cleared vector store service cache")


# Global service instance (process-wide singleton)
_vector_store_service: Optional[VectorStoreService] = None
_vector_store_service_lock = threading.Lock()


def get_vector_store_service() -> VectorStoreService:
    """Get the global vector store service instance.

    Thread-safe. Under a threaded Celery worker pool (``--pool=threads``) many
    threads can race on first access. Two guarantees prevent a half-initialized
    service from ever being observed:

    * **Build-before-publish**: the service is fully initialized into a local
      variable and only then assigned to the module global, so a concurrent
      caller never sees an instance whose ``provider`` is still ``None``.
    * **Double-checked locking**: the steady-state path stays lock-free; only
      the first build (and a transient-failure retry) takes the lock.

    If a previous build left the provider unset because the underlying factory
    hit a transient error (the factory no longer caches such ``None`` results),
    a subsequent call retries provider initialization instead of staying
    permanently unavailable for the process lifetime.
    """
    global _vector_store_service
    service = _vector_store_service
    if service is not None and service.provider is not None:
        return service
    with _vector_store_service_lock:
        if _vector_store_service is None:
            service = VectorStoreService()
            service.initialize_provider()
            _vector_store_service = service
        elif _vector_store_service.provider is None:
            # Earlier build got a None provider (e.g. a transient factory error).
            # Retry so a one-off failure does not pin the worker as unavailable.
            _vector_store_service.initialize_provider(force=True)
        return _vector_store_service
