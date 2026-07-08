#!/usr/bin/env python3
"""
Migrate flat-namespace memories to CoALA category-based namespaces.

This script reads all memories from the legacy namespace (user_id, "memories")
and re-writes them into CoALA-structured namespaces:
  - (user_id, "semantic")       for preferences, facts, goals, general
  - (user_id, "episodic")       for context, session_summary, interaction
  - (user_id, "procedural")     for instructions, workflow, skill

Usage:
    # Dry-run (preview what would be migrated, no writes)
    python scripts/migrate_memory_to_coala.py --dry-run

    # Execute migration
    python scripts/migrate_memory_to_coala.py

    # Specify custom legacy namespace
    python scripts/migrate_memory_to_coala.py --legacy-namespace memories

Environment:
    Reads DATABASE_URL from .env (or SQLALCHEMY_DATABASE_URI).
    Requires the same embedding model used by the application.

Source: CoALA whitepaper – arxiv:2309.02427
"""

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_core.services.agent_memory_service import (
    INDEXED_FIELDS,
    MEMORIES_STORE_NAME,
    MemoryCategory,
    MemoryNamespaceBuilder,
    get_category_for_type,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_store(db_url: str):
    """Initialize LangGraph Store from database URL with embeddings."""
    from sentence_transformers import SentenceTransformer

    from inference_core.core.config import get_settings

    settings = get_settings()
    model_name = settings.vector_embedding_model
    model = SentenceTransformer(model_name)
    dims = settings.vector_dim

    sample = model.encode(["test"], convert_to_tensor=False)
    try:
        if hasattr(sample, "shape") and len(getattr(sample, "shape")) >= 2:
            dims = int(sample.shape[1])
        elif isinstance(sample, (list, tuple)) and sample:
            dims = len(sample[0])
    except Exception:
        pass

    def embed_fn(texts: list[str]) -> list[list[float]]:
        # Decrypt-before-embed: stored content/topic may be enc.v1 tokens.
        from inference_core.services.content_cipher import dec_field

        plain = [dec_field(t, purpose="memory") if isinstance(t, str) else t
                 for t in texts]
        return model.encode(plain, convert_to_tensor=False).tolist()

    index_config = {"embed": embed_fn, "dims": dims}

    if "sqlite" in db_url:
        from langgraph.store.sqlite import SqliteStore

        store = SqliteStore.from_conn_string(db_url, index=index_config)
    elif "postgresql" in db_url:
        from langgraph.store.postgres import PostgresStore

        store = PostgresStore.from_conn_string(db_url, index=index_config)
    else:
        raise ValueError(f"Unsupported database URL for migration: {db_url}")

    if hasattr(store, "__enter__"):
        store.__enter__()
    if hasattr(store, "setup"):
        store.setup()

    return store


def discover_user_ids(store, legacy_ns_name: str) -> list[str]:
    """Discover user IDs by listing namespaces that match the legacy pattern.

    Since LangGraph stores don't have a direct list_namespaces for all users,
    we do a broad search with empty query and collect unique user_id prefixes.
    """
    # This is a heuristic; for large stores, consider querying the DB directly.
    # We'll try searching with an empty/wildcard query in the legacy namespace.
    logger.info("Discovering user IDs from legacy namespace '%s'...", legacy_ns_name)

    # Try to list all items with list_namespaces if available
    if hasattr(store, "list_namespaces"):
        try:
            namespaces = store.list_namespaces(prefix=(), suffix=(legacy_ns_name,))
            user_ids = list({ns[0] for ns in namespaces if len(ns) >= 2})
            logger.info("Found %d user IDs via list_namespaces", len(user_ids))
            return user_ids
        except Exception as e:
            logger.warning("list_namespaces failed: %s, falling back", e)

    logger.warning(
        "Could not auto-discover user IDs. "
        "Pass --user-ids explicitly or check store backend."
    )
    return []


def migrate_user(
    store,
    user_id: str,
    legacy_ns_name: str,
    ns_builder: MemoryNamespaceBuilder,
    dry_run: bool = True,
) -> dict[str, int]:
    """Migrate all memories for one user from legacy to CoALA namespaces.

    Returns dict of {category: count_migrated}.
    """
    legacy_ns = (user_id, legacy_ns_name)
    stats: dict[str, int] = {cat.value: 0 for cat in MemoryCategory}

    # Fetch all items from legacy namespace
    try:
        results = store.search(legacy_ns, query="", limit=1000)
    except TypeError:
        results = store.search(legacy_ns, query=None, limit=1000)

    items = list(results or [])
    logger.info(
        "User %s: found %d items in legacy namespace %s",
        user_id,
        len(items),
        legacy_ns,
    )

    for item in items:
        key = getattr(item, "key", getattr(item, "id", None))
        value = getattr(item, "value", {})
        if not key or not value:
            continue

        memory_type = value.get("memory_type", "general")
        category = get_category_for_type(memory_type)

        # Enrich with category field if missing
        if "memory_category" not in value:
            value["memory_category"] = category.value

        target_ns = ns_builder.namespace_for(user_id, category)

        if dry_run:
            logger.info(
                "  [DRY-RUN] Would migrate key=%s type=%s → %s (ns=%s)",
                key,
                memory_type,
                category.value,
                target_ns,
            )
        else:
            try:
                store.put(target_ns, key, value, index=INDEXED_FIELDS)
                logger.debug(
                    "  Migrated key=%s → %s",
                    key,
                    target_ns,
                )
            except Exception as e:
                logger.error("  Failed to migrate key=%s: %s", key, e)
                continue

        stats[category.value] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate flat-namespace memories to CoALA categories."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without writing data.",
    )
    parser.add_argument(
        "--legacy-namespace",
        default=MEMORIES_STORE_NAME,
        help=f"Name of the legacy namespace segment (default: {MEMORIES_STORE_NAME}).",
    )
    parser.add_argument(
        "--user-ids",
        nargs="*",
        help="Explicit list of user IDs to migrate. If omitted, auto-discovers.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Database URL override. Reads from settings if omitted.",
    )
    args = parser.parse_args()

    # Load settings
    from inference_core.core.config import get_settings

    settings = get_settings()
    db_url = args.database_url or str(settings.sqlalchemy_database_uri)

    logger.info("Database: %s", db_url[:50] + "..." if len(db_url) > 50 else db_url)
    logger.info("Legacy namespace segment: %s", args.legacy_namespace)
    logger.info("Mode: %s", "DRY-RUN" if args.dry_run else "EXECUTE")

    store = get_store(db_url)
    ns_builder = MemoryNamespaceBuilder(
        base_collection=settings.agent_memory_collection,
        agent_name=None,  # Migration uses shared namespaces; agent-scoping is opt-in later
    )

    user_ids = args.user_ids or discover_user_ids(store, args.legacy_namespace)
    if not user_ids:
        logger.warning("No user IDs found. Nothing to migrate.")
        return

    total_stats: dict[str, int] = {cat.value: 0 for cat in MemoryCategory}

    for uid in user_ids:
        stats = migrate_user(
            store, uid, args.legacy_namespace, ns_builder, args.dry_run
        )
        for cat, count in stats.items():
            total_stats[cat] += count

    logger.info("=" * 50)
    logger.info("Migration summary (%s):", "DRY-RUN" if args.dry_run else "EXECUTED")
    for cat, count in total_stats.items():
        logger.info("  %s: %d memories", cat, count)
    logger.info(
        "Total: %d memories across %d users", sum(total_stats.values()), len(user_ids)
    )


if __name__ == "__main__":
    main()
