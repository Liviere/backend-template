"""
Agent Memory Service (Store-based) – CoALA Architecture

Implements long-term memory operations using LangGraph Store backends
following the Cognitive Architectures for Language Agents (CoALA) framework.

Memory is organized into three CoALA categories:
  - Semantic: stable world-knowledge & user facts (preferences, facts, goals, general)
  - Episodic: sequences of past experiences (context, session_summary, interaction)
  - Procedural: operational rules & skills (instructions, workflow, skill)

Each category has its own namespace to enable independent retrieval and
scoped isolation (shared vs per-agent).

Source: CoALA whitepaper – arxiv:2309.02427
"""

import logging
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from inference_core.services.agents_service import AgentService

logger = logging.getLogger(__name__)


MEMORIES_STORE_NAME = os.getenv("MEMORIES_STORE_NAME", "memories")
INDEXED_FIELDS = ["content", "topic"]


# =============================================================================
# CoALA Memory Categories
# =============================================================================


class MemoryCategory(str, Enum):
    """Top-level CoALA memory categories.

    Semantic – stable facts about the world and the user.
    Episodic  – sequences of past agent/user interactions.
    Procedural – operational rules, workflows, and learned skills.
    """

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


MEMORY_CATEGORIES = [mc.value for mc in MemoryCategory]


class MemoryType(str, Enum):
    """Fine-grained memory types, each mapped to a CoALA category."""

    # --- Semantic (stable world-knowledge) ---
    PREFERENCES = "preferences"
    FACTS = "facts"
    GOALS = "goals"
    GENERAL = "general"

    # --- Episodic (past experiences / interactions) ---
    CONTEXT = "context"
    SESSION_SUMMARY = "session_summary"
    INTERACTION = "interaction"

    # --- Procedural (rules / skills / workflows) ---
    INSTRUCTION = "instructions"
    WORKFLOW = "workflow"
    SKILL = "skill"


MEMORY_TYPES = [mt.value for mt in MemoryType]


# Canonical mapping from MemoryType → MemoryCategory
MEMORY_TYPE_TO_CATEGORY: Dict[str, MemoryCategory] = {
    # Semantic
    MemoryType.PREFERENCES.value: MemoryCategory.SEMANTIC,
    MemoryType.FACTS.value: MemoryCategory.SEMANTIC,
    MemoryType.GOALS.value: MemoryCategory.SEMANTIC,
    MemoryType.GENERAL.value: MemoryCategory.SEMANTIC,
    # Episodic
    MemoryType.CONTEXT.value: MemoryCategory.EPISODIC,
    MemoryType.SESSION_SUMMARY.value: MemoryCategory.EPISODIC,
    MemoryType.INTERACTION.value: MemoryCategory.EPISODIC,
    # Procedural
    MemoryType.INSTRUCTION.value: MemoryCategory.PROCEDURAL,
    MemoryType.WORKFLOW.value: MemoryCategory.PROCEDURAL,
    MemoryType.SKILL.value: MemoryCategory.PROCEDURAL,
}


def get_category_for_type(memory_type: str) -> MemoryCategory:
    """Resolve CoALA category for a given memory type.

    Falls back to SEMANTIC for unknown types to preserve backward compatibility.
    """
    return MEMORY_TYPE_TO_CATEGORY.get(memory_type, MemoryCategory.SEMANTIC)


def get_types_for_category(category: MemoryCategory) -> List[str]:
    """Return all memory types belonging to the given CoALA category."""
    return [t for t, c in MEMORY_TYPE_TO_CATEGORY.items() if c == category]


# =============================================================================
# Namespace Builder (CoALA-aware)
# =============================================================================


class MemoryNamespaceBuilder:
    """Builds store namespaces following the CoALA isolation model.

    Namespace structure:
      Shared:     (user_id, "semantic")
      Per-agent:  (user_id, "procedural", agent_name)

    Rules:
      - Semantic memory is ALWAYS shared across agents (user-global facts).
      - Episodic memory is per-agent by default (each agent has its own history).
      - Procedural memory is per-agent by default (agent-specific skills/rules).
      - When agent_name is None, per-agent categories fall back to shared.
    """

    def __init__(
        self,
        base_collection: str = "agent_memory",
        agent_name: Optional[str] = None,
    ) -> None:
        self.base_collection = base_collection
        self.agent_name = agent_name

    def namespace_for(
        self,
        user_id: str,
        category: MemoryCategory,
    ) -> Tuple[str, ...]:
        """Build namespace tuple for a given user and CoALA category."""
        user_id = str(user_id)

        if category == MemoryCategory.SEMANTIC:
            # Always shared – user-global knowledge
            return (user_id, MemoryCategory.SEMANTIC.value)

        if category == MemoryCategory.EPISODIC:
            if self.agent_name:
                return (user_id, MemoryCategory.EPISODIC.value, self.agent_name)
            return (user_id, MemoryCategory.EPISODIC.value)

        if category == MemoryCategory.PROCEDURAL:
            if self.agent_name:
                return (user_id, MemoryCategory.PROCEDURAL.value, self.agent_name)
            return (user_id, MemoryCategory.PROCEDURAL.value)

        # Fallback (should not happen with Enum)
        return (user_id, MEMORIES_STORE_NAME)

    def namespace_for_type(
        self,
        user_id: str,
        memory_type: str,
    ) -> Tuple[str, ...]:
        """Convenience: resolve category from memory_type, then build namespace."""
        category = get_category_for_type(memory_type)
        return self.namespace_for(user_id, category)

    def legacy_namespace(self, user_id: str) -> Tuple[str, ...]:
        """Return old-style namespace for migration compatibility."""
        return (str(user_id), MEMORIES_STORE_NAME)


MEMORY_TYPES = [mt.value for mt in MemoryType]


def get_memory_type_literal():
    """Generate Literal type hint from MemoryType enum for runtime consistency.

    Use this in tool signatures instead of hardcoding Literal values.
    Example: memory_type: get_memory_type_literal() = MemoryType.GENERAL.value
    """
    from typing import Literal, get_args

    return Literal[tuple(mt.value for mt in MemoryType)]


def get_memory_category_literal():
    """Generate Literal type hint from MemoryCategory enum."""
    from typing import Literal

    return Literal[tuple(mc.value for mc in MemoryCategory)]


def validate_memory_type(value: str) -> str:
    """Runtime validator ensuring memory_type matches canonical enum.

    Raises ValueError if invalid type provided.
    Use in Pydantic validators or explicit checks.
    """
    # Unwrap Enum if passed instead of str
    if not isinstance(value, str):
        if hasattr(value, "value"):
            value = value.value

    if value not in MEMORY_TYPES:
        valid_types = ", ".join(MEMORY_TYPES)
        raise ValueError(
            f"Invalid memory_type '{value}'. Must be one of: {valid_types}"
        )
    return value


def validate_memory_category(value: str) -> str:
    """Runtime validator ensuring memory category matches canonical enum."""
    if not isinstance(value, str):
        if hasattr(value, "value"):
            value = value.value

    if value not in MEMORY_CATEGORIES:
        valid = ", ".join(MEMORY_CATEGORIES)
        raise ValueError(f"Invalid memory_category '{value}'. Must be one of: {valid}")
    return value


def format_memory_types_for_description() -> str:
    """Generate formatted list of memory types grouped by CoALA category.

    Returns multi-line string suitable for embedding in tool description.
    """
    lines = ["Available memory types (grouped by CoALA category):"]
    for category in MemoryCategory:
        lines.append(f"\n  [{category.value.upper()}]")
        for mtype in MemoryType:
            if MEMORY_TYPE_TO_CATEGORY.get(mtype.value) == category:
                desc_enum = MemoryTypeDescription[mtype.name]
                first_sentence = desc_enum.value.split(".")[0] + "."
                lines.append(f"    - {mtype.value}: {first_sentence}")
    return "\n".join(lines)


class MemoryTypeDescription(str, Enum):
    """Descriptions for memory type categories."""

    # --- Semantic ---

    PREFERENCES = """Personal preferences, communication styles, and behavioral patterns that should be remembered for future interactions. These memories help maintain consistency in responses and adapt to individual needs and expectations.
    Examples: 'Prefers concise answers with bullet points', 'Enjoys humor in casual conversations'"""

    FACTS = """Factual information about entities, relationships, locations, or any objective data that remains stable over time. These memories store verifiable information that can be referenced to provide accurate and personalized responses.
    Examples: 'Lives in Warsaw, Poland', 'Company headquarters located in New York'"""

    GOALS = """Objectives, aspirations, and desired outcomes that represent what someone or something is working towards. These memories help maintain focus on long-term vision and provide context for prioritizing actions and decisions.
    Examples: 'Improve team collaboration and communication', 'Reduce system response time by 30%'"""

    GENERAL = """Miscellaneous observations, notes, and information that may be useful but doesn't fit into other specific categories. These memories serve as a repository for various insights and details that could be relevant in future contexts.
    Examples: 'Noticed increased activity during evening hours', 'Client mentioned budget concerns during last meeting'"""

    # --- Episodic ---

    CONTEXT = """Situational awareness and environmental information that provides background for current or recent activities. These memories capture the immediate circumstances and ongoing situations that may influence decision-making.
    Examples: 'Meeting scheduled for tomorrow at 2 PM', 'Weather forecast shows rain this week'"""

    SESSION_SUMMARY = """Condensed summary of a completed conversation session, capturing key decisions, action items, and outcomes. Provides efficient recall of past interactions without replaying full transcripts.
    Examples: 'Session on 2025-01-15: discussed Q1 goals, decided to prioritize API redesign', 'Reviewed deployment pipeline, identified 3 bottlenecks'"""

    INTERACTION = """Notable interaction events or exchanges worth remembering for continuity across sessions. Captures specific moments, questions, or patterns that inform future behavior.
    Examples: 'User asked about async patterns twice — may need a tutorial', 'Successfully helped debug CORS issue using proxy approach'"""

    # --- Procedural ---

    INSTRUCTION = """Operational guidelines, rules, or specific directions about how tasks should be performed or processes should be followed. These memories ensure consistent execution of established procedures and methodologies.
    Examples: 'Always backup files before making changes', 'Send weekly reports every Friday morning'"""

    WORKFLOW = """Multi-step processes or standard operating procedures that the agent should follow for recurring tasks. Captures the sequence and dependencies between steps.
    Examples: 'Deploy process: run tests → build image → push to staging → smoke test → promote to prod', 'Code review checklist: types, tests, docs, perf'"""

    SKILL = """Learned capabilities or techniques that the agent has acquired through interaction, including tool usage patterns, domain-specific approaches, or optimized strategies.
    Examples: 'Use pandas .query() instead of boolean indexing for readability', 'For this user, always include type annotations in code examples'"""


@dataclass
class MemoryMetadata:
    """Consistent metadata schema for stored memories."""

    user_id: str
    memory_type: str = MemoryType.GENERAL.value
    memory_category: str = MemoryCategory.SEMANTIC.value
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    session_id: Optional[str] = None
    topic: Optional[str] = None
    agent_name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Flatten metadata for storage and filtering."""

        result = {
            "user_id": self.user_id,
            "memory_type": self.memory_type,
            "memory_category": self.memory_category,
            "created_at": self.created_at,
        }
        if self.session_id:
            result["session_id"] = self.session_id
        if self.topic:
            result["topic"] = self.topic
        if self.agent_name:
            result["agent_name"] = self.agent_name
        if self.extra:
            result.update(self.extra)
        return result


@dataclass
class MemoryStoreDocument:
    """Lightweight record wrapper for store search results.

    Includes temporal fields (created_at, updated_at) extracted from
    LangGraph Store Item metadata for time-aware memory operations.
    """

    id: str
    value: Dict[str, Any]
    metadata: Dict[str, Any]
    score: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def content(self) -> str:
        from inference_core.services.content_cipher import dec_field

        return dec_field(self.value.get("content", ""), purpose="memory")

    @property
    def topic(self) -> Optional[str]:
        from inference_core.services.content_cipher import dec_field

        return dec_field(self.value.get("topic", None), purpose="memory")

    @property
    def memory_type(self) -> Optional[str]:
        return self.value.get("memory_type", None)

    @property
    def memory_category(self) -> Optional[str]:
        """Resolve CoALA category from stored memory_type."""
        mt = self.memory_type
        if mt:
            return get_category_for_type(mt).value
        return None

    @property
    def created_at_iso(self) -> Optional[str]:
        """Return created_at as ISO 8601 string or None."""
        if self.created_at:
            return self.created_at.isoformat()
        return None

    @property
    def updated_at_iso(self) -> Optional[str]:
        """Return updated_at as ISO 8601 string or None."""
        if self.updated_at:
            return self.updated_at.isoformat()
        return None


@dataclass
class MemoryData:
    """Structured memory data for storage."""

    content: str
    memory_type: str
    memory_category: str = ""
    session_id: Optional[str] = None
    topic: Optional[str] = None
    agent_name: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self):
        """Auto-resolve category from memory_type if not explicitly set."""
        if not self.memory_category:
            self.memory_category = get_category_for_type(self.memory_type).value


class AgentMemoryStoreService:
    """Store-backed memory service following CoALA architecture.

    Uses MemoryNamespaceBuilder to route memories into category-specific
    namespaces (semantic / episodic / procedural) while maintaining
    backward compatibility with the flat namespace layout.
    """

    def __init__(
        self,
        store: Any,
        base_namespace: Sequence[str] = (MEMORIES_STORE_NAME,),
        max_results: int = 5,
        agent_name: Optional[str] = None,
    ) -> None:
        self.store = store
        self.base_namespace = tuple(base_namespace)
        self.max_results = max_results
        self.agent_name = agent_name
        self.ns_builder = MemoryNamespaceBuilder(
            base_collection=base_namespace[0] if base_namespace else "agent_memory",
            agent_name=agent_name,
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _namespace(self, user_id: str) -> tuple:
        """Build a namespaced tuple for store isolation per user (legacy)."""

        return (str(user_id), *self.base_namespace)

    async def save_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str = MemoryType.GENERAL.value,
        session_id: Optional[str] = None,
        topic: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        """Persist a memory item, routed to the correct CoALA namespace.

        The category is auto-resolved from memory_type when not provided.
        """

        assert (
            memory_type in MemoryType._value2member_map_
        ), f"Invalid memory type: {memory_type}"

        user_id = str(user_id)

        # Resolve category
        resolved_category = (
            MemoryCategory(category) if category else get_category_for_type(memory_type)
        )
        namespace_for_memory = self.ns_builder.namespace_for(user_id, resolved_category)

        if not extra_metadata:
            extra_metadata = {}

        if not memory_id:
            memory_id = str(uuid.uuid4())

        memory_data = MemoryData(
            content=content,
            memory_type=memory_type,
            memory_category=resolved_category.value,
            session_id=session_id,
            topic=topic,
            agent_name=self.agent_name,
            extra_metadata=extra_metadata,
        )

        # Encrypt content/topic at rest (flag-gated). Semantic indexing stays
        # correct because the embed_fn wrapped by
        # EmbeddingService.get_embed_fn() decrypts tokens before embedding —
        # `index=INDEXED_FIELDS` is REQUIRED on every put so only those
        # fields are extracted (never the whole JSON with mixed ciphertext).
        from inference_core.services.content_cipher import (
            enc_field,
            memory_enc_enabled,
        )

        value = dict(memory_data.__dict__)
        value["content"] = enc_field(
            value.get("content"), purpose="memory", enabled=memory_enc_enabled()
        )
        if value.get("topic"):
            value["topic"] = enc_field(
                value["topic"], purpose="memory", enabled=memory_enc_enabled()
            )

        self.store.put(
            namespace_for_memory,
            memory_id,
            value,
            index=INDEXED_FIELDS,
        )

        return memory_id

    async def recall_memories(
        self,
        user_id: str,
        query: str,
        k: Optional[int] = None,
        memory_type: Optional[str] = None,
        include_scores: bool = True,
        sort_by_time: bool = False,
        category: Optional[str] = None,
    ) -> List[MemoryStoreDocument]:
        """Retrieve relevant memories from the store for the user.

        When category is specified, searches only that CoALA namespace.
        When memory_type is specified, category is auto-resolved.
        When neither is specified, searches across all categories.

        Args:
            user_id: User identifier for namespace isolation.
            query: Semantic search query string.
            k: Maximum number of results to return.
            memory_type: Optional filter by memory type.
            include_scores: Whether to include relevance scores.
            sort_by_time: If True, sort results by created_at descending (newest first).
            category: Optional CoALA category to search within.

        Returns:
            List of MemoryStoreDocument with temporal metadata (created_at, updated_at).
        """
        limit = k or self.max_results

        # Determine which namespaces to search
        if category:
            cat = MemoryCategory(category)
            namespaces = [self.ns_builder.namespace_for(user_id, cat)]
        elif memory_type:
            cat = get_category_for_type(memory_type)
            namespaces = [self.ns_builder.namespace_for(user_id, cat)]
        else:
            # Search all categories
            namespaces = [
                self.ns_builder.namespace_for(user_id, cat) for cat in MemoryCategory
            ]

        filters = {}
        if memory_type:
            filters["memory_type"] = memory_type

        documents: List[MemoryStoreDocument] = []
        for ns in namespaces:
            results = self.store.search(
                ns,
                query=query,
                limit=limit,
                filter=filters,
            )

            for item in results or []:
                item_created_at = getattr(item, "created_at", None)
                item_updated_at = getattr(item, "updated_at", None)

                documents.append(
                    MemoryStoreDocument(
                        id=getattr(item, "key", getattr(item, "id", "")),
                        value=getattr(item, "value", {}),
                        metadata=getattr(item, "value", {}).get("extra_metadata", {}),
                        score=getattr(item, "score", None) if include_scores else None,
                        created_at=item_created_at,
                        updated_at=item_updated_at,
                    )
                )

        # Re-sort by score across namespaces when not sorting by time
        if not sort_by_time:
            documents.sort(
                key=lambda d: (d.score or 0.0),
                reverse=True,
            )
            documents = documents[:limit]
        else:
            documents.sort(
                key=lambda d: (
                    d.created_at is not None,
                    d.created_at or datetime.min.replace(tzinfo=timezone.utc),
                ),
                reverse=True,
            )
            documents = documents[:limit]

        self.logger.debug(
            "Recalled %d memories for user=%s, query='%s', sort_by_time=%s",
            len(documents),
            user_id,
            query[:50] + "..." if len(query) > 50 else query,
            sort_by_time,
        )
        return documents

    async def recall_by_category(
        self,
        user_id: str,
        category: MemoryCategory,
        query: str,
        k: Optional[int] = None,
        memory_type: Optional[str] = None,
    ) -> List[MemoryStoreDocument]:
        """Convenience: recall memories from a single CoALA category namespace."""
        return await self.recall_memories(
            user_id=user_id,
            query=query,
            k=k,
            memory_type=memory_type,
            category=category.value,
        )

    async def get_user_context(
        self,
        user_id: str,
        memory_types: Optional[List[str]] = None,
        max_per_type: int = 3,
    ) -> Dict[str, List[str]]:
        """Aggregate memories per type for prompt construction (legacy, content-only).

        For time-aware context, use get_user_context_with_timestamps() instead.
        """
        if memory_types is None:
            memory_types = [
                MemoryType.PREFERENCES.value,
                MemoryType.FACTS.value,
                MemoryType.INSTRUCTION.value,
            ]

        context: Dict[str, List[str]] = {}
        for mtype in memory_types:
            memories = await self.recall_memories(
                user_id=user_id,
                query=mtype,
                k=max_per_type,
                memory_type=mtype,
            )
            if memories:
                context[mtype] = [m.content for m in memories]

        return context

    async def get_user_context_with_timestamps(
        self,
        user_id: str,
        memory_types: Optional[List[str]] = None,
        max_per_type: int = 10,
    ) -> Dict[str, List[MemoryStoreDocument]]:
        """Aggregate memories per type with full temporal metadata.

        Args:
            user_id: User identifier.
            memory_types: List of memory types to include.
            max_per_type: Maximum memories per type.

        Returns:
            Dict mapping memory_type to list of MemoryStoreDocument objects
            (includes created_at, updated_at for temporal processing).
        """
        if memory_types is None:
            memory_types = [
                MemoryType.PREFERENCES.value,
                MemoryType.FACTS.value,
                MemoryType.INSTRUCTION.value,
            ]

        context: Dict[str, List[MemoryStoreDocument]] = {}
        for mtype in memory_types:
            memories = await self.recall_memories(
                user_id=user_id,
                query=mtype,
                k=max_per_type,
                memory_type=mtype,
                sort_by_time=True,
            )
            if memories:
                context[mtype] = memories

        return context

    async def delete_memory(
        self,
        user_id: str,
        memory_id: str,
        category: Optional[str] = None,
    ) -> bool:
        """Remove a specific memory key from the store.

        When category is unknown, searches all namespaces to locate the key.
        """
        if not hasattr(self.store, "delete"):
            self.logger.warning("Store does not support delete; skipping")
            return False

        if category:
            ns = self.ns_builder.namespace_for(user_id, MemoryCategory(category))
            try:
                self.store.delete(ns, memory_id)
                self.logger.info("Deleted memory id=%s for user=%s", memory_id, user_id)
                return True
            except Exception as exc:
                self.logger.error("Failed to delete memory id=%s: %s", memory_id, exc)
                return False

        # Category unknown – try all namespaces
        for cat in MemoryCategory:
            ns = self.ns_builder.namespace_for(user_id, cat)
            try:
                self.store.delete(ns, memory_id)
                self.logger.info(
                    "Deleted memory id=%s from %s for user=%s",
                    memory_id,
                    cat.value,
                    user_id,
                )
                return True
            except Exception:
                continue

        self.logger.warning("Memory id=%s not found in any namespace", memory_id)
        return False

    async def delete_user_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        category: Optional[str] = None,
    ) -> int:
        """Bulk-delete user memories by type, category, or all."""

        if category:
            cats = [MemoryCategory(category)]
        elif memory_type:
            cats = [get_category_for_type(memory_type)]
        else:
            cats = list(MemoryCategory)

        filters: Dict[str, Any] = {}
        if memory_type:
            filters["memory_type"] = memory_type

        deleted = 0
        for cat in cats:
            ns = self.ns_builder.namespace_for(user_id, cat)
            try:
                results = self.store.search(ns, query="", filter=filters)
            except TypeError:
                results = self.store.search(ns, query=None, filter=filters)

            if hasattr(self.store, "delete"):
                for item in results or []:
                    key = getattr(item, "key", getattr(item, "id", None))
                    if key is None:
                        continue
                    try:
                        self.store.delete(ns, key)
                        deleted += 1
                    except Exception as exc:  # pragma: no cover
                        self.logger.error("Failed to delete memory %s: %s", key, exc)

        self.logger.info(
            "Deleted %d memories for user=%s, type=%s, category=%s",
            deleted,
            user_id,
            memory_type or "all",
            category or "all",
        )
        return deleted

    async def format_context_for_prompt(
        self,
        user_id: str,
        query: Optional[str] = None,
        include_types: Optional[List[str]] = None,
        max_memories: int = 30,
        include_categories: Optional[List[str]] = None,
    ) -> str:
        """Render stored memories as CoALA-structured XML context for prompts.

        Memories are organized first by CoALA category (semantic / episodic / procedural),
        then within each category by temporal buckets for rich temporal awareness.

        Args:
            user_id: User identifier.
            query: Optional semantic search query for relevance-based recall.
            include_types: Memory types to include in context.
            max_memories: Maximum total memories to retrieve for bucketing.
            include_categories: CoALA categories to include. Defaults to all.

        Returns:
            Formatted string with memories organized by category and time buckets.
        """
        # Resolve which categories & types to search
        if include_categories:
            cats = [MemoryCategory(c) for c in include_categories]
        else:
            cats = list(MemoryCategory)

        if include_types is None:
            include_types = [
                MemoryType.PREFERENCES.value,
                MemoryType.FACTS.value,
                MemoryType.INSTRUCTION.value,
                MemoryType.GOALS.value,
                MemoryType.CONTEXT.value,
            ]

        # Collect memories per category
        category_memories: Dict[str, List[MemoryStoreDocument]] = {}

        per_cat_limit = max(max_memories // len(cats), 5) if cats else max_memories

        for cat in cats:
            cat_types = [t for t in get_types_for_category(cat) if t in include_types]
            if not cat_types and not query:
                continue

            cat_docs: List[MemoryStoreDocument] = []

            if query:
                docs = await self.recall_memories(
                    user_id=user_id,
                    query=query,
                    k=per_cat_limit,
                    category=cat.value,
                    sort_by_time=False,
                )
                cat_docs.extend(
                    d
                    for d in docs
                    if d.memory_type in include_types or d.memory_type is None
                )
            else:
                for mtype in cat_types:
                    docs = await self.recall_memories(
                        user_id=user_id,
                        query=mtype,
                        k=per_cat_limit // max(len(cat_types), 1),
                        memory_type=mtype,
                        category=cat.value,
                        sort_by_time=True,
                    )
                    cat_docs.extend(docs)

            if cat_docs:
                category_memories[cat.value] = cat_docs

        if not category_memories:
            return ""

        return self._format_coala_context(category_memories, query)

    def _bucket_memories_by_time(
        self,
        memories: List[MemoryStoreDocument],
    ) -> Dict[str, List[MemoryStoreDocument]]:
        """Group memories into temporal buckets.

        Buckets:
        - today: Memories from today
        - yesterday: Memories from yesterday
        - prev5_before_yesterday: Memories from D-2 to D-6 (5 days before yesterday)
        - prev90_excluding_above: Memories from D-7 to D-96
        - older: Memories older than 96 days

        Args:
            memories: List of MemoryStoreDocument with created_at timestamps.

        Returns:
            Dict mapping bucket name to list of memories, each bucket sorted newest-first.
        """
        now = datetime.now(timezone.utc)
        today = now.date()
        yesterday = today - timedelta(days=1)
        # D-2 to D-6 (5 days counting back from the day before yesterday)
        prev5_start = today - timedelta(days=6)  # D-6
        prev5_end = today - timedelta(days=2)  # D-2
        # D-7 to D-96 (90 days excluding the above)
        prev90_start = today - timedelta(days=96)  # D-96
        prev90_end = today - timedelta(days=7)  # D-7

        buckets: Dict[str, List[MemoryStoreDocument]] = {
            "today": [],
            "yesterday": [],
            "prev5_before_yesterday": [],
            "prev90_excluding_above": [],
            "older": [],
        }

        # De-duplicate by content
        seen_contents: set[str] = set()

        for mem in memories:
            content = mem.content
            if not content or content in seen_contents:
                continue
            seen_contents.add(content)

            # Parse timestamp
            mem_date = self._get_memory_date(mem)

            # Assign to bucket
            if mem_date == today:
                buckets["today"].append(mem)
            elif mem_date == yesterday:
                buckets["yesterday"].append(mem)
            elif mem_date and prev5_start <= mem_date <= prev5_end:
                buckets["prev5_before_yesterday"].append(mem)
            elif mem_date and prev90_start <= mem_date <= prev90_end:
                buckets["prev90_excluding_above"].append(mem)
            else:
                buckets["older"].append(mem)

        # Sort each bucket by timestamp descending (newest first)
        for bucket_name in buckets:
            buckets[bucket_name].sort(
                key=lambda m: (
                    m.created_at is not None,
                    m.created_at or datetime.min.replace(tzinfo=timezone.utc),
                ),
                reverse=True,
            )

        return buckets

    def _get_memory_date(self, mem: MemoryStoreDocument) -> Optional[datetime.date]:
        """Extract date from memory's created_at timestamp.

        Args:
            mem: MemoryStoreDocument instance.

        Returns:
            Date object or None if timestamp unavailable.
        """
        if mem.created_at:
            dt = mem.created_at
            # Ensure timezone-aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.date()
        return None

    def _format_bucketed_memories(
        self,
        buckets: Dict[str, List[MemoryStoreDocument]],
        query: Optional[str] = None,
    ) -> str:
        """Format bucketed memories as text for prompt injection.

        Args:
            buckets: Dict of bucket_name -> list of MemoryStoreDocument.
            query: Optional original query for context header.

        Returns:
            Formatted string with temporal sections, prefixed with current date.
        """
        # Current date/time for temporal context awareness
        now = datetime.now(timezone.utc)
        current_datetime_str = now.isoformat().replace("+00:00", "Z")

        header = (
            f"Current datetime (UTC): {current_datetime_str}\n\n"
            "Relevant information from previous conversations. Each memory includes created_at (UTC) "
            "indicating when the information was remembered/saved. Pay attention to the temporal context: "
            "some information remains valid regardless of age (e.g., user's name, stable preferences, permanent facts), "
            "while other information may be time-sensitive (e.g., current projects, temporary preferences, scheduled events). "
            "When memories conflict, consider both recency AND the nature of the information to determine which is most relevant. "
            "Use the timestamps to understand the timeline of user's interactions and evolving context:"
        )

        sections: List[str] = [header]

        def render_bucket(title: str, items: List[MemoryStoreDocument]) -> None:
            """Render a single bucket section."""
            if not items:
                return
            sections.append(f"\n{title}:")
            for item in items:
                ts_label = ""
                if item.created_at:
                    ts_label = f"[{item.created_at.isoformat()}] "
                type_label = f" (type: {item.memory_type})" if item.memory_type else ""
                topic_label = f" [topic: {item.topic}]" if item.topic else ""
                sections.append(f"- {ts_label}{item.content}{type_label}{topic_label}")

        render_bucket("Data from today", buckets.get("today", []))
        render_bucket("Data from yesterday", buckets.get("yesterday", []))
        render_bucket(
            "Data from the previous 5 days (counting back from yesterday)",
            buckets.get("prev5_before_yesterday", []),
        )
        render_bucket(
            "Data from the previous 90 days (excluding the above)",
            buckets.get("prev90_excluding_above", []),
        )
        render_bucket("Older entries", buckets.get("older", []))

        return "\n".join(sections) if len(sections) > 1 else ""

    # -----------------------------------------------------------------
    # CoALA-structured formatting
    # -----------------------------------------------------------------

    def _format_coala_context(
        self,
        category_memories: Dict[str, List[MemoryStoreDocument]],
        query: Optional[str] = None,
    ) -> str:
        """Format memories grouped by CoALA category with temporal buckets inside."""
        now = datetime.now(timezone.utc)
        current_datetime_str = now.isoformat().replace("+00:00", "Z")

        header = (
            f"Current datetime (UTC): {current_datetime_str}\n\n"
            "Relevant information retrieved from long-term memory, organized by CoALA category. "
            "Semantic = stable user facts & preferences; Episodic = past interaction history; "
            "Procedural = operational rules & learned skills. "
            "Timestamps indicate when information was saved. Use both recency and information "
            "nature to judge relevance:"
        )

        sections: List[str] = [header]

        category_labels = {
            MemoryCategory.SEMANTIC.value: "SEMANTIC MEMORY (facts, preferences, goals)",
            MemoryCategory.EPISODIC.value: "EPISODIC MEMORY (past experiences, sessions)",
            MemoryCategory.PROCEDURAL.value: "PROCEDURAL MEMORY (instructions, workflows, skills)",
        }

        for cat_value, label in category_labels.items():
            docs = category_memories.get(cat_value, [])
            if not docs:
                continue

            sections.append(f"\n<{cat_value}_memory>")
            sections.append(f"## {label}")

            # Bucket within category
            buckets = self._bucket_memories_by_time(docs)

            def render_cat_bucket(title: str, items: List[MemoryStoreDocument]) -> None:
                if not items:
                    return
                sections.append(f"\n  {title}:")
                for item in items:
                    ts_label = ""
                    if item.created_at:
                        ts_label = f"[{item.created_at.isoformat()}] "
                    type_label = (
                        f" (type: {item.memory_type})" if item.memory_type else ""
                    )
                    topic_label = f" [topic: {item.topic}]" if item.topic else ""
                    sections.append(
                        f"  - {ts_label}{item.content}{type_label}{topic_label}"
                    )

            render_cat_bucket("Recent (today)", buckets.get("today", []))
            render_cat_bucket("Yesterday", buckets.get("yesterday", []))
            render_cat_bucket(
                "Previous 5 days", buckets.get("prev5_before_yesterday", [])
            )
            render_cat_bucket(
                "Previous 90 days", buckets.get("prev90_excluding_above", [])
            )
            render_cat_bucket("Older", buckets.get("older", []))

            sections.append(f"</{cat_value}_memory>")

        return "\n".join(sections) if len(sections) > 1 else ""


# =============================================================================
# Standalone purge / count helpers (user-facing memory management)
# =============================================================================

# Batch size for store listing, and a runaway guard for the purge loop in case a
# delete silently fails and the same items keep being returned.
_PURGE_BATCH_SIZE = 100
_MAX_PURGE_BATCHES = 1000


def _store_search(
    store: Any,
    prefix: Tuple[str, ...],
    *,
    limit: int,
    offset: int = 0,
) -> List[Any]:
    """Namespace-prefix listing that tolerates differing store search signatures.

    Uses ``query=None`` so this is a plain listing (no vector search) and works
    even when the store was opened without an embedding index.
    """
    try:
        return store.search(prefix, query=None, limit=limit, offset=offset) or []
    except TypeError:
        # Older/alternate signature without keyword query support.
        return store.search(prefix, limit=limit, offset=offset) or []


def purge_user_memories(
    store: Any,
    user_id: str,
    category: Optional[str] = None,
) -> int:
    """Delete a user's long-term memories from *store*.

    Deletes by each item's OWN namespace (not the search prefix), so per-agent
    episodic/procedural entries stored under ``(user_id, category, agent_name)``
    are removed correctly.

    Args:
        store: A LangGraph BaseStore-compatible instance.
        user_id: Owner of the memories.
        category: One of the CoALA categories (semantic/episodic/procedural).
            When ``None``, every category is purged.

    Returns:
        Number of memory items deleted.
    """
    uid = str(user_id)
    if category is not None:
        category = validate_memory_category(category)
        prefix: Tuple[str, ...] = (uid, category)
    else:
        prefix = (uid,)

    if not hasattr(store, "delete"):
        logger.warning("Store does not support delete; cannot purge memories")
        return 0

    deleted = 0
    for _ in range(_MAX_PURGE_BATCHES):
        items = _store_search(store, prefix, limit=_PURGE_BATCH_SIZE)
        if not items:
            break
        progressed = False
        for item in items:
            namespace = getattr(item, "namespace", None) or prefix
            key = getattr(item, "key", getattr(item, "id", None))
            if key is None:
                continue
            try:
                store.delete(tuple(namespace), key)
                deleted += 1
                progressed = True
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to delete memory %s/%s: %s", namespace, key, exc)
        if not progressed:
            # Nothing could be deleted this round — avoid an infinite loop.
            break

    logger.info(
        "Purged %d memories for user=%s, category=%s",
        deleted,
        uid,
        category or "all",
    )
    return deleted


def count_user_memories(store: Any, user_id: str) -> Dict[str, int]:
    """Count a user's stored memories per CoALA category.

    Returns a dict with ``semantic``, ``episodic``, ``procedural`` and ``total``.
    """
    uid = str(user_id)
    counts: Dict[str, int] = {c: 0 for c in MEMORY_CATEGORIES}

    for category in MEMORY_CATEGORIES:
        prefix = (uid, category)
        offset = 0
        while True:
            items = _store_search(store, prefix, limit=_PURGE_BATCH_SIZE, offset=offset)
            counts[category] += len(items)
            if len(items) < _PURGE_BATCH_SIZE:
                break
            offset += _PURGE_BATCH_SIZE

    counts["total"] = sum(counts[c] for c in MEMORY_CATEGORIES)
    return counts


def _memory_item_to_dict(item: Any) -> Dict[str, Any]:
    """Flatten a store SearchItem into a serializable memory record.

    The CoALA category and agent are derived from the namespace tuple
    (`(user_id, category[, agent_name])`), falling back to the stored value.
    """
    namespace = list(getattr(item, "namespace", ()) or ())
    value = getattr(item, "value", {}) or {}
    key = getattr(item, "key", getattr(item, "id", ""))

    category = namespace[1] if len(namespace) > 1 else value.get("memory_category")
    agent_name = namespace[2] if len(namespace) > 2 else value.get("agent_name")

    def _iso(dt: Any) -> Optional[str]:
        if dt is None:
            return None
        if hasattr(dt, "isoformat"):
            return dt.isoformat()
        if isinstance(dt, str):
            return dt
        return None

    created_at = _iso(getattr(item, "created_at", None)) or value.get("created_at")
    updated_at = _iso(getattr(item, "updated_at", None))

    from inference_core.services.content_cipher import dec_field

    return {
        "namespace": namespace,
        "key": key,
        "content": dec_field(value.get("content", ""), purpose="memory"),
        "memory_type": value.get("memory_type"),
        "memory_category": category,
        "topic": dec_field(value.get("topic"), purpose="memory"),
        "agent_name": agent_name,
        "session_id": value.get("session_id"),
        "created_at": created_at,
        "updated_at": updated_at,
    }


def list_user_memories(
    store: Any,
    user_id: str,
    *,
    category: Optional[str] = None,
    memory_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """List a user's stored memories with filtering, sorting and pagination.

    Loads every item under the namespace prefix (`(user_id,)` or
    `(user_id, category)`), filters by ``memory_type`` and a case-insensitive
    substring on content/topic, sorts newest-first, and slices for pagination.
    Returns ``(page_items, total)`` where ``total`` is the filtered count.
    Memory volume per user is small, so in-Python filtering/sort is acceptable.
    """
    uid = str(user_id)
    if category is not None:
        category = validate_memory_category(category)
        prefix: Tuple[str, ...] = (uid, category)
    else:
        prefix = (uid,)

    raw: List[Any] = []
    search_offset = 0
    while True:
        batch = _store_search(
            store, prefix, limit=_PURGE_BATCH_SIZE, offset=search_offset
        )
        if not batch:
            break
        raw.extend(batch)
        if len(batch) < _PURGE_BATCH_SIZE:
            break
        search_offset += _PURGE_BATCH_SIZE

    items = [_memory_item_to_dict(it) for it in raw]

    if memory_type:
        items = [d for d in items if d.get("memory_type") == memory_type]

    if search:
        needle = search.strip().lower()
        if needle:
            items = [
                d
                for d in items
                if needle in (d.get("content") or "").lower()
                or needle in (d.get("topic") or "").lower()
            ]

    # ISO-8601 UTC strings sort chronologically as plain strings.
    items.sort(key=lambda d: d.get("created_at") or "", reverse=True)

    total = len(items)
    if offset:
        items = items[offset:]
    if limit is not None:
        items = items[:limit]
    return items, total


def delete_user_memory_entry(
    store: Any,
    user_id: str,
    namespace: Sequence[str],
    key: str,
) -> bool:
    """Delete a single memory item, guarding that it belongs to ``user_id``."""
    ns = tuple(namespace)
    if not ns or str(ns[0]) != str(user_id):
        logger.warning("Refusing to delete memory outside user's namespace: %s", ns)
        return False
    if not hasattr(store, "delete"):
        logger.warning("Store does not support delete; cannot delete memory")
        return False
    try:
        store.delete(ns, key)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to delete memory %s/%s: %s", ns, key, exc)
        return False


def delete_user_memory_entries(
    store: Any,
    user_id: str,
    entries: Sequence[Tuple[Sequence[str], str]],
) -> int:
    """Delete multiple memory items; returns the number actually deleted."""
    deleted = 0
    for namespace, key in entries:
        if delete_user_memory_entry(store, user_id, namespace, key):
            deleted += 1
    return deleted


@contextmanager
def open_memory_store() -> Iterator[Any]:
    """Open the process-configured LangGraph memory store for one-off operations.

    Builds a sync store from ``settings.database_url`` (mapping async drivers to
    their sync counterparts) WITHOUT an embedding index — list and delete
    operations need no vectors.  Unknown dialects yield a fresh in-memory store
    (nothing is persisted there, so counts/purges are no-ops).

    Mirrors the store construction in
    ``inference_core.agents.graph_builder._init_memory_store``.
    """
    from inference_core.core.config import get_settings

    settings = get_settings()
    url = settings.database_url
    for async_drv, sync_drv in (
        ("+aiosqlite", ""),
        ("+asyncpg", ""),
        ("+aiomysql", ""),
    ):
        if async_drv in url:
            url = url.replace(async_drv, sync_drv)
            break

    if "sqlite" in url:
        from langgraph.store.sqlite import SqliteStore

        with SqliteStore.from_conn_string(url) as store:
            if hasattr(store, "setup") and callable(store.setup):
                store.setup()
            yield store
    elif "postgresql" in url:
        from langgraph.store.postgres import PostgresStore

        with PostgresStore.from_conn_string(url) as store:
            if hasattr(store, "setup") and callable(store.setup):
                store.setup()
            yield store
    else:
        from langgraph.store.memory import InMemoryStore

        logger.warning(
            "Unknown DB dialect for memory store (%s); using ephemeral InMemoryStore",
            url,
        )
        yield InMemoryStore()


def purge_user_long_term_memory(
    user_id: str,
    category: Optional[str] = None,
) -> int:
    """Open the configured store and delete the user's memories. Returns count."""
    with open_memory_store() as store:
        return purge_user_memories(store, user_id, category)


def count_user_long_term_memory(user_id: str) -> Dict[str, int]:
    """Open the configured store and count the user's memories per category."""
    with open_memory_store() as store:
        return count_user_memories(store, user_id)


def list_user_long_term_memory(
    user_id: str,
    *,
    category: Optional[str] = None,
    memory_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """Open the configured store and list the user's memories. Returns (items, total)."""
    with open_memory_store() as store:
        return list_user_memories(
            store,
            user_id,
            category=category,
            memory_type=memory_type,
            search=search,
            limit=limit,
            offset=offset,
        )


def delete_user_long_term_memory_entry(
    user_id: str,
    namespace: Sequence[str],
    key: str,
) -> bool:
    """Open the configured store and delete one memory item for the user."""
    with open_memory_store() as store:
        return delete_user_memory_entry(store, user_id, namespace, key)


def delete_user_long_term_memory_entries(
    user_id: str,
    entries: Sequence[Tuple[Sequence[str], str]],
) -> int:
    """Open the configured store and delete several memory items for the user."""
    with open_memory_store() as store:
        return delete_user_memory_entries(store, user_id, entries)


__all__ = [
    "AgentMemoryStoreService",
    "MemoryCategory",
    "MemoryData",
    "MemoryMetadata",
    "MemoryNamespaceBuilder",
    "MemoryStoreDocument",
    "MemoryType",
    "MemoryTypeDescription",
    "MEMORY_CATEGORIES",
    "MEMORY_TYPE_TO_CATEGORY",
    "validate_memory_type",
    "validate_memory_category",
    "format_memory_types_for_description",
    "get_category_for_type",
    "get_types_for_category",
    "get_memory_type_literal",
    "get_memory_category_literal",
    "purge_user_memories",
    "count_user_memories",
    "list_user_memories",
    "delete_user_memory_entry",
    "delete_user_memory_entries",
    "open_memory_store",
    "purge_user_long_term_memory",
    "count_user_long_term_memory",
    "list_user_long_term_memory",
    "delete_user_long_term_memory_entry",
    "delete_user_long_term_memory_entries",
]
