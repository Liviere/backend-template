import logging
import uuid
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Dict, Literal, Optional

from deepagents import CompiledSubAgent, SubAgent
from deepagents.backends import StateBackend, StoreBackend
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.backends.utils import create_file_data
from deepagents.middleware import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import InterruptOnConfig
from langchain.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables.config import RunnableConfig, merge_configs
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore
from langgraph.store.sqlite import SqliteStore
from pydantic import BaseModel

from inference_core.agents.middleware import (
    CostTrackingMiddleware,
    MemoryMiddleware,
    ToolBasedModelSwitchMiddleware,
    build_model_fallback_middleware,
    canonicalize_fallback_overrides,
    create_tool_model_switch_middleware,
)
from inference_core.agents.tools.memory_tools import (
    generate_memory_tools_system_instructions,
    get_memory_tools,
)
from inference_core.agents.tools.skill_reader import (
    add_skill_reader_tool,
    create_skills_middleware,
)
from inference_core.celery.async_utils import run_async_safely
from inference_core.core.config import get_settings
from inference_core.database.sql.models.user_agent_instance import UserAgentInstance
from inference_core.llm.config import LLMConfig, get_llm_config
from inference_core.llm.emulation import activate_emulation_session
from inference_core.llm.models import LLMModelFactory, get_model_factory
from inference_core.llm.tools import get_registered_providers, load_tools_for_agent
from inference_core.services._cancel import AgentCancelled
from inference_core.services.agent_memory_service import AgentMemoryStoreService
from inference_core.services.agent_snapshot_service import (
    AgentRunSnapshot,
    AgentSnapshotMatcher,
    AgentSnapshotResolvedConfig,
    AgentSnapshotTokenSegment,
    AgentSnapshotTraceEvent,
    AgentSnapshotInput,
    build_snapshot_fingerprint,
    build_snapshot_query_text,
    get_agent_snapshot_store,
    rebuild_agent_response,
    serialize_snapshot_value,
)


class AgentMetadata(BaseModel):
    model_name: str
    tools_used: list[str]
    start_time: datetime
    end_time: Optional[datetime] = None


class AgentCostMetrics(BaseModel):
    """Cost and usage metrics from an agent run."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    extra_tokens: dict[str, int] = {}
    model_call_count: int = 0
    tool_call_count: int = 0


class AgentResponse(BaseModel):
    result: dict[str, Any]
    steps: list[dict[str, Any]]
    metadata: AgentMetadata
    cost_metrics: Optional[AgentCostMetrics] = None
    # Validated structured payload returned by LangChain when the agent is
    # configured with ``response_format``. None when no schema is configured
    # or when the model did not emit a structured response for this run.
    structured_response: Optional[Any] = None


# Cost-tracking field names emitted by CostTrackingMiddleware in state updates.
_COST_FIELDS = (
    "accumulated_input_tokens",
    "accumulated_output_tokens",
    "accumulated_total_tokens",
    "accumulated_extra_tokens",
    "model_call_count",
    "tool_call_count",
    "usage_session_id",
)

_TRACE_MODES = frozenset({"separate", "nested", "nested-only"})


@dataclass
class RunAccumulator:
    """Mutable state accumulated during an agent streaming run.

    WHY: Both sync and async execution paths accumulate identical data
    (steps, cost fields, messages) while iterating over the stream.
    Centralising this state and its mutation logic eliminates duplication
    and ensures the message-accumulation breaking change is applied
    consistently everywhere.
    """

    steps: list[dict[str, Any]] = field(default_factory=list)
    accumulated_state: dict[str, Any] = field(default_factory=dict)
    all_messages: list[Any] = field(default_factory=list)
    # Latest structured payload emitted by LangChain's create_agent when
    # ``response_format`` is configured. Captured from ``updates`` chunks so
    # both local streaming and Agent Server v2 stream paths populate it.
    structured_response: Any = None

    def process_updates_chunk(
        self,
        chunk_data: dict[str, Any],
        on_step: Callable[[str, Any], None] | None,
    ) -> None:
        """Process a single ``updates`` chunk — shared between sync and async."""
        for step_name, data in chunk_data.items():
            self.steps.append({"name": step_name, "data": data})

            if on_step:
                try:
                    on_step(step_name, data)
                except Exception:
                    pass  # best-effort, never break agent execution

            if isinstance(data, dict):
                for key in _COST_FIELDS:
                    if key in data:
                        self.accumulated_state[key] = data[key]

                if "messages" in data:
                    self.all_messages.extend(data["messages"])

                if "structured_response" in data:
                    self.structured_response = data["structured_response"]

            elif step_name == "__interrupt__":
                self.all_messages.append({"__interrupt__": data})

    def extract_result(self) -> dict[str, Any]:
        """Build the final result dict from accumulated messages."""
        if self.all_messages:
            return {"messages": self.all_messages}
        return {}

    def build_response(
        self,
        model_name: str,
        start_time: datetime,
        enable_cost_tracking: bool,
    ) -> AgentResponse:
        """Assemble a complete ``AgentResponse`` from accumulated data."""
        result = self.extract_result()

        tools_used = [step["name"] for step in self.steps]
        metadata = AgentMetadata(
            model_name=model_name,
            tools_used=tools_used,
            start_time=start_time,
            end_time=datetime.now(UTC),
        )

        cost_metrics = None
        if self.accumulated_state and enable_cost_tracking:
            cost_metrics = AgentCostMetrics(
                input_tokens=self.accumulated_state.get("accumulated_input_tokens", 0),
                output_tokens=self.accumulated_state.get(
                    "accumulated_output_tokens", 0
                ),
                total_tokens=self.accumulated_state.get("accumulated_total_tokens", 0),
                extra_tokens=self.accumulated_state.get("accumulated_extra_tokens", {}),
                model_call_count=self.accumulated_state.get("model_call_count", 0),
                tool_call_count=self.accumulated_state.get("tool_call_count", 0),
            )

        return AgentResponse(
            result=result,
            steps=self.steps,
            metadata=metadata,
            cost_metrics=cost_metrics,
            structured_response=self.structured_response,
        )


@dataclass(frozen=True)
class InstanceContext:
    """Identity of a UserAgentInstance backing this agent run.

    WHY: Bundles instance-level identity so that memory namespacing,
    cost tracking, and logging can distinguish between different user
    instances of the same base agent.

    subagent_configs is a dict mapping base_agent_name → override dict
    (primary_model, system_prompt_override, system_prompt_append, etc.)
    for subagents that have user instance overrides in the DB.
    Forwarded to Agent Server configurable so SubagentConfigMiddleware
    can apply per-subagent overrides at runtime.
    """

    instance_id: uuid.UUID
    instance_name: str
    base_agent_name: str
    subagent_configs: dict[str, dict[str, Any]] = field(default_factory=dict)


class AgentService:
    def __init__(
        self,
        agent_name: Optional[str] = "default_agent",
        tools: Optional[list[Callable]] = None,
        use_checkpoints: bool = False,
        use_memory: bool = False,
        checkpoint_config: Optional[dict[str, Any]] = None,
        context_schema: Optional[Any] = None,
        middleware: Optional[list[Any]] = None,
        enable_cost_tracking: bool = True,
        user_id: Optional[uuid.UUID] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        config: Optional[LLMConfig] = None,
        instance_context: Optional[InstanceContext] = None,
        system_prompt_override: Optional[str] = None,
        system_prompt_append: Optional[str] = None,
        user_skills: Optional[list[dict[str, str]]] = None,
        memory_postrun_analysis: Optional[bool] = None,
        memory_postrun_model: Optional[str] = None,
        memory_tools: Optional[list[str]] = None,
        memory_session_context_enabled: Optional[bool] = None,
        memory_tool_instructions_enabled: Optional[bool] = None,
        response_format_override: Optional[Any] = None,
    ):
        """Initialize the AgentService.

        Args:
            agent_name: Name of the agent configuration to use (YAML config key).
            tools: List of tools to provide to the agent.
            use_checkpoints: Whether to enable checkpointing for conversation history.
            use_memory: Whether to automatically add memory tools and MemoryMiddleware.
                        Requires agent_memory_enabled=True in settings and user_id.
            checkpoint_config: Configuration for the checkpointer (required if use_checkpoints=True).
            context_schema: Optional schema for agent context.
            middleware: List of middleware to apply to the agent. If None and
                       enable_cost_tracking=True, CostTrackingMiddleware is added automatically.
            enable_cost_tracking: Whether to automatically add CostTrackingMiddleware.
                                  Set to False to disable cost tracking.
            user_id: Optional user ID for cost tracking and memory attribution.
            session_id: Optional session ID for grouping related requests.
            request_id: Optional correlation ID (e.g., Celery task ID).
            config: Optional LLMConfig instance with overrides (e.g. user preferences).
            instance_context: Optional identity of the UserAgentInstance backing this run.
                              When set, enables per-instance memory isolation, cost tracking
                              attribution, and enriched logging.
            system_prompt_override: When set, fully replaces the system_prompt passed to
                create_agent().  Used by from_user_instance() when the DB instance
                has a complete custom prompt.
            system_prompt_append: When set, appended to the system_prompt passed to
                create_agent().  Used when the DB instance extends the base prompt.
            memory_postrun_analysis: Override for post-run memory analysis flag.
                When None (default), uses settings.agent_memory_postrun_analysis_enabled.
            memory_postrun_model: Optional model name for the post-run extraction LLM
                call.  When None, uses settings.agent_memory_postrun_analysis_model
                (typically the agent's own model).
            memory_tools: Override for which memory tools are exposed to the model.
                When None, inherits from AgentConfig.memory_tools or exposes all.
                Empty list disables model-facing memory tools while keeping the
                MemoryMiddleware active for after_agent postrun analysis.
            memory_session_context_enabled: Override for MemoryMiddleware session
                context injection.  When None, inherits from
                AgentConfig.memory_session_context_enabled or global auto_recall.
            memory_tool_instructions_enabled: Override for memory tool instructions
                in the system prompt.  When None, inherits from
                AgentConfig.memory_tool_instructions_enabled or default behaviour.
            response_format_override: Optional per-call structured-output schema.
                Forwarded as-is to LangChain ``create_agent(response_format=...)``
                when set, taking precedence over ``AgentConfig.response_format``.
                Accepts a Pydantic model class, dataclass, TypedDict, JSON Schema
                dict, ``ToolStrategy``, or ``ProviderStrategy``. Use this to keep
                the schema as a Python type (so LangChain can return a validated
                Pydantic instance via ``structured_response``) instead of going
                through a YAML JSON-Schema round-trip.
        """
        # Model and tools setup
        self.agent_name = agent_name
        self.instance_context = instance_context
        self.config = config

        if self.config:
            self.model_factory = LLMModelFactory(self.config)
        else:
            self.model_factory = get_model_factory()

        self.agent_config = self.model_factory.config.get_specific_agent_config(
            agent_name
        )
        self.tools = tools or []
        self.context_schema = context_schema

        self.model_name = self.model_factory.get_agent_model_name(agent_name)
        assert self.model_name is not None, "Model must have a valid name."
        self.model = self.model_factory.get_model_for_agent(agent_name)
        assert (
            self.model is not None
        ), f"Model for agent {agent_name} could not be found."

        # Exit stack to handle context managers for checkpointers
        self._exit_stack = ExitStack()

        # Checkpointing setup
        self.use_checkpoints = use_checkpoints
        self.checkpoint_config = checkpoint_config
        if use_checkpoints:
            assert checkpoint_config is not None, "Checkpoint config must be provided."
            self.checkpointer = self._initialize_checkpointer()
        else:
            self.checkpointer = None

        # Memory setup
        self.use_memory = use_memory
        self._enable_memory = False
        if use_memory:
            self.memory_store = self._initialize_memory_store()
            self._enable_memory = True
        else:
            self.memory_store = None

        # Middleware setup
        self._middleware = middleware or []
        self._enable_cost_tracking = enable_cost_tracking
        self._user_id = user_id
        self._session_id = session_id
        self._request_id = request_id

        # Memory service reference (store-backed)
        self._memory_service = None
        self._memory_store_service = None
        self._enhanced_system_prompt = None

        # DB-backed prompt customisation: stored here, applied in create_agent()
        self._system_prompt_override = system_prompt_override
        self._system_prompt_append = system_prompt_append

        # User-defined skills: list of {name, description, content} dicts
        self._user_skills = user_skills or []

        # Post-run memory analysis overrides (None = use settings defaults)
        self._memory_postrun_analysis = memory_postrun_analysis
        self._memory_postrun_model = memory_postrun_model

        # Memory surface configuration overrides (None = inherit from AgentConfig/global)
        self._memory_tools_override = memory_tools
        self._memory_session_context_enabled_override = memory_session_context_enabled
        self._memory_tool_instructions_enabled_override = (
            memory_tool_instructions_enabled
        )

        # Per-call structured-output schema override (None = use AgentConfig.response_format).
        # Forwarded as-is to ``create_agent(response_format=...)`` so callers
        # can pass Pydantic models / dataclasses / TypedDicts and benefit from
        # native LangChain validation into the ``structured_response`` state key.
        self._response_format_override = response_format_override

        # Tools whose invocation must be delegated to ``multimodal_support_model``
        # because the primary model does not support multimodal inputs. Populated
        # by ``_load_providers_tools`` and consumed by
        # ``_add_tool_model_switch_middleware``.
        self._delegated_multimodal_tools: list[str] = []
        self._resolved_middleware: list[Any] = []
        self._resolved_response_format: Any = None
        self.agent = None

    @property
    def display_name(self) -> str:
        """Human-readable agent identity for logging and observability.

        Returns instance_name when running as a user agent instance,
        otherwise falls back to the YAML agent_name.
        """
        if self.instance_context:
            return self.instance_context.instance_name
        return self.agent_name

    @property
    def _is_remote(self) -> bool:
        """Whether this agent should delegate to the remote Agent Server.

        WHY: Central check used by run/arun/stream to decide execution
        path.  Returns True only when both the global feature flag AND
        the per-agent YAML setting are active.
        """
        settings = get_settings()
        return (
            settings.agent_server_enabled
            and self.agent_config.execution_mode == "remote"
        )

    def set_cancel_check(self, cancel_check: Callable[[], bool] | None) -> None:
        """Propagate a cancellation callback to middleware.

        When set, the middleware will check this callback before each model
        and tool call, raising ``AgentCancelled`` if it returns ``True``.
        """
        for mw in self._middleware:
            if isinstance(mw, CostTrackingMiddleware):
                mw.cancel_check = cancel_check
            elif hasattr(mw, "cancel_check"):
                mw.cancel_check = cancel_check

    # ------------------------------------------------------------------
    # Memory surface resolution
    # ------------------------------------------------------------------

    def _resolve_memory_tools(self) -> list[str] | None:
        """Resolve which memory tool names to expose to the model.

        Resolution order (first non-None wins):
            runtime override → AgentConfig (YAML / config_overrides) → None (all tools).

        Returns:
            Explicit list of tool names, or None meaning "expose all tools".
        """
        if self._memory_tools_override is not None:
            return self._memory_tools_override
        if self.agent_config and self.agent_config.memory_tools is not None:
            return self.agent_config.memory_tools
        return None

    def _resolve_memory_session_context_enabled(self) -> bool:
        """Resolve whether MemoryMiddleware should inject recalled context.

        Resolution order (first non-None wins):
            runtime override → AgentConfig → global agent_memory_auto_recall.
        """
        if self._memory_session_context_enabled_override is not None:
            return self._memory_session_context_enabled_override
        if (
            self.agent_config
            and self.agent_config.memory_session_context_enabled is not None
        ):
            return self.agent_config.memory_session_context_enabled
        return get_settings().agent_memory_auto_recall

    def _resolve_memory_tool_instructions_enabled(self) -> bool:
        """Resolve whether memory tool instructions are appended to the system prompt.

        Resolution order (first non-None wins):
            runtime override → AgentConfig → True when any model-facing tools exist.
        """
        if self._memory_tool_instructions_enabled_override is not None:
            return self._memory_tool_instructions_enabled_override
        if (
            self.agent_config
            and self.agent_config.memory_tool_instructions_enabled is not None
        ):
            return self.agent_config.memory_tool_instructions_enabled
        # Default: instructions enabled when memory tools are active
        resolved_tools = self._resolve_memory_tools()
        # None means all tools → instructions on; empty list → instructions off
        return resolved_tools is None or len(resolved_tools) > 0

    def _resolve_response_format(self) -> Optional[Any]:
        """Return the structured-output schema configured for this agent, or None.

        WHY: structured output is opt-in per agent. Resolution order:
        1. ``response_format_override`` passed to the constructor (per-call) -
           accepts any value supported by ``create_agent`` (Pydantic class,
           dataclass, TypedDict, JSON Schema dict, ``ToolStrategy``,
           ``ProviderStrategy``).
        2. ``AgentConfig.response_format`` from YAML / DB override (JSON Schema
           dict).
        Centralising the lookup keeps create_agent() lean and gives
        DeepAgentService a single hook to override without re-resolving the
        whole AgentConfig.
        """
        if self._response_format_override is not None:
            return self._response_format_override
        if self.agent_config is None:
            return None
        return self.agent_config.response_format

    async def create_agent(
        self, system_prompt: Optional[str] = None, **kwargs
    ) -> Callable:
        """Create and configure the agent with tools and middleware.

        This method:
        1. Loads tools from registered providers
        2. Loads MCP tools if configured
        3. Builds middleware list (including CostTrackingMiddleware if enabled)
        4. Creates the agent with all configurations
        5. Appends memory tools instructions to system_prompt if memory is enabled

        Args:
            system_prompt: Optional system prompt/instructions for the agent.
                          If memory is enabled, memory tools usage instructions
                          will be automatically appended.
            **kwargs: Additional arguments passed to create_agent.

        Returns:
            The configured agent callable.
        """
        # Load tools from registered providers if any are configured
        provider_context = self._build_provider_user_context(kwargs.get("user_context"))
        await self._load_providers_tools(user_context=provider_context)

        # Load tools from MCP if configured
        await self._load_mcp_tools()

        # Load tools from memory if configured
        await self._load_memory_tools()

        # Apply DB-backed prompt overrides, then build enhanced system prompt
        # (memory tool instructions are appended by _build_system_prompt if enabled)
        effective_prompt = self._apply_prompt_overrides(system_prompt)
        self._enhanced_system_prompt = self._build_system_prompt(effective_prompt)

        # Build middleware list
        middleware = self._build_middleware()

        # Add SkillsMiddleware if user-defined skills are present
        if self._user_skills:
            store, skill_sources = self._build_skills_store(self._user_skills)
            skills_backend = lambda rt: StoreBackend(rt)
            skills_middleware = create_skills_middleware(
                backend=skills_backend,
                sources=skill_sources,
            )
            middleware.append(skills_middleware)
            add_skill_reader_tool(
                self.tools,
                backend=skills_backend,
                sources=skill_sources,
            )
            # Inject store into kwargs so create_agent passes it to the graph
            kwargs.setdefault("store", store)

        # Create the agent
        # Resolve structured-output schema from AgentConfig.  Pass via kwargs
        # only when present and not already provided by the caller so explicit
        # overrides via **kwargs continue to win.
        response_format = self._resolve_response_format()
        if response_format is not None and "response_format" not in kwargs:
            kwargs["response_format"] = response_format
        self._resolved_middleware = middleware
        self._resolved_response_format = response_format

        self.agent = create_agent(
            self.model,
            tools=self.tools,
            checkpointer=self.checkpointer,
            context_schema=self.context_schema,
            middleware=middleware,
            system_prompt=self._enhanced_system_prompt,
            **kwargs,
        )
        self.model_params = self.model_factory.config.get_model_params(self.model_name)
        return self.agent

    def _build_middleware(self) -> list[Any]:
        """Build the middleware list for the agent.

        Adds CostTrackingMiddleware automatically if enabled and not already present.
        Adds MemoryMiddleware if memory is enabled and service is available.

        Returns:
            List of middleware instances.
        """
        middleware = list(self._middleware)

        # Add CostTrackingMiddleware if enabled and not already present
        if self._enable_cost_tracking:
            # Check if CostTrackingMiddleware is already in the list
            has_cost_tracking = any(
                isinstance(m, CostTrackingMiddleware) for m in middleware
            )

            if not has_cost_tracking:
                # Get pricing config for the model
                pricing_config = None
                provider = None
                try:
                    llm_config = get_llm_config()
                    model_cfg = llm_config.models.get(self.model_name)
                    if model_cfg:
                        pricing_config = model_cfg.pricing
                        provider = model_cfg.provider
                except Exception as e:
                    logging.debug(f"Could not load pricing config: {e}")

                cost_middleware = CostTrackingMiddleware(
                    pricing_config=pricing_config,
                    user_id=self._user_id,
                    session_id=self._session_id,
                    request_id=self._request_id,
                    task_type="agent",
                    request_mode="sync",
                    provider=provider,
                    model_name=self.model_name,
                    instance_id=(
                        self.instance_context.instance_id
                        if self.instance_context
                        else None
                    ),
                    instance_name=(
                        self.instance_context.instance_name
                        if self.instance_context
                        else None
                    ),
                )
                # Append at the end so CostTracking runs FIRST in the
                # reversed after_model phase — before any interrupt-causing
                # middleware (e.g. HumanInTheLoopMiddleware, SubAgentMiddleware).
                # Source: LangChain middleware docs – execution order, 2025-08
                middleware.append(cost_middleware)
                logging.debug(
                    f"Added CostTrackingMiddleware for agent '{self.display_name}'"
                )

        # Add MemoryMiddleware if enabled and service available
        if self.use_memory and self._memory_service and self._user_id:
            has_memory = any(isinstance(m, MemoryMiddleware) for m in middleware)

            if not has_memory:
                try:
                    settings = get_settings()
                    resolved_auto_recall = (
                        self._resolve_memory_session_context_enabled()
                    )
                    memory_middleware = MemoryMiddleware(
                        memory_service=self._memory_service,
                        user_id=str(self._user_id),
                        auto_recall=resolved_auto_recall,
                        max_recall_results=settings.agent_memory_max_results,
                        postrun_analysis=(
                            self._memory_postrun_analysis
                            if self._memory_postrun_analysis is not None
                            else settings.agent_memory_postrun_analysis_enabled
                        ),
                        postrun_analysis_model=(
                            self._memory_postrun_model
                            or settings.agent_memory_postrun_analysis_model
                        ),
                        # Dedicated memory model (independent of the session
                        # model).  This is the single place a future per-user
                        # choice would read a user preference instead of the
                        # global default.
                        memory_model=settings.agent_memory_model,
                        # Attribute memory-handling LLM cost to this session.
                        session_id=self._session_id,
                        request_id=self._request_id,
                    )
                    # Insert before CostTracking (which sits at the end)
                    # so memory context enrichment runs in before_model first.
                    ct_index = next(
                        (
                            i
                            for i, m in enumerate(middleware)
                            if isinstance(m, CostTrackingMiddleware)
                        ),
                        len(middleware),
                    )
                    middleware.insert(ct_index, memory_middleware)
                    logging.debug(
                        f"Added MemoryMiddleware for agent '{self.display_name}'"
                    )
                except Exception as e:
                    logging.error(
                        f"Failed to add MemoryMiddleware for agent '{self.display_name}': {e}",
                        exc_info=True,
                    )

        # Add no-cost tool emulation only after tools are loaded, so selection
        # can respect include/exclude lists and tool metadata.
        try:
            from inference_core.llm.emulation import build_tool_emulation_middleware

            tool_emulator = build_tool_emulation_middleware(self.tools)
            if tool_emulator is not None:
                middleware.append(tool_emulator)
                logging.debug(
                    "Added LLMToolEmulator for agent '%s' in emulation mode",
                    self.display_name,
                )
        except Exception as e:
            logging.error(
                "Failed to add LLMToolEmulator for agent '%s': %s",
                self.display_name,
                e,
                exc_info=True,
            )

        # Add ToolBasedModelSwitchMiddleware if configured in agent config
        self._add_tool_model_switch_middleware(middleware)

        # Add ToolCallLimitMiddleware instances if configured in agent config
        self._add_tool_call_limit_middleware(middleware)

        # Add ModelFallbackMiddleware if fallback models are configured.
        self._add_model_fallback_middleware(middleware)
        self._ensure_cost_tracking_last(middleware)

        return middleware

    @staticmethod
    def _build_skills_store(
        user_skills: list[dict[str, str]],
        base_path: str = "/skills/",
    ) -> tuple[InMemoryStore, list[str]]:
        """Build an InMemoryStore populated with user-defined skills.

        Each skill entry is a dict with 'name', 'description', 'content'.
        The content is stored as a SKILL.md file inside the store under
        ``{base_path}{name}/SKILL.md``.

        Returns:
            A tuple of (store, [base_path]) to pass to StoreBackend/SkillsMiddleware.
            The source path must be the *parent* directory of all skill subdirectories,
            because SkillsMiddleware.ls_info scans one level down for skill dirs.
        """
        store = InMemoryStore()

        for skill in user_skills:
            skill_name = skill["name"]
            store_key = f"{base_path}{skill_name}/SKILL.md"

            store.put(
                namespace=("filesystem",),
                key=store_key,
                value=create_file_data(skill["content"]),
            )

        # Return the parent directory as the single source so SkillsMiddleware
        # can discover each skill subdirectory via ls_info(base_path).
        return store, [base_path]

    def _apply_prompt_overrides(self, base_prompt: Optional[str]) -> Optional[str]:
        """Apply system_prompt_override or system_prompt_append from a UserAgentInstance.

        WHY: DB-backed instances let users fully replace or extend the caller-supplied
        system prompt without the caller needing to know about DB configuration.

        Precedence (highest to lowest):
          1. system_prompt_override  – replaces base_prompt entirely
          2. system_prompt_append    – concatenated onto base_prompt (or used alone)
          3. base_prompt             – unchanged if neither DB field is set
        """
        if self._system_prompt_override is not None:
            return self._system_prompt_override
        if self._system_prompt_append is not None:
            if base_prompt:
                return f"{base_prompt}\n\n{self._system_prompt_append}"
            return self._system_prompt_append
        return base_prompt

    def _build_system_prompt(self, base_prompt: Optional[str] = None) -> Optional[str]:
        """Build enhanced system prompt with memory and tool-call-limit instructions.

        Appends instruction blocks when the corresponding features are active:
        - Memory tool usage instructions (when memory is enabled)
        - Tool-call limit policy instructions (when tool_call_limits is configured)

        Args:
            base_prompt: Base system prompt provided by user.

        Returns:
            Enhanced system prompt with appended instructions,
            or None if no prompt and no instructions.
        """
        prompt = base_prompt

        # --- Memory tool instructions ---
        if self._memory_service and self.use_memory:
            if self._resolve_memory_tool_instructions_enabled():
                resolved_tools = self._resolve_memory_tools()
                memory_instructions = generate_memory_tools_system_instructions(
                    active_tool_names=resolved_tools,
                )

                if memory_instructions:
                    if prompt:
                        prompt = f"{prompt}\n\n{memory_instructions}"
                    else:
                        prompt = memory_instructions

        # --- Tool-call limit instructions ---
        from inference_core.llm.config import ToolCallLimitsConfig

        tool_call_limits = getattr(self.agent_config, "tool_call_limits", None)
        if isinstance(tool_call_limits, ToolCallLimitsConfig):
            from inference_core.agents.middleware.tool_call_limits import (
                generate_tool_call_limits_instructions,
            )

            limit_instructions = generate_tool_call_limits_instructions(
                tool_call_limits
            )
            if limit_instructions:
                if prompt:
                    prompt = f"{prompt}\n\n{limit_instructions}"
                else:
                    prompt = limit_instructions

        return prompt

    def _add_tool_model_switch_middleware(self, middleware: list[Any]) -> None:
        """Add ToolBasedModelSwitchMiddleware if tool_model_overrides is configured.

        Checks the agent configuration for tool_model_overrides and creates
        the middleware if any overrides are defined.

        Also synthesizes overrides for tools retained under the ``delegate``
        capability strategy so that multimodal-requiring tools are routed to
        the configured ``multimodal_support_model`` without losing any
        user-defined overrides (yaml config wins on name collision).

        Args:
            middleware: List of middleware to append to (modified in place).
        """
        if not self.agent_config:
            return

        overrides = list(self.agent_config.tool_model_overrides or [])

        # Merge in synthetic overrides for delegated multimodal tools.
        support_model = getattr(self.agent_config, "multimodal_support_model", None)
        if self._delegated_multimodal_tools and support_model:
            from inference_core.llm.config import ToolModelOverrideConfig

            existing_names = {o.tool_name for o in overrides}
            for tool_name in self._delegated_multimodal_tools:
                if tool_name in existing_names:
                    continue
                overrides.append(
                    ToolModelOverrideConfig(
                        tool_name=tool_name,
                        model=support_model,
                        trigger="before_tool",
                        description=(
                            "Auto-delegated: tool requires multimodal and primary "
                            "model is not multimodal."
                        ),
                    )
                )
                logging.info(
                    "Agent '%s': auto-delegating tool '%s' to multimodal "
                    "support model '%s'",
                    self.display_name,
                    tool_name,
                    support_model,
                )

        if not overrides:
            return

        # Check if already present
        has_tool_model_switch = any(
            isinstance(m, ToolBasedModelSwitchMiddleware) for m in middleware
        )
        if has_tool_model_switch:
            return

        try:
            # Convert Pydantic models to dicts for the factory function
            override_dicts = [
                {
                    "tool_name": o.tool_name,
                    "model": o.model,
                    "trigger": o.trigger,
                    "description": o.description,
                }
                for o in overrides
            ]

            tool_model_middleware = create_tool_model_switch_middleware(
                overrides=override_dicts,
                default_model=self.model_name,
                model_factory=self.model_factory,
                cache_models=True,
                reasoning_output=getattr(self.agent_config, "reasoning_output", False),
            )

            # Insert at the end (after cost tracking and memory)
            # This ensures model switching happens during the actual model call
            middleware.append(tool_model_middleware)

            logging.info(
                f"Added ToolBasedModelSwitchMiddleware for agent '{self.display_name}' "
                f"with {len(overrides)} override(s)"
            )
        except Exception as e:
            logging.error(
                f"Failed to add ToolBasedModelSwitchMiddleware for agent "
                f"'{self.display_name}': {e}",
                exc_info=True,
            )

    def _add_tool_call_limit_middleware(self, middleware: list[Any]) -> None:
        """Add ToolCallLimitMiddleware instances from agent config.

        Reads ``tool_call_limits`` from the agent's YAML config and builds
        middleware via the shared ``build_tool_call_limit_middleware`` factory.
        Inserts before CostTrackingMiddleware so limits are enforced before
        cost accounting.
        """
        if not self.agent_config:
            return

        from inference_core.llm.config import ToolCallLimitsConfig

        tool_call_limits = getattr(self.agent_config, "tool_call_limits", None)
        if not isinstance(tool_call_limits, ToolCallLimitsConfig):
            return

        from inference_core.agents.middleware.tool_call_limits import (
            build_tool_call_limit_middleware,
        )

        limiters = build_tool_call_limit_middleware(tool_call_limits)
        if limiters:
            # Insert before CostTracking (which must stay last).
            ct_index = next(
                (
                    i
                    for i, m in enumerate(middleware)
                    if isinstance(m, CostTrackingMiddleware)
                ),
                len(middleware),
            )
            for j, limiter in enumerate(limiters):
                middleware.insert(ct_index + j, limiter)

            logging.info(
                "Added %d ToolCallLimitMiddleware instance(s) for agent '%s'",
                len(limiters),
                self.display_name,
            )

    def _add_model_fallback_middleware(self, middleware: list[Any]) -> None:
        """Add LangChain ModelFallbackMiddleware from agent fallback config."""
        if not self.agent_config:
            return

        _agent_gen = getattr(self.agent_config, "generation_params", None)
        fallback_middleware = build_model_fallback_middleware(
            model_factory=self.model_factory,
            fallback_models=getattr(self.agent_config, "fallback", None),
            primary_model=self.model_name,
            reasoning_output=getattr(self.agent_config, "reasoning_output", False),
            generation_params=_agent_gen.as_overrides() if _agent_gen else None,
            owner=self.display_name,
        )
        if fallback_middleware is None:
            return

        middleware.append(fallback_middleware)
        logging.info(
            "Added ModelFallbackMiddleware for agent '%s'",
            self.display_name,
        )

    @staticmethod
    def _ensure_cost_tracking_last(middleware: list[Any]) -> None:
        """Move CostTrackingMiddleware to the absolute end of the list.

        WHY: In the reversed ``after_model`` phase, the last middleware runs
        first.  Placing CostTracking last guarantees its state-accumulation
        hook executes before any interrupt-causing middleware
        (HumanInTheLoopMiddleware, SubAgentMiddleware) can halt the graph.

        The primary DB persistence already happens in ``wrap_model_call``
        (which is interrupt-safe), so this is a defensive measure for the
        best-effort state accumulation in ``after_model``.
        """
        ct_instances = [m for m in middleware if isinstance(m, CostTrackingMiddleware)]
        if not ct_instances:
            return
        for ct in ct_instances:
            middleware.remove(ct)
            middleware.append(ct)

    def close(self) -> None:
        try:
            self._exit_stack.close()
        finally:
            close_method = getattr(self.model_factory, "close", None)
            if callable(close_method):
                close_method()

    async def aclose(self) -> None:
        """Close resources owned by this agent run.

        WHY: API routes execute agents inside async request handlers.  An async
        close path lets LLM/provider clients expose ``aclose()`` without being
        forced through a synchronous cleanup bridge.
        """
        try:
            self._exit_stack.close()
        finally:
            async_close_method = getattr(self.model_factory, "aclose", None)
            if callable(async_close_method):
                await async_close_method()
            else:
                close_method = getattr(self.model_factory, "close", None)
                if callable(close_method):
                    close_method()

    def __enter__(self) -> "AgentService":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    async def __aenter__(self) -> "AgentService":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()

    def _get_checkpointer(self):
        """Initialize and return a checkpointer for the agent.

        Savers are constructed manually (not via ``from_conn_string``, which
        does not accept a serde) so the checkpoint-encryption serializer can
        be injected. ``build_checkpoint_serializer`` returns a serializer
        that always dual-reads (legacy plaintext blobs + encrypted ones) and
        encrypts new writes only behind ENCRYPT_CHECKPOINTS_WRITES; when no
        key is configured it returns None and the saver's default serde is
        used.
        """
        from inference_core.services.checkpoint_cipher import (
            build_checkpoint_serializer,
        )

        serde = build_checkpoint_serializer()
        url = self._sync_connection_string()
        if "sqlite" in url:
            return _sqlite_saver_with_serde(url, serde)
        elif "postgresql" in url:
            return _postgres_saver_with_serde(url, serde)
        elif "mysql" in url:
            logging.warning(
                "MySQL checkpointer not implemented, using in-memory saver."
            )
            return InMemorySaver()
        else:
            logging.warning(
                "Unknown database type for checkpointer, using in-memory saver."
            )
            return InMemorySaver()

    def _build_provider_user_context(
        self, user_context: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Build a stable context payload for provider-level tool scoping.

        This exists to keep per-request and per-user identity fields in one place
        so every provider gets the same context shape, regardless of how the agent
        was instantiated.
        """
        merged_context = dict(user_context or {})

        if self._user_id and "user_id" not in merged_context:
            merged_context["user_id"] = str(self._user_id)

        if self._session_id and "session_id" not in merged_context:
            merged_context["session_id"] = self._session_id

        if self._request_id and "request_id" not in merged_context:
            merged_context["request_id"] = self._request_id

        if self.instance_context:
            merged_context.setdefault(
                "instance_id", str(self.instance_context.instance_id)
            )
            merged_context.setdefault(
                "instance_name", self.instance_context.instance_name
            )

        return merged_context

    async def _load_providers_tools(
        self,
        user_context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Load tools from registered providers for the agent.

        WHY: the local FastAPI execution path does not automatically bootstrap
        the provider registry like the Agent Server graph builder does. Lazily
        registering providers from ``llm_config.yaml`` keeps local agent runs
        aligned with the same declarative configuration instead of silently
        dropping every ``local_tool_providers`` entry.
        """
        # Collect tools from local providers if configured
        configured_providers = self.agent_config.local_tool_providers or []
        if not configured_providers:
            return

        registered_providers = get_registered_providers()
        if not any(name in registered_providers for name in configured_providers):
            try:
                from inference_core.agents.graph_registry import (
                    register_providers_from_config,
                )

                registered_names = register_providers_from_config(
                    self.model_factory.config
                )
                if registered_names:
                    registered_providers = get_registered_providers()
            except Exception:
                logging.exception(
                    "Failed to register tool providers from llm_config.yaml for agent '%s'",
                    self.display_name,
                )

        confirmed_providers = []
        for registered_provider_name in registered_providers.keys():
            if registered_provider_name in configured_providers:
                logging.info(
                    f"Loading tools from provider '{registered_provider_name}' "
                    f"for agent '{self.display_name}'"
                )
                confirmed_providers.append(registered_provider_name)

        if not confirmed_providers:
            logging.warning(
                "No registered tool providers found; skipping tool loading."
            )
            return

        try:
            provider_context = self._build_provider_user_context(user_context)
            model_multimodal = self._primary_model_is_multimodal()
            on_missing_capability = getattr(
                self.agent_config, "on_missing_capability", "skip"
            )
            multimodal_support_model = getattr(
                self.agent_config, "multimodal_support_model", None
            )
            settings = get_settings()
            tool_environment = getattr(
                self.agent_config, "tool_environment", "production"
            )
            if settings.agent_tool_environment != "production":
                tool_environment = settings.agent_tool_environment
            require_test_doubles = bool(
                getattr(self.agent_config, "require_test_doubles", False)
                or settings.agent_require_test_doubles
                or tool_environment == "strict_test"
            )
            tool_double_strategy = getattr(
                self.agent_config,
                "tool_double_strategy",
                settings.agent_tool_double_strategy,
            )
            if settings.agent_tool_double_strategy != "replace":
                tool_double_strategy = settings.agent_tool_double_strategy
            provider_tools = await load_tools_for_agent(
                self.agent_name,
                confirmed_providers,
                allowed_tools=self.agent_config.allowed_tools,
                model_multimodal=model_multimodal,
                on_missing_capability=on_missing_capability,
                multimodal_support_model=multimodal_support_model,
                tool_environment=tool_environment,
                require_test_doubles=require_test_doubles,
                tool_double_strategy=tool_double_strategy,
                user_context=provider_context,
                user_id=provider_context.get("user_id"),
                session_id=provider_context.get("session_id"),
                request_id=provider_context.get("request_id"),
            )

            # Track any multimodal tools that were retained under 'delegate'
            # strategy so _add_tool_model_switch_middleware can route them to
            # the configured support model.
            if not model_multimodal and on_missing_capability == "delegate":
                from inference_core.llm.tools import tool_requires_multimodal

                for t in provider_tools:
                    if tool_requires_multimodal(t):
                        t_name = getattr(t, "name", None)
                        if t_name:
                            self._delegated_multimodal_tools.append(t_name)

            self.tools.extend(provider_tools)
        except Exception as e:
            logging.error(
                f"Error loading tools from provider '{registered_provider_name}': {e}",
                exc_info=True,
            )

    def _primary_model_is_multimodal(self) -> bool:
        """Return True if the agent's primary model declares multimodal support."""
        try:
            model_cfg = self.model_factory.config.get_model_config(self.model_name)
        except Exception:
            return False
        return bool(getattr(model_cfg, "multimodal", False))

    async def _load_mcp_tools(self) -> None:
        """Load tools from MCP if configured for this agent."""
        if not self.agent_config.mcp_profile:
            return

        try:
            from inference_core.agents.agent_mcp_tools import get_agent_mcp_manager

            manager = get_agent_mcp_manager()
            mcp_tools = await manager.get_tools_for_profile(
                self.agent_config.mcp_profile
            )

            if mcp_tools:
                logging.info(
                    f"Adding {len(mcp_tools)} MCP tools to agent '{self.display_name}'"
                )
                self.tools.extend(mcp_tools)

        except Exception as e:
            logging.error(
                f"Error loading MCP tools for agent '{self.display_name}': {e}",
                exc_info=True,
            )

    async def _load_memory_tools(self) -> None:
        """Load memory tools if memory is enabled for this agent.

        WHY: Initialises the memory service (needed by MemoryMiddleware for
        after_agent postrun analysis) even when no model-facing tools are
        exposed.  The resolved tool list controls which tools the model sees.

        Memory tools are added when:
        - use_memory=True was passed to constructor
        - agent_memory_enabled=True in settings
        - user_id is available for namespace isolation
        - Store backend is configured
        """
        if not self._enable_memory:
            return

        if not self._user_id:
            logging.warning(
                f"Memory enabled for agent '{self.display_name}' but no user_id provided. "
                "Memory tools require user_id for namespace isolation."
            )
            return

        settings = get_settings()

        # Store-based memory only
        if not (self.use_memory and self.memory_store):
            logging.warning(
                "Memory enabled but store not initialized; set use_memory=True and ensure store setup."
            )
            return

        if not self._memory_store_service:
            try:
                base_namespace = (settings.agent_memory_collection,)
                self._memory_store_service = AgentMemoryStoreService(
                    store=self.memory_store,
                    base_namespace=base_namespace,
                    max_results=settings.agent_memory_max_results,
                    agent_name=self.display_name,
                )
            except Exception as exc:
                logging.error(
                    "Failed to initialize AgentMemoryStoreService: %s",
                    exc,
                    exc_info=True,
                )
                return

        # Expose store service as _memory_service so MemoryMiddleware can use it
        self._memory_service = self._memory_store_service

        try:
            resolved_tools = self._resolve_memory_tools()
            memory_tools = get_memory_tools(
                memory_service=self._memory_store_service,
                user_id=str(self._user_id),
                session_id=self._session_id,
                max_recall_results=settings.agent_memory_max_results,
                include_tools=resolved_tools,
            )
            if memory_tools:
                self.tools.extend(memory_tools)
            logging.info(
                "Added %d store memory tools to agent '%s' (filter=%s)",
                len(memory_tools),
                self.display_name,
                resolved_tools,
            )
        except Exception as exc:
            logging.error(
                "Error loading store memory tools for agent '%s': %s",
                self.display_name,
                exc,
                exc_info=True,
            )

    def _initialize_checkpointer(self):
        """Enter checkpointer context managers and run setup if available."""
        checkpointer_obj = self._get_checkpointer()
        if hasattr(checkpointer_obj, "__enter__") and hasattr(
            checkpointer_obj, "__exit__"
        ):
            checkpointer = self._exit_stack.enter_context(checkpointer_obj)
        else:
            checkpointer = checkpointer_obj

        if hasattr(checkpointer, "setup") and callable(checkpointer.setup):
            checkpointer.setup()

        return checkpointer

    def get_memory_store(self):
        """Return the initialized memory store."""
        if self.memory_store is None:
            raise ValueError("Memory store is not initialized.")
        return self.memory_store

    def _get_memory_store(self):
        """Initialize and return a memory store for the agent."""
        url = self._sync_connection_string()
        embed_fn, dims, embeddings_obj = self._init_embeddings()
        if "sqlite" in url:
            return SqliteStore.from_conn_string(
                url, index={"embed": embed_fn, "dims": dims}
            )
        elif "postgresql" in url:
            return PostgresStore.from_conn_string(
                url, index={"embed": embed_fn, "dims": dims}
            )
        elif "mysql" in url:
            logging.warning(
                "MySQL memory store not implemented, using in-memory store."
            )
            return InMemoryStore(index={"embed": embed_fn, "dims": dims})
        else:
            logging.warning(
                "Unknown database type for memory store, using in-memory store."
            )
            return InMemoryStore(index={"embed": embed_fn, "dims": dims})

    def _initialize_memory_store(self):
        """Enter memory store context managers and run setup if available."""
        store_obj = self._get_memory_store()
        if hasattr(store_obj, "__enter__") and hasattr(store_obj, "__exit__"):
            store = self._exit_stack.enter_context(store_obj)
        else:
            store = store_obj

        if hasattr(store, "setup") and callable(store.setup):
            store.setup()

        return store

    def _init_embeddings(self):
        """Prepare embedding function and dimension for store semantic search.

        Delegates to the global EmbeddingService instead of loading
        SentenceTransformer directly. The actual compute happens either
        on a Celery prefork worker (local) or via an API provider (remote).
        """
        from inference_core.services.embedding_service import get_embedding_service

        service = get_embedding_service()
        embed_fn = service.get_embed_fn()
        dims = service.get_dimension()

        return embed_fn, dims, service

    @staticmethod
    def _sync_connection_string() -> str:
        """Return a sync SQLAlchemy connection string for SQLChatMessageHistory.

        Map async drivers to their sync counterparts for sync-mode history.
        """
        settings = get_settings()
        url = settings.database_url
        if "+aiosqlite" in url:
            return url.replace("+aiosqlite", "")
        if "+asyncpg" in url:
            return url.replace("+asyncpg", "")
        if "+aiomysql" in url:
            return url.replace("+aiomysql", "")
        return url

    def _merge_params(
        self, task: str, runtime_params: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge default model params with runtime params (runtime wins)."""
        base = dict(self._default_model_params.get(task, {}))
        base.update(runtime_params)
        return base

    def get_agent(self) -> Callable:
        """Return a callable LangChain agent.

        The agent is created in the constructor and can execute model/tool workflows.
        """
        return self.agent

    # ------------------------------------------------------------------
    # Shared helpers for run_agent_steps / arun_agent_steps
    # ------------------------------------------------------------------

    def _make_configurable(self) -> dict[str, Any]:
        """Build the ``configurable`` dict shared by sync and async execution paths."""
        configurable: dict[str, Any] = {}
        if self.checkpoint_config:
            configurable.update(self.checkpoint_config)
        if getattr(self.agent_config, "reasoning_output", False):
            configurable["reasoning_output"] = True
        return configurable

    @staticmethod
    def _parent_trace_config_available(
        parent_runnable_config: RunnableConfig | None,
    ) -> bool:
        """Detect whether a parent config can preserve trace nesting.

        WHY: Nested LangSmith traces depend on inheriting the outer callback
        stack. Metadata or tags alone are not enough to preserve a parent-child
        run tree when AgentService creates a new inner runnable execution.
        """
        if not parent_runnable_config:
            return False
        return parent_runnable_config.get("callbacks") is not None

    def _build_stream_config(
        self,
        configurable: dict[str, Any],
        *,
        on_token: Callable | None,
        on_custom: Callable | None,
        graceful_cancel: bool,
        parent_runnable_config: RunnableConfig | None = None,
        trace_mode: Literal["separate", "nested", "nested-only"] = "separate",
        sync: bool = False,
    ) -> tuple[
        dict[str, Any],
        Any | None,
        list[str],
    ]:
        """Build stream config, cancel callback, and stream mode list.

        WHY: Both sync and async execution paths need identical setup —
        dynamic stream_mode based on which callbacks are provided, and
        optional cancel-callback injection for graceful cancellation.

        Args:
            parent_runnable_config: Outer LangGraph/LangChain runnable config to
                inherit when nested tracing should be preserved.
            trace_mode: ``separate`` keeps AgentService as its own trace root.
                ``nested`` reuses parent callbacks when available and otherwise
                falls back to separate tracing. ``nested-only`` requires a
                parent trace config and raises when one is unavailable.
            sync: When ``True``, use :class:`SyncStreamCancelCallback`
                (``BaseCallbackHandler``) so that ``GraphInterrupt``
                propagates synchronously.  When ``False`` (default), use
                the async :class:`StreamCancelCallback`.
        """
        from inference_core.services.stream_utils import (
            StreamCancelCallback,
            SyncStreamCancelCallback,
        )

        stream_modes: list[str] = ["updates"]
        if on_token is not None:
            stream_modes.append("messages")
        if on_custom is not None:
            stream_modes.append("custom")

        if trace_mode not in _TRACE_MODES:
            raise ValueError(
                f"Unsupported trace_mode={trace_mode!r}. Expected one of: "
                f"{sorted(_TRACE_MODES)}"
            )

        cancel_cb: Any | None = None
        stream_config: dict[str, Any] = {"configurable": configurable}
        if graceful_cancel:
            cancel_cb = SyncStreamCancelCallback() if sync else StreamCancelCallback()
            stream_config["callbacks"] = [cancel_cb]

        trace_link_available = self._parent_trace_config_available(
            parent_runnable_config
        )
        if trace_mode == "nested-only" and not trace_link_available:
            raise ValueError(
                "trace_mode='nested-only' requires a parent RunnableConfig "
                "with callbacks so AgentService can stay inside the outer trace."
            )
        if trace_mode != "separate" and trace_link_available:
            stream_config = merge_configs(parent_runnable_config, stream_config)

        return stream_config, cancel_cb, stream_modes

    @staticmethod
    def _check_cancel(
        cancel_check: Callable[[], bool] | None,
    ) -> None:
        """Raise ``AgentCancelled`` if the caller requested cancellation.

        WHY: This pattern is repeated 4+ times across sync/async paths —
        centralising it eliminates copy-paste errors and keeps the
        streaming loop body focused on chunk processing.
        """
        if not cancel_check:
            return
        try:
            if cancel_check():
                raise AgentCancelled("Agent execution cancelled by user")
        except AgentCancelled:
            raise
        except Exception:
            pass  # best-effort — never break on check errors

    def _build_remote_metadata(self) -> dict[str, Any]:
        """Build the metadata dict forwarded to the Agent Server for remote execution."""
        metadata: dict[str, Any] = {
            "agent_name": self.agent_name,
            "model_name": self.model_name,
        }
        if self._user_id:
            metadata["user_id"] = str(self._user_id)
        if self._session_id:
            metadata["session_id"] = self._session_id
        if self._request_id:
            metadata["request_id"] = self._request_id
        if self.instance_context:
            metadata["instance_id"] = str(self.instance_context.instance_id)
            metadata["instance_name"] = self.instance_context.instance_name
        if self._system_prompt_override is not None:
            metadata["system_prompt_override"] = self._system_prompt_override
        if self._system_prompt_append is not None:
            metadata["system_prompt_append"] = self._system_prompt_append
        if (
            self.agent_config
            and getattr(self.agent_config, "fallback", None) is not None
        ):
            metadata["fallback_models"] = list(self.agent_config.fallback or [])
        if self.instance_context and self.instance_context.subagent_configs:
            metadata["subagent_configs"] = self.instance_context.subagent_configs
        if self.config:
            base_config = get_llm_config()
            base_model = base_config.agent_models.get(self.agent_name)
            if self.model_name and self.model_name != base_model:
                metadata["primary_model"] = self.model_name
        if getattr(self.agent_config, "reasoning_output", False):
            metadata["reasoning_output"] = True

        # Memory surface per-request overrides (forwarded to MemoryMiddleware
        # via configurable → contextvars on the Agent Server).
        resolved_session_ctx = self._resolve_memory_session_context_enabled()
        # Only forward explicit overrides — None means "use compile-time default"
        if self._memory_session_context_enabled_override is not None:
            metadata["memory_session_context_enabled"] = resolved_session_ctx
        if self._memory_tool_instructions_enabled_override is not None:
            metadata["memory_tool_instructions_enabled"] = (
                self._resolve_memory_tool_instructions_enabled()
            )

        return metadata

    @staticmethod
    def build_instance_context(instance: "UserAgentInstance") -> InstanceContext:
        """Build instance identity metadata without constructing an AgentService.

        WHY: The Agent Server run-bundle endpoint only needs identity and
        subagent override metadata. Building that data separately avoids the
        AgentService constructor, which creates provider-specific LLM clients.
        """
        subagent_configs: dict[str, dict[str, Any]] = {}
        for sub in getattr(instance, "subagents", None) or []:
            sub_config = {
                "instance_id": str(sub.id),
                "instance_name": sub.instance_name,
                "primary_model": sub.primary_model,
                "system_prompt_override": sub.system_prompt_override,
                "system_prompt_append": sub.system_prompt_append,
            }
            sub_overrides = canonicalize_fallback_overrides(
                getattr(sub, "config_overrides", None) or {}
            )
            if "fallback" in sub_overrides:
                sub_config["fallback_models"] = sub_overrides["fallback"]
            subagent_configs[sub.base_agent_name] = sub_config

        return InstanceContext(
            instance_id=instance.id,
            instance_name=instance.instance_name,
            base_agent_name=instance.base_agent_name,
            subagent_configs=subagent_configs,
        )

    @classmethod
    def build_remote_metadata_for_user_instance(
        cls,
        instance: "UserAgentInstance",
        user_id: Optional[uuid.UUID] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        base_config: Optional[LLMConfig] = None,
    ) -> tuple[dict[str, Any], bool]:
        """Build Agent Server run metadata without creating a local model.

        WHY: Frontend run-bundle calls are a handshake with the Agent Server.
        They must forward the same resolved instance metadata as AgentService,
        but they should not require the API process to instantiate a Gemini,
        OpenAI, Claude, or DeepInfra client just to return configuration.
        """
        resolved_config = cls.build_config_for_instance(
            instance, base_config=base_config
        )
        agent_name = instance.base_agent_name
        agent_config = resolved_config.get_specific_agent_config(agent_name)
        if agent_config is None:
            raise ValueError(
                f"Base agent '{agent_name}' is not present in the resolved LLM config"
            )

        model_name = resolved_config.get_agent_model(agent_name)
        if not model_name:
            raise ValueError(f"Model for agent '{agent_name}' could not be resolved")

        instance_context = cls.build_instance_context(instance)
        metadata: dict[str, Any] = {
            "agent_name": agent_name,
            "model_name": model_name,
        }
        if user_id:
            metadata["user_id"] = str(user_id)
        if session_id:
            metadata["session_id"] = session_id
        if request_id:
            metadata["request_id"] = request_id

        metadata["instance_id"] = str(instance_context.instance_id)
        metadata["instance_name"] = instance_context.instance_name

        if instance.system_prompt_override is not None:
            metadata["system_prompt_override"] = instance.system_prompt_override
        if instance.system_prompt_append is not None:
            metadata["system_prompt_append"] = instance.system_prompt_append

        if getattr(agent_config, "fallback", None) is not None:
            metadata["fallback_models"] = list(agent_config.fallback or [])
        if instance_context.subagent_configs:
            metadata["subagent_configs"] = instance_context.subagent_configs

        if base_config is not None:
            base_model = get_llm_config().agent_models.get(agent_name)
            if model_name != base_model:
                metadata["primary_model"] = model_name

        if getattr(agent_config, "reasoning_output", False):
            metadata["reasoning_output"] = True

        config_overrides = canonicalize_fallback_overrides(
            getattr(instance, "config_overrides", None) or {}
        )
        memory_session_context_enabled = config_overrides.get(
            "memory_session_context_enabled"
        )
        if memory_session_context_enabled is not None:
            metadata["memory_session_context_enabled"] = memory_session_context_enabled

        memory_tool_instructions_enabled = config_overrides.get(
            "memory_tool_instructions_enabled"
        )
        if memory_tool_instructions_enabled is not None:
            metadata["memory_tool_instructions_enabled"] = (
                memory_tool_instructions_enabled
            )

        settings = get_settings()
        is_remote = (
            settings.agent_server_enabled
            and getattr(agent_config, "execution_mode", None) == "remote"
        )

        return metadata, is_remote

    def _process_stream_chunk(
        self,
        chunk: dict[str, Any],
        accumulator: "RunAccumulator",
        *,
        on_step: Callable[[str, Any], None] | None,
        on_token: Callable[[str, dict[str, Any]], None] | None,
        on_custom: Callable[[Any], None] | None,
        cancel_check: Callable[[], bool] | None,
    ) -> None:
        """Process a ``StreamPart`` dict from the agent stream."""
        chunk_type = chunk.get("type", "")

        if chunk_type == "updates":
            accumulator.process_updates_chunk(chunk["data"], on_step)
        elif chunk_type == "messages":
            if on_token:
                token, meta = chunk["data"]
                try:
                    self._process_message_chunk(token, meta, on_token)
                except Exception:
                    pass  # best-effort
        elif chunk_type == "custom":
            if on_custom:
                try:
                    on_custom(chunk["data"])
                except Exception:
                    pass  # best-effort

        self._check_cancel(cancel_check)

    @staticmethod
    def _process_message_chunk(
        token: Any,
        metadata: dict[str, Any],
        on_token: Callable[[str, dict[str, Any]], None],
    ) -> None:
        """Extract text/reasoning from a ``messages`` stream chunk and forward to *on_token*.

        Called for every ``(token, metadata)`` pair emitted when ``stream_mode``
        includes ``"messages"``.  The *token* is typically an ``AIMessageChunk``.

        Content blocks are inspected to distinguish between regular text,
        reasoning/thinking tokens and partial tool-call arguments.  Each
        non-empty piece is forwarded to the caller-provided *on_token*
        callback with a ``meta`` dict containing:

        * ``type`` – ``"text"`` | ``"reasoning"`` | ``"tool_call"``
        * ``node`` – the LangGraph node name that produced the token
        * ``agent_name`` – agent name if available (for sub-agent disambiguation)
        """
        for text, meta in AgentService._extract_message_segments(token, metadata):
            on_token(text, meta)

    @staticmethod
    def _extract_message_segments(
        token: Any,
        metadata: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Return normalized text segments extracted from one message chunk."""

        if not isinstance(token, AIMessageChunk):
            return []

        node = metadata.get("langgraph_node", "")

        # Skip tokens emitted by middleware nodes (e.g.
        # PromptInjectionGuardMiddleware, CostTrackingMiddleware) — only
        # the main agent / model tokens should reach the UI.
        if "Middleware" in node:
            return []
        agent_name = metadata.get("lc_agent_name")

        base_meta: dict[str, Any] = {"node": node}
        if agent_name:
            base_meta["agent_name"] = agent_name
        segments: list[tuple[str, dict[str, Any]]] = []

        # Prefer content_blocks (normalised across providers) when available.
        content_blocks = getattr(token, "content_blocks", None)
        if content_blocks:
            for block in content_blocks:
                btype = block.get("type", "")
                if btype == "text" and block.get("text"):
                    segments.append((block["text"], {**base_meta, "type": "text"}))
                elif btype == "reasoning" and block.get("reasoning"):
                    segments.append(
                        (
                            block["reasoning"],
                            {**base_meta, "type": "reasoning"},
                        )
                    )
                # elif btype == "tool_call_chunk":
                elif "tool_call" in btype:
                    # Emit the tool name as a reasoning token so the
                    # UI shows "calling `tool`" in the Thinking section.
                    # Raw args are dropped to prevent them from leaking
                    # into the chat bubble as visible text.
                    name = block.get("name")
                    if name:
                        segments.append(
                            (
                                f"calling `{name}`\n",
                                {**base_meta, "type": "reasoning"},
                            )
                        )
            return segments

        # Fallback: plain string content (older providers / simple models).
        if token.content and isinstance(token.content, str):
            segments.append((token.content, {**base_meta, "type": "text"}))
        return segments

    def _snapshot_capture_enabled(self) -> bool:
        return bool(getattr(get_settings(), "agent_snapshot_capture_enabled", False))

    def _snapshot_replay_enabled(self) -> bool:
        return bool(getattr(get_settings(), "agent_snapshot_replay_enabled", False))

    def _build_snapshot_resolved_config(self) -> AgentSnapshotResolvedConfig:
        tools_summary = []
        for tool in self.tools:
            tools_summary.append(
                {
                    "name": getattr(tool, "name", type(tool).__name__),
                    "description": getattr(tool, "description", ""),
                    "requires_multimodal": bool(
                        getattr(tool, "requires_multimodal", False)
                    ),
                    "requires_network": bool(getattr(tool, "requires_network", False)),
                    "external_side_effects": bool(
                        getattr(tool, "external_side_effects", False)
                    ),
                }
            )

        middleware_summary = []
        for middleware in self._resolved_middleware or self._middleware:
            middleware_summary.append({"type": type(middleware).__name__})

        instance_context = None
        if self.instance_context is not None:
            instance_context = serialize_snapshot_value(
                {
                    "instance_id": self.instance_context.instance_id,
                    "instance_name": self.instance_context.instance_name,
                    "base_agent_name": self.instance_context.base_agent_name,
                    "subagent_configs": self.instance_context.subagent_configs,
                }
            )

        execution_mode = "local"
        if self.agent_config is not None:
            resolved_execution_mode = getattr(self.agent_config, "execution_mode", None)
            if isinstance(resolved_execution_mode, str) and resolved_execution_mode:
                execution_mode = resolved_execution_mode

        return AgentSnapshotResolvedConfig(
            agent_name=self.agent_name or "default_agent",
            model_name=self.model_name,
            execution_mode=execution_mode,
            system_prompt=self._enhanced_system_prompt,
            response_format=serialize_snapshot_value(self._resolved_response_format),
            tools=tools_summary,
            middleware=middleware_summary,
            instance_context=instance_context,
        )

    def _record_snapshot_trace_event(
        self,
        chunk: dict[str, Any],
        trace_events: list[AgentSnapshotTraceEvent],
    ) -> None:
        if not self._snapshot_capture_enabled():
            return

        event_type = chunk.get("type")
        namespace = list(chunk.get("ns", []))
        if event_type == "updates":
            trace_events.append(
                AgentSnapshotTraceEvent(
                    event_type="updates",
                    payload=serialize_snapshot_value(chunk.get("data", {})),
                    namespace=namespace,
                )
            )
            return

        if event_type == "custom":
            trace_events.append(
                AgentSnapshotTraceEvent(
                    event_type="custom",
                    payload=serialize_snapshot_value(chunk.get("data")),
                    namespace=namespace,
                )
            )
            return

        if event_type == "messages":
            token, metadata = chunk.get("data", (None, {}))
            segments = [
                AgentSnapshotTokenSegment(
                    text=text,
                    metadata=serialize_snapshot_value(meta),
                )
                for text, meta in self._extract_message_segments(token, metadata or {})
            ]
            if segments:
                trace_events.append(
                    AgentSnapshotTraceEvent(
                        event_type="messages",
                        segments=segments,
                        namespace=namespace,
                    )
                )

    def _save_agent_run_snapshot(
        self,
        *,
        user_input: str,
        context: Any,
        configurable: dict[str, Any],
        trace_events: list[AgentSnapshotTraceEvent],
        response: AgentResponse,
    ) -> None:
        settings = get_settings()
        if not self._snapshot_capture_enabled() or self._is_remote:
            return

        resolved_config = self._build_snapshot_resolved_config()
        snapshot = AgentRunSnapshot(
            snapshot_id=str(uuid.uuid4()),
            created_at=datetime.now(UTC),
            resolved_config=resolved_config,
            input=AgentSnapshotInput(
                user_input=user_input,
                context=serialize_snapshot_value(context),
                configurable=serialize_snapshot_value(configurable) or {},
                fingerprint=build_snapshot_fingerprint(
                    agent_name=resolved_config.agent_name,
                    model_name=resolved_config.model_name,
                    system_prompt=resolved_config.system_prompt,
                    response_format=resolved_config.response_format,
                    user_input=user_input,
                    context=context,
                ),
                query_text=build_snapshot_query_text(user_input, context),
            ),
            trace_events=trace_events,
            response=serialize_snapshot_value(response.model_dump(mode="python")),
        )
        try:
            get_agent_snapshot_store(settings).save_snapshot(snapshot)
        except Exception:
            logging.exception(
                "Failed to persist agent snapshot for '%s'",
                self.display_name,
            )

    def _maybe_replay_snapshot(
        self,
        *,
        user_input: str,
        context: Any,
        on_step: Callable[[str, Any], None] | None,
        on_token: Callable[[str, dict[str, Any]], None] | None,
        on_custom: Callable[[Any], None] | None,
    ) -> AgentResponse | None:
        if not self._snapshot_replay_enabled() or self._is_remote:
            return None

        settings = get_settings()
        resolved_config = self._build_snapshot_resolved_config()
        matcher = AgentSnapshotMatcher(
            get_agent_snapshot_store(settings),
            settings=settings,
        )
        match = matcher.find_best_match(
            agent_name=resolved_config.agent_name,
            model_name=resolved_config.model_name,
            system_prompt=resolved_config.system_prompt,
            response_format=resolved_config.response_format,
            user_input=user_input,
            context=context,
            match_mode=getattr(
                settings,
                "agent_snapshot_replay_match_mode",
                "exact_or_semantic",
            ),
            min_score=float(getattr(settings, "agent_snapshot_replay_min_score", 0.92)),
        )
        if match is None:
            return None

        for event in match.snapshot.trace_events:
            if event.event_type == "updates" and isinstance(event.payload, dict):
                for step_name, data in event.payload.items():
                    if on_step:
                        try:
                            on_step(step_name, data)
                        except Exception:
                            pass
            elif event.event_type == "messages" and on_token is not None:
                for segment in event.segments:
                    try:
                        on_token(segment.text, segment.metadata)
                    except Exception:
                        pass
            elif event.event_type == "custom" and on_custom is not None:
                try:
                    on_custom(event.payload)
                except Exception:
                    pass

        return AgentResponse.model_validate(
            rebuild_agent_response(match.snapshot.response)
        )

    def run_agent_steps(
        self,
        user_input: str,
        context=None,
        on_step: Callable[[str, Any], None] | None = None,
        on_token: Callable[[str, dict[str, Any]], None] | None = None,
        on_custom: Callable[[Any], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        graceful_cancel: bool = True,
        parent_runnable_config: RunnableConfig | None = None,
        trace_mode: Literal["separate", "nested", "nested-only"] = "separate",
    ) -> AgentResponse:
        """Execute the agent synchronously and return the response.

        Streams agent execution using v2 format (``StreamPart`` dicts),
        collecting steps and results.  Supports ``updates``, ``messages``,
        and ``custom`` stream modes simultaneously.

        Args:
            user_input: The user's input message.
            context: Optional context to pass to the agent.
            on_step: Callback for ``updates`` chunks — ``(step_name, data)``.
            on_token: Callback for ``messages`` chunks — ``(text, meta)``.
            on_custom: Callback for ``custom`` chunks — ``(data,)``.
                       Tools can emit custom data via ``get_stream_writer()``.
            cancel_check: Returns ``True`` to request cancellation.
            graceful_cancel: Use ``GraphInterrupt`` for clean LangSmith traces.
            parent_runnable_config: Parent runnable config to reuse when
                preserving outer trace context.
            trace_mode: Controls whether the inner agent keeps its own trace or
                stays nested under the caller's runnable trace.

        Raises:
            AgentCancelled: If *cancel_check* returns ``True``.
            RuntimeError: If agent is configured for remote execution.
        """
        if self._is_remote:
            raise RuntimeError(
                f"Agent '{self.display_name}' is configured for remote execution "
                f"(execution_mode='remote'). Use arun_agent_steps() instead — "
                f"the sync run_agent_steps() does not support remote delegation."
            )

        self._ensure_local_agent_initialized_sync()

        from inference_core.services.stream_utils import SyncInterruptibleStream

        start_time = datetime.now(UTC)
        accumulator = RunAccumulator()
        trace_events: list[AgentSnapshotTraceEvent] = []

        configurable = self._make_configurable()
        stream_config, cancel_cb, stream_modes = self._build_stream_config(
            configurable,
            on_token=on_token,
            on_custom=on_custom,
            graceful_cancel=graceful_cancel,
            parent_runnable_config=parent_runnable_config,
            trace_mode=trace_mode,
            sync=True,
        )

        replay_response = self._maybe_replay_snapshot(
            user_input=user_input,
            context=context,
            on_step=on_step,
            on_token=on_token,
            on_custom=on_custom,
        )
        if replay_response is not None:
            return replay_response

        with self._emulation_session_scope():
            raw_stream = self.agent.stream(
                {"messages": [HumanMessage(content=user_input)]},
                stream_config,
                stream_mode=stream_modes,
                context=context,
                version="v2",
            )

            if graceful_cancel:
                stream: Any = SyncInterruptibleStream(
                    raw_stream,
                    cancel_callback=cancel_cb,
                )
            else:
                stream = raw_stream

            try:
                for chunk in stream:
                    self._record_snapshot_trace_event(chunk, trace_events)
                    self._process_stream_chunk(
                        chunk,
                        accumulator,
                        on_step=on_step,
                        on_token=on_token,
                        on_custom=on_custom,
                        cancel_check=cancel_check,
                    )
            except AgentCancelled:
                if graceful_cancel and isinstance(stream, SyncInterruptibleStream):
                    stream.close()
                raise
            finally:
                if hasattr(raw_stream, "close"):
                    try:
                        raw_stream.close()
                    except Exception:
                        pass

        response = accumulator.build_response(
            self.model_name, start_time, self._enable_cost_tracking
        )
        self._save_agent_run_snapshot(
            user_input=user_input,
            context=context,
            configurable=configurable,
            trace_events=trace_events,
            response=response,
        )
        return response

    async def arun_agent_steps(
        self,
        user_input: str,
        context=None,
        on_step: Callable[[str, Any], None] | None = None,
        on_token: Callable[[str, dict[str, Any]], None] | None = None,
        on_custom: Callable[[Any], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        graceful_cancel: bool = True,
        parent_runnable_config: RunnableConfig | None = None,
        trace_mode: Literal["separate", "nested", "nested-only"] = "separate",
    ) -> AgentResponse:
        """Async version of run_agent_steps using astream.

        Use this in async contexts (e.g. FastAPI endpoints) to avoid
        asyncio event-loop conflicts.  See :meth:`run_agent_steps` for
        full parameter documentation.

        Args:
            graceful_cancel: Use ``GraphInterrupt`` for clean LangSmith
                traces.  Set to ``False`` for instant cancellation via
                ``aclose()`` at the cost of a failed LangSmith trace.

        Raises:
            AgentCancelled: If *cancel_check* returns ``True``.
        """
        # --- Remote execution path ---
        if self._is_remote:
            if trace_mode != "separate":
                raise RuntimeError(
                    f"trace_mode={trace_mode!r} is not supported for remote "
                    "Agent Server execution because the inner run is delegated "
                    "over HTTP and cannot inherit local parent callbacks. "
                    "Use trace_mode='separate' or force local agent execution."
                )
            return await self._arun_agent_steps_remote(
                user_input=user_input,
                on_step=on_step,
                on_token=on_token,
                on_custom=on_custom,
                cancel_check=cancel_check,
            )

        # --- Local execution path ---
        await self._ensure_local_agent_initialized_async()

        from inference_core.services.stream_utils import InterruptibleStream

        start_time = datetime.now(UTC)
        accumulator = RunAccumulator()
        trace_events: list[AgentSnapshotTraceEvent] = []

        configurable = self._make_configurable()
        stream_config, cancel_cb, stream_modes = self._build_stream_config(
            configurable,
            on_token=on_token,
            on_custom=on_custom,
            graceful_cancel=graceful_cancel,
            parent_runnable_config=parent_runnable_config,
            trace_mode=trace_mode,
        )

        replay_response = self._maybe_replay_snapshot(
            user_input=user_input,
            context=context,
            on_step=on_step,
            on_token=on_token,
            on_custom=on_custom,
        )
        if replay_response is not None:
            return replay_response

        with self._emulation_session_scope():
            raw_stream = self.agent.astream(
                {"messages": [HumanMessage(content=user_input)]},
                stream_config,
                stream_mode=stream_modes,
                context=context,
                version="v2",
            )

            if graceful_cancel:
                stream: Any = InterruptibleStream(
                    raw_stream,
                    cancel_callback=cancel_cb,
                )
            else:
                stream = raw_stream

            try:
                async for chunk in stream:
                    self._record_snapshot_trace_event(chunk, trace_events)
                    self._process_stream_chunk(
                        chunk,
                        accumulator,
                        on_step=on_step,
                        on_token=on_token,
                        on_custom=on_custom,
                        cancel_check=cancel_check,
                    )
            except AgentCancelled:
                if graceful_cancel and isinstance(stream, InterruptibleStream):
                    await stream.stop()
                    await stream.close()
                raise
            finally:
                if hasattr(raw_stream, "aclose"):
                    try:
                        await raw_stream.aclose()
                    except Exception:
                        pass

        response = accumulator.build_response(
            self.model_name, start_time, self._enable_cost_tracking
        )
        self._save_agent_run_snapshot(
            user_input=user_input,
            context=context,
            configurable=configurable,
            trace_events=trace_events,
            response=response,
        )
        return response

    def _ensure_local_agent_initialized_sync(self) -> None:
        """Ensure the local in-process agent graph exists before sync execution."""

        if self.agent is not None:
            return
        run_async_safely(self.create_agent())

    async def _ensure_local_agent_initialized_async(self) -> None:
        """Ensure the local in-process agent graph exists before async execution."""

        if self.agent is not None:
            return
        await self.create_agent()

    def _emulation_session_scope(self):
        """Return a run-scoped context for local emulated LLM timing.

        WHY: one agent run should share a single latency plan so consecutive
        model calls behave like one coherent mock session instead of isolated
        constant-delay invocations.
        """

        if not get_settings().llm_emulation_enabled:
            return nullcontext()
        return activate_emulation_session()

    async def _arun_agent_steps_remote(
        self,
        user_input: str,
        on_step: Callable[[str, Any], None] | None = None,
        on_token: Callable[[str, dict[str, Any]], None] | None = None,
        on_custom: Callable[[Any], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> AgentResponse:
        """Execute agent via the remote LangGraph Agent Server.

        Delegates the full agent run to the Agent Server while preserving
        the AgentResponse interface.
        """
        from inference_core.services.agent_server_client import (
            run_remote,
            stream_remote,
        )

        start_time = datetime.now(UTC)
        accumulator = RunAccumulator()

        remote_graph_id = self.agent_config.remote_graph_id
        metadata = self._build_remote_metadata()

        # Determine interrupt config from YAML
        interrupt_before = None
        interrupt_after = None
        if self.agent_config.interrupt_on:
            interrupt_before = self.agent_config.interrupt_on.get("before")
            interrupt_after = self.agent_config.interrupt_on.get("after")

        use_streaming = (
            on_token is not None or on_step is not None or on_custom is not None
        )

        def _capture_step(name: str, data: Any) -> None:
            accumulator.process_updates_chunk({name: data}, on_step)

        if use_streaming:
            result = await stream_remote(
                agent_name=self.agent_name,
                remote_graph_id=remote_graph_id,
                user_input=user_input,
                thread_id=(
                    self.checkpoint_config.get("thread_id")
                    if self.checkpoint_config
                    else None
                ),
                checkpoint_config=self.checkpoint_config,
                metadata=metadata,
                on_token=on_token,
                on_step=_capture_step,
                on_custom=on_custom,
                cancel_check=cancel_check,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
            )
        else:
            result = await run_remote(
                agent_name=self.agent_name,
                remote_graph_id=remote_graph_id,
                user_input=user_input,
                thread_id=(
                    self.checkpoint_config.get("thread_id")
                    if self.checkpoint_config
                    else None
                ),
                checkpoint_config=self.checkpoint_config,
                metadata=metadata,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
            )
            # For non-streaming, populate accumulator from the final result
            if isinstance(result, dict) and "messages" in result:
                accumulator.all_messages.extend(result["messages"])
            if isinstance(result, dict) and "structured_response" in result:
                accumulator.structured_response = result["structured_response"]

        agent_metadata = AgentMetadata(
            model_name=self.model_name,
            tools_used=[s["name"] for s in accumulator.steps],
            start_time=start_time,
            end_time=datetime.now(UTC),
        )

        return AgentResponse(
            result=result if isinstance(result, dict) else {"raw": result},
            steps=accumulator.steps,
            metadata=agent_metadata,
            cost_metrics=None,
            structured_response=accumulator.structured_response,
        )

    @staticmethod
    def build_config_for_instance(
        agent_instance_config: "dict | UserAgentInstance",
        base_config: Optional[LLMConfig] = None,
    ) -> LLMConfig:
        """Build an LLMConfig with all DB overrides from a UserAgentInstance applied.

        WHY: UserAgentInstance stores user customisations (model, tools, prompt params)
        as flat ORM columns and a JSON blob.  LLMConfig.with_overrides() needs those
        translated into typed dicts keyed by agent_name and model_name.  This method
        owns that translation so callers deal only with ORM objects or ready configs.

        Args:
            agent_instance_config: A dict (from UserAgentInstance.to_dict()) or an ORM
                UserAgentInstance object.  ORM objects are auto-converted via to_dict().
            base_config: Base LLMConfig to apply overrides onto.  When None, falls back
                to the global YAML config.  Pass the result of
                LLMConfigService.get_config_with_overrides() to include admin/user
                dynamic configuration in the resolution chain.

        Translation rules:
          primary_model                    → agent_overrides[base_agent_name]["primary"]
          config_overrides.fallback        → agent_overrides[...]["fallback"]
          config_overrides.allowed_tools   → agent_overrides[...]["allowed_tools"]
          config_overrides.mcp_profile     → agent_overrides[...]["mcp_profile"]
          config_overrides.temperature     → model_overrides[effective_model]["temperature"]
          config_overrides.max_tokens      → model_overrides[effective_model]["max_tokens"]
          instance.description             → agent_overrides[...]["description"]
        """
        # Normalise input: accept both ORM objects and plain dicts.
        if not isinstance(agent_instance_config, dict):
            agent_instance_config = agent_instance_config.to_dict()

        config = base_config if base_config is not None else get_llm_config()
        agent_name = agent_instance_config.get("base_agent_name")

        agent_overrides: dict[str, Any] = {}
        model_overrides: dict[str, dict[str, Any]] = {}

        primary_model = agent_instance_config.get("primary_model")
        if primary_model:
            agent_overrides["primary"] = primary_model

        description = agent_instance_config.get("description")
        if description:
            agent_overrides["description"] = description

        # Keys from config_overrides that map directly to AgentConfig fields
        _AGENT_LEVEL_KEYS = (
            "fallback",
            "allowed_tools",
            "mcp_profile",
            "memory_tools",
            "memory_session_context_enabled",
            "memory_tool_instructions_enabled",
            "response_format",
        )
        # Keys from config_overrides that map to ModelConfig / ModelParams fields
        _MODEL_LEVEL_KEYS = (
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "reasoning_effort",
        )

        config_overrides = canonicalize_fallback_overrides(
            agent_instance_config.get("config_overrides", {})
        )
        if config_overrides:
            for key in _AGENT_LEVEL_KEYS:
                if key in config_overrides:
                    agent_overrides[key] = config_overrides[key]

            model_params = {
                k: config_overrides[k]
                for k in _MODEL_LEVEL_KEYS
                if k in config_overrides
            }
            if model_params:
                # Target the effective model name: DB override if set, else YAML primary.
                # Must be resolved BEFORE with_overrides() is called so the correct
                # model entry is patched regardless of whether primary_model changes it.
                target_model = primary_model or config.agent_models.get(agent_name)
                if target_model:
                    model_overrides[target_model] = model_params

        if not agent_overrides and not model_overrides:
            return config  # Nothing to override; return base config unchanged

        return config.with_overrides(
            agent_overrides={agent_name: agent_overrides} if agent_overrides else None,
            model_overrides=model_overrides if model_overrides else None,
        )

    @classmethod
    def from_user_instance(
        cls,
        instance: "UserAgentInstance",
        use_checkpoints: bool = False,
        use_memory: bool = False,
        checkpoint_config: Optional[dict[str, Any]] = None,
        context_schema: Optional[Any] = None,
        middleware: Optional[list[Any]] = None,
        enable_cost_tracking: bool = True,
        user_id: Optional[uuid.UUID] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        base_config: Optional[LLMConfig] = None,
    ) -> "AgentService":
        """Create an AgentService fully configured from a UserAgentInstance ORM object.

        Steps:
          1. Apply all DB overrides (model, params, tools) to LLMConfig via
             build_config_for_instance() layered on top of base_config.
          2. Construct InstanceContext from the ORM instance fields.
          3. Pluck system_prompt_override / system_prompt_append from the DB row
             and store on the service so create_agent() applies them transparently.

        Args:
            instance: A UserAgentInstance ORM object.
            use_checkpoints / use_memory / ... : Runtime params forwarded to __init__.
            base_config: Base LLMConfig with admin/user dynamic overrides already
                         applied (e.g. from LLMConfigService.get_config_with_overrides()).
                         When None, falls back to raw YAML config.

        Returns:
            AgentService ready for ``await service.create_agent(system_prompt=...)``
            (or usable directly as a context manager).
        """
        resolved_config = cls.build_config_for_instance(
            instance, base_config=base_config
        )

        instance_ctx = cls.build_instance_context(instance)
        if instance_ctx.subagent_configs:
            logging.debug(
                "Built subagent_configs for '%s': %s",
                instance.instance_name,
                list(instance_ctx.subagent_configs.keys()),
            )

        return cls(
            agent_name=instance.base_agent_name,
            use_checkpoints=use_checkpoints,
            use_memory=use_memory,
            checkpoint_config=checkpoint_config,
            context_schema=context_schema,
            middleware=middleware,
            enable_cost_tracking=enable_cost_tracking,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            config=resolved_config,
            instance_context=instance_ctx,
            system_prompt_override=instance.system_prompt_override,
            system_prompt_append=instance.system_prompt_append,
            user_skills=instance.skills,
        )


class DeepAgentService(AgentService):
    """Service for creating agents with sub-agent orchestration via SubAgentMiddleware.

    Uses the standard langchain `create_agent` with deepagents middleware
    (SubAgentMiddleware, SkillsMiddleware) instead of the opinionated
    `create_deep_agent` factory.  This gives full control over the parent
    agent's middleware stack while still supporting:

    * Declarative subagent specs (SubAgent TypedDict) built from YAML config
    * Pre-compiled subagent runnables (CompiledSubAgent)
    * Per-subagent interrupt_on / skills / middleware
    * SkillsMiddleware for the parent agent itself
    """

    # Source: deepagents v0.4.4 – SubAgentMiddleware, SkillsMiddleware docs

    def __init__(
        self,
        agent_name: str,
        tools: Optional[list[Callable]] = None,
        subagents: Optional[list["AgentService | SubAgent | CompiledSubAgent"]] = None,
        interrupt_on: Optional[dict[str, bool | InterruptOnConfig]] = None,
        use_checkpoints: bool = False,
        use_memory: bool = False,
        checkpoint_config: Optional[dict[str, Any]] = None,
        context_schema: Optional[Any] = None,
        middleware: Optional[list[Any]] = None,
        enable_cost_tracking: bool = True,
        user_id: Optional[uuid.UUID] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        config: Optional[LLMConfig] = None,
        backend: Optional[BackendProtocol | BackendFactory] = None,
        system_prompt_override: Optional[str] = None,
        system_prompt_append: Optional[str] = None,
        instance_context: Optional[InstanceContext] = None,
        user_skills: Optional[list[dict[str, str]]] = None,
        _visited_agents: Optional[set[str]] = None,
    ):
        """Initialize the DeepAgentService.

        Args:
            agent_name: Name of the agent configuration to use (YAML config key).
            tools: List of tools to provide to the parent agent.
            subagents: Subagent specifications.  Accepts a mix of:
                - AgentService instances (compiled into CompiledSubAgent)
                - SubAgent dicts  (passed directly to SubAgentMiddleware)
                - CompiledSubAgent dicts (passed directly to SubAgentMiddleware)
            interrupt_on: Default interrupt_on config applied to config-based
                subagents that do not declare their own.
            use_checkpoints: Whether to enable checkpointing.
            use_memory: Whether to add memory tools and MemoryMiddleware.
            checkpoint_config: Checkpointer settings.
            context_schema: Optional schema for agent context.
            middleware: Extra middleware for the parent agent.
            enable_cost_tracking: Auto-add CostTrackingMiddleware.
            user_id: User ID for cost tracking / memory isolation.
            session_id: Session ID for grouping related requests.
            request_id: Correlation ID (e.g. Celery task ID).
            config: LLMConfig instance with overrides.
            backend: Backend for SubAgentMiddleware / SkillsMiddleware.
                Defaults to StateBackend (lazy factory).
            system_prompt_override: Fully replaces the system_prompt passed to
                create_agent().  Used by from_user_instance() when the DB instance
                has a complete custom prompt.
            system_prompt_append: Appended to the system_prompt passed to
                create_agent().  Used when the DB instance extends the base prompt.
            instance_context: Optional identity of the UserAgentInstance backing this run.
            _visited_agents: Internal recursion guard.
        """
        super().__init__(
            agent_name,
            tools,
            use_checkpoints=use_checkpoints,
            use_memory=use_memory,
            checkpoint_config=checkpoint_config,
            context_schema=context_schema,
            middleware=middleware,
            enable_cost_tracking=enable_cost_tracking,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            config=config,
            instance_context=instance_context,
            system_prompt_override=system_prompt_override,
            system_prompt_append=system_prompt_append,
            user_skills=user_skills,
        )
        self._explicit_subagents = subagents or []

        # Merge interrupt_on: YAML config base + dynamic overrides on top
        self.interrupt_on: dict[str, Any] = {}
        if self.agent_config and self.agent_config.interrupt_on:
            self.interrupt_on.update(self.agent_config.interrupt_on)
        if interrupt_on:
            self.interrupt_on.update(interrupt_on)

        # Backend for SubAgentMiddleware / SkillsMiddleware; StateBackend is
        # a lightweight factory that requires no filesystem root.
        self._backend: BackendProtocol | BackendFactory = backend or StateBackend

        self._visited_agents = _visited_agents or set()
        self._visited_agents.add(agent_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_agent(
        self, system_prompt: Optional[str] = None, **kwargs
    ) -> Callable:
        """Create a standard agent with SubAgentMiddleware & SkillsMiddleware.

        Steps:
        1. Load parent's own tools (providers, MCP, memory)
        2. Build subagent specs (explicit + YAML-config-based)
        3. Inject SubAgentMiddleware into middleware stack
        4. Optionally inject SkillsMiddleware
        5. Call langchain create_agent (NOT create_deep_agent)

        Args:
            system_prompt: System prompt for the parent agent.
            **kwargs: Forwarded to ``create_agent``.

        Returns:
            The compiled agent graph.
        """
        # 1. Load parent's own tools
        provider_context = self._build_provider_user_context(kwargs.get("user_context"))
        await self._load_providers_tools(user_context=provider_context)
        await self._load_mcp_tools()
        await self._load_memory_tools()

        # 2. Apply DB-backed prompt overrides, then build enhanced system prompt
        # (memory tool instructions are appended by _build_system_prompt if enabled)
        effective_prompt = self._apply_prompt_overrides(system_prompt)
        self._enhanced_system_prompt = self._build_system_prompt(effective_prompt)

        # 3. Build base middleware (cost tracking, memory, tool-model switch)
        middleware = self._build_middleware()

        # Extract CostTrackingMiddleware and MemoryMiddleware to propagate to subagents
        propagated_middleware = [
            m
            for m in middleware
            if isinstance(m, (CostTrackingMiddleware, MemoryMiddleware))
        ]

        # 4. Build subagent specs and add SubAgentMiddleware
        subagent_specs = await self._build_subagent_specs(
            inherited_middleware=propagated_middleware
        )
        if subagent_specs:
            sub_middleware = SubAgentMiddleware(
                backend=self._backend,
                subagents=subagent_specs,
            )
            middleware.append(sub_middleware)

        # 5. Add SkillsMiddleware if parent has skills configured (YAML or user-defined)
        yaml_skills = (self.agent_config.skills if self.agent_config else None) or []
        user_skills = self._user_skills or []

        if yaml_skills or user_skills:
            if user_skills:
                # User skills require a StoreBackend populated with skill files.
                # Merge YAML skill sources with user-defined skill sources.
                store, user_skill_sources = self._build_skills_store(user_skills)
                all_sources = list(yaml_skills) + user_skill_sources
                skills_backend = lambda rt: StoreBackend(rt)
                kwargs.setdefault("store", store)
            else:
                all_sources = list(yaml_skills)
                skills_backend = self._backend

            skills_middleware = create_skills_middleware(
                backend=skills_backend,
                sources=all_sources,
            )
            middleware.append(skills_middleware)
            add_skill_reader_tool(
                self.tools,
                backend=skills_backend,
                sources=all_sources,
            )

        # 6. Ensure CostTrackingMiddleware is absolute last so its
        #    after_model (state accumulation) runs FIRST in reverse order,
        #    before any interrupt-causing middleware.
        self._ensure_cost_tracking_last(middleware)

        # 7. Create the agent via standard create_agent
        # Resolve structured-output schema (inherited from AgentService helper).
        # Forward via kwargs only when set and not explicitly overridden by caller.
        response_format = self._resolve_response_format()
        if response_format is not None and "response_format" not in kwargs:
            kwargs["response_format"] = response_format

        self.agent = create_agent(
            self.model,
            tools=self.tools,
            checkpointer=self.checkpointer,
            context_schema=self.context_schema,
            middleware=middleware,
            system_prompt=self._enhanced_system_prompt,
            **kwargs,
        )
        self.model_params = self.model_factory.config.get_model_params(self.model_name)
        return self.agent

    # ------------------------------------------------------------------
    # Subagent spec builders
    # ------------------------------------------------------------------

    @classmethod
    async def from_user_instance(
        cls,
        instance: UserAgentInstance,
        use_checkpoints: bool = False,
        use_memory: bool = False,
        checkpoint_config: Optional[dict[str, Any]] = None,
        context_schema: Optional[Any] = None,
        middleware: Optional[list[Any]] = None,
        enable_cost_tracking: bool = True,
        user_id: Optional[uuid.UUID] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        backend: Optional[BackendProtocol | BackendFactory] = None,
        base_config: Optional[LLMConfig] = None,
        _visited_agents: Optional[set[str]] = None,
    ) -> "DeepAgentService":
        """Create a DeepAgentService fully configured from a UserAgentInstance ORM object.

        WHY: UserAgentInstance stores user-specific agent customisations in the DB
        (custom model, prompt overrides, JSON config blob, M2M subagent relationships).
        This factory bridges the ORM layer to the runtime service by:
          1. Applying all DB overrides to LLMConfig so that model/config resolution
             downstream uses DB values instead of raw YAML values.
          2. Resolving the M2M subagents relationship recursively so nested deep agents
             are pre-compiled and simple agents become declarative SubAgent specs.
          3. Storing prompt override/append on the returned service so create_agent()
             applies them transparently without the caller knowing about DB config.

        Args:
            instance: A UserAgentInstance with .subagents loaded.  The default
                      lazy="selectin" on the relationship means subagents are loaded
                      automatically as long as the SQLAlchemy session is still open.
            use_checkpoints / use_memory / ... : Runtime params forwarded to __init__.
            base_config: Base LLMConfig with admin/user dynamic overrides already
                         applied.  Pass LLMConfigService.get_config_with_overrides()
                         result to include the full dynamic config resolution chain.
                         When None, falls back to raw YAML config.
            _visited_agents: Internal recursion guard.  Callers must not set this.
                             Holds both YAML agent names and instance UUID strings.

        Returns:
            DeepAgentService ready for ``await service.create_agent(system_prompt=...)``.

        Raises:
            ValueError: When a circular subagent reference is detected.
        """
        visited: set[str] = set(_visited_agents or set())
        instance_key = str(instance.id)

        if instance_key in visited:
            raise ValueError(
                f"Circular subagent reference detected for instance "
                f"'{instance.instance_name}' (id={instance.id})"
            )
        visited.add(instance_key)

        # 1. Build an LLMConfig that incorporates all DB overrides for this instance.
        #    When base_config is provided (e.g. from LLMConfigService), admin overrides
        #    and user preferences are already baked in; instance overrides go on top.
        resolved_config = cls.build_config_for_instance(
            instance, base_config=base_config
        )

        # 2. Resolve the DB subagents relationship into middleware-ready specs.
        db_subagents: list[SubAgent | CompiledSubAgent] = []

        for sub_instance in instance.subagents or []:
            sub_key = str(sub_instance.id)
            if sub_key in visited:
                logging.debug(
                    "Skipping DB subagent '%s' (id=%s) – already visited (recursion guard)",
                    sub_instance.instance_name,
                    sub_instance.id,
                )
                continue

            if sub_instance.is_deepagent or sub_instance.subagents:
                # Nested deep agent: recursively build and pre-compile its graph so the
                # parent's SubAgentMiddleware receives a CompiledSubAgent runnable.
                nested_service = await cls.from_user_instance(
                    sub_instance,
                    use_checkpoints=use_checkpoints,
                    use_memory=use_memory,
                    checkpoint_config=checkpoint_config,
                    context_schema=context_schema,
                    middleware=middleware,
                    enable_cost_tracking=enable_cost_tracking,
                    user_id=user_id,
                    session_id=session_id,
                    request_id=request_id,
                    backend=backend,
                    base_config=base_config,
                    _visited_agents=visited | {sub_key},
                )
                # system_prompt=None: the nested service applies its own
                # _system_prompt_override / _system_prompt_append internally.
                await nested_service.create_agent()
                db_subagents.append(nested_service)
            else:
                # Simple (non-deep) subagent: build a declarative SubAgent dict.
                # Model is resolved via LLMModelFactory so SubAgentMiddleware
                # receives a BaseChatModel instance and skips init_chat_model().
                sub_config = cls.build_config_for_instance(
                    sub_instance, base_config=base_config
                )
                sub_model = LLMModelFactory(sub_config).get_model_for_agent(
                    sub_instance.base_agent_name
                )
                if sub_model is None:
                    logging.warning(
                        "Could not create model for DB subagent '%s' (id=%s) – skipping",
                        sub_instance.instance_name,
                        sub_instance.id,
                    )
                    continue

                effective_system_prompt: Optional[str] = (
                    sub_instance.system_prompt_override
                    or sub_instance.description
                    or f"You are {sub_instance.instance_name}."
                )
                if (
                    sub_instance.system_prompt_append
                    and not sub_instance.system_prompt_override
                ):
                    effective_system_prompt = f"{effective_system_prompt}\n\n{sub_instance.system_prompt_append}"

                spec = cls._build_declarative_subagent_spec(
                    name=sub_instance.instance_name,
                    description=sub_instance.description
                    or f"Subagent {sub_instance.instance_name}",
                    system_prompt=effective_system_prompt,
                    model=sub_model,
                    tools=[],
                    base_agent_name=sub_instance.base_agent_name,
                )
                db_subagents.append(spec)

        # 3. Construct and return the service.  resolved_config propagates through
        #    LLMModelFactory so all model/config lookups use DB-overridden values.
        # Build subagent_configs for Agent Server remote execution (same as
        # base AgentService.from_user_instance — needed when DeepAgentService
        # is used with execution_mode='remote').
        subagent_configs: dict[str, dict[str, Any]] = {}
        for sub in instance.subagents or []:
            subagent_configs[sub.base_agent_name] = {
                "instance_id": str(sub.id),
                "instance_name": sub.instance_name,
                "primary_model": sub.primary_model,
                "system_prompt_override": sub.system_prompt_override,
                "system_prompt_append": sub.system_prompt_append,
            }

        instance_ctx = InstanceContext(
            instance_id=instance.id,
            instance_name=instance.instance_name,
            base_agent_name=instance.base_agent_name,
            subagent_configs=subagent_configs,
        )
        return cls(
            agent_name=instance.base_agent_name,
            subagents=db_subagents,
            use_checkpoints=use_checkpoints,
            use_memory=use_memory,
            checkpoint_config=checkpoint_config,
            context_schema=context_schema,
            middleware=middleware,
            enable_cost_tracking=enable_cost_tracking,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            config=resolved_config,
            backend=backend,
            system_prompt_override=instance.system_prompt_override,
            system_prompt_append=instance.system_prompt_append,
            instance_context=instance_ctx,
            user_skills=instance.skills,
            _visited_agents=visited,
        )

    async def _build_subagent_specs(
        self,
        inherited_middleware: Optional[list[Any]] = None,
    ) -> list[SubAgent | CompiledSubAgent]:
        """Merge explicit subagents with YAML-config-based ones.

        Explicit subagents are processed first; config-based ones are added
        only if their name has not already been claimed by an explicit spec.
        A visited-agents set prevents infinite recursion.
        """
        specs: list[SubAgent | CompiledSubAgent] = []
        excluded_base_names: set[str] = set()

        # --- Explicit subagents (passed via constructor) ---
        for subagent in self._explicit_subagents:
            if isinstance(subagent, AgentService):
                if subagent.instance_context:
                    excluded_base_names.add(subagent.instance_context.base_agent_name)
                elif subagent.agent_name:
                    excluded_base_names.add(subagent.agent_name)
            elif isinstance(subagent, dict) and "base_agent_name" in subagent:
                excluded_base_names.add(subagent["base_agent_name"])
                # Remove it so we don't leak non-standard keys to downstream middlewares
                subagent = {k: v for k, v in subagent.items() if k != "base_agent_name"}

            spec = await self._resolve_explicit_subagent(
                subagent, inherited_middleware=inherited_middleware
            )
            specs.append(spec)

        # --- Config-defined subagents ---
        if self.agent_config and self.agent_config.subagents:
            added_names = self._collect_spec_names(specs)
            for subagent_name in self.agent_config.subagents:
                if subagent_name in added_names or subagent_name in excluded_base_names:
                    continue
                if subagent_name in self._visited_agents:
                    logging.debug(
                        "Skipping subagent '%s' – already visited (recursion guard)",
                        subagent_name,
                    )
                    continue

                spec = await self._build_subagent_from_config(
                    subagent_name, inherited_middleware=inherited_middleware
                )
                if spec is not None:
                    specs.append(spec)

        return specs

    async def _resolve_explicit_subagent(
        self,
        subagent: "AgentService | SubAgent | CompiledSubAgent",
        inherited_middleware: Optional[list[Any]] = None,
    ) -> SubAgent | CompiledSubAgent:
        """Turn an explicit subagent into a middleware-ready spec.

        AgentService instances are compiled and wrapped as CompiledSubAgent.
        Raw SubAgent / CompiledSubAgent dicts are passed through.
        """
        if isinstance(subagent, AgentService):
            if not hasattr(subagent, "agent") or subagent.agent is None:
                await subagent.create_agent()
            return CompiledSubAgent(
                name=subagent.display_name,
                description=(
                    subagent.agent_config.description
                    or f"Subagent {subagent.display_name}"
                ),
                runnable=subagent.agent,
            )

        elif isinstance(subagent, dict) and "model" in subagent:
            subagent_name = subagent.get("name", "Unnamed Subagent")

            model = subagent.get("model")
            if model:
                if isinstance(model, str):
                    model = self.model_factory.create_model(model)
            else:
                model = self.model_factory.get_model_for_agent(subagent_name)

            if model is None:
                logging.warning(
                    "Could not create model for subagent '%s' – skipping",
                    subagent_name,
                )
                return None

            tools = subagent.get("tools", [])
            middleware = subagent.get("middleware", [])

            # Combine inherited middleware
            if inherited_middleware:
                for m in inherited_middleware:
                    # Avoid duplicates by type
                    if not any(
                        isinstance(existing, type(m)) for existing in middleware
                    ):
                        middleware.append(m)

            # CostTracking must be last so it runs first in after_model
            self._ensure_cost_tracking_last(middleware)

            # Per-subagent interrupt_on: own config wins, parent's as fallback
            sub_interrupt = subagent.get("interrupt_on") or self.interrupt_on
            description = subagent.get("description") or f"Subagent {subagent_name}"
            system_prompt = (
                subagent.get("system_prompt")
                or description
                or f"You are {subagent_name}."
            )
            skills = subagent.get("skills", [])

            spec = self._build_declarative_subagent_spec(
                name=subagent_name,
                description=description,
                system_prompt=system_prompt,
                model=model,
                tools=tools,
                interrupt_on=sub_interrupt,
                skills=skills,
                middleware=middleware,
            )

            return spec

        # Assume it's already a valid SubAgent or CompiledSubAgent
        return subagent

    async def _build_subagent_from_config(
        self,
        subagent_name: str,
        inherited_middleware: Optional[list[Any]] = None,
    ) -> Optional[SubAgent | CompiledSubAgent]:
        """Build a SubAgent spec from YAML agent configuration.

        Constructs a declarative SubAgent dict so that SubAgentMiddleware
        handles compilation internally.  Falls back to compiling via
        AgentService if the subagent itself has nested subagents.
        """
        sub_config = self.model_factory.config.get_specific_agent_config(subagent_name)
        if not sub_config:
            logging.warning(
                "No config found for subagent '%s' – skipping", subagent_name
            )
            return None

        # If the subagent itself has nested subagents, recursively create a
        # DeepAgentService and wrap it as CompiledSubAgent.
        if sub_config.subagents:
            return await self._compile_nested_deep_subagent(
                subagent_name, sub_config, inherited_middleware=inherited_middleware
            )

        # Otherwise build a declarative SubAgent spec
        model = self.model_factory.get_model_for_agent(subagent_name)
        if model is None:
            logging.warning(
                "Could not create model for subagent '%s' – skipping",
                subagent_name,
            )
            return None

        tools = await self._load_tools_for_subagent(subagent_name, sub_config)

        middleware = []
        if inherited_middleware:
            middleware.extend(inherited_middleware)

        # Add subagent's own tool-call limits from its YAML config
        from inference_core.llm.config import ToolCallLimitsConfig as _TCL

        sub_tool_call_limits = getattr(sub_config, "tool_call_limits", None)
        if isinstance(sub_tool_call_limits, _TCL):
            from inference_core.agents.middleware.tool_call_limits import (
                build_tool_call_limit_middleware,
            )

            middleware.extend(build_tool_call_limit_middleware(sub_tool_call_limits))

        _sub_gen = getattr(sub_config, "generation_params", None)
        fallback_middleware = build_model_fallback_middleware(
            model_factory=self.model_factory,
            fallback_models=getattr(sub_config, "fallback", None),
            primary_model=self.model_factory.get_agent_model_name(subagent_name),
            reasoning_output=getattr(sub_config, "reasoning_output", False),
            generation_params=_sub_gen.as_overrides() if _sub_gen else None,
            owner=f"subagent {subagent_name}",
        )
        if fallback_middleware is not None:
            middleware.append(fallback_middleware)

        # CostTracking must be last so it runs first in after_model
        self._ensure_cost_tracking_last(middleware)

        # Per-subagent interrupt_on: own config wins, parent's as fallback
        sub_interrupt = sub_config.interrupt_on or self.interrupt_on

        spec = self._build_declarative_subagent_spec(
            name=subagent_name,
            description=sub_config.description or f"Subagent {subagent_name}",
            system_prompt=(
                sub_config.system_prompt
                or sub_config.description
                or f"You are {subagent_name}."
            ),
            model=model,
            tools=tools,
            interrupt_on=sub_interrupt,
            skills=sub_config.skills,
            middleware=middleware,
        )

        return spec

    @staticmethod
    def _build_declarative_subagent_spec(
        name: str,
        description: str,
        system_prompt: str,
        model: Any,
        tools: list[Any],
        base_agent_name: Optional[str] = None,
        interrupt_on: Optional[dict[str, bool | InterruptOnConfig]] = None,
        skills: Optional[list[Any]] = None,
        middleware: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        """Creates a standardized declarative SubAgent TypedDict.

        Using this ensures that SubAgentMiddleware receives a consistent dictionary
        with properly initialized instances, explicitly skipping string-based LangChain `init_chat_model` bugs.
        """
        spec: dict[str, Any] = {
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
            "model": model,
            "tools": tools,
        }

        if base_agent_name:
            spec["base_agent_name"] = base_agent_name
        if interrupt_on is not None:
            spec["interrupt_on"] = interrupt_on
        if skills is not None:
            spec["skills"] = skills
        if middleware is not None:
            spec["middleware"] = middleware

        return spec

    async def _compile_nested_deep_subagent(
        self,
        subagent_name: str,
        sub_config: Any,
        inherited_middleware: Optional[list[Any]] = None,
    ) -> CompiledSubAgent:
        """Recursively compile a subagent that itself has subagents.

        Creates a nested DeepAgentService, compiles its graph, and wraps
        the result as CompiledSubAgent for the parent's SubAgentMiddleware.
        """
        nested_service = DeepAgentService(
            agent_name=subagent_name,
            use_checkpoints=self.use_checkpoints,
            use_memory=self.use_memory,
            checkpoint_config=self.checkpoint_config,
            context_schema=self.context_schema,
            middleware=inherited_middleware,
            enable_cost_tracking=self._enable_cost_tracking,
            user_id=self._user_id,
            session_id=self._session_id,
            request_id=self._request_id,
            config=self.config,
            backend=self._backend,
            _visited_agents=set(self._visited_agents),
        )
        await nested_service.create_agent(
            system_prompt=(
                sub_config.system_prompt
                or sub_config.description
                or f"You are {subagent_name}."
            )
        )
        return CompiledSubAgent(
            name=subagent_name,
            description=sub_config.description or f"Subagent {subagent_name}",
            runnable=nested_service.agent,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _load_tools_for_subagent(
        self, agent_name: str, agent_config: Any
    ) -> list[Any]:
        """Load tools from registered providers for a config-based subagent.

        Mirrors the tool-loading logic of AgentService._load_providers_tools
        but returns the tools instead of mutating self.tools.  The parent's
        user context (user_id, session_id, request_id) is forwarded so that
        user-scoped providers (e.g. email tools) can resolve per-user
        resources.
        """
        configured_providers = agent_config.local_tool_providers or []
        if not configured_providers:
            return []

        registered_providers = get_registered_providers()
        confirmed = [
            name for name in configured_providers if name in registered_providers
        ]
        if not confirmed:
            return []

        # Build the same user context that _load_providers_tools passes,
        # reusing the parent agent's identity fields.
        provider_context = self._build_provider_user_context()

        # Resolve multimodal capability for the subagent's primary model so
        # the loader can filter/delegate tools consistently.
        sub_model_name = self.model_factory.get_agent_model_name(agent_name)
        sub_model_cfg = (
            self.model_factory.config.get_model_config(sub_model_name)
            if sub_model_name
            else None
        )
        model_multimodal = bool(getattr(sub_model_cfg, "multimodal", False))
        on_missing_capability = getattr(agent_config, "on_missing_capability", "skip")

        try:
            return await load_tools_for_agent(
                agent_name,
                provider_names=confirmed,
                allowed_tools=agent_config.allowed_tools,
                model_multimodal=model_multimodal,
                on_missing_capability=on_missing_capability,
                user_context=provider_context,
                user_id=provider_context.get("user_id"),
                session_id=provider_context.get("session_id"),
                request_id=provider_context.get("request_id"),
            )
        except Exception as e:
            logging.warning("Failed to load tools for subagent '%s': %s", agent_name, e)
            return []

    @staticmethod
    def _collect_spec_names(
        specs: list[SubAgent | CompiledSubAgent],
    ) -> set[str]:
        """Extract the 'name' field from a list of subagent specs."""
        names: set[str] = set()
        for s in specs:
            if isinstance(s, dict):
                name = s.get("name")
            else:
                name = getattr(s, "name", None)
            if name:
                names.add(name)
        return names


# ---------------------------------------------------------------------------
# Checkpointer construction with an injectable serde (privacy Faza 2).
# ``from_conn_string`` does not accept a serializer, so these replicate its
# exact connection setup and pass ``serde=`` through to the saver.
# ---------------------------------------------------------------------------


@contextmanager
def _postgres_saver_with_serde(url: str, serde):
    from psycopg import Connection
    from psycopg.rows import dict_row

    with Connection.connect(
        url, autocommit=True, prepare_threshold=0, row_factory=dict_row
    ) as conn:
        yield PostgresSaver(conn, serde=serde)


@contextmanager
def _sqlite_saver_with_serde(url: str, serde):
    import sqlite3
    from contextlib import closing

    with closing(
        sqlite3.connect(
            url,
            # https://ricardoanderegg.com/posts/python-sqlite-thread-safety/
            check_same_thread=False,
        )
    ) as conn:
        yield SqliteSaver(conn, serde=serde)
