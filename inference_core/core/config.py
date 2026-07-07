import os
from functools import lru_cache
from typing import Any, List, Literal, Optional

from dotenv import dotenv_values
from pydantic import Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import (
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
)

from inference_core.core.env import get_project_dotenv_path

###################################
#            Classess             #
###################################


@lru_cache(maxsize=4)
def _configured_dotenv_keys(dotenv_path: str) -> frozenset[str]:
    """Return the setting keys explicitly defined in the selected dotenv file.

    WHY: testing-safe defaults must respect both shell-provided environment
    variables and values declared in `.env` / `.env.test`, otherwise the app
    would silently override operator intent.
    """

    values = dotenv_values(dotenv_path)
    return frozenset(key for key in values if key)


def _setting_is_explicitly_configured(env_name: str) -> bool:
    """Return whether a setting was provided by env vars or the active dotenv.

    WHY: `ENVIRONMENT=testing` should enable no-cost defaults only when the
    caller did not already choose a concrete runtime behavior.
    """

    if env_name in os.environ:
        return True
    dotenv_path = str(get_project_dotenv_path())
    return env_name in _configured_dotenv_keys(dotenv_path)


class ListParsingEnvSource(EnvSettingsSource):
    """Custom settings source that handles comma-separated lists in environment variables"""

    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        """
        Parse comma-separated strings into lists for specific CORS and ALLOWED_HOSTS fields
        """

        # Handle CORS list fields
        if field_name in (
            "cors_methods",
            "cors_origins",
            "cors_headers",
            "allowed_hosts",
            "llm_tool_emulation_include",
            "llm_tool_emulation_exclude",
        ) and isinstance(value, str):
            # Handle special case of "*" which should remain as single item
            if value.strip() == "*":
                return ["*"]
            # Split by comma and clean whitespace, remove quotes
            cleaned = value.strip().strip('"').strip("'")
            return [item.strip() for item in cleaned.split(",") if item.strip()]

        # For other complex types, use default JSON parsing
        if value_is_complex and value is not None:
            return self.decode_complex_value(field_name, field, value)

        return value


class ListParsingDotEnvSource(DotEnvSettingsSource):
    """Custom dotenv settings source that handles comma-separated lists"""

    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        """
        Parse comma-separated strings into lists for specific CORS and ALLOWED_HOSTS fields
        """
        # Handle CORS and ALLOWED_HOSTS list fields
        if field_name in (
            "cors_methods",
            "cors_origins",
            "cors_headers",
            "allowed_hosts",
            "llm_tool_emulation_include",
            "llm_tool_emulation_exclude",
        ) and isinstance(value, str):
            # Handle special case of "*" which should remain as single item
            if value.strip() == "*":
                return ["*"]
            # Split by comma and clean whitespace, remove quotes
            cleaned = value.strip().strip('"').strip("'")
            return [item.strip() for item in cleaned.split(",") if item.strip()]

        # For other complex types, use default JSON parsing
        if value_is_complex and value is not None:
            return self.decode_complex_value(field_name, field, value)

        return value


class Settings(BaseSettings):
    """Application settings - all configuration in one place"""

    ###################################
    #     APPLICATION SETTINGS        #
    ###################################
    app_name: str = Field(
        default="Backend Template API", description="Application name"
    )
    app_title: str = Field(
        default="Backend Template API", description="Application title"
    )
    app_description: str = Field(
        default="A production-ready FastAPI backend template with LLM integration",
        description="Application description",
    )
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(
        default="development", description="Application environment"
    )
    debug: bool = Field(default=True, description="Debug mode")
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    app_public_url: str = Field(
        default="http://localhost:8000",
        description="Public URL for the application (used in emails)",
    )

    ###################################
    #           CORS SETTINGS         #
    ###################################

    cors_methods: List[str] = Field(default=["*"], description="CORS allowed methods")
    cors_headers: List[str] = Field(default=["*"], description="CORS allowed headers")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")

    ###################################
    #       TRUSTED HOST SETTINGS     #
    ###################################

    allowed_hosts: Optional[List[str]] = Field(
        default=None,
        description="Trusted hosts for TrustedHostMiddleware (separate from CORS). "
        "If not set, will derive from cors_origins by normalizing hostnames.",
    )

    ###################################
    #           DATABASE              #
    ###################################
    # database_url: str = Field(
    #     default="sqlite+aiosqlite:///./inference_core.db",
    #     description="Database connection URL",
    # )

    database_service: str = Field(
        default="sqlite+aiosqlite",
        description="Database service type (sqlite, postgresql, mysql)",
    )

    database_host: str = Field(
        default="localhost",
        description="Database host",
    )

    database_port: int = Field(
        default=5432,
        description="Database port",
    )

    database_name: str = Field(
        default="app_db",
        description="Database name",
    )

    database_user: Optional[str] = Field(
        default="db_user",
        description="Database user",
    )
    database_password: Optional[str] = Field(
        default="db_password",
        description="Database password",
    )

    database_echo: bool = Field(
        default=False,
        description="Echo SQL queries (development only)",
    )
    database_pool_size: int = Field(
        default=20,
        description="Database connection pool size",
    )
    database_max_overflow: int = Field(
        default=30,
        description="Maximum database connection overflow",
    )
    database_pool_timeout: int = Field(
        default=30,
        description="Pool connection timeout in seconds",
    )
    database_pool_recycle: int = Field(
        default=3600,
        description="Connection recycle time in seconds",
    )
    database_pool_class: Optional[Literal["default", "null"]] = Field(
        default="default",
        description="Connection pool class. Use 'null' (NullPool) for Jupyter notebooks or multi-event-loop environments to avoid 'different loop' errors. 'default' uses standard QueuePool.",
    )
    database_mysql_charset: str = Field(
        default="utf8mb4",
        description="MySQL character set",
    )
    database_url: str = Field(
        default="sqlite+aiosqlite:///./inference_core.db",
        description="Database connection URL",
    )

    @model_validator(mode="after")
    def set_database_url(self) -> "Settings":
        """Construct database URL based on service type and other parameters"""
        if self.database_service == "sqlite+aiosqlite":
            self.database_url = f"{self.database_service}:///./inference_core.db"
        else:
            self.database_url = (
                f"{self.database_service}://{self.database_user}:{self.database_password}"
                f"@{self.database_host}:{self.database_port}/{self.database_name}"
            )
        return self

    @model_validator(mode="after")
    def apply_testing_safe_defaults(self) -> "Settings":
        """Apply no-cost runtime defaults for testing when not configured.

        WHY: starting the API with only `ENVIRONMENT=testing` should not try to
        instantiate provider-backed chat models or real network tools. Explicit
        env or dotenv values must still win so operators can opt back into local
        embeddings or provider-backed paths when needed.
        """

        if not self.is_testing:
            return self

        testing_defaults = {
            "LLM_EMULATION_ENABLED": ("llm_emulation_enabled", True),
            "LLM_TOOL_EMULATION_MODE": ("llm_tool_emulation_mode", "external"),
            "AGENT_TOOL_ENVIRONMENT": ("agent_tool_environment", "strict_test"),
            "AGENT_REQUIRE_TEST_DOUBLES": ("agent_require_test_doubles", True),
            "AGENT_TOOL_DOUBLE_STRATEGY": (
                "agent_tool_double_strategy",
                "replace",
            ),
            "EMBEDDING_BACKEND": ("embedding_backend", "fake"),
            "PII_BACKEND": ("pii_backend", "regex"),
        }

        for env_name, (field_name, value) in testing_defaults.items():
            if not _setting_is_explicitly_configured(env_name):
                setattr(self, field_name, value)

        return self

    ###################################
    #           MONITORING            #
    ###################################
    sentry_dsn: Optional[str] = Field(
        default=None, description="Sentry DSN for error monitoring"
    )
    sentry_traces_sample_rate: float = Field(
        default=1.0,
        description="Sentry traces sample rate (0.0 to 1.0)",
    )
    sentry_profiles_sample_rate: float = Field(
        default=1.0,
        description="Sentry profiles sample rate (0.0 to 1.0)",
    )

    ###################################
    #           VALIDATORS            #
    ###################################
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        allowed_envs = ["development", "staging", "production", "testing"]
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of: {allowed_envs}")
        return v

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate and enhance database URL"""
        if not v:
            raise ValueError("Database URL cannot be empty")

        # Ensure async drivers are used
        url_mappings = {
            "postgresql://": "postgresql+asyncpg://",
            "mysql://": "mysql+aiomysql://",
            "sqlite:///": "sqlite+aiosqlite:///",
        }

        for old_prefix, new_prefix in url_mappings.items():
            if v.startswith(old_prefix):
                v = v.replace(old_prefix, new_prefix, 1)
                break

        return v

    ###################################
    #           PROPERTIES            #
    ###################################
    # Security / Auth
    secret_key: str = Field(
        default="change-me-in-production",
        description="Secret key for JWT signing",
    )
    algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    access_token_expire_minutes: int = Field(
        default=30, description="Access token expiration in minutes"
    )
    refresh_token_expire_days: int = Field(
        default=7, description="Refresh token expiration in days"
    )

    # LLM API Access Control
    llm_api_access_mode: Literal["public", "user", "superuser"] = Field(
        default="superuser",
        description="Access control mode for LLM API endpoints. 'public' allows no authentication, 'user' requires authenticated active users, 'superuser' requires superuser privileges. Default: 'superuser' for security.",
    )

    ###################################
    #       LLM EMULATION             #
    ###################################
    llm_config_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional path to the LLM YAML config. Relative paths are resolved "
            "from the project root. Useful for fully emulated test profiles."
        ),
    )
    llm_emulation_enabled: bool = Field(
        default=False,
        description=(
            "When true, chat model creation is routed to an in-process "
            "emulated model instead of real LLM providers."
        ),
    )
    llm_emulation_profile: str = Field(
        default="deterministic",
        description="Named response profile used by the emulated chat model.",
    )
    llm_emulation_response: str = Field(
        default="This is an emulated LLM response.",
        description="Default assistant message returned by deterministic emulation.",
    )
    llm_emulation_latency_ms: int = Field(
        default=0,
        ge=0,
        le=60000,
        description="Artificial latency in milliseconds added to emulated LLM calls.",
    )
    llm_emulation_latency_jitter_ms: int = Field(
        default=0,
        ge=0,
        le=60000,
        description=(
            "Maximum plus/minus jitter in milliseconds added per emulated LLM "
            "call when a session context is active."
        ),
    )
    llm_emulation_session_scale_min: float = Field(
        default=1.0,
        ge=0.0,
        le=20.0,
        description=(
            "Lower bound of the per-session latency multiplier applied to "
            "emulated LLM calls."
        ),
    )
    llm_emulation_session_scale_max: float = Field(
        default=1.0,
        ge=0.0,
        le=20.0,
        description=(
            "Upper bound of the per-session latency multiplier applied to "
            "emulated LLM calls."
        ),
    )
    llm_emulation_step_latency_growth: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description=(
            "Linear growth factor applied to later emulated LLM calls within "
            "the same session."
        ),
    )
    llm_emulation_stream_first_chunk_ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of total emulated stream latency spent before the first "
            "chunk is yielded."
        ),
    )
    llm_emulation_error_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability that an emulated LLM call raises a test error.",
    )
    llm_tool_emulation_mode: Literal["off", "all", "configured", "external"] = Field(
        default="off",
        description=(
            "Tool emulation policy for agent tests. 'all' emulates every tool, "
            "'configured' uses the include list, and 'external' targets tools "
            "marked as network/cost/side-effecting."
        ),
    )
    llm_tool_emulation_include: Optional[List[str]] = Field(
        default=None,
        description="Optional comma-separated list of tool names to emulate.",
    )
    llm_tool_emulation_exclude: Optional[List[str]] = Field(
        default=None,
        description="Optional comma-separated list of tool names never emulated.",
    )
    agent_tool_environment: Literal["production", "emulated", "strict_test"] = Field(
        default="production",
        description=(
            "Global agent tool exposure mode. strict_test exposes only tools "
            "with explicit test doubles."
        ),
    )
    agent_require_test_doubles: bool = Field(
        default=False,
        description="When true, tools without test doubles are not exposed to agents.",
    )
    agent_tool_double_strategy: Literal["replace", "disable", "passthrough"] = Field(
        default="replace",
        description=(
            "How test doubles are applied in non-production tool environments."
        ),
    )
    agent_snapshot_capture_enabled: bool = Field(
        default=False,
        description=(
            "When true, local AgentService runs persist replayable snapshots "
            "containing resolved config, streamed events, and final output."
        ),
    )
    agent_snapshot_replay_enabled: bool = Field(
        default=False,
        description=(
            "When true, local AgentService runs try to replay a previously "
            "captured snapshot before executing the live graph."
        ),
    )
    agent_snapshot_storage_path: str = Field(
        default="debug/agent_run_snapshots",
        description=(
            "Filesystem path used to store captured agent snapshots. Relative "
            "paths are resolved from the project root."
        ),
    )
    agent_snapshot_replay_match_mode: Literal[
        "exact", "exact_or_semantic", "semantic"
    ] = Field(
        default="exact_or_semantic",
        description=(
            "Snapshot replay matching strategy. exact requires a fingerprint "
            "match, semantic requires EmbeddingService-backed similarity with "
            "a real embedding backend, and exact_or_semantic tries exact "
            "before semantic fallback."
        ),
    )
    agent_snapshot_replay_min_score: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity required for semantic snapshot replay "
            "when EMBEDDING_BACKEND is local or remote."
        ),
    )

    ###################################
    #        VECTOR STORE             #
    ###################################
    vector_backend: Optional[Literal["qdrant", "memory"]] = Field(
        default=None,
        description="Vector store backend. None disables vector store features. 'qdrant' for production, 'memory' for testing/development.",
    )
    vector_collection_default: str = Field(
        default="default_documents",
        description="Default collection name for vector storage",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key for authentication (optional)",
    )
    vector_distance: Literal["cosine", "euclidean", "dot"] = Field(
        default="cosine",
        description="Distance metric for vector similarity",
    )
    vector_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model for text vectorization",
    )
    vector_dim: int = Field(
        default=384,
        description="Vector dimension size - must match embedding model output",
        ge=1,
        le=4096,
    )
    vector_ingest_max_batch_size: int = Field(
        default=1000,
        description="Maximum batch size for vector ingestion",
        ge=1,
        le=10000,
    )

    ###################################
    #      EMBEDDING SETTINGS         #
    ###################################
    embedding_backend: Literal["local", "remote", "fake"] = Field(
        default="local",
        description="Embedding backend: 'local' (Celery prefork worker) "
        "or 'remote' (API provider via LangChain), or 'fake' for deterministic tests.",
    )
    embedding_local_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="SentenceTransformer model for local embedding backend. "
        "Runs on Celery prefork workers, never in the API process.",
    )
    embedding_local_timeout: int = Field(
        default=60,
        description="Timeout in seconds for local Celery embedding task.",
        ge=5,
        le=300,
    )

    ###################################
    #        PII DETECTION            #
    ###################################
    pii_backend: Literal["regex", "gliner-local", "remote", "fake"] = Field(
        default="regex",
        description="PII detection backend: 'regex' (generic patterns + "
        "product-injected locale packs — no model), 'gliner-local' (in-process "
        "GLiNER multilingual PII model, lazy per process; needs the optional "
        "'pii' dependency group), 'remote' (external PII HTTP service), or "
        "'fake' (no model; tests). Default 'regex' keeps today's behavior with "
        "no new dependency or model load.",
    )
    pii_local_model: str = Field(
        default="urchade/gliner_multi_pii-v1",
        description="Multilingual GLiNER PII model id for the 'gliner-local' "
        "backend and the 'pii.detect' batch task. Runs on CPU.",
    )
    pii_local_timeout: int = Field(
        default=60,
        description="Timeout in seconds for the off-process 'pii.detect' "
        "Celery batch task.",
        ge=5,
        le=300,
    )
    pii_egress_timeout_ms: int = Field(
        default=50,
        description="Hard timeout (ms) for ML-backed redact_fast() on the "
        "synchronous Sentry/Langfuse egress path; on timeout it falls back to "
        "pure-regex redaction so an error event is never blocked or dropped.",
        ge=1,
        le=5000,
    )
    pii_remote_url: Optional[str] = Field(
        default=None,
        description="Base URL of the remote PII service (only read by the "
        "'remote' backend).",
    )
    pii_remote_timeout: int = Field(
        default=10,
        description="HTTP timeout in seconds for the 'remote' PII backend.",
        ge=1,
        le=120,
    )

    ###################################
    #        AGENT MEMORY             #
    ###################################
    agent_memory_enabled: bool = Field(
        default=False,
        description="Enable long-term memory for agents. Requires vector_backend to be configured.",
    )
    agent_memory_collection: str = Field(
        default="agent_memory",
        description="Collection name for agent memory storage (separate from RAG documents)",
    )
    agent_memory_max_results: int = Field(
        default=5,
        description="Maximum number of memory items to retrieve per category during recall",
        ge=1,
        le=50,
    )
    agent_memory_auto_recall: bool = Field(
        default=True,
        description="Automatically recall relevant memories in before_agent middleware hook",
    )
    agent_memory_categories: str = Field(
        default="semantic,episodic,procedural",
        description=(
            "Comma-separated list of CoALA memory categories to enable for auto-recall. "
            "Valid values: semantic, episodic, procedural"
        ),
    )
    agent_memory_agent_scope_enabled: bool = Field(
        default=True,
        description=(
            "When True, episodic and procedural memories are scoped per agent_name. "
            "When False, all categories are shared across agents."
        ),
    )
    agent_memory_postrun_analysis_enabled: bool = Field(
        default=True,
        description=(
            "Enable post-run analysis in MemoryMiddleware.after_agent hook. "
            "When True, the middleware calls the LLM to extract and silently persist "
            "session-level memories that the agent may not have explicitly saved."
        ),
    )
    agent_memory_model: Optional[str] = Field(
        default="gemini-3.1-flash-lite-preview",
        description=(
            "Dedicated model used for all memory-handling LLM mechanisms "
            "(currently post-run analysis), independent of the model that runs "
            "the main session. This is the single source of truth for the memory "
            "model. Set to None to fall back to the agent's own model. "
            "In the future this default can be overridden per-user."
        ),
    )
    agent_memory_postrun_analysis_model: Optional[str] = Field(
        default=None,
        description=(
            "Optional finer-grained override for the post-run extraction LLM "
            "call only. When set, it takes precedence over 'agent_memory_model'. "
            "When None (default), 'agent_memory_model' is used."
        ),
    )

    ###################################
    #     LANGGRAPH AGENT SERVER      #
    ###################################
    agent_server_url: Optional[str] = Field(
        default=None,
        description=(
            "Base URL of the LangGraph Agent Server (e.g. http://localhost:8123). "
            "When set together with agent_server_enabled=True, agents with "
            "execution_mode='remote' in YAML config will delegate to this server."
        ),
    )
    agent_server_api_key: Optional[str] = Field(
        default=None,
        description="API key for authenticating with the LangGraph Agent Server.",
    )
    agent_server_enabled: bool = Field(
        default=False,
        description=(
            "Master switch for remote agent execution. When False, all agents "
            "run locally regardless of their execution_mode setting."
        ),
    )
    agent_server_timeout: int = Field(
        default=1920,
        ge=10,
        le=3600,
        description="HTTP timeout in seconds for Agent Server requests.",
    )

    # Cookie settings for refresh tokens
    refresh_cookie_name: str = Field(
        default="refresh_token", description="Name of the refresh token cookie"
    )
    refresh_cookie_path: str = Field(
        default="/", description="Path scope for refresh token cookie"
    )
    refresh_cookie_samesite: str = Field(
        default="lax", description="SameSite setting for refresh token cookie"
    )
    refresh_cookie_domain: Optional[str] = Field(
        default=None, description="Domain for refresh token cookie"
    )

    # User activation and verification settings
    auth_register_default_active: bool = Field(
        default=True,
        description="If false, newly registered users are inactive by default",
    )
    auth_register_endpoint_enabled: bool = Field(
        default=True,
        description=(
            "If false, the built-in POST /auth/register endpoint returns 404. "
            "Use when a deployment ships its own registration endpoint."
        ),
    )
    auth_forgot_password_endpoint_enabled: bool = Field(
        default=True,
        description=(
            "If false, the built-in POST /auth/forgot-password endpoint returns "
            "404. Use when a deployment ships its own password-reset request flow."
        ),
    )
    auth_send_verification_email_on_register: bool = Field(
        default=False, description="If true, send verification email after registration"
    )
    auth_login_require_active: bool = Field(
        default=True, description="If true, login is denied for inactive users"
    )
    auth_login_require_verified: bool = Field(
        default=False, description="If true, login is denied for unverified users"
    )
    auth_email_verification_token_ttl_minutes: int = Field(
        default=60, description="Token lifetime for email verification links in minutes"
    )
    auth_email_verification_url_base: Optional[str] = Field(
        default=None,
        description="Base URL for verification links (if not set, use backend endpoint)",
    )
    auth_email_verification_makes_active: bool = Field(
        default=True,
        description="If true, verifying email also activates the user",
    )

    ###################################
    #              REDIS              #
    ###################################
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for sessions/locks",
    )
    redis_refresh_prefix: str = Field(
        default="auth:refresh:", description="Key prefix for refresh sessions"
    )

    ###################################
    #           PROPERTIES            #
    ###################################
    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_testing(self) -> bool:
        return self.environment == "testing"

    @property
    def refresh_cookie_secure(self) -> bool:
        """Determine if refresh token cookie should be secure (HTTPS only)"""
        return self.is_production

    @property
    def refresh_cookie_max_age(self) -> int:
        """Calculate max age for refresh token cookie in seconds"""
        return self.refresh_token_expire_days * 24 * 60 * 60

    @property
    def is_sqlite(self) -> bool:
        return "sqlite" in self.database_url.lower()

    @property
    def is_mysql(self) -> bool:
        return "mysql" in self.database_url.lower()

    @property
    def is_postgresql(self) -> bool:
        return "postgresql" in self.database_url.lower()

    @property
    def database_type(self) -> str:
        if self.is_sqlite:
            return "sqlite"
        elif self.is_mysql:
            return "mysql"
        elif self.is_postgresql:
            return "postgresql"
        else:
            return "unknown"

    def get_database_engine_args(self) -> dict:
        """Get database engine arguments based on database type.

        Returns connection pool configuration. Use database_pool_class='null'
        for Jupyter notebooks or multi-event-loop environments to avoid
        'Task got Future attached to a different loop' errors.
        """
        from sqlalchemy.pool import NullPool

        base_args = {
            "echo": self.database_echo,
        }

        # NullPool support for Jupyter/multi-loop environments
        # Source: SQLAlchemy docs - asyncio engine with multiple event loops
        if self.database_pool_class == "null":
            base_args["poolclass"] = NullPool

        # SQLite doesn't support pooling
        if self.is_sqlite:
            base_args.update({"connect_args": {"check_same_thread": False}})
        elif self.database_pool_class != "null":
            # PostgreSQL and MySQL pooling settings (skip if NullPool)
            base_args.update(
                {
                    "pool_size": self.database_pool_size,
                    "max_overflow": self.database_max_overflow,
                    "pool_timeout": self.database_pool_timeout,
                    "pool_recycle": self.database_pool_recycle,
                    "pool_pre_ping": True,  # Validate connections
                }
            )

            # MySQL specific settings
            if self.is_mysql:
                base_args["connect_args"] = {
                    "charset": self.database_mysql_charset,
                    "use_unicode": True,
                    "autocommit": False,
                }
        elif self.is_mysql:
            # MySQL charset settings still needed with NullPool
            base_args["connect_args"] = {
                "charset": self.database_mysql_charset,
                "use_unicode": True,
                "autocommit": False,
            }

        return base_args

    def _normalize_hostname(self, url_or_hostname: str) -> str:
        """
        Extract hostname from a URL or return hostname as-is.

        Examples:
            "https://app.example.com:8080" -> "app.example.com"
            "http://localhost:3000" -> "localhost"
            "example.com" -> "example.com"
            "*" -> "*"
        """
        if url_or_hostname == "*":
            return "*"

        # Remove scheme if present
        if "://" in url_or_hostname:
            url_or_hostname = url_or_hostname.split("://", 1)[1]

        # Remove path if present (do this before port handling)
        if "/" in url_or_hostname:
            url_or_hostname = url_or_hostname.split("/", 1)[0]

        # Remove port if present
        if ":" in url_or_hostname:
            # Split only on the first colon to handle cases like IPv6
            parts = url_or_hostname.split(":", 1)
            if len(parts) == 2:
                hostname, port_part = parts
                # Only treat as port if it's purely numeric
                if port_part.isdigit():
                    url_or_hostname = hostname
                # For non-numeric or complex cases, keep as is for now

        return url_or_hostname.strip()

    def get_effective_allowed_hosts(self) -> List[str]:
        """
        Get the effective allowed hosts for TrustedHostMiddleware.

        If allowed_hosts is explicitly set, use it.
        Otherwise, derive hostnames from cors_origins.

        Returns:
            List of hostnames/IPs for TrustedHostMiddleware
        """
        if self.allowed_hosts is not None and len(self.allowed_hosts) > 0:
            return self.allowed_hosts

        # Fallback: derive from cors_origins
        if self.cors_origins == ["*"]:
            return ["*"]

        normalized_hosts = []
        for origin in self.cors_origins:
            normalized = self._normalize_hostname(origin)
            if normalized and normalized not in normalized_hosts:
                normalized_hosts.append(normalized)

        # Always include localhost/127.0.0.1 for local development and health checks
        if not self.is_production:
            for local_host in ["localhost", "127.0.0.1"]:
                if local_host not in normalized_hosts:
                    normalized_hosts.append(local_host)

        return normalized_hosts if normalized_hosts else ["*"]

    @property
    def is_vector_store_enabled(self) -> bool:
        """Check if vector store is enabled"""
        return self.vector_backend is not None

    model_config = SettingsConfigDict(
        env_file=str(get_project_dotenv_path()),
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Customize settings sources to use our custom list parsing for environment variables
        """

        dotenv_path = str(get_project_dotenv_path())

        if os.getenv("ENVIRONMENT") == "testing":
            return (
                init_settings,
                ListParsingDotEnvSource(settings_cls, env_file=dotenv_path),
                ListParsingEnvSource(settings_cls),
                file_secret_settings,
            )

        return (
            init_settings,
            ListParsingEnvSource(settings_cls),
            ListParsingDotEnvSource(settings_cls, env_file=dotenv_path),
            file_secret_settings,
        )


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (cached)

    Returns:
        Settings instance
    """
    return Settings()


# -------------------------------------------------------------
# Helpers for tests / scenarios requiring pure default settings
# -------------------------------------------------------------
class PureDefaultsSettings(Settings):  # type: ignore
    """Settings variant ignoring ENV, .env and secrets.

    Used in tests to obtain pure defaults (with validators) and
    to create instances with overrides that go through full validation.
    """

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Only use init_settings source – no ENV, no .env, no secrets
        return (init_settings,)


def get_settings_pure_defaults() -> Settings:
    """Return instance with default values only (validators active, no ENV/.env)."""
    return PureDefaultsSettings()


def build_pure_defaults_with_overrides(**overrides: Any) -> Settings:
    """Create PureDefaultsSettings instance with overrides, all validated."""
    return PureDefaultsSettings(**overrides)


from contextlib import contextmanager


@contextmanager
def isolated_settings_environment(clear: bool = True):
    """Test context isolating ENV and get_settings cache.

    Usage:
        with isolated_settings_environment():
            s = get_settings_pure_defaults()
            ... assertions ...

    Args:
        clear: if True, removes all existing environment variables (restored after exit)
    """
    from copy import deepcopy

    original_env = deepcopy(os.environ)
    try:
        if clear:
            os.environ.clear()
        # Clear cache so get_settings() does not return a stale instance
        try:
            get_settings.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass
        yield
    finally:
        os.environ.clear()
        os.environ.update(original_env)
        try:
            get_settings.cache_clear()  # refresh after exit
        except Exception:
            pass
