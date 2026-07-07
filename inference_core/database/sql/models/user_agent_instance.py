"""
User Agent Instance Model

Database model for user-created agent configuration instances.
Users can create custom agent configurations based on base agent templates
(defined in agents_config.yaml), overriding model, tools, prompt, etc.

Each instance represents a personalized agent variant that can be selected
when starting a chat session.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Index,
    String,
    Table,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base import Base, BaseModel, SmartJSON

# Association table for deep agent subagents
user_agent_subagents = Table(
    "user_agent_subagents",
    Base.metadata,
    Column(
        "parent_id",
        Uuid(as_uuid=True),
        ForeignKey("user_agent_instances.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "subagent_id",
        Uuid(as_uuid=True),
        ForeignKey("user_agent_instances.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class UserAgentInstance(BaseModel):
    """
    User-created agent configuration instance.

    WHY: Allows users to create personalized variants of base agents
    (e.g., "Creative Assistant" with high temperature, or "Precise Coder"
    with specific tools enabled) without modifying the global YAML config.

    Each instance references a base agent from agents_config.yaml and
    stores overrides as JSON. When a chat session uses this instance,
    the overrides are merged with the base config via LLMConfig.with_overrides().

    Attributes:
        id: Unique identifier (UUID)
        user_id: Owner of this instance
        instance_name: Unique name per user (slug-like, used as identifier)
        display_name: Human-readable name shown in UI
        base_agent_name: Name of the base agent from YAML config
        description: User-provided description of what this instance is for
        primary_model: Override for the primary LLM model (None = use base)
        system_prompt_override: Custom system prompt (None = use base default)
        system_prompt_append: Text appended to the base system prompt
        config_overrides: JSON dict of additional overrides (fallback, tools, etc.)
        is_default: Whether this is the user's default agent for new chats
        is_active: Whether this instance is currently usable
    """

    __tablename__ = "user_agent_instances"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique identifier",
    )

    # Owner reference
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="User who owns this agent instance",
    )

    # Instance identity
    instance_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
        doc="Unique instance name per user (slug-like identifier; stays "
        "plaintext — SQL lookup key; Text is preventive widening)",
    )

    display_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Human-readable display name (plaintext; Text is preventive)",
    )

    # Base agent reference
    base_agent_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        doc="Name of the base agent from agents_config.yaml (e.g., 'assistant_agent')",
    )

    # Description (encrypted at rest behind ENCRYPT_INSTANCES_WRITES — the
    # `description` property below is the read/write boundary)
    _description: Mapped[Optional[str]] = mapped_column(
        "description",
        Text,
        nullable=True,
        doc="User-provided description of this agent instance",
    )

    # Model override
    primary_model: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="Override for the primary LLM model (None = use base agent's model)",
    )

    # System prompt overrides (encrypted at rest — see properties below)
    _system_prompt_override: Mapped[Optional[str]] = mapped_column(
        "system_prompt_override",
        Text,
        nullable=True,
        doc="Full system prompt override (replaces base prompt entirely)",
    )

    _system_prompt_append: Mapped[Optional[str]] = mapped_column(
        "system_prompt_append",
        Text,
        nullable=True,
        doc="Text appended to the base system prompt (additive customization)",
    )

    # Flexible JSON overrides for other agent config fields (encrypted at
    # rest as a JSON string scalar — see property below). SQL-side access
    # must use the underscore column and dual-read the value.
    _config_overrides: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "config_overrides",
        SmartJSON,
        nullable=True,
        doc=(
            "JSON dict of additional overrides. Supported keys: "
            "'fallback' (list[str]), 'allowed_tools' (list[str]), "
            "'mcp_profile' (str), 'temperature' (float), 'max_tokens' (int)"
        ),
    )

    # Default flag
    is_default: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Whether this is the user's default agent for new chats",
    )

    # User-defined skills (encrypted at rest — see property below)
    _skills: Mapped[Optional[list]] = mapped_column(
        "skills",
        SmartJSON,
        nullable=True,
        doc=(
            "User-defined skills for the agent. Each entry is a dict with "
            "'name' (str), 'description' (str), and 'content' (str, full SKILL.md)."
        ),
    )

    # Deep agent flag
    is_deepagent: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether this instance is a deep agent",
    )

    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        doc="Whether this instance is currently usable",
    )

    # Relationships
    user = relationship("User", foreign_keys=[user_id])

    subagents: Mapped[List["UserAgentInstance"]] = relationship(
        "UserAgentInstance",
        secondary=user_agent_subagents,
        primaryjoin=id == user_agent_subagents.c.parent_id,
        secondaryjoin=id == user_agent_subagents.c.subagent_id,
        backref="parent_agents",
        lazy="selectin",
        doc="Subagents assigned to this deep agent instance",
    )

    __table_args__ = (
        # One instance name per user
        UniqueConstraint(
            "user_id",
            "instance_name",
            name="uq_user_agent_instance_user_name",
        ),
        Index("ix_user_agent_instances_user_active", "user_id", "is_active"),
        Index(
            "ix_user_agent_instances_user_default",
            "user_id",
            "is_default",
            "is_active",
        ),
    )

    # ------------------------------------------------------------------
    # Encrypted-field boundaries (privacy Faza 2). Every ORM attribute
    # access — services, agent builders, API serializers, builder patchers —
    # routes through these properties, so callers always see plaintext while
    # the stored value may be an ``enc.v1.…`` token. Writes encrypt only
    # when ENCRYPT_INSTANCES_WRITES is on; reads dual-read (legacy plaintext
    # passes through). SQL-side column access must use the underscore
    # attributes and dual-read explicitly.
    # ------------------------------------------------------------------

    _ENCRYPTED_KWARGS = (
        "description",
        "system_prompt_override",
        "system_prompt_append",
        "config_overrides",
        "skills",
    )

    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        encrypted = {
            key: kwargs.pop(key) for key in list(kwargs)
            if key in self._ENCRYPTED_KWARGS
        }
        super().__init__(**kwargs)
        for key, value in encrypted.items():
            setattr(self, key, value)

    @property
    def description(self) -> Optional[str]:
        from inference_core.services.content_cipher import dec_field

        return dec_field(self._description)

    @description.setter
    def description(self, value: Optional[str]) -> None:
        from inference_core.services.content_cipher import (
            enc_field,
            instances_enc_enabled,
        )

        self._description = enc_field(value, enabled=instances_enc_enabled())

    @property
    def system_prompt_override(self) -> Optional[str]:
        from inference_core.services.content_cipher import dec_field

        return dec_field(self._system_prompt_override)

    @system_prompt_override.setter
    def system_prompt_override(self, value: Optional[str]) -> None:
        from inference_core.services.content_cipher import (
            enc_field,
            instances_enc_enabled,
        )

        self._system_prompt_override = enc_field(
            value, enabled=instances_enc_enabled()
        )

    @property
    def system_prompt_append(self) -> Optional[str]:
        from inference_core.services.content_cipher import dec_field

        return dec_field(self._system_prompt_append)

    @system_prompt_append.setter
    def system_prompt_append(self, value: Optional[str]) -> None:
        from inference_core.services.content_cipher import (
            enc_field,
            instances_enc_enabled,
        )

        self._system_prompt_append = enc_field(
            value, enabled=instances_enc_enabled()
        )

    @property
    def config_overrides(self) -> Optional[Dict[str, Any]]:
        from inference_core.services.content_cipher import dec_json

        return dec_json(self._config_overrides)

    @config_overrides.setter
    def config_overrides(self, value: Optional[Dict[str, Any]]) -> None:
        from inference_core.services.content_cipher import (
            assert_decryptable_json,
            enc_json,
            instances_enc_enabled,
        )

        assert_decryptable_json(getattr(self, "_config_overrides", None))
        self._config_overrides = enc_json(value, enabled=instances_enc_enabled())

    @property
    def skills(self) -> Optional[list]:
        from inference_core.services.content_cipher import dec_json

        return dec_json(self._skills)

    @skills.setter
    def skills(self, value: Optional[list]) -> None:
        from inference_core.services.content_cipher import (
            assert_decryptable_json,
            enc_json,
            instances_enc_enabled,
        )

        assert_decryptable_json(getattr(self, "_skills", None))
        self._skills = enc_json(value, enabled=instances_enc_enabled())

    def __repr__(self) -> str:
        return (
            f"<UserAgentInstance(id={self.id}, user_id={self.user_id}, "
            f"instance_name={self.instance_name}, base={self.base_agent_name})>"
        )
