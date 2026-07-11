"""
User Agent Instance Service

CRUD service for managing user-created agent configuration instances.
Handles validation, default management, and template listing.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from inference_core.agents.middleware.model_fallback import (
    canonicalize_fallback_overrides,
)
from inference_core.database.sql.models.user_agent_instance import UserAgentInstance
from inference_core.schemas.agent_prompt_limits import validate_agent_prompt_lengths
from inference_core.services.llm_config_service import LLMConfigService

logger = logging.getLogger(__name__)


INSTANCE_LOAD_OPTIONS = (
    selectinload(UserAgentInstance.subagents).selectinload(UserAgentInstance.subagents),
)


class UserAgentInstanceService:
    """
    Service for managing user agent instances.

    WHY: Provides CRUD operations for user-created agent configurations,
    validates against available base agents and models, and manages
    the "default" flag (only one default per user).
    """

    def __init__(self, db: AsyncSession, llm_config_service: LLMConfigService):
        self.db = db
        self.llm_config_service = llm_config_service

    # =========================================================
    # Read Operations
    # =========================================================

    async def list_instances(
        self,
        user_id: UUID,
        active_only: bool = True,
    ) -> List[UserAgentInstance]:
        """List all agent instances for a user."""
        filters = [
            UserAgentInstance.user_id == user_id,
        ]
        if active_only:
            filters.append(UserAgentInstance.is_active == True)  # noqa: E712

        query = (
            select(UserAgentInstance)
            .options(*INSTANCE_LOAD_OPTIONS)
            .where(and_(*filters))
            .order_by(
                UserAgentInstance.is_default.desc(), UserAgentInstance.display_name
            )
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_instance(
        self,
        user_id: UUID,
        instance_id: UUID,
    ) -> Optional[UserAgentInstance]:
        """Get a specific agent instance by ID."""
        query = (
            select(UserAgentInstance)
            .options(*INSTANCE_LOAD_OPTIONS)
            .where(
                and_(
                    UserAgentInstance.id == instance_id,
                    UserAgentInstance.user_id == user_id,
                )
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_instance_by_name(
        self,
        user_id: UUID,
        instance_name: str,
    ) -> Optional[UserAgentInstance]:
        """Get a specific agent instance by name."""
        query = (
            select(UserAgentInstance)
            .options(*INSTANCE_LOAD_OPTIONS)
            .where(
                and_(
                    UserAgentInstance.instance_name == instance_name,
                    UserAgentInstance.user_id == user_id,
                )
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_default_instance(
        self,
        user_id: UUID,
    ) -> Optional[UserAgentInstance]:
        """Get the user's default agent instance."""
        query = (
            select(UserAgentInstance)
            .options(*INSTANCE_LOAD_OPTIONS)
            .where(
                and_(
                    UserAgentInstance.user_id == user_id,
                    UserAgentInstance.is_default == True,  # noqa: E712
                    UserAgentInstance.is_active == True,  # noqa: E712
                )
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    # =========================================================
    # Write Operations
    # =========================================================

    async def create_instance(
        self,
        user_id: UUID,
        instance_name: str,
        display_name: str,
        base_agent_name: str,
        description: Optional[str] = None,
        primary_model: Optional[str] = None,
        system_prompt_override: Optional[str] = None,
        system_prompt_append: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        skills: Optional[List[Dict[str, str]]] = None,
        is_default: bool = False,
        is_deepagent: bool = False,
        subagent_ids: Optional[List[UUID]] = None,
    ) -> UserAgentInstance:
        """Create a new agent instance for a user.

        Validates that:
        - base_agent_name exists in resolved config (with admin overrides)
        - primary_model (if provided) exists in available models from resolved config
        - instance_name is unique for this user
        - subagent_ids (if provided) exist and belong to the user

        If is_default=True, clears any existing default for the user.
        """
        validate_agent_prompt_lengths(
            system_prompt_override=system_prompt_override,
            system_prompt_append=system_prompt_append,
        )

        # Get resolved config for validation
        resolved = await self.llm_config_service.get_resolved_config(user_id)

        # Validate base agent exists
        if base_agent_name not in resolved.agents:
            available = list(resolved.agents.keys())
            raise ValueError(
                f"Base agent '{base_agent_name}' not found. "
                f"Available agents: {available}"
            )

        # Reject internal-only agents (Agent Builder stage agents, etc.).
        # WHY: ``user_selectable=False`` means the agent is reserved for system
        # orchestration and must never back a ``UserAgentInstance`` even though
        # it appears in the resolved config registry.
        base_agent_config = resolved.agents[base_agent_name]
        if not getattr(base_agent_config, "user_selectable", True):
            raise ValueError(
                f"Base agent '{base_agent_name}' is not available for user " "instances"
            )

        # Validate primary model if provided
        if primary_model and primary_model not in resolved.available_models:
            available = resolved.available_models
            raise ValueError(
                f"Model '{primary_model}' not found. " f"Available models: {available}"
            )

        config_overrides = self._normalize_and_validate_config_overrides(
            config_overrides,
            resolved.available_models,
        )

        # Check uniqueness
        existing = await self.get_instance_by_name(user_id, instance_name)
        if existing:
            raise ValueError(
                f"Agent instance '{instance_name}' already exists for this user"
            )

        # Validate subagents
        subagents = []
        if is_deepagent and subagent_ids:
            query = select(UserAgentInstance).where(
                and_(
                    UserAgentInstance.id.in_(subagent_ids),
                    UserAgentInstance.user_id == user_id,
                )
            )
            result = await self.db.execute(query)
            subagents = list(result.scalars().all())
            if len(subagents) != len(subagent_ids):
                raise ValueError(
                    "One or more subagents not found or do not belong to the user"
                )

        # Clear existing default if setting new one
        if is_default:
            await self._clear_user_defaults(user_id)

        instance = UserAgentInstance(
            user_id=user_id,
            instance_name=instance_name,
            display_name=display_name,
            base_agent_name=base_agent_name,
            description=description,
            primary_model=primary_model,
            system_prompt_override=system_prompt_override,
            system_prompt_append=system_prompt_append,
            config_overrides=config_overrides,
            skills=skills,
            is_default=is_default,
            is_deepagent=is_deepagent,
            is_active=True,
            subagents=subagents,
        )

        self.db.add(instance)
        await self.db.commit()

        # Refresh with eager loading of subagents to avoid MissingGreenlet error
        query = (
            select(UserAgentInstance)
            .options(*INSTANCE_LOAD_OPTIONS)
            .where(UserAgentInstance.id == instance.id)
        )
        result = await self.db.execute(query)
        instance = result.scalar_one()

        logger.info(
            f"Created agent instance '{instance_name}' (base={base_agent_name}) "
            f"for user {user_id}"
        )
        return instance

    async def update_instance(
        self,
        user_id: UUID,
        instance_id: UUID,
        **updates: Any,
    ) -> Optional[UserAgentInstance]:
        """Update an existing agent instance.

        Validates model names and manages default flag.
        """
        validate_agent_prompt_lengths(
            system_prompt_override=updates.get("system_prompt_override"),
            system_prompt_append=updates.get("system_prompt_append"),
        )

        # We need to eager load subagents if we are going to update them
        query = (
            select(UserAgentInstance)
            .options(*INSTANCE_LOAD_OPTIONS)
            .where(
                and_(
                    UserAgentInstance.id == instance_id,
                    UserAgentInstance.user_id == user_id,
                )
            )
        )
        result = await self.db.execute(query)
        instance = result.scalar_one_or_none()

        if not instance:
            return None

        # Validate primary_model if being updated
        if "primary_model" in updates and updates["primary_model"] is not None:
            resolved = await self.llm_config_service.get_resolved_config(user_id)
            if updates["primary_model"] not in resolved.available_models:
                available = resolved.available_models
                raise ValueError(
                    f"Model '{updates['primary_model']}' not found. "
                    f"Available models: {available}"
                )

        # Validate base_agent_name if being updated (mirrors create_instance):
        # ensure the new base exists and is user-selectable. Without this guard
        # callers could swap a valid instance over to an internal-only agent
        # (e.g. an Agent Builder stage agent) via PATCH.
        if "base_agent_name" in updates and updates["base_agent_name"] is not None:
            resolved = await self.llm_config_service.get_resolved_config(user_id)
            new_base = updates["base_agent_name"]
            if new_base not in resolved.agents:
                available = list(resolved.agents.keys())
                raise ValueError(
                    f"Base agent '{new_base}' not found. "
                    f"Available agents: {available}"
                )
            new_base_config = resolved.agents[new_base]
            if not getattr(new_base_config, "user_selectable", True):
                raise ValueError(
                    f"Base agent '{new_base}' is not available for user " "instances"
                )

        if "config_overrides" in updates and updates["config_overrides"] is not None:
            resolved = await self.llm_config_service.get_resolved_config(user_id)
            updates["config_overrides"] = self._normalize_and_validate_config_overrides(
                updates["config_overrides"],
                resolved.available_models,
            )

        # Handle default flag
        if updates.get("is_default") is True and not instance.is_default:
            await self._clear_user_defaults(user_id)

        # Handle subagents update
        if updates.get("subagent_ids"):
            subagent_ids = updates.pop("subagent_ids")
            if subagent_ids is not None:
                if not updates.get("is_deepagent", instance.is_deepagent):
                    raise ValueError("Cannot assign subagents to a non-deep agent")

                subagents_query = select(UserAgentInstance).where(
                    and_(
                        UserAgentInstance.id.in_(subagent_ids),
                        UserAgentInstance.user_id == user_id,
                    )
                )
                subagents_result = await self.db.execute(subagents_query)
                subagents = list(subagents_result.scalars().all())
                if len(subagents) != len(subagent_ids):
                    raise ValueError(
                        "One or more subagents not found or do not belong to the user"
                    )
                instance.subagents = subagents
            else:
                instance.subagents = []

        # Apply updates
        for key, value in updates.items():
            if value is not None and hasattr(instance, key):
                setattr(instance, key, value)

        await self.db.commit()

        # Refresh with eager loading of subagents to avoid MissingGreenlet error
        query = (
            select(UserAgentInstance)
            .options(*INSTANCE_LOAD_OPTIONS)
            .where(UserAgentInstance.id == instance.id)
        )
        result = await self.db.execute(query)
        instance = result.scalar_one()

        logger.info(
            f"Updated agent instance '{instance.instance_name}' "
            f"(id={instance_id}) for user {user_id}"
        )
        return instance

    async def delete_instance(
        self,
        user_id: UUID,
        instance_id: UUID,
    ) -> bool:
        """Hard-delete an agent instance."""
        instance = await self.get_instance(user_id, instance_id)
        if not instance:
            return False

        await self.db.delete(instance)
        await self.db.commit()

        logger.info(
            f"Deleted agent instance '{instance.instance_name}' "
            f"(id={instance_id}) for user {user_id}"
        )
        return True

    # =========================================================
    # Template Operations (read-only from YAML)
    # =========================================================

    async def list_templates(self) -> List[Dict[str, Any]]:
        """List available agent templates from resolved config.

        Returns list of dicts with agent config details that can be
        used as base for creating instances, filtered by admin overrides.
        """
        resolved = await self.llm_config_service.get_resolved_config(None)
        templates = []
        for agent_name, agent_config in resolved.agents.items():
            # Skip internal-only agents so the user-facing template list never
            # exposes Agent Builder stage agents or other system agents.
            if not getattr(agent_config, "user_selectable", True):
                continue
            templates.append(
                {
                    "agent_name": agent_name,
                    "primary_model": agent_config.primary_model,
                    "fallback_models": agent_config.fallback_models,
                    "description": agent_config.description,
                    "allowed_tools": agent_config.allowed_tools,
                    "mcp_profile": agent_config.mcp_profile,
                    "local_tool_providers": agent_config.local_tool_providers or [],
                    "skills": agent_config.skills,
                    "subagents": agent_config.subagents,
                }
            )
        return templates

    async def list_available_models(self) -> List[str]:
        """List available model names from resolved config."""
        resolved = await self.llm_config_service.get_resolved_config(None)
        return resolved.available_models

    # =========================================================
    # Helpers
    # =========================================================

    async def _clear_user_defaults(self, user_id: UUID) -> None:
        """Clear the is_default flag for all user's instances."""
        stmt = (
            update(UserAgentInstance)
            .where(
                and_(
                    UserAgentInstance.user_id == user_id,
                    UserAgentInstance.is_default == True,  # noqa: E712
                )
            )
            .values(is_default=False)
        )
        await self.db.execute(stmt)

    @staticmethod
    def _normalize_and_validate_config_overrides(
        config_overrides: Optional[Dict[str, Any]],
        available_models: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Normalize fallback aliases and validate referenced fallback models."""
        if config_overrides is None:
            return None

        normalized = canonicalize_fallback_overrides(config_overrides)

        # Validate per-assistant memory gates (app-layer flags stored in the
        # config_overrides bag). Reject malformed values at the API boundary.
        if "memory_access_enabled" in normalized:
            if not isinstance(normalized["memory_access_enabled"], bool):
                raise ValueError(
                    "config_overrides.memory_access_enabled must be a boolean"
                )
        if "memory_scope" in normalized:
            if normalized["memory_scope"] not in ("shared", "dedicated"):
                raise ValueError(
                    "config_overrides.memory_scope must be 'shared' or 'dedicated'"
                )

        # Validate response_format: must be a non-empty JSON Schema dict.
        # WHY: keep DB overrides aligned with AgentConfig.validate_response_format
        # so invalid schemas are rejected at the API boundary, not at agent build.
        if "response_format" in normalized:
            rf = normalized["response_format"]
            if rf is not None:
                if not isinstance(rf, dict) or not rf:
                    raise ValueError(
                        "config_overrides.response_format must be a non-empty "
                        "JSON Schema dict or null"
                    )
                if not any(
                    key in rf for key in ("type", "$ref", "oneOf", "anyOf", "allOf")
                ):
                    raise ValueError(
                        "config_overrides.response_format must look like a JSON "
                        "Schema: expected at least one of 'type', '$ref', "
                        "'oneOf', 'anyOf', or 'allOf' at the top level"
                    )

        if "fallback" not in normalized:
            return normalized

        fallback = normalized.get("fallback")
        if fallback is None:
            fallback = []
            normalized["fallback"] = fallback
            normalized["fallback_models"] = fallback

        if not isinstance(fallback, list) or not all(
            isinstance(model, str) for model in fallback
        ):
            raise ValueError("config_overrides.fallback must be a list of model names")

        invalid = [model for model in fallback if model not in available_models]
        if invalid:
            raise ValueError(
                f"Fallback model(s) not found: {invalid}. "
                f"Available models: {available_models}"
            )

        return normalized


# =========================================================
# Factory / Dependency
# =========================================================


def get_user_agent_instance_service(
    db: AsyncSession, llm_config_service: LLMConfigService
) -> UserAgentInstanceService:
    """Factory for creating UserAgentInstanceService instances."""
    return UserAgentInstanceService(db, llm_config_service)
