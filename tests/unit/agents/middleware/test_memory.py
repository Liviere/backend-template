"""Tests for MemoryMiddleware (CoALA Architecture).

Covers lifecycle hooks (before_agent, wrap_model_call) and helper methods
(_extract_user_input, _recall_and_format).
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference_core.agents.middleware.memory import (
    MemoryMiddleware,
    MemoryState,
    _MemoryMiddlewareContext,
    create_memory_middleware,
)
from inference_core.services.agent_memory_service import MemoryCategory, MemoryType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_memory_service():
    """Mock AgentMemoryStoreService with async methods."""
    service = MagicMock()
    # recall_memories is used by _prefetch_existing_memories in post-run analysis
    service.recall_memories = AsyncMock(return_value=[])
    return service


@pytest.fixture
def middleware(mock_memory_service):
    """MemoryMiddleware wired to mocked memory service."""
    return MemoryMiddleware(
        memory_service=mock_memory_service,
        user_id="user-123",
        auto_recall=True,
        max_recall_results=5,
    )


# ---------------------------------------------------------------------------
# _extract_user_input
# ---------------------------------------------------------------------------


class TestExtractUserInput:
    """Verify extraction of last user message from agent state."""

    def test_human_message_with_type_attr(self, middleware):
        """Finds message with type='human'."""
        msg = MagicMock()
        msg.type = "human"
        msg.content = "Hello agent"

        state = MemoryState(messages=[msg])
        assert middleware._extract_user_input(state) == "Hello agent"

    def test_user_message_with_role_attr(self, middleware):
        """Finds message with role='user'."""
        msg = MagicMock(spec=[])
        msg.role = "user"
        msg.content = "What is 2+2?"

        state = MemoryState(messages=[msg])
        assert middleware._extract_user_input(state) == "What is 2+2?"

    def test_dict_message_with_type(self, middleware):
        """Finds dict message with type='human'."""
        state = MemoryState(messages=[{"type": "human", "content": "Dict message"}])
        assert middleware._extract_user_input(state) == "Dict message"

    def test_dict_message_with_role(self, middleware):
        """Finds dict message with role='user'."""
        state = MemoryState(messages=[{"role": "user", "content": "User role dict"}])
        assert middleware._extract_user_input(state) == "User role dict"

    def test_picks_last_human_message(self, middleware):
        """When multiple messages exist, picks the last human one."""
        sys_msg = MagicMock()
        sys_msg.type = "system"

        human1 = MagicMock()
        human1.type = "human"
        human1.content = "first"

        ai_msg = MagicMock()
        ai_msg.type = "ai"

        human2 = MagicMock()
        human2.type = "human"
        human2.content = "second"

        state = MemoryState(messages=[sys_msg, human1, ai_msg, human2])
        assert middleware._extract_user_input(state) == "second"

    def test_empty_messages_returns_none(self, middleware):
        """Empty message list yields None."""
        state = MemoryState(messages=[])
        assert middleware._extract_user_input(state) is None

    def test_fallback_to_last_message_content(self, middleware):
        """When no human/user message found, falls back to last message."""
        msg = MagicMock(spec=[])
        msg.content = "fallback content"

        state = MemoryState(messages=[msg])
        assert middleware._extract_user_input(state) == "fallback content"


# ---------------------------------------------------------------------------
# before_agent
# ---------------------------------------------------------------------------


class TestBeforeAgent:
    """Verify before_agent recall, caching, and state updates."""

    def test_returns_none_when_auto_recall_disabled(self, mock_memory_service):
        """Skips recall if auto_recall=False."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            auto_recall=False,
        )
        state = MemoryState(messages=[MagicMock(type="human", content="hi")])
        result = mw.before_agent(state, MagicMock())

        assert result is None

    def test_returns_none_when_no_user_input(self, middleware):
        """Skips recall when no user input can be extracted."""
        state = MemoryState(messages=[])
        result = middleware.before_agent(state, MagicMock())

        assert result is None

    def test_returns_zero_recall_when_no_memories(self, middleware):
        """Returns count=0 when _recall_and_format returns empty context."""
        with patch.object(
            middleware, "_recall_and_format", return_value=("", {"latency_ms": 5.0})
        ):
            msg = MagicMock()
            msg.type = "human"
            msg.content = "test"
            state = MemoryState(messages=[msg])

            result = middleware.before_agent(state, MagicMock())

        assert result["memories_recalled"] == 0

    def test_returns_full_state_when_memories_found(self, middleware):
        """Returns complete state update with memory context and metrics."""
        metrics = {
            "count": 3,
            "latency_ms": 12.5,
            "types": ["preferences", "facts"],
            "categories": ["semantic"],
        }
        with patch.object(
            middleware,
            "_recall_and_format",
            return_value=("<memory>some context</memory>", metrics),
        ):
            msg = MagicMock()
            msg.type = "human"
            msg.content = "Tell me about X"
            state = MemoryState(messages=[msg])

            result = middleware.before_agent(state, MagicMock())

        # memory_context must NOT be persisted into graph state (it would inline
        # plaintext recalled-memory content into the checkpoint JSONB). It lives
        # only in the in-process cache used for prompt injection.
        assert "memory_context" not in result
        assert middleware._cached_memory_context == "<memory>some context</memory>"
        assert result["memories_recalled"] == 3
        assert result["memory_recall_latency_ms"] == 12.5
        assert result["memory_types_recalled"] == ["preferences", "facts"]
        assert result["memory_categories_recalled"] == ["semantic"]

    def test_handles_recall_error_gracefully(self, middleware):
        """Returns empty state on _recall_and_format exception."""
        with patch.object(
            middleware,
            "_recall_and_format",
            side_effect=Exception("DB down"),
        ):
            # This will be caught by the try/except in before_agent
            msg = MagicMock()
            msg.type = "human"
            msg.content = "test"
            state = MemoryState(messages=[msg])

            result = middleware.before_agent(state, MagicMock())

        assert result["memories_recalled"] == 0


# ---------------------------------------------------------------------------
# wrap_model_call
# ---------------------------------------------------------------------------


class TestWrapModelCall:
    """Verify system prompt injection of memory context."""

    def test_passthrough_when_no_cached_context(self, middleware):
        """Handler called unchanged when no memory context was cached."""
        middleware._cached_memory_context = None
        request = MagicMock()
        response = MagicMock()
        handler = MagicMock(return_value=response)

        result = middleware.wrap_model_call(request, handler)

        handler.assert_called_once_with(request)
        assert result is response

    def test_injects_memory_into_system_message(self, middleware):
        """Memory context is appended as content block to system message."""
        middleware._cached_memory_context = "User likes Python"

        # Prepare request with system message
        original_block = {"type": "text", "text": "You are a helpful assistant."}
        request = MagicMock()
        request.system_message.content_blocks = [original_block]

        captured_request = None

        def capture_handler(r):
            nonlocal captured_request
            captured_request = r
            return MagicMock()

        handler = capture_handler

        with patch(
            "inference_core.agents.middleware.memory.SystemMessage"
        ) as MockSysMsg:
            mock_sys_msg = MagicMock()
            MockSysMsg.return_value = mock_sys_msg
            request.override.return_value = MagicMock()

            middleware.wrap_model_call(request, handler)

            # Verify SystemMessage was created with memory block appended
            call_args = MockSysMsg.call_args
            new_content = call_args[1]["content"] if call_args[1] else call_args[0][0]

            # Should have original block + memory block
            assert len(new_content) == 2
            memory_block = new_content[1]
            assert "<user_memory_context>" in memory_block["text"]
            assert "User likes Python" in memory_block["text"]

    def test_falls_back_on_injection_error(self, middleware):
        """On error during injection, falls back to original request."""
        middleware._cached_memory_context = "some context"

        request = MagicMock()
        # Make content_blocks access raise to trigger the except branch
        request.system_message.content_blocks = None  # Will fail on list()

        response = MagicMock()
        handler = MagicMock(return_value=response)

        result = middleware.wrap_model_call(request, handler)

        # Handler should be called with original request
        handler.assert_called_once_with(request)
        assert result is response


# ---------------------------------------------------------------------------
# _recall_and_format
# ---------------------------------------------------------------------------


class TestRecallAndFormat:
    """Verify async memory recall via run_async_safely."""

    def test_returns_context_and_metrics(self, middleware, mock_memory_service):
        """Successful recall returns formatted context and metrics dict."""

        async def mock_format_context(**kwargs):
            return "<semantic>facts here</semantic>"

        async def mock_get_user_context(**kwargs):
            return {
                "preferences": [MagicMock(), MagicMock()],
                "facts": [MagicMock()],
            }

        mock_memory_service.format_context_for_prompt = mock_format_context
        mock_memory_service.get_user_context = mock_get_user_context

        with patch(
            "inference_core.agents.middleware.memory.run_async_safely"
        ) as mock_run:
            mock_run.return_value = (
                "<semantic>facts here</semantic>",
                {"preferences": [1, 2], "facts": [3]},
            )

            with patch(
                "inference_core.services.agent_memory_service.get_category_for_type",
                return_value=MemoryCategory.SEMANTIC,
            ):
                context, metrics = middleware._recall_and_format("Tell me about X")

        assert context == "<semantic>facts here</semantic>"
        assert metrics["count"] == 3  # 2 + 1
        assert "latency_ms" in metrics

    def test_returns_empty_on_timeout(self, middleware):
        """TimeoutError returns empty context and zero metrics."""
        with patch(
            "inference_core.agents.middleware.memory.run_async_safely",
            side_effect=TimeoutError("Memory recall timed out"),
        ):
            context, metrics = middleware._recall_and_format("query")

        assert context == ""
        assert metrics["count"] == 0

    def test_returns_empty_on_generic_error(self, middleware):
        """Generic exception returns empty context and zero metrics."""
        with patch(
            "inference_core.agents.middleware.memory.run_async_safely",
            side_effect=RuntimeError("connection refused"),
        ):
            context, metrics = middleware._recall_and_format("query")

        assert context == ""
        assert metrics["count"] == 0


# ---------------------------------------------------------------------------
# create_memory_middleware factory
# ---------------------------------------------------------------------------


class TestCreateMemoryMiddleware:
    """Verify factory function produces correctly configured middleware."""

    def test_creates_with_defaults(self, mock_memory_service):
        """Factory creates middleware with default CoALA categories."""
        mw = create_memory_middleware(
            memory_service=mock_memory_service,
            user_id="user-42",
        )

        assert mw.user_id == "user-42"
        assert mw.auto_recall is True
        assert mw.max_recall_results == 5
        assert MemoryCategory.SEMANTIC.value in mw.include_categories
        assert MemoryCategory.EPISODIC.value in mw.include_categories
        assert MemoryCategory.PROCEDURAL.value in mw.include_categories

    def test_creates_with_custom_categories(self, mock_memory_service):
        """Factory passes custom categories and types through."""
        mw = create_memory_middleware(
            memory_service=mock_memory_service,
            user_id="user-42",
            include_categories=["semantic"],
            include_memory_types=["facts"],
            max_recall_results=10,
        )

        assert mw.include_categories == ["semantic"]
        assert mw.include_memory_types == ["facts"]
        assert mw.max_recall_results == 10

    def test_creates_with_none_user_id(self, mock_memory_service):
        """Factory accepts user_id=None for Agent Server deferred resolution."""
        mw = create_memory_middleware(
            memory_service=mock_memory_service,
            user_id=None,
        )
        assert mw.user_id is None
        assert mw.auto_recall is True


# ---------------------------------------------------------------------------
# Deferred user_id resolution (Agent Server)
# ---------------------------------------------------------------------------


class TestDeferredUserId:
    """Verify _resolve_user_id falls back to runtime context var."""

    def test_returns_instance_user_id_when_set(self, mock_memory_service):
        """Uses self.user_id when it is not None."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="explicit-uid",
        )
        assert mw._resolve_user_id() == "explicit-uid"

    def test_falls_back_to_context_var(self, mock_memory_service):
        """Resolves from _runtime_context when self.user_id is None."""
        import uuid

        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id=None,
        )
        uid = uuid.uuid4()
        with patch(
            "inference_core.agents.middleware.memory._ctx_get_user_id",
            return_value=uid,
        ):
            assert mw._resolve_user_id() == str(uid)

    def test_returns_none_when_both_empty(self, mock_memory_service):
        """Returns None when neither instance attr nor context var is set."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id=None,
        )
        with patch(
            "inference_core.agents.middleware.memory._ctx_get_user_id",
            return_value=None,
        ):
            assert mw._resolve_user_id() is None

    def test_before_agent_skips_when_no_user_id(self, mock_memory_service):
        """before_agent returns None when user_id cannot be resolved."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id=None,
        )
        msg = MagicMock()
        msg.type = "human"
        msg.content = "hi"
        state = MemoryState(messages=[msg])

        with patch(
            "inference_core.agents.middleware.memory._ctx_get_user_id",
            return_value=None,
        ):
            result = mw.before_agent(state, MagicMock())

        assert result is None

    def test_before_agent_resolves_from_ctx(self, mock_memory_service):
        """before_agent uses context var user_id and passes it to _recall_and_format."""
        import uuid

        uid = uuid.uuid4()
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id=None,
        )
        msg = MagicMock()
        msg.type = "human"
        msg.content = "test query"
        state = MemoryState(messages=[msg])

        with (
            patch(
                "inference_core.agents.middleware.memory._ctx_get_user_id",
                return_value=uid,
            ),
            patch.object(
                mw,
                "_recall_and_format",
                return_value=(
                    "ctx",
                    {"count": 1, "latency_ms": 1, "types": [], "categories": []},
                ),
            ) as mock_recall,
        ):
            result = mw.before_agent(state, MagicMock())

        mock_recall.assert_called_once_with("test query", str(uid))
        assert result["memories_recalled"] == 1


# ---------------------------------------------------------------------------
# _arecall_and_format (async recall path)
# ---------------------------------------------------------------------------


class TestAsyncRecallAndFormat:
    """Verify async recall path for Agent Server context."""

    async def test_returns_context_and_metrics(self, mock_memory_service):
        """_arecall_and_format directly awaits memory service."""
        mock_memory_service.format_context_for_prompt = AsyncMock(
            return_value="<semantic>data</semantic>"
        )
        mock_memory_service.get_user_context = AsyncMock(
            return_value={"facts": [MagicMock(), MagicMock()]}
        )

        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u-async",
        )

        with patch(
            "inference_core.services.agent_memory_service.get_category_for_type",
            return_value=MemoryCategory.SEMANTIC,
        ):
            context, metrics = await mw._arecall_and_format("some query")

        assert context == "<semantic>data</semantic>"
        assert metrics["count"] == 2
        assert "latency_ms" in metrics

    async def test_returns_empty_on_error(self, mock_memory_service):
        """_arecall_and_format returns empty on exception."""
        mock_memory_service.format_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )

        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u-err",
        )

        context, metrics = await mw._arecall_and_format("query")
        assert context == ""
        assert metrics["count"] == 0

    async def test_uses_provided_user_id_over_self(self, mock_memory_service):
        """Explicit user_id param takes precedence over self.user_id."""
        mock_memory_service.format_context_for_prompt = AsyncMock(return_value="")
        mock_memory_service.get_user_context = AsyncMock(return_value={})

        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="self-uid",
        )
        await mw._arecall_and_format("q", user_id="override-uid")

        call_kwargs = mock_memory_service.format_context_for_prompt.call_args[1]
        assert call_kwargs["user_id"] == "override-uid"


# ---------------------------------------------------------------------------
# _has_memory_save_in_run
# ---------------------------------------------------------------------------


class TestHasMemorySaveInRun:
    """Verify detection of save_memory_store tool calls in run history."""

    def test_returns_false_when_no_messages(self, middleware):
        state = MemoryState(messages=[])
        assert middleware._has_memory_save_in_run(state) is False

    def test_detects_tool_result_message_by_name(self, middleware):
        """ToolMessage with name='save_memory_store' is detected."""
        msg = MagicMock()
        msg.name = "save_memory_store"
        msg.tool_calls = None
        state = MemoryState(messages=[msg])
        assert middleware._has_memory_save_in_run(state) is True

    def test_detects_ai_message_with_tool_call_dict(self, middleware):
        """AI message with tool_calls list of dicts is detected."""
        msg = MagicMock()
        msg.name = None
        msg.tool_calls = [{"name": "save_memory_store", "args": {}}]
        state = MemoryState(messages=[msg])
        assert middleware._has_memory_save_in_run(state) is True

    def test_detects_ai_message_with_tool_call_object(self, middleware):
        """AI message with tool_calls list of objects is detected."""
        tc = MagicMock()
        tc.name = "save_memory_store"
        msg = MagicMock()
        msg.name = None
        msg.tool_calls = [tc]
        state = MemoryState(messages=[msg])
        assert middleware._has_memory_save_in_run(state) is True

    def test_returns_false_for_other_tool_names(self, middleware):
        """Other tool names are not confused with save_memory_store."""
        msg = MagicMock()
        msg.name = "recall_memories_store"
        msg.tool_calls = [{"name": "recall_memories_store", "args": {}}]
        state = MemoryState(messages=[msg])
        assert middleware._has_memory_save_in_run(state) is False

    def test_returns_false_when_tool_calls_is_none(self, middleware):
        msg = MagicMock()
        msg.name = None
        msg.tool_calls = None
        state = MemoryState(messages=[msg])
        assert middleware._has_memory_save_in_run(state) is False


# ---------------------------------------------------------------------------
# Helpers for message construction
# ---------------------------------------------------------------------------


def _make_human(content: str):
    m = MagicMock()
    m.type = "human"
    m.role = None
    m.content = content
    return m


def _make_ai(content: str):
    m = MagicMock()
    m.type = "ai"
    m.role = None
    m.content = content
    return m


# ---------------------------------------------------------------------------
# _get_analysis_model
# ---------------------------------------------------------------------------


class TestGetAnalysisModel:
    """Verify model selection for post-run extraction.

    ``_get_analysis_model`` returns a ``(model_instance, model_name)`` tuple.
    The name is resolved via ``_resolve_memory_model_name`` (dedicated memory
    model, defaulting to ``settings.agent_memory_model``) and the instance is
    built through the global model factory.  The agent's own captured model is
    only an emergency fallback used when the configured model cannot be created.
    """

    def test_returns_dedicated_memory_model(self, middleware):
        """Default path resolves the configured memory model and instantiates it."""
        from inference_core.core.config import get_settings

        model, name = middleware._get_analysis_model()
        assert model is not None
        assert name == get_settings().agent_memory_model

    def test_falls_back_to_captured_when_creation_fails(self, middleware):
        """Captured model is the emergency fallback when the factory returns None."""
        captured = MagicMock()
        middleware._captured_model = captured
        with patch("inference_core.llm.models.get_model_factory") as mock_factory:
            mock_factory.return_value.create_model.return_value = None
            assert middleware._get_analysis_model() == (captured, None)

    def test_postrun_analysis_model_name_stored(self, mock_memory_service):
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            postrun_analysis_model="gpt-4.1-mini",
        )
        assert mw.postrun_analysis_model == "gpt-4.1-mini"

    def test_returns_none_when_no_model_anywhere(self, mock_memory_service):
        """(None, None) when no name resolves and no captured model exists."""
        mw = MemoryMiddleware(memory_service=mock_memory_service, user_id="u1")
        with patch.object(mw, "_resolve_memory_model_name", return_value=None):
            assert mw._get_analysis_model() == (None, None)


# ---------------------------------------------------------------------------
# _analyse_via_tool_call
# ---------------------------------------------------------------------------


class TestAnalyseViaToolCall:
    """Verify the model-based tool-calling extraction path."""

    async def test_returns_zero_for_empty_messages(self, middleware):
        state = MemoryState(messages=[])
        result = await middleware._analyse_via_tool_call(
            state, MagicMock(), None, "u1", False
        )
        assert result == 0

    async def test_returns_zero_when_model_returns_no_tool_calls(self, middleware):
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        mock_response = MagicMock()
        mock_response.tool_calls = []
        model = MagicMock()
        model.bind_tools.return_value.ainvoke = AsyncMock(return_value=mock_response)
        result = await middleware._analyse_via_tool_call(
            state, model, None, "u1", False
        )
        assert result == 0

    async def test_calls_arun_for_save_memory_store_tool_call(
        self, middleware, mock_memory_service
    ):
        """When model issues one save_memory_store tool_call, _arun is invoked."""
        mock_memory_service.save_memory = AsyncMock(return_value="mem-id-abc")
        state = MemoryState(messages=[_make_human("I use Python"), _make_ai("Got it.")])
        tc = {
            "name": "save_memory_store",
            "args": {"content": "Uses Python", "memory_type": "general"},
        }
        mock_response = MagicMock()
        mock_response.tool_calls = [tc]
        done_response = MagicMock()
        done_response.tool_calls = []
        model = MagicMock()
        model.bind_tools.return_value.ainvoke = AsyncMock(
            side_effect=[mock_response, done_response]
        )
        result = await middleware._analyse_via_tool_call(
            state, model, None, "u1", False
        )
        assert result == 1
        mock_memory_service.save_memory.assert_awaited_once()

    async def test_saves_multiple_tool_calls(self, middleware, mock_memory_service):
        """When model issues two tool calls, both are executed and count returned."""
        mock_memory_service.save_memory = AsyncMock(return_value="mem-id")
        state = MemoryState(
            messages=[_make_human("I like Python and dark mode"), _make_ai("Noted.")]
        )
        tcs = [
            {"name": "save_memory_store", "args": {"content": "Likes Python"}},
            {"name": "save_memory_store", "args": {"content": "Prefers dark mode"}},
        ]
        mock_response = MagicMock()
        mock_response.tool_calls = tcs
        done_response = MagicMock()
        done_response.tool_calls = []
        model = MagicMock()
        model.bind_tools.return_value.ainvoke = AsyncMock(
            side_effect=[mock_response, done_response]
        )
        result = await middleware._analyse_via_tool_call(
            state, model, None, "u1", False
        )
        assert result == 2
        assert mock_memory_service.save_memory.await_count == 2

    async def test_ignores_other_tool_calls(self, middleware):
        """Non-save_memory_store tool calls are silently skipped."""
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        tc = {"name": "some_other_tool", "args": {"x": 1}}
        mock_response = MagicMock()
        mock_response.tool_calls = [tc]
        model = MagicMock()
        model.bind_tools.return_value.ainvoke = AsyncMock(return_value=mock_response)
        result = await middleware._analyse_via_tool_call(
            state, model, None, "u1", False
        )
        assert result == 0

    async def test_continues_even_when_save_memory_raises(
        self, middleware, mock_memory_service
    ):
        """_arun swallows save_memory exceptions internally; both tool calls still complete.

        WHY: SaveMemoryStoreTool._arun catches all exceptions and returns an error string
        rather than re-raising. So from _analyse_via_tool_call's perspective both calls
        "complete" (no exception propagates), and saved_count counts invocations not
        DB-level success.
        """
        call_count = 0

        async def flaky(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("DB down")
            return "ok"

        mock_memory_service.save_memory = flaky
        state = MemoryState(messages=[_make_human("msg"), _make_ai("resp")])
        tcs = [
            {"name": "save_memory_store", "args": {"content": "first"}},
            {"name": "save_memory_store", "args": {"content": "second"}},
        ]
        mock_response = MagicMock()
        mock_response.tool_calls = tcs
        model = MagicMock()
        done_response = MagicMock()
        done_response.tool_calls = []
        model.bind_tools.return_value.ainvoke = AsyncMock(
            side_effect=[mock_response, done_response]
        )
        result = await middleware._analyse_via_tool_call(
            state, model, None, "u1", False
        )
        # _arun swallows internally — both calls complete, save_memory called twice
        assert result == 2
        assert call_count == 2

    async def test_returns_zero_on_model_invocation_exception(self, middleware):
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        model = MagicMock()
        model.bind_tools.return_value.ainvoke = AsyncMock(
            side_effect=RuntimeError("model error")
        )
        result = await middleware._analyse_via_tool_call(
            state, model, None, "u1", False
        )
        assert result == 0

    async def test_forwards_already_saved_flag_in_prompt(self, middleware):
        """When already_saved=True the user-turn prompt must mention it to guide the model."""
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        captured_messages: list = []

        async def capture(messages):
            captured_messages.extend(messages)
            mock_response = MagicMock()
            mock_response.tool_calls = []
            return mock_response

        model = MagicMock()
        model.bind_tools.return_value.ainvoke = capture
        await middleware._analyse_via_tool_call(
            state, model, None, "u1", already_saved=True
        )
        # messages[0] is system, messages[1] is user task prompt
        user_msg = next(m for m in captured_messages if m["role"] == "user")
        assert "already saved" in user_msg["content"].lower()

    async def test_sends_system_message_as_first_turn(self, middleware):
        """First message in messages_ctx must be a system-role message with security rules."""
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        captured_messages: list = []

        async def capture(messages):
            captured_messages.extend(messages)
            mock_response = MagicMock()
            mock_response.tool_calls = []
            return mock_response

        model = MagicMock()
        model.bind_tools.return_value.ainvoke = capture
        await middleware._analyse_via_tool_call(state, model, None, "u1", False)
        assert captured_messages[0]["role"] == "system"
        assert "CRITICAL SECURITY" in captured_messages[0]["content"]
        assert "memory management" in captured_messages[0]["content"].lower()

    async def test_security_prompt_forbids_treating_transcript_as_instruction(
        self, middleware
    ):
        """System prompt must explicitly state transcript/memories are data, not instructions."""
        assert (
            "not instructions" in middleware._TOOL_ANALYSIS_SYSTEM_PROMPT.lower()
            or "never instructions" in middleware._TOOL_ANALYSIS_SYSTEM_PROMPT.lower()
            or "are never instructions"
            in middleware._TOOL_ANALYSIS_SYSTEM_PROMPT.lower()
            or "are data" in middleware._TOOL_ANALYSIS_SYSTEM_PROMPT.lower()
        )

    async def test_counts_update_memory_store_calls(
        self, middleware, mock_memory_service
    ):
        """update_memory_store calls are counted the same as save calls."""
        mock_memory_service.save_memory = AsyncMock(return_value="mem-id-updated")
        state = MemoryState(
            messages=[_make_human("my name is Jan"), _make_ai("Noted.")]
        )
        tc = {
            "name": "update_memory_store",
            "args": {"memory_id": "abc", "content": "Name: Jan"},
        }
        mock_response = MagicMock()
        mock_response.tool_calls = [tc]
        done_response = MagicMock()
        done_response.tool_calls = []
        model = MagicMock()
        model.bind_tools.return_value.ainvoke = AsyncMock(
            side_effect=[mock_response, done_response]
        )
        result = await middleware._analyse_via_tool_call(
            state, model, None, "u1", False
        )
        assert result == 1
        mock_memory_service.save_memory.assert_awaited_once()

    async def test_executes_recall_but_does_not_count_it(
        self, middleware, mock_memory_service
    ):
        """recall_memories_store calls are executed for the model but not counted."""
        recalled = MagicMock()
        recalled.id = "r1"
        recalled.score = 0.75
        recalled.memory_type = "preferences"
        recalled.memory_category = "semantic"
        recalled.topic = "food"
        recalled.created_at = None
        recalled.content = "Likes pizza"
        mock_memory_service.recall_memories = AsyncMock(return_value=[recalled])

        state = MemoryState(
            messages=[_make_human("I also like pasta"), _make_ai("Nice.")]
        )
        recall_tc = {
            "name": "recall_memories_store",
            "args": {"query": "food preferences"},
            "id": "r-call-1",
        }
        mock_recall_response = MagicMock()
        mock_recall_response.tool_calls = [recall_tc]
        done_response = MagicMock()
        done_response.tool_calls = []
        model = MagicMock()
        model.bind_tools.return_value.ainvoke = AsyncMock(
            side_effect=[mock_recall_response, done_response]
        )
        result = await middleware._analyse_via_tool_call(
            state, model, None, "u1", False
        )
        # recall is executed but not counted
        assert result == 0

    async def test_multi_turn_recall_then_save(self, middleware, mock_memory_service):
        """Model recalls in first turn, saves in second — multi-turn loop works."""
        recalled = MagicMock()
        recalled.id = "r1"
        recalled.score = 0.75
        recalled.memory_type = "preferences"
        recalled.memory_category = "semantic"
        recalled.topic = "food"
        recalled.created_at = None
        recalled.content = "Likes pizza"
        # This recall_memories is for the tool call (not the pre-fetch — pre-fetch uses service directly)
        mock_memory_service.recall_memories = AsyncMock(return_value=[recalled])
        mock_memory_service.save_memory = AsyncMock(return_value="new-mem-id")

        state = MemoryState(
            messages=[_make_human("I also like sushi"), _make_ai("Noted.")]
        )

        recall_tc = {
            "name": "recall_memories_store",
            "args": {"query": "food"},
            "id": "recall-1",
        }
        save_tc = {"name": "save_memory_store", "args": {"content": "Likes sushi"}}

        recall_response = MagicMock()
        recall_response.tool_calls = [recall_tc]
        save_response = MagicMock()
        save_response.tool_calls = [save_tc]
        done_response = MagicMock()
        done_response.tool_calls = []

        model = MagicMock()
        model.bind_tools.return_value.ainvoke = AsyncMock(
            side_effect=[recall_response, save_response, done_response]
        )
        result = await middleware._analyse_via_tool_call(
            state, model, None, "u1", False
        )
        assert result == 1
        mock_memory_service.save_memory.assert_awaited_once()

    async def test_binds_all_three_tools(self, middleware, mock_memory_service):
        """bind_tools must receive save, recall, and update tool instances."""
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        mock_response = MagicMock()
        mock_response.tool_calls = []
        model = MagicMock()
        model.bind_tools.return_value.ainvoke = AsyncMock(return_value=mock_response)
        await middleware._analyse_via_tool_call(state, model, None, "u1", False)
        tools_passed = model.bind_tools.call_args[0][0]
        tool_names = [t.name for t in tools_passed]
        assert "save_memory_store" in tool_names
        assert "recall_memories_store" in tool_names
        assert "update_memory_store" in tool_names


# ---------------------------------------------------------------------------
# _prefetch_existing_memories
# ---------------------------------------------------------------------------


class TestPrefetchExistingMemories:
    """Verify semantic pre-fetch of relevant existing memories."""

    async def test_returns_empty_string_for_no_memories(
        self, middleware, mock_memory_service
    ):
        mock_memory_service.recall_memories = AsyncMock(return_value=[])
        result = await middleware._prefetch_existing_memories("some transcript", "u1")
        assert result == ""

    async def test_formats_memories_with_id_and_content(
        self, middleware, mock_memory_service
    ):
        """Pre-fetched memories include id (for update calls) and readable content."""
        mem = MagicMock()
        mem.id = "abc123"
        mem.score = 0.75
        mem.memory_type = "preferences"
        mem.memory_category = "semantic"
        mem.topic = "cooking"
        mem.created_at = None
        mem.content = "Likes aniagotuje.pl"
        mock_memory_service.recall_memories = AsyncMock(return_value=[mem])
        result = await middleware._prefetch_existing_memories(
            "cooking transcript", "u1"
        )
        assert "abc123" in result
        assert "Likes aniagotuje.pl" in result
        assert "preferences" in result

    async def test_returns_empty_on_recall_exception(
        self, middleware, mock_memory_service
    ):
        """Fail-open: exception during recall returns empty string, never raises."""
        mock_memory_service.recall_memories = AsyncMock(
            side_effect=RuntimeError("store down")
        )
        result = await middleware._prefetch_existing_memories("transcript", "u1")
        assert result == ""

    async def test_truncates_transcript_to_2000_chars_for_query(
        self, middleware, mock_memory_service
    ):
        """Only first 2000 chars of transcript are sent as the recall query."""
        mock_memory_service.recall_memories = AsyncMock(return_value=[])
        long_transcript = "x" * 5000
        await middleware._prefetch_existing_memories(long_transcript, "u1")
        call_kwargs = mock_memory_service.recall_memories.call_args
        query = call_kwargs[1].get("query") or call_kwargs[0][1]
        assert len(query) == 2000

    async def test_filters_by_score_threshold(self, middleware, mock_memory_service):
        """Memories with score < 0.60 are excluded from the pre-fetch output."""
        high = MagicMock()
        high.id = "high"
        high.score = 0.75
        high.memory_type = "preferences"
        high.memory_category = "semantic"
        high.topic = "cooking"
        high.created_at = None
        high.content = "Likes aniagotuje.pl"

        low = MagicMock()
        low.id = "low"
        low.score = 0.42
        low.memory_type = "preferences"
        low.memory_category = "semantic"
        low.topic = "cooking"
        low.created_at = None
        low.content = "Loosely related entry"

        mock_memory_service.recall_memories = AsyncMock(return_value=[high, low])
        result = await middleware._prefetch_existing_memories(
            "cooking transcript", "u1"
        )
        assert "high" in result
        assert "low" not in result
        assert "Loosely related entry" not in result

    async def test_returns_empty_when_all_memories_below_threshold(
        self, middleware, mock_memory_service
    ):
        """When all memories score < 0.60, return empty string (nothing to show)."""
        mem = MagicMock()
        mem.score = 0.30
        mock_memory_service.recall_memories = AsyncMock(return_value=[mem])
        result = await middleware._prefetch_existing_memories("transcript", "u1")
        assert result == ""

    async def test_includes_created_at_in_output(self, middleware, mock_memory_service):
        """created_at timestamp is included when available."""
        import datetime

        mem = MagicMock()
        mem.id = "xyz"
        mem.score = 0.75
        mem.memory_type = "facts"
        mem.memory_category = "episodic"
        mem.topic = None
        mem.created_at = datetime.datetime(2026, 3, 24, 10, 10)
        mem.content = "User prefers dark mode"
        mock_memory_service.recall_memories = AsyncMock(return_value=[mem])
        result = await middleware._prefetch_existing_memories("dark mode", "u1")
        assert "2026-03-24" in result
        assert "User prefers dark mode" in result


# ---------------------------------------------------------------------------
# after_agent (sync)
# ---------------------------------------------------------------------------


class TestAfterAgent:
    """Verify post-run persistence hook — sync path."""

    def test_skips_when_postrun_analysis_disabled(self, mock_memory_service):
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            postrun_analysis=False,
        )
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        assert mw.after_agent(state, MagicMock()) is None

    def test_skips_when_no_user_id(self, mock_memory_service):
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id=None,
        )
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        with patch(
            "inference_core.agents.middleware.memory._ctx_get_user_id",
            return_value=None,
        ):
            assert mw.after_agent(state, MagicMock()) is None

    def test_skips_when_fewer_than_two_messages(self, middleware):
        state = MemoryState(messages=[_make_human("hi")])
        assert middleware.after_agent(state, MagicMock()) is None

    def test_skips_when_no_model_available(self, middleware):
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        with patch.object(middleware, "_get_analysis_model", return_value=(None, None)):
            assert middleware.after_agent(state, MagicMock()) is None

    def test_returns_none_when_analyse_returns_zero(self, middleware):
        """When run_async_safely returns 0 (nothing saved), after_agent returns None."""
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        middleware._captured_model = MagicMock()
        with patch(
            "inference_core.agents.middleware.memory.run_async_safely",
            return_value=0,
        ):
            assert middleware.after_agent(state, MagicMock()) is None

    def test_saves_and_returns_state_on_success(self, middleware):
        """When run_async_safely returns saved_count > 0, state dict is returned."""
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        middleware._captured_model = MagicMock()
        with patch(
            "inference_core.agents.middleware.memory.run_async_safely",
            return_value=1,
        ):
            result = middleware.after_agent(state, MagicMock())
        assert result is not None
        assert result["session_analysis_saved"] is True
        assert "session_analysis_latency_ms" in result

    def test_returns_none_on_exception(self, middleware):
        """run_async_safely raising is swallowed (fail-open)."""
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        middleware._captured_model = MagicMock()
        with patch(
            "inference_core.agents.middleware.memory.run_async_safely",
            side_effect=RuntimeError("DB down"),
        ):
            assert middleware.after_agent(state, MagicMock()) is None


# ---------------------------------------------------------------------------
# aafter_agent (async)
# ---------------------------------------------------------------------------


class TestAAfterAgent:
    """Verify post-run persistence hook — async path."""

    async def test_skips_when_postrun_analysis_disabled(self, mock_memory_service):
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            postrun_analysis=False,
        )
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        assert await mw.aafter_agent(state, MagicMock()) is None

    async def test_skips_when_no_user_id(self, mock_memory_service):
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id=None,
        )
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        with patch(
            "inference_core.agents.middleware.memory._ctx_get_user_id",
            return_value=None,
        ):
            assert await mw.aafter_agent(state, MagicMock()) is None

    async def test_skips_when_no_model_available(self, mock_memory_service):
        mw = MemoryMiddleware(memory_service=mock_memory_service, user_id="u1")
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        with patch.object(mw, "_get_analysis_model", return_value=(None, None)):
            assert await mw.aafter_agent(state, MagicMock()) is None

    async def test_returns_none_when_analyse_returns_zero(self, mock_memory_service):
        mw = MemoryMiddleware(memory_service=mock_memory_service, user_id="u1")
        mw._captured_model = MagicMock()
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        with patch.object(mw, "_analyse_via_tool_call", new=AsyncMock(return_value=0)):
            assert await mw.aafter_agent(state, MagicMock()) is None

    async def test_saves_and_returns_state_on_success(self, mock_memory_service):
        mw = MemoryMiddleware(memory_service=mock_memory_service, user_id="u1")
        mw._captured_model = MagicMock()
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        with patch.object(mw, "_analyse_via_tool_call", new=AsyncMock(return_value=2)):
            result = await mw.aafter_agent(state, MagicMock())
        assert result is not None
        assert result["session_analysis_saved"] is True

    async def test_returns_none_on_exception(self, mock_memory_service):
        mw = MemoryMiddleware(memory_service=mock_memory_service, user_id="u1")
        mw._captured_model = MagicMock()
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        with patch.object(
            mw,
            "_analyse_via_tool_call",
            new=AsyncMock(side_effect=RuntimeError("DB down")),
        ):
            assert await mw.aafter_agent(state, MagicMock()) is None

    async def test_passes_already_saved_flag(self, mock_memory_service):
        """already_saved flag from _has_memory_save_in_run is forwarded to _analyse_via_tool_call."""
        mw = MemoryMiddleware(memory_service=mock_memory_service, user_id="u1")
        mw._captured_model = MagicMock()
        # Build a state where a save was already made
        tc = MagicMock()
        tc.name = "save_memory_store"
        ai_msg = MagicMock()
        ai_msg.name = None
        ai_msg.tool_calls = [tc]
        state = MemoryState(messages=[_make_human("msg"), ai_msg])
        captured_already_saved: list[bool] = []

        async def capture(st, model, model_name, uid, already_saved):
            captured_already_saved.append(already_saved)
            return 0

        with patch.object(mw, "_analyse_via_tool_call", new=capture):
            await mw.aafter_agent(state, MagicMock())
        assert captured_already_saved == [True]


# ---------------------------------------------------------------------------
# postrun_analysis init param + factory
# ---------------------------------------------------------------------------


class TestPostrunAnalysisParam:
    """Verify postrun_analysis and postrun_analysis_model are wired correctly."""

    def test_default_postrun_analysis_is_true(self, mock_memory_service):
        mw = MemoryMiddleware(memory_service=mock_memory_service, user_id="u1")
        assert mw.postrun_analysis is True

    def test_postrun_analysis_can_be_disabled(self, mock_memory_service):
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            postrun_analysis=False,
        )
        assert mw.postrun_analysis is False

    def test_postrun_analysis_model_default_none(self, mock_memory_service):
        mw = MemoryMiddleware(memory_service=mock_memory_service, user_id="u1")
        assert mw.postrun_analysis_model is None

    def test_postrun_analysis_model_stored(self, mock_memory_service):
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            postrun_analysis_model="gpt-4.1-mini",
        )
        assert mw.postrun_analysis_model == "gpt-4.1-mini"

    def test_factory_passes_postrun_analysis(self, mock_memory_service):
        mw = create_memory_middleware(
            memory_service=mock_memory_service,
            user_id="u42",
            postrun_analysis=False,
        )
        assert mw.postrun_analysis is False

    def test_factory_passes_postrun_analysis_model(self, mock_memory_service):
        mw = create_memory_middleware(
            memory_service=mock_memory_service,
            user_id="u42",
            postrun_analysis_model="gpt-4.1-mini",
        )
        assert mw.postrun_analysis_model == "gpt-4.1-mini"

    def test_factory_default_postrun_analysis_is_true(self, mock_memory_service):
        mw = create_memory_middleware(
            memory_service=mock_memory_service,
            user_id="u42",
        )
        assert mw.postrun_analysis is True


# ===========================================================================
# _should_inject_tool_instructions resolution
# ===========================================================================


class TestShouldInjectToolInstructions:
    """Verify resolution order: contextvar → compile-time → default."""

    def test_default_true_when_tools_present(self, mock_memory_service):
        """Default is True when active_tool_names has tools."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            active_tool_names=["save_memory_store"],
        )
        assert mw._should_inject_tool_instructions() is True

    def test_default_true_when_tools_none(self, mock_memory_service):
        """Default is True when active_tool_names is None (all tools)."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            active_tool_names=None,
        )
        assert mw._should_inject_tool_instructions() is True

    def test_default_false_when_tools_empty(self, mock_memory_service):
        """Default is False when active_tool_names is empty list."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            active_tool_names=[],
        )
        assert mw._should_inject_tool_instructions() is False

    def test_compile_time_override(self, mock_memory_service):
        """Compile-time tool_instructions_enabled overrides default."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            active_tool_names=["save_memory_store"],
            tool_instructions_enabled=False,
        )
        assert mw._should_inject_tool_instructions() is False

    def test_contextvar_overrides_compile_time(self, mock_memory_service):
        """Per-request contextvar overrides compile-time setting."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            active_tool_names=["save_memory_store"],
            tool_instructions_enabled=False,
        )
        with patch(
            "inference_core.agents.middleware.memory._ctx_get_memory_tool_instructions_enabled",
            return_value=True,
        ):
            assert mw._should_inject_tool_instructions() is True

    def test_contextvar_false_overrides_default_true(self, mock_memory_service):
        """Per-request contextvar=False overrides default=True."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            active_tool_names=["save_memory_store"],
        )
        with patch(
            "inference_core.agents.middleware.memory._ctx_get_memory_tool_instructions_enabled",
            return_value=False,
        ):
            assert mw._should_inject_tool_instructions() is False


# ===========================================================================
# _build_injection_blocks
# ===========================================================================


class TestBuildInjectionBlocks:
    """Verify block construction for memory context + tool instructions."""

    def test_no_blocks_when_nothing_cached(self, mock_memory_service):
        """Returns empty list when no cached context and no tool instructions needed."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
        )
        mw._cached_memory_context = None
        blocks = mw._build_injection_blocks()
        assert blocks == []

    def test_memory_context_block_only(self, mock_memory_service):
        """Returns memory context block when cached and no active_tool_names."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            active_tool_names=None,
        )
        mw._cached_memory_context = "User prefers Python"
        blocks = mw._build_injection_blocks()
        assert len(blocks) == 1
        assert "<user_memory_context>" in blocks[0]["text"]
        assert "User prefers Python" in blocks[0]["text"]

    def test_both_blocks_on_agent_server(self, mock_memory_service):
        """Returns memory context + tool instructions when on Agent Server."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            active_tool_names=["save_memory_store", "recall_memories_store"],
        )
        mw._cached_memory_context = "User prefers Python"
        blocks = mw._build_injection_blocks()
        assert len(blocks) == 2
        assert "<user_memory_context>" in blocks[0]["text"]
        assert "save_memory_store" in blocks[1]["text"]

    def test_tool_instructions_only_when_no_cached_context(self, mock_memory_service):
        """Returns only tool instructions when no cached context but tools active."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            active_tool_names=["save_memory_store"],
        )
        mw._cached_memory_context = None
        blocks = mw._build_injection_blocks()
        assert len(blocks) == 1
        assert "save_memory_store" in blocks[0]["text"]

    def test_no_tool_instructions_when_disabled(self, mock_memory_service):
        """No tool instruction block when tool_instructions_enabled=False."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            active_tool_names=["save_memory_store"],
            tool_instructions_enabled=False,
        )
        mw._cached_memory_context = "context"
        blocks = mw._build_injection_blocks()
        assert len(blocks) == 1
        assert "<user_memory_context>" in blocks[0]["text"]

    def test_no_tool_instructions_when_local_path(self, mock_memory_service):
        """No tool instruction block on local path (active_tool_names=None)."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            active_tool_names=None,
        )
        mw._cached_memory_context = "context"
        blocks = mw._build_injection_blocks()
        assert len(blocks) == 1
        assert "<user_memory_context>" in blocks[0]["text"]


# ===========================================================================
# before_agent — per-request session context override
# ===========================================================================


class TestBeforeAgentSessionContextOverride:
    """Verify contextvar override disables before_agent recall."""

    def test_contextvar_false_skips_recall(self, mock_memory_service):
        """before_agent returns None when contextvar disables session context."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            auto_recall=True,
        )
        msg = MagicMock()
        msg.type = "human"
        msg.content = "test"
        state = MemoryState(messages=[msg])

        with patch(
            "inference_core.agents.middleware.memory._ctx_get_memory_session_context_enabled",
            return_value=False,
        ):
            result = mw.before_agent(state, MagicMock())

        assert result is None

    def test_contextvar_none_allows_recall(self, mock_memory_service):
        """before_agent proceeds normally when contextvar is None (not overridden)."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            auto_recall=True,
        )
        msg = MagicMock()
        msg.type = "human"
        msg.content = "test query"
        state = MemoryState(messages=[msg])

        with (
            patch(
                "inference_core.agents.middleware.memory._ctx_get_memory_session_context_enabled",
                return_value=None,
            ),
            patch.object(
                mw, "_recall_and_format", return_value=("", {"latency_ms": 1.0})
            ),
        ):
            result = mw.before_agent(state, MagicMock())

        assert result is not None
        assert result["memories_recalled"] == 0
