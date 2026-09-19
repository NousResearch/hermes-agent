"""SamplingHandler tests for the MCP client support.

Extracted byte-for-byte from ``tests/tools/test_mcp_tool.py`` (2K-law
fracture of a god-file) - no assertion, mock target, or fixture body was
changed by the move.

Part of #79959, #78647.
"""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stop_reason(result):
    """Read a sampling result's stop reason across the mcp 1.x -> 2.x rename.

    ``CreateMessageResult.stopReason`` became ``.stop_reason`` in mcp 2.0
    (camelCase survives only as the serialization alias, which pydantic does
    not expose to attribute access).
    """
    from tools.mcp_tool import mcp_field

    return mcp_field(result, "stop_reason", "stopReason")


# ===========================================================================
# SamplingHandler tests
# ===========================================================================


class _CompatType:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


try:
    from mcp.types import (
        CreateMessageResult,
        ErrorData,
        SamplingCapability,
        TextContent,
    )
except ImportError:
    CreateMessageResult = _CompatType
    ErrorData = _CompatType
    SamplingCapability = _CompatType
    TextContent = _CompatType

try:
    from mcp.types import CreateMessageResultWithTools
except ImportError:
    CreateMessageResultWithTools = _CompatType

try:
    from mcp.types import SamplingToolsCapability
except ImportError:
    SamplingToolsCapability = _CompatType

try:
    from mcp.types import ToolUseContent
except ImportError:
    ToolUseContent = _CompatType

from tools.mcp_tool import CreateMessageResultWithTools, SamplingHandler, SamplingToolsCapability, ToolUseContent
from tools.mcp_tool_common import _safe_numeric


# ---------------------------------------------------------------------------
# Helpers for sampling tests
# ---------------------------------------------------------------------------

def _make_sampling_params(
    messages=None,
    max_tokens=100,
    system_prompt=None,
    model_preferences=None,
    temperature=None,
    stop_sequences=None,
    tools=None,
    tool_choice=None,
):
    """Create a fake CreateMessageRequestParams using SimpleNamespace.

    Each message must have a ``content_as_list`` attribute that mirrors
    the SDK helper so that ``_convert_messages`` works correctly.
    """
    if messages is None:
        content = SimpleNamespace(text="Hello")
        msg = SimpleNamespace(role="user", content=content, content_as_list=[content])
        messages = [msg]

    params = SimpleNamespace(
        messages=messages,
        maxTokens=max_tokens,
        modelPreferences=model_preferences,
        temperature=temperature,
        stopSequences=stop_sequences,
        tools=tools,
        toolChoice=tool_choice,
    )
    if system_prompt is not None:
        params.systemPrompt = system_prompt
    return params


def _make_llm_response(
    content="LLM response",
    model="test-model",
    finish_reason="stop",
    tool_calls=None,
):
    """Create a fake OpenAI chat completion response (text)."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(
        finish_reason=finish_reason,
        message=message,
    )
    usage = SimpleNamespace(total_tokens=42)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def _make_llm_tool_response(tool_calls_data=None, model="test-model"):
    """Create a fake response with tool_calls.

    ``tool_calls_data``: list of (id, name, arguments_json) tuples.
    """
    if tool_calls_data is None:
        tool_calls_data = [("call_1", "get_weather", '{"city": "London"}')]

    tc_list = [
        SimpleNamespace(
            id=tc_id,
            function=SimpleNamespace(name=name, arguments=args),
        )
        for tc_id, name, args in tool_calls_data
    ]
    return _make_llm_response(
        content=None,
        model=model,
        finish_reason="tool_calls",
        tool_calls=tc_list,
    )


# ---------------------------------------------------------------------------
# 1. _safe_numeric helper
# ---------------------------------------------------------------------------

class TestSafeNumeric:
    def test_coercion_clamping_and_fallbacks(self):
        # (value, default, caster, kwargs, expected)
        cases = [
            (10, 5, int, {}, 10),
            ("20", 5, int, {}, 20),
            ("3.5", 1.0, float, {}, 3.5),
            (None, 7, int, {}, 7),
            ("abc", 42, int, {}, 42),
            (float("inf"), 3.0, float, {}, 3.0),
            (float("nan"), 4.0, float, {}, 4.0),
            (-5, 10, int, {"minimum": 1}, 1),
            (0, 10, int, {"minimum": 0}, 0),
        ]
        for value, default, caster, kwargs, expected in cases:
            assert _safe_numeric(value, default, caster, **kwargs) == expected, value

# ---------------------------------------------------------------------------
# 2. SamplingHandler initialization and config parsing
# ---------------------------------------------------------------------------

class TestSamplingHandlerInit:
    def test_defaults(self):
        h = SamplingHandler("srv", {})
        assert h.server_name == "srv"
        assert h.max_rpm == 10
        assert h.timeout == 30
        assert h.max_tokens_cap == 4096
        assert h.max_tool_rounds == 5
        assert h.model_override is None
        assert h.allowed_models == []
        assert h.metrics == {"requests": 0, "errors": 0, "tokens_used": 0, "tool_use_count": 0}

    def test_custom_config(self):
        cfg = {
            "max_rpm": 20,
            "timeout": 60,
            "max_tokens_cap": 2048,
            "max_tool_rounds": 3,
            "model": "gpt-4o",
            "allowed_models": ["gpt-4o", "gpt-3.5-turbo"],
            "log_level": "debug",
        }
        h = SamplingHandler("custom", cfg)
        assert h.max_rpm == 20
        assert h.timeout == 60.0
        assert h.max_tokens_cap == 2048
        assert h.max_tool_rounds == 3
        assert h.model_override == "gpt-4o"
        assert h.allowed_models == ["gpt-4o", "gpt-3.5-turbo"]

# ---------------------------------------------------------------------------
# 3. Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimit:
    def setup_method(self):
        self.handler = SamplingHandler("rl", {"max_rpm": 3})

    def test_rejects_over_limit(self):
        for _ in range(3):
            self.handler._check_rate_limit()
        assert self.handler._check_rate_limit() is False

    def test_window_expiry(self):
        """Old timestamps should be purged from the sliding window."""
        for _ in range(3):
            self.handler._check_rate_limit()
        # Simulate timestamps from 61 seconds ago
        self.handler._rate_timestamps[:] = [time.time() - 61] * 3
        assert self.handler._check_rate_limit() is True


# ---------------------------------------------------------------------------
# 4. Model resolution
# ---------------------------------------------------------------------------

class TestResolveModel:
    def setup_method(self):
        self.handler = SamplingHandler("mr", {})

    def test_config_override_wins(self):
        self.handler.model_override = "override-model"
        prefs = SimpleNamespace(hints=[SimpleNamespace(name="hint-model")])
        assert self.handler._resolve_model(prefs) == "override-model"

    def test_hint_used_when_no_override(self):
        prefs = SimpleNamespace(hints=[SimpleNamespace(name="hint-model")])
        assert self.handler._resolve_model(prefs) == "hint-model"

# ---------------------------------------------------------------------------
# 5. Message conversion
# ---------------------------------------------------------------------------

class TestConvertMessages:
    def setup_method(self):
        self.handler = SamplingHandler("mc", {})

    def test_single_text_message(self):
        content = SimpleNamespace(text="Hello world")
        msg = SimpleNamespace(role="user", content=content, content_as_list=[content])
        params = _make_sampling_params(messages=[msg])
        result = self.handler._convert_messages(params)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello world"}


    def test_tool_use_message(self):
        tu_block = SimpleNamespace(
            id="call_2", name="get_weather", input={"city": "London"}
        )
        msg = SimpleNamespace(
            role="assistant",
            content=[tu_block],
            content_as_list=[tu_block],
        )
        params = _make_sampling_params(messages=[msg])
        result = self.handler._convert_messages(params)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["tool_calls"][0]["function"]["arguments"]) == {"city": "London"}

# ---------------------------------------------------------------------------
# 6. Text-only sampling callback (full flow)
# ---------------------------------------------------------------------------

class TestSamplingCallbackText:
    def setup_method(self):
        self.handler = SamplingHandler("txt", {})

    def test_text_response(self):
        """Full flow: text response returns CreateMessageResult."""
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response(
            content="Hello from LLM"
        )

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            params = _make_sampling_params()
            result = asyncio.run(self.handler(None, params))

        assert isinstance(result, CreateMessageResult)
        assert isinstance(result.content, TextContent)
        assert result.content.text == "Hello from LLM"
        assert result.model == "test-model"
        assert result.role == "assistant"
        assert _stop_reason(result) == "endTurn"

    def test_server_tools_with_object_schema_are_normalized(self):
        """Server-provided tools should gain empty properties for object schemas."""
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response()
        server_tool = SimpleNamespace(
            name="ask",
            description="Ask Crawl4AI",
            inputSchema={"type": "object"},
        )

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ) as mock_call:
            params = _make_sampling_params(tools=[server_tool])
            asyncio.run(self.handler(None, params))

        tools = mock_call.call_args.kwargs["tools"]
        assert tools == [{
            "type": "function",
            "function": {
                "name": "ask",
                "description": "Ask Crawl4AI",
                "parameters": {"type": "object", "properties": {}},
            },
        }]

# ---------------------------------------------------------------------------
# 7. Tool use sampling callback
# ---------------------------------------------------------------------------

class TestSamplingCallbackToolUse:
    def setup_method(self):
        self.handler = SamplingHandler("tu", {})

    def test_tool_use_response(self):
        """LLM tool_calls response returns CreateMessageResultWithTools."""
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_tool_response()

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            params = _make_sampling_params()
            result = asyncio.run(self.handler(None, params))

        assert isinstance(result, CreateMessageResultWithTools)
        assert _stop_reason(result) == "toolUse"
        assert result.model == "test-model"
        assert len(result.content) == 1
        tc = result.content[0]
        assert isinstance(tc, ToolUseContent)
        assert tc.name == "get_weather"
        assert tc.id == "call_1"
        assert tc.input == {"city": "London"}

# ---------------------------------------------------------------------------
# 8. Tool loop governance
# ---------------------------------------------------------------------------

class TestToolLoopGovernance:
    def test_max_tool_rounds_enforcement(self):
        """After max_tool_rounds consecutive tool responses, an error is returned."""
        handler = SamplingHandler("tl", {"max_tool_rounds": 2})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_tool_response()

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            params = _make_sampling_params()
            # Round 1, 2: allowed
            r1 = asyncio.run(handler(None, params))
            assert isinstance(r1, CreateMessageResultWithTools)
            r2 = asyncio.run(handler(None, params))
            assert isinstance(r2, CreateMessageResultWithTools)
            # Round 3: exceeds limit
            r3 = asyncio.run(handler(None, params))
            assert isinstance(r3, ErrorData)
            assert "Tool loop limit exceeded" in r3.message

    def test_max_tool_rounds_zero_disables(self):
        """max_tool_rounds=0 means tool loops are disabled entirely."""
        handler = SamplingHandler("tl3", {"max_tool_rounds": 0})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_tool_response()

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(result, ErrorData)
            assert "Tool loops disabled" in result.message


# ---------------------------------------------------------------------------
# 9. Error paths: rate limit, timeout, no provider
# ---------------------------------------------------------------------------

class TestSamplingErrors:
    def test_rate_limit_error(self):
        handler = SamplingHandler("rle", {"max_rpm": 1})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response()

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            # First call succeeds
            r1 = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(r1, CreateMessageResult)
            # Second call is rate limited
            r2 = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(r2, ErrorData)
            assert "rate limit" in r2.message.lower()
            assert handler.metrics["errors"] == 1

    def test_timeout_error(self):
        # Config values clamp to a 1s floor (_safe_numeric minimum), so set the
        # attribute directly to exercise the timeout branch without a 1s wait.
        handler = SamplingHandler("to", {})
        handler.timeout = 0.05

        def slow_call(**kwargs):
            import threading
            evt = threading.Event()
            # Outlives the 0.05s handler timeout, but short enough that the
            # abandoned worker thread doesn't stall loop shutdown.
            evt.wait(0.15)
            return _make_llm_response()

        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=slow_call,
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(result, ErrorData)
            assert "timed out" in result.message.lower()
            assert handler.metrics["errors"] == 1

    def test_empty_choices_returns_error(self):
        """LLM returning choices=[] is handled gracefully, not IndexError."""
        handler = SamplingHandler("ec", {})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[],
            model="test-model",
            usage=SimpleNamespace(total_tokens=0),
        )

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))

        assert isinstance(result, ErrorData)
        assert "empty response" in result.message.lower()
        assert handler.metrics["errors"] == 1

# ---------------------------------------------------------------------------
# 10. Model whitelist
# ---------------------------------------------------------------------------

class TestModelWhitelist:
    def test_allowed_model_passes(self):
        handler = SamplingHandler("wl", {"allowed_models": ["gpt-4o", "test-model"]})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response()

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(result, CreateMessageResult)

    def test_disallowed_model_rejected(self):
        handler = SamplingHandler("wl2", {"allowed_models": ["gpt-4o"], "model": "test-model"})
        fake_client = MagicMock()

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(result, ErrorData)
            assert "not allowed" in result.message
            assert handler.metrics["errors"] == 1

# ---------------------------------------------------------------------------
# 11. Malformed tool_call arguments
# ---------------------------------------------------------------------------

class TestMalformedToolCallArgs:
    def test_invalid_json_wrapped_as_raw(self):
        """Malformed JSON arguments get wrapped in {"_raw": ...}."""
        handler = SamplingHandler("mf", {})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_tool_response(
            tool_calls_data=[("call_x", "some_tool", "not valid json {{{")]
        )

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))

        assert isinstance(result, CreateMessageResultWithTools)
        tc = result.content[0]
        assert isinstance(tc, ToolUseContent)
        assert tc.input == {"_raw": "not valid json {{{"}

# ---------------------------------------------------------------------------
# 12. Metrics tracking
# ---------------------------------------------------------------------------

class TestMetricsTracking:
    def test_request_and_token_metrics(self):
        handler = SamplingHandler("met", {})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response()

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            asyncio.run(handler(None, _make_sampling_params()))

        assert handler.metrics["requests"] == 1
        assert handler.metrics["tokens_used"] == 42
        assert handler.metrics["errors"] == 0

# ---------------------------------------------------------------------------
# 13. session_kwargs()
# ---------------------------------------------------------------------------

class TestSessionKwargs:
    def test_returns_correct_keys(self):
        handler = SamplingHandler("sk", {})
        kwargs = handler.session_kwargs()
        assert "sampling_callback" in kwargs
        assert "sampling_capabilities" in kwargs
        assert kwargs["sampling_callback"] is handler

# ---------------------------------------------------------------------------
# 14. MCPServerTask integration
# ---------------------------------------------------------------------------

class TestMCPServerTaskSamplingIntegration:
    def test_sampling_handler_created_when_enabled(self):
        """MCPServerTask.run() creates a SamplingHandler when sampling is enabled."""
        from tools.mcp_tool import MCPServerTask, _MCP_SAMPLING_TYPES

        server = MCPServerTask("int_test")
        config = {
            "command": "fake",
            "sampling": {"enabled": True, "max_rpm": 5},
        }
        # We only need to test the setup logic, not the actual connection.
        # Calling run() would attempt a real connection, so we test the
        # sampling setup portion directly.
        server._config = config
        sampling_config = config.get("sampling", {})
        if sampling_config.get("enabled", True) and _MCP_SAMPLING_TYPES:
            server._sampling = SamplingHandler(server.name, sampling_config)
        else:
            server._sampling = None

        assert server._sampling is not None
        assert isinstance(server._sampling, SamplingHandler)
        assert server._sampling.server_name == "int_test"
        assert server._sampling.max_rpm == 5

    def test_sampling_handler_none_when_disabled(self):
        """MCPServerTask._sampling is None when sampling is disabled."""
        from tools.mcp_tool import MCPServerTask, _MCP_SAMPLING_TYPES

        server = MCPServerTask("int_test2")
        config = {
            "command": "fake",
            "sampling": {"enabled": False},
        }
        server._config = config
        sampling_config = config.get("sampling", {})
        if sampling_config.get("enabled", True) and _MCP_SAMPLING_TYPES:
            server._sampling = SamplingHandler(server.name, sampling_config)
        else:
            server._sampling = None

        assert server._sampling is None

