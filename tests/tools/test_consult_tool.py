"""Tests for the consult tool (second opinion from a reference model).

``consult`` wraps ``agent.auxiliary_client.call_llm(task="consult", ...)``.
Refusals / empty responses / call failures from the reference model must
degrade gracefully to ``{"unavailable": true, ...}`` rather than raising --
that's the whole point of the feature (reasoning-heavy or safety-tuned
frontier models refuse often enough that a hard failure here would make the
tool useless).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.consult_tool import consult_tool, MAX_CONTEXT_CHARS, MAX_QUESTION_CHARS


def _fake_response(content="", finish_reason="stop"):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.reasoning = None
    resp.choices[0].message.reasoning_content = None
    resp.choices[0].message.reasoning_details = None
    resp.choices[0].finish_reason = finish_reason
    return resp


class TestConsultToolSuccess:
    def test_returns_answer_on_success(self):
        resp = _fake_response("Looks sound; watch the race on shutdown.")
        with patch("agent.auxiliary_client.call_llm", return_value=resp):
            result = json.loads(consult_tool("Is this plan sound?"))
        assert result["unavailable"] is False
        assert result["answer"] == "Looks sound; watch the race on shutdown."

    def test_passes_task_consult_to_call_llm(self):
        resp = _fake_response("ok")
        with patch("agent.auxiliary_client.call_llm", return_value=resp) as mock_call:
            consult_tool("Is this plan sound?")
        assert mock_call.call_args.kwargs.get("task") == "consult"

    def test_includes_context_in_user_message(self):
        resp = _fake_response("ok")
        with patch("agent.auxiliary_client.call_llm", return_value=resp) as mock_call:
            consult_tool("Is this plan sound?", context="def foo(): pass")
        messages = mock_call.call_args.kwargs.get("messages")
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "def foo(): pass" in user_msg["content"]
        assert "Is this plan sound?" in user_msg["content"]

    def test_no_context_omits_separator(self):
        resp = _fake_response("ok")
        with patch("agent.auxiliary_client.call_llm", return_value=resp) as mock_call:
            consult_tool("Is this plan sound?")
        messages = mock_call.call_args.kwargs.get("messages")
        user_msg = next(m for m in messages if m["role"] == "user")
        assert user_msg["content"] == "Is this plan sound?"

    def test_strips_whitespace_from_answer(self):
        resp = _fake_response("  padded answer  \n")
        with patch("agent.auxiliary_client.call_llm", return_value=resp):
            result = json.loads(consult_tool("q"))
        assert result["answer"] == "padded answer"


class TestConsultToolGracefulDegradation:
    """The core design requirement: refusal/failure != exception."""

    def test_empty_content_is_unavailable_not_error(self):
        resp = _fake_response("")
        with patch("agent.auxiliary_client.call_llm", return_value=resp):
            result = json.loads(consult_tool("q"))
        assert result["unavailable"] is True
        assert result["answer"] is None
        assert "reason" in result and result["reason"]

    def test_whitespace_only_content_is_unavailable(self):
        resp = _fake_response("   \n\t  ")
        with patch("agent.auxiliary_client.call_llm", return_value=resp):
            result = json.loads(consult_tool("q"))
        assert result["unavailable"] is True

    def test_content_filter_finish_reason_is_unavailable(self):
        # Even if content is non-empty, a content_filter finish reason
        # (Anthropic-native refusal mapping) means "don't trust this".
        resp = _fake_response("partial thing", finish_reason="content_filter")
        with patch("agent.auxiliary_client.call_llm", return_value=resp):
            result = json.loads(consult_tool("q"))
        assert result["unavailable"] is True

    def test_call_llm_exception_is_unavailable_not_raised(self):
        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=RuntimeError("no provider configured"),
        ):
            result = json.loads(consult_tool("q"))
        assert result["unavailable"] is True
        assert result["answer"] is None
        assert "no provider configured" in result["reason"]

    def test_malformed_response_object_is_unavailable(self):
        # A response missing .choices entirely must not crash the tool.
        with patch("agent.auxiliary_client.call_llm", return_value=object()):
            result = json.loads(consult_tool("q"))
        assert result["unavailable"] is True


class TestConsultToolValidation:
    def test_empty_question_is_error(self):
        result = json.loads(consult_tool(""))
        assert "error" in result

    def test_whitespace_only_question_is_error(self):
        result = json.loads(consult_tool("   "))
        assert "error" in result

    def test_none_question_is_error(self):
        result = json.loads(consult_tool(None))
        assert "error" in result

    def test_long_question_is_truncated(self):
        resp = _fake_response("ok")
        long_q = "x" * (MAX_QUESTION_CHARS + 500)
        with patch("agent.auxiliary_client.call_llm", return_value=resp) as mock_call:
            consult_tool(long_q)
        messages = mock_call.call_args.kwargs.get("messages")
        user_msg = next(m for m in messages if m["role"] == "user")
        assert len(user_msg["content"]) < len(long_q)
        assert "truncated" in user_msg["content"]

    def test_long_context_is_truncated(self):
        resp = _fake_response("ok")
        long_ctx = "y" * (MAX_CONTEXT_CHARS + 500)
        with patch("agent.auxiliary_client.call_llm", return_value=resp) as mock_call:
            consult_tool("q", context=long_ctx)
        messages = mock_call.call_args.kwargs.get("messages")
        user_msg = next(m for m in messages if m["role"] == "user")
        assert len(user_msg["content"]) < len(long_ctx) + len("q") + 50
        assert "truncated" in user_msg["content"]


class TestConsultToolRegistration:
    def test_registered_in_registry(self):
        import model_tools  # noqa: F401  (triggers discover_builtin_tools)
        from tools.registry import registry

        entry = registry.get_entry("consult")
        assert entry is not None
        assert entry.toolset == "consult"
        assert entry.schema["name"] == "consult"
        assert "question" in entry.schema["parameters"]["properties"]
        assert entry.schema["parameters"]["required"] == ["question"]

    def test_dispatch_through_registry(self):
        import model_tools  # noqa: F401
        from tools.registry import registry

        resp = _fake_response("dispatched ok")
        with patch("agent.auxiliary_client.call_llm", return_value=resp):
            result = json.loads(
                registry.dispatch("consult", {"question": "q"})
            )
        assert result["unavailable"] is False
        assert result["answer"] == "dispatched ok"


class TestConsultToolset:
    """consult is a registered, opt-in toolset -- not in the default
    always-on tool list (_HERMES_CORE_TOOLS), matching how vision/video are
    opt-in rather than core."""

    def test_consult_toolset_resolves_to_consult_tool(self):
        from toolsets import resolve_toolset

        assert resolve_toolset("consult") == ["consult"]

    def test_consult_not_in_default_core_tools(self):
        from toolsets import _HERMES_CORE_TOOLS

        assert "consult" not in _HERMES_CORE_TOOLS
