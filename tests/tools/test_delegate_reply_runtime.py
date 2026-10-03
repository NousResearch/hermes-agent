"""Delivery contracts through the real child loop and completion pipeline."""

import json
import queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from tools.delegate_tool import _run_single_child
from tools.delegate_tool_reply import DELEGATE_TOOL_REPLY_SCHEMA


def _response(content, tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=tool_calls),
            finish_reason="tool_calls" if tool_calls else "stop",
        )], model="test/model", usage=None,
    )


def _call(name, arguments="{}", call_id="cleanup"):
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


@pytest.fixture
def child():
    definitions = [{"type": "function", "function": {
        "name": "terminal", "description": "test cleanup",
        "parameters": {"type": "object", "properties": {}},
    }}, {"type": "function", "function": DELEGATE_TOOL_REPLY_SCHEMA}]
    with (
        patch("model_tools.get_tool_definitions", return_value=definitions),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1/",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    agent._cached_system_prompt = "Complete the delegated audit."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.client = MagicMock()
    agent._delegate_depth = 1
    agent._delegate_role = "leaf"
    agent._delegate_reply_chunks = []
    with patch("model_tools.handle_function_call", return_value="cleanup succeeded"):
        yield agent
    agent.close()


@pytest.mark.parametrize("explicit", [False, True])
def test_cleanup_does_not_replace_explicit_deliverable(child, explicit):
    calls = []
    if explicit:
        calls.append(_call("delegate_tool_reply", json.dumps({"content": "FULL AUDIT REPORT"}), "delivery"))
    calls.append(_call("terminal"))
    child.client.chat.completions.create.side_effect = [
        _response("FULL AUDIT REPORT", calls),
        _response("Cleanup complete."),
    ]
    entry = _run_single_child(0, "Audit the changes", child, None)
    # Without an explicit call the existing final-response fallback is unchanged.
    assert entry["summary"] == ("FULL AUDIT REPORT" if explicit else "Cleanup complete.")
    assert entry["status"] == "completed"
    assert entry["api_calls"] == 2


@pytest.mark.parametrize("in_place", [True, False])
def test_delivery_survives_real_compaction_and_retry_reset(child, in_place):
    from agent.context_compressor import SUMMARY_PREFIX, is_compaction_summary_message
    from agent.turn_context_compaction import run_turn_start_compaction
    from tools.delegate_tool_child_run import _extract_reply_deliverable

    child.compression_enabled = True
    child.compression_in_place = in_place
    compressor = child.context_compressor
    compressor.protect_first_n = 1
    compressor.protect_last_n = 3
    compressor.tail_token_budget = 500
    messages = [{"role": "user", "content": "Audit the changes"}]
    call = _call("delegate_tool_reply", json.dumps({"content": "FULL AUDIT REPORT"}), "delivery")
    messages.append({"role": "assistant", "content": "", "tool_calls": [{
        "id": call.id, "type": "function",
        "function": {"name": call.function.name, "arguments": call.function.arguments},
    }]})
    child._execute_tool_calls_sequential(SimpleNamespace(content="", tool_calls=[call]), messages, "child-task")
    for i in range(30):
        messages.extend([
            {"role": "user", "content": f"Follow-up {i}: " + "context " * 100},
            {"role": "assistant", "content": f"Investigation {i}: " + "findings " * 100},
        ])
    messages.append({"role": "user", "content": "Finish cleanup"})
    child._last_content_with_tools = "stale fallback"
    child._last_content_tools_all_housekeeping = True
    child._request_pressure_anchored = True
    with (
        patch("agent.turn_context._preflight_request_tokens", side_effect=[200_000, 1_000]),
        patch.object(compressor, "_generate_summary", return_value=SUMMARY_PREFIX + "Earlier investigation summarized.") as summarize,
    ):
        out = run_turn_start_compaction(
            child, messages=messages, system_message="Audit instructions", active_system_prompt="Audit instructions",
            conversation_history=None, current_turn_user_idx=len(messages) - 1,
            user_message="Finish cleanup", effective_task_id="child-task",
        )
    summarize.assert_called_once()
    assert out.compressed and out.messages is not messages
    assert any(is_compaction_summary_message(msg) for msg in out.messages)
    assert not any(tc["function"]["name"] == "delegate_tool_reply"
                   for msg in out.messages for tc in msg.get("tool_calls", []))
    assert "FULL AUDIT REPORT" not in json.dumps(out.messages)
    assert child._last_content_with_tools is None
    assert child._last_content_tools_all_housekeeping is False
    child._invoke_tool("delegate_tool_reply", {"content": "SECOND CHUNK"}, "child-task")
    assert _extract_reply_deliverable(child) == "FULL AUDIT REPORT\n\nSECOND CHUNK"
    child.compression_enabled = False
    child.client.chat.completions.create.return_value = _response("Cleanup complete.")
    entry = _run_single_child(0, "Finish cleanup", child, None)
    assert entry["summary"] == "FULL AUDIT REPORT\n\nSECOND CHUNK"


@pytest.mark.parametrize("background", [False, True])
def test_delivery_reaches_sync_and_async_transport(child, background, monkeypatch):
    from tools import async_delegation as ad
    import tools.delegate_tool as dt
    from tools.process_registry import process_registry
    from tools.process_registry_notifications import format_process_notification

    completion_queue = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", completion_queue)
    ad._reset_for_tests()
    child.client.chat.completions.create.side_effect = [
        _response("", [_call("delegate_tool_reply", json.dumps({"content": "PART A"}), "a")]),
        _response("", [_call("delegate_tool_reply", json.dumps({"content": "PART B"}), "b"), _call("terminal")]),
        _response("Cleanup complete."),
    ]
    parent = SimpleNamespace(
        _delegate_depth=0, session_id="parent-delivery", _active_children=[],
        _active_children_lock=None, _interrupt_requested=False,
    )
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **kw: {
        "model": "test/model", "provider": None, "base_url": None, "api_key": None, "api_mode": None,
    })
    monkeypatch.setattr("gateway.session_context.async_delivery_supported", lambda: True)
    try:
        result = json.loads(dt.delegate_task(goal="Audit the changes", parent_agent=parent, background=background))
        if background:
            assert result["status"] == "dispatched"
            event = completion_queue.get(timeout=10)
            assert event["delegation_id"] == result["delegation_id"]
            assert event["parent_session_id"] == parent.session_id
            result = event
            formatted = format_process_notification(event)
            assert "PART A\n\nPART B" in formatted
            assert "Cleanup complete." not in formatted
        assert result["results"][0]["summary"] == "PART A\n\nPART B"
        assert result["results"][0]["status"] == "completed"
    finally:
        ad._reset_for_tests()


@pytest.mark.parametrize("dispatch", ["concurrent", "sequential"])
@pytest.mark.parametrize("blocked", [False, True])
def test_delivery_obeys_middleware_and_emits_one_post_hook(child, dispatch, blocked, monkeypatch):
    from hermes_cli.plugins import get_plugin_manager

    seen = []

    def rewrite(tool_name, args, **kw):
        assert tool_name == "delegate_tool_reply"
        seen.append("request")
        return {"args": {"content": "rewritten"}}

    def execute(tool_name, args, next_call, **kw):
        seen.append("execution")
        assert args == {"content": "rewritten"}
        if blocked:
            return json.dumps({"error": "delivery blocked"})
        return next_call({"content": args["content"] + " by middleware"})

    monkeypatch.setattr(get_plugin_manager(), "_middleware", {
        "tool_request": [rewrite], "tool_execution": [execute],
    })
    with (
        patch("hermes_cli.lifecycle.invoke_hook", return_value=[]) as hooks,
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
    ):
        messages = []
        executor = child._execute_tool_calls_concurrent if dispatch == "concurrent" else child._execute_tool_calls_sequential
        executor(
            SimpleNamespace(content="", tool_calls=[_call("delegate_tool_reply", '{"content":"original"}', "d1")]),
            messages, "child-task",
        )
        result = messages[0]["content"]
    assert seen == ["request", "execution"]
    assert child._delegate_reply_chunks == ([] if blocked else ["rewritten by middleware"])
    if blocked:
        assert "delivery blocked" in result
    else:
        assert json.loads(result)["acknowledged"] is True
    post_calls = [call for call in hooks.call_args_list if call.args[0] == "post_tool_call"]
    assert len(post_calls) == 1
    assert post_calls[0].kwargs["tool_call_id"] == "d1"


def test_segmented_batch_preserves_delivery_and_cleanup_order(child, tmp_path):
    from agent.tool_dispatch_helpers import _plan_tool_batch_segments

    child.valid_tool_names.add("read_file")
    calls = [
        _call("read_file", json.dumps({"path": str(tmp_path / "a")}), "read-a"),
        _call("read_file", json.dumps({"path": str(tmp_path / "b")}), "read-b"),
        _call("delegate_tool_reply", '{"content":"PART A"}', "delivery-a"),
        _call("terminal", "{}", "cleanup"),
        _call("delegate_tool_reply", '{"content":"PART B"}', "delivery-b"),
    ]
    assert [kind for kind, _ in _plan_tool_batch_segments(calls)] == ["parallel", "sequential"]
    at_cleanup = []

    def tool(name, *args, **kw):
        if name == "terminal":
            at_cleanup.append(list(child._delegate_reply_chunks))
        return "ok"

    messages = []
    with patch("model_tools.handle_function_call", side_effect=tool):
        child._execute_tool_calls(SimpleNamespace(content="", tool_calls=calls), messages, "child-task", 0)
    assert at_cleanup == [["PART A"]]
    assert child._delegate_reply_chunks == ["PART A", "PART B"]
    assert [msg["tool_call_id"] for msg in messages] == [call.id for call in calls]
