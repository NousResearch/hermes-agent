"""review tool — the agent starts the same reviewer as /review.

The contract: the inline executor hands the turn's messages to the /review engine, the reviewer is
routed through ``auxiliary.review`` (a different model from the author is the point), it rides the
background async-delegation rail, and no subagent, leaf or orchestrator, is ever offered the tool.
"""

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry

_MSGS = [
    {"role": "user", "content": "fix the reaper bug"},
    {"role": "assistant", "content": "Pushed the fix to PR #42: protects notify processes from eviction"},
]


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    deadline = time.monotonic() + 2.0
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _parent(depth=0, session_messages=None):
    parent = MagicMock()
    parent._delegate_depth = depth
    parent.session_id = "review-tool-parent"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    parent._session_messages = session_messages or []
    return parent


@pytest.fixture
def captured(monkeypatch):
    """Real start_review → real delegate_task; only child construction/execution is faked."""
    import tools.delegate_tool as dt

    seen = {}

    def fake_resolve(routing_cfg, parent_agent):
        seen["routing_cfg"] = routing_cfg
        return {"model": routing_cfg.get("model") or "parent-model", "provider": routing_cfg.get("provider"),
                "base_url": None, "api_key": None, "api_mode": None, "command": None, "args": None}

    def fake_build(**kw):
        seen["context"] = kw.get("context")
        child = MagicMock()
        child._delegate_role = "leaf"
        return child

    monkeypatch.setattr(dt, "_resolve_delegation_credentials", fake_resolve)
    monkeypatch.setattr(dt, "_build_child_agent", fake_build)
    monkeypatch.setattr(dt, "_run_single_child", lambda *a, **k: {
        "task_index": 0, "status": "completed", "summary": "VERDICT: no blockers", "api_calls": 1,
        "duration_seconds": 0.1, "model": "reviewer-model", "exit_reason": "completed",
    })
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"auxiliary": {"review": {"provider": "nous", "model": "openai/gpt-6.1-sol"}}},
    )
    return seen


def _await_completion():
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            return process_registry.completion_queue.get(timeout=0.2)
        except Exception:
            continue
    return None


def _inline(agent, args, messages):
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext

    return INLINE_TOOL_EXECUTORS["review"](agent, args, InlineToolContext(effective_task_id="t", messages=messages))


def test_inline_review_routes_to_reviewer_model_in_background(captured):
    result = json.loads(_inline(_parent(), {"focus": "check eviction edge cases"}, _MSGS))

    assert result["status"] == "dispatched"
    assert captured["routing_cfg"]["model"] == "openai/gpt-6.1-sol"
    assert "PR #42" in captured["context"]
    assert "The primary agent requested this review" in captured["context"]
    assert "Review focus from the primary agent:\ncheck eviction edge cases" in captured["context"]
    assert "from the user" not in captured["context"]
    evt = _await_completion()
    assert evt is not None and evt["type"] == "async_delegation"
    assert evt["results"][0]["summary"] == "VERDICT: no blockers"


def test_registry_fallback_reviews_persisted_history(captured):
    from tools.registry import registry

    raw = registry.dispatch("review", {"focus": ""}, parent_agent=_parent(session_messages=_MSGS))
    assert json.loads(raw)["status"] == "dispatched"
    assert "PR #42" in captured["context"]


def test_subagent_call_is_refused(captured):
    result = json.loads(_inline(_parent(depth=1), {}, _MSGS))
    assert "Subagents cannot start reviews" in result["error"]
    assert "routing_cfg" not in captured


def test_empty_conversation_is_a_tool_error(captured):
    assert "Nothing to review" in json.loads(_inline(_parent(), {}, []))["error"]


@pytest.mark.parametrize("role", ["leaf", "orchestrator"])
def test_no_subagent_is_offered_review(role):
    import model_tools
    from tools.delegate_tool_toolsets import _resolve_child_toolsets

    parent = SimpleNamespace(enabled_toolsets=["hermes-cli"], disabled_toolsets=[])
    enabled, disabled = _resolve_child_toolsets(parent, None, role)
    names = {d["function"]["name"] for d in model_tools.get_tool_definitions(
        enabled_toolsets=enabled, disabled_toolsets=disabled, quiet_mode=True)}
    assert "review" not in names
    parent_names = {d["function"]["name"] for d in model_tools.get_tool_definitions(
        enabled_toolsets=["hermes-cli"], quiet_mode=True)}
    assert "review" in parent_names


def test_review_stays_inline_and_emits_one_post_hook_per_executor_path(monkeypatch):
    """Both executor paths (concurrent invoke_tool, sequential) dispatch review inline — it needs the live agent and
    turn messages — and each fires exactly one post_tool_call."""
    from unittest.mock import patch

    from agent.agent_runtime_helpers import AGENT_RUNTIME_POST_HOOK_TOOL_NAMES
    from run_agent import AIAgent

    assert "review" in AGENT_RUNTIME_POST_HOOK_TOOL_NAMES
    with patch("model_tools.get_tool_definitions", return_value=[]), \
            patch("model_tools.check_toolset_requirements", return_value={}), patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent.valid_tool_names = {"review"}
    agent._memory_manager = None
    seen, posts = [], []
    monkeypatch.setattr("tools.review_tool.review_tool", lambda **kw: seen.append(kw) or '{"ok":true}')
    monkeypatch.setattr("hermes_cli.plugins._dispatch_pre_tool_call_hooks", lambda *a, **k: (None, None))
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook",
                        lambda name, **kw: posts.append(kw["tool_call_id"]) if name == "post_tool_call" else None)
    call = SimpleNamespace(id="seq", type="function", function=SimpleNamespace(name="review", arguments='{"focus":"x"}'))
    with patch("model_tools.handle_function_call", side_effect=AssertionError("review must stay inline")):
        agent._invoke_tool("review", {"focus": "x"}, "t", tool_call_id="conc", messages=_MSGS)
        agent._execute_tool_calls_sequential(SimpleNamespace(content="", tool_calls=[call]), list(_MSGS), "t")
    assert posts == ["conc", "seq"]
    assert all(kw["parent_agent"] is agent and kw["focus"] == "x" for kw in seen)
    assert seen[0]["messages"] is _MSGS
