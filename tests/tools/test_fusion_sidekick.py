"""Behavior-contract tests for Fusion persistent sidekick delegation.

These are intentionally black-box around delegate_task and existing child-agent
construction helpers. They encode the v1 contracts:
`fusion.enabled: false` must be behavior-invisible, and `sidekick=true` must
reuse one persistent child per parent session without per-call teardown.
"""

from __future__ import annotations

import inspect
import json
import os
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import tools.delegate_tool as delegate_tool


def _delegation_config() -> dict[str, Any]:
    return {
        "max_iterations": 3,
        "max_concurrent_children": 3,
        "max_spawn_depth": 1,
    }


def _full_config(*, fusion_enabled: bool = True) -> dict[str, Any]:
    return {
        "delegation": _delegation_config(),
        "fusion": {
            "enabled": fusion_enabled,
            "sidekick_model": "openai/gpt-4.1-mini",
            "sidekick_provider": "openrouter",
            "sidekick_base_url": "https://openrouter.ai/api/v1",
            "sidekick_api_key": "«redacted:sk-…»",
            "sidekick_api_mode": "chat_completions",
            "sidekick_toolsets": ["terminal", "file"],
            "compaction_routing": False,
            "routing_model": "",
            "routing_provider": "",
            "routing_base_url": "",
            "routing_api_key": "",
            "routing_api_mode": "",
            "frontier_model": "",
        },
    }


def _make_parent(session_id: str = "parent-session") -> MagicMock:
    parent = MagicMock()
    parent.session_id = session_id
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "«redacted:sk-…»"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent.openrouter_min_coding_score = None
    parent.enabled_toolsets = ["terminal", "file", "delegation"]
    parent.valid_tool_names = []
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent._current_task_id = "parent-task"
    parent._current_turn_id = "turn-1"
    return parent


class FakeSidekick:
    """Small child-agent double that behaves like one persistent sidekick."""

    def __init__(self) -> None:
        self.session_id = "sidekick-session"
        self.model = "openai/gpt-4.1-mini"
        self.provider = "openrouter"
        self.tool_progress_callback = None
        self._credential_pool = None
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._subagent_id = "sidekick-subagent"
        self._parent_subagent_id = None
        self._delegate_saved_tool_names = []
        self._is_persistent_sidekick = True
        self._use_prompt_caching = True
        self._cached_system_prompt = "stable-sidekick-prompt"
        self.enabled_toolsets = ["terminal", "file"]
        self.conversation_history: list[dict[str, str]] = []
        self.run_inputs: list[str] = []
        self.closed = False
        self.close_count = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0

    def run_conversation(self, user_message=None, task_id=None, stream_callback=None):
        goal_text = str(user_message)
        self.run_inputs.append(goal_text)
        self.conversation_history.extend(
            [
                {"role": "user", "content": goal_text},
                {"role": "assistant", "content": f"done: {goal_text}"},
            ]
        )
        return {
            "final_response": f"done: {goal_text}",
            "completed": True,
            "api_calls": 1,
            "messages": list(self.conversation_history),
        }

    def get_activity_summary(self) -> dict[str, Any]:
        return {"current_tool": None, "api_call_count": len(self.run_inputs), "max_iterations": 3}

    def close(self) -> None:
        self.close_count += 1
        self.closed = True


def _decode_tool_result(raw: str) -> dict[str, Any]:
    return json.loads(raw)


@pytest.fixture(autouse=True)
def clear_sidekick_registries():
    for name in ("_sidekick_agents", "_sidekick_locks"):
        registry = getattr(delegate_tool, name, None)
        if isinstance(registry, dict):
            registry.clear()
    yield
    for name in ("_sidekick_agents", "_sidekick_locks"):
        registry = getattr(delegate_tool, name, None)
        if isinstance(registry, dict):
            registry.clear()


def test_sidekick_schema_is_advertised_only_when_fusion_enabled():
    """E4: disabled means invisible; enabled exposes the single new parameter."""

    with (
        patch("tools.delegate_tool._load_config", return_value=_delegation_config()),
        patch("hermes_cli.config.load_config", return_value=_full_config(fusion_enabled=True)),
    ):
        enabled_schema = delegate_tool._build_dynamic_schema_overrides()

    props = enabled_schema["parameters"]["properties"]
    assert "sidekick" in props
    assert props["sidekick"]["type"] == "boolean"
    assert "persistent sidekick" in props["sidekick"]["description"].lower()

    with (
        patch("tools.delegate_tool._load_config", return_value=_delegation_config()),
        patch("hermes_cli.config.load_config", return_value=_full_config(fusion_enabled=False)),
    ):
        disabled_schema = delegate_tool._build_dynamic_schema_overrides()

    assert "sidekick" not in disabled_schema["parameters"]["properties"]
    assert "sidekick" not in delegate_tool.DELEGATE_TASK_SCHEMA["parameters"]["properties"]


def test_delegate_task_accepts_sidekick_keyword():
    """A1: sidekick is a runtime argument, not a separate core tool."""

    assert "sidekick" in inspect.signature(delegate_tool.delegate_task).parameters


@pytest.mark.parametrize(
    ("kwargs", "expected_error"),
    [
        ({"tasks": [{"goal": "one"}], "sidekick": True}, "sidekick accepts a single goal"),
        ({"goal": "one", "sidekick": True, "background": True}, "sidekick runs synchronously in v1"),
        ({"goal": "one", "sidekick": True, "role": "orchestrator"}, "sidekick is a leaf"),
    ],
)
def test_sidekick_validation_errors_are_tool_results(kwargs: dict[str, Any], expected_error: str):
    """E7: invalid v1 combinations return clear tool-result errors, not exceptions."""

    parent = _make_parent()
    with (
        patch("tools.delegate_tool._load_config", return_value=_delegation_config()),
        patch("hermes_cli.config.load_config", return_value=_full_config(fusion_enabled=True)),
    ):
        result = _decode_tool_result(delegate_tool.delegate_task(parent_agent=parent, **kwargs))

    assert expected_error in result.get("error", "")


def test_delegate_dispatch_treats_string_false_sidekick_as_false(monkeypatch):
    """Tool-arg defensive parsing: provider string 'false' must not force sync sidekick mode."""

    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    setattr(agent, "_delegate_depth", 0)
    captured: dict[str, Any] = {}

    def fake_delegate_task(**kwargs):
        captured.update(kwargs)
        return json.dumps({"ok": True})

    monkeypatch.setattr("tools.delegate_tool.delegate_task", fake_delegate_task)

    result = AIAgent._dispatch_delegate_task(agent, {"goal": "normal child", "sidekick": "false"})

    assert _decode_tool_result(result) == {"ok": True}
    assert captured["sidekick"] is False
    assert captured["background"] is True


def test_sidekick_reuses_one_child_keeps_history_and_skips_per_call_close():
    """E1/E2/E3: sequential sidekick calls share identity, prompt/tools, and resources."""

    parent = _make_parent()
    child = FakeSidekick()

    def build_child(*args, **kwargs):
        parent._active_children.append(child)
        return child

    with (
        patch("tools.delegate_tool._load_config", return_value=_delegation_config()),
        patch("hermes_cli.config.load_config", return_value=_full_config(fusion_enabled=True)),
        patch(
            "tools.delegate_tool._resolve_delegation_credentials",
            return_value={"provider": None, "base_url": None, "api_key": None, "api_mode": None, "model": None},
        ),
        patch("tools.delegate_tool._build_child_agent", side_effect=build_child) as build_mock,
    ):
        first = _decode_tool_result(delegate_tool.delegate_task(goal="inspect failing tests", sidekick=True, parent_agent=parent))
        prompt_hash = hash(child._cached_system_prompt)
        toolsets = tuple(child.enabled_toolsets)
        second = _decode_tool_result(delegate_tool.delegate_task(goal="apply lint fix", sidekick=True, parent_agent=parent))

    assert first["results"][0]["status"] == "completed"
    assert second["results"][0]["status"] == "completed"
    assert build_mock.call_count == 1
    assert child.run_inputs == ["inspect failing tests", "apply lint fix"]
    assert len(child.conversation_history) == 4
    assert hash(child._cached_system_prompt) == prompt_hash
    assert tuple(child.enabled_toolsets) == toolsets
    assert child.closed is False
    assert child.close_count == 0
    assert child in parent._active_children


def test_sidekick_context_is_folded_into_each_call_goal_not_persistent_prompt():
    """Per-call context must reach the sidekick via the user message on every
    delegation, and must NOT be baked into the persistent system prompt (which
    would pin the first call's context for the whole session)."""

    parent = _make_parent()
    child = FakeSidekick()
    captured_kwargs: dict[str, Any] = {}

    def build_child(*args, **kwargs):
        captured_kwargs.update(kwargs)
        parent._active_children.append(child)
        return child

    with (
        patch("tools.delegate_tool._load_config", return_value=_delegation_config()),
        patch("hermes_cli.config.load_config", return_value=_full_config(fusion_enabled=True)),
        patch(
            "tools.delegate_tool._resolve_delegation_credentials",
            return_value={"provider": None, "base_url": None, "api_key": None, "api_mode": None, "model": None},
        ),
        patch("tools.delegate_tool._build_child_agent", side_effect=build_child),
    ):
        delegate_tool.delegate_task(
            goal="fix the failing test",
            context="tests/test_foo.py::test_bar fails with KeyError",
            sidekick=True,
            parent_agent=parent,
        )
        delegate_tool.delegate_task(
            goal="run lint",
            context="use ruff, config in pyproject.toml",
            sidekick=True,
            parent_agent=parent,
        )

    assert child.run_inputs[0].startswith("fix the failing test")
    assert "KeyError" in child.run_inputs[0]
    assert child.run_inputs[1].startswith("run lint")
    assert "pyproject.toml" in child.run_inputs[1]
    # Persistent system prompt was built without per-call context.
    assert captured_kwargs.get("context") is None


def test_sidekick_busy_lock_returns_status_without_second_run():
    """E6: non-blocking lock prevents concurrent mutation of shared sidekick history."""

    parent = _make_parent()
    child = FakeSidekick()

    def build_child(*args, **kwargs):
        parent._active_children.append(child)
        return child

    with (
        patch("tools.delegate_tool._load_config", return_value=_delegation_config()),
        patch("hermes_cli.config.load_config", return_value=_full_config(fusion_enabled=True)),
        patch(
            "tools.delegate_tool._resolve_delegation_credentials",
            return_value={"provider": None, "base_url": None, "api_key": None, "api_mode": None, "model": None},
        ),
        patch("tools.delegate_tool._build_child_agent", side_effect=build_child),
    ):
        _decode_tool_result(delegate_tool.delegate_task(goal="first", sidekick=True, parent_agent=parent))
        lock = delegate_tool._sidekick_locks[parent.session_id]
        assert lock.acquire(blocking=False) is True
        try:
            busy = _decode_tool_result(delegate_tool.delegate_task(goal="second", sidekick=True, parent_agent=parent))
        finally:
            lock.release()

    assert busy["status"] == "sidekick_busy"
    assert "working on another task" in busy["summary"]
    assert child.run_inputs == ["first"]


def test_sidekick_lock_released_when_child_build_raises():
    """A failed sidekick build must not leave the per-session lock held —
    otherwise every later sidekick call is a permanent 'sidekick_busy'."""

    parent = _make_parent()
    child = FakeSidekick()
    build_attempts = {"n": 0}

    def build_child(*args, **kwargs):
        build_attempts["n"] += 1
        if build_attempts["n"] == 1:
            raise RuntimeError("transient build failure")
        parent._active_children.append(child)
        return child

    with (
        patch("tools.delegate_tool._load_config", return_value=_delegation_config()),
        patch("hermes_cli.config.load_config", return_value=_full_config(fusion_enabled=True)),
        patch(
            "tools.delegate_tool._resolve_delegation_credentials",
            return_value={"provider": None, "base_url": None, "api_key": None, "api_mode": None, "model": None},
        ),
        patch("tools.delegate_tool._build_child_agent", side_effect=build_child),
    ):
        with pytest.raises(RuntimeError, match="transient build failure"):
            delegate_tool.delegate_task(goal="first", sidekick=True, parent_agent=parent)
        # The lock must be free again: the retry runs instead of 'sidekick_busy'.
        retry = _decode_tool_result(
            delegate_tool.delegate_task(goal="retry", sidekick=True, parent_agent=parent)
        )

    assert retry["results"][0]["status"] == "completed"
    assert child.run_inputs == ["retry"]


def test_close_sidekick_for_session_closes_child_and_clears_registries():
    """A4/E3: parent close tears down its persistent sidekick exactly once."""

    parent = _make_parent()
    child = FakeSidekick()

    def build_child(*args, **kwargs):
        parent._active_children.append(child)
        return child

    with (
        patch("tools.delegate_tool._load_config", return_value=_delegation_config()),
        patch("hermes_cli.config.load_config", return_value=_full_config(fusion_enabled=True)),
        patch(
            "tools.delegate_tool._resolve_delegation_credentials",
            return_value={"provider": None, "base_url": None, "api_key": None, "api_mode": None, "model": None},
        ),
        patch("tools.delegate_tool._build_child_agent", side_effect=build_child),
    ):
        _decode_tool_result(delegate_tool.delegate_task(goal="first", sidekick=True, parent_agent=parent))
        closed_child = delegate_tool.close_sidekick_for_session(parent.session_id)

    assert closed_child is child
    assert child.closed is True
    assert child.close_count == 1
    assert parent.session_id not in delegate_tool._sidekick_agents
    assert parent.session_id not in delegate_tool._sidekick_locks


def test_reset_session_state_closes_old_sidekick_and_resets_fusion_state():
    """Session rotation must not orphan old sidekicks or leak routing state."""

    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.session_id = "new-session"
    agent.context_compressor = None
    agent._fusion_routing_suspended = True
    agent._fusion_routing_verdict = "MECHANICAL"
    transitions: list[dict[str, Any]] = []
    agent._transition_context_engine_session = lambda **kwargs: transitions.append(kwargs)

    child = FakeSidekick()
    delegate_tool._sidekick_agents["old-session"] = child
    delegate_tool._sidekick_locks["old-session"] = threading.Lock()

    AIAgent.reset_session_state(agent, old_session_id="old-session")

    assert child.closed is True
    assert child.close_count == 1
    assert "old-session" not in delegate_tool._sidekick_agents
    assert "old-session" not in delegate_tool._sidekick_locks
    assert agent._fusion_routing_verdict is None
    assert agent._fusion_routing_suspended is False
    assert transitions and transitions[-1]["old_session_id"] == "old-session"
    assert transitions[-1]["new_session_id"] == "new-session"


def test_agent_close_does_not_double_close_registered_sidekick(monkeypatch):
    """E3: integrated AIAgent.close path closes the registered sidekick exactly once."""

    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    setattr(agent, "session_id", "parent-session")
    setattr(agent, "_active_children_lock", threading.Lock())
    child = FakeSidekick()
    setattr(agent, "_active_children", [child])
    agent.client = None
    setattr(agent, "_session_db", None)
    setattr(agent, "_end_session_on_close", False)
    setattr(agent, "_session_messages", [{"role": "user", "content": "hi"}])

    delegate_tool._sidekick_agents[getattr(agent, "session_id")] = child
    delegate_tool._sidekick_locks[getattr(agent, "session_id")] = threading.Lock()

    monkeypatch.setattr("run_agent.cleanup_vm", lambda *args, **kwargs: None)
    monkeypatch.setattr("run_agent.cleanup_browser", lambda *args, **kwargs: None)
    monkeypatch.setattr("tools.process_registry.process_registry.kill_all", lambda *args, **kwargs: None)

    AIAgent.close(agent)

    assert child.closed is True
    assert child.close_count == 1
    assert getattr(agent, "_active_children") == []
    assert delegate_tool._sidekick_agents == {}
    assert delegate_tool._sidekick_locks == {}


def test_sidekick_restart_lazy_rebuilds_after_module_registries_are_cleared():
    """E9: process/module-dict reset loses old sidekick and next call builds a fresh one."""

    parent = _make_parent()
    first_child = FakeSidekick()
    second_child = FakeSidekick()
    second_child.session_id = "sidekick-session-2"
    built_children = [first_child, second_child]

    def build_child(*args, **kwargs):
        child = built_children.pop(0)
        parent._active_children.append(child)
        return child

    with (
        patch("tools.delegate_tool._load_config", return_value=_delegation_config()),
        patch("hermes_cli.config.load_config", return_value=_full_config(fusion_enabled=True)),
        patch(
            "tools.delegate_tool._resolve_delegation_credentials",
            return_value={"provider": None, "base_url": None, "api_key": None, "api_mode": None, "model": None},
        ),
        patch("tools.delegate_tool._build_child_agent", side_effect=build_child) as build_mock,
    ):
        first = _decode_tool_result(delegate_tool.delegate_task(goal="first", sidekick=True, parent_agent=parent))
        delegate_tool._sidekick_agents.clear()
        delegate_tool._sidekick_locks.clear()
        second = _decode_tool_result(delegate_tool.delegate_task(goal="second", sidekick=True, parent_agent=parent))

    assert first["results"][0]["status"] == "completed"
    assert second["results"][0]["status"] == "completed"
    assert build_mock.call_count == 2
    assert first_child.run_inputs == ["first"]
    assert second_child.run_inputs == ["second"]
    assert delegate_tool._sidekick_agents[parent.session_id] is second_child


def test_sidekick_session_rows_cascade_delete_with_parent(tmp_path):
    """E3b: delegate session rows marked with _delegate_from delete with the parent."""

    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("parent", source="cli", model="frontier-model")
        db.create_session(
            "sidekick",
            source="subagent",
            model="sidekick-model",
            parent_session_id="parent",
            model_config={"_delegate_from": "parent"},
        )
        db.append_message("sidekick", role="assistant", content="sidekick summary")

        assert db.delete_session("parent") is True
        assert db.get_session("parent") is None
        assert db.get_session("sidekick") is None
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Upstream-readiness caveats
# ---------------------------------------------------------------------------

class _LongResponseSidekick:
    """FakeSidekick variant that produces an over-budget final_response so the
    summary budget path is exercised end-to-end through delegate_task."""

    def __init__(self, long_text: str) -> None:
        self.session_id = "sidekick-long"
        self.model = "openai/gpt-4.1-mini"
        self.provider = "openrouter"
        self.tool_progress_callback = None
        self._credential_pool = None
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._subagent_id = "sidekick-long-id"
        self._parent_subagent_id = None
        self._delegate_saved_tool_names = []
        self._is_persistent_sidekick = True
        self.enabled_toolsets = ["terminal", "file"]
        self.run_inputs = []
        self.closed = False
        self.close_count = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self._long_text = long_text

    def run_conversation(self, user_message=None, task_id=None, stream_callback=None):
        goal_text = str(user_message)
        self.run_inputs.append(goal_text)
        return {
            "final_response": self._long_text,
            "completed": True,
            "api_calls": 1,
            "messages": [
                {"role": "user", "content": goal_text},
                {"role": "assistant", "content": self._long_text},
            ],
        }

    def get_activity_summary(self):
        return {"current_tool": None, "api_call_count": len(self.run_inputs), "max_iterations": 3}

    def close(self):
        self.close_count += 1
        self.closed = True


def test_sidekick_summary_budget_applied_in_real_delegate_path(tmp_path, monkeypatch):
    """Caveat #1: persistent sidekick results must flow through
    _apply_summary_budget on the actual delegate_task(sidekick=True) path.

    The FakeSidekick produces a long final_response; delegation.max_summary_chars
    is patched small enough to trigger trimming. HERMES_HOME is isolated to the
    tmp_path so the spill file lands in the temp delegation cache.
    """
    parent = _make_parent()
    head_marker = "UNIQUE_HEAD_MARKER_LINE_1\n"
    tail_marker = "\nUNIQUE_TAIL_MARKER_LINE_END"
    long_text = head_marker + ("Y" * 50_000) + tail_marker

    child = _LongResponseSidekick(long_text)

    def build_child(*args, **kwargs):
        parent._active_children.append(child)
        return child

    fake_home = tmp_path / "hermes_home_budget"
    fake_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    deleg_cfg = _delegation_config()
    deleg_cfg["max_summary_chars"] = 2000

    with (
        patch("tools.delegate_tool._load_config", return_value=deleg_cfg),
        patch("hermes_cli.config.load_config", return_value=_full_config(fusion_enabled=True)),
        patch(
            "tools.delegate_tool._resolve_delegation_credentials",
            return_value={"provider": None, "base_url": None, "api_key": None, "api_mode": None, "model": None},
        ),
        patch("tools.delegate_tool._build_child_agent", side_effect=build_child),
    ):
        result = _decode_tool_result(
            delegate_tool.delegate_task(goal="summarize the huge file", sidekick=True, parent_agent=parent)
        )

    entry = result["results"][0]
    assert entry["status"] == "completed"
    # The summary was trimmed (budget applied), not passed through verbatim.
    assert entry.get("summary_truncated") is True
    assert "summary_full_path" in entry
    spill_path = entry["summary_full_path"]
    assert os.path.exists(spill_path)
    assert os.path.join("cache", "delegation") in spill_path
    # Spill file holds the full original final_response — nothing is lost.
    with open(spill_path, encoding="utf-8") as fh:
        assert fh.read() == long_text
    # Head + tail + footer markers all survive in the in-context summary.
    assert "UNIQUE_HEAD_MARKER_LINE_1" in entry["summary"]
    assert "UNIQUE_TAIL_MARKER_LINE_END" in entry["summary"]
    assert "[SUMMARY TRUNCATED]" in entry["summary"]
    assert "read_file" in entry["summary"]
    assert "offset=" in entry["summary"]


def test_sidekick_construction_uses_standard_child_path_with_compression():
    """Caveat #2: persistent sidekick construction goes through the standard
    _build_child_agent path (persistent_sidekick=True) and the cached child
    retains compression_enabled=True + a context_compressor object, proving
    Fusion does not bypass self-compression wiring.
    """

    class _ChildWithCompression(FakeSidekick):
        """Persistent child double carrying the standard AIAgent compression contract."""

        compression_enabled = True

        def __init__(self):
            super().__init__()
            self.context_compressor = SimpleNamespace(context_length=200_000, max_tokens=8_000)
            self.session_id = "sidekick-compression"
            self._is_persistent_sidekick = True

    parent = _make_parent()
    child = _ChildWithCompression()

    captured_kwargs = {}

    def build_child(*args, **kwargs):
        captured_kwargs.update(kwargs)
        parent._active_children.append(child)
        return child

    with (
        patch("tools.delegate_tool._load_config", return_value=_delegation_config()),
        patch("hermes_cli.config.load_config", return_value=_full_config(fusion_enabled=True)),
        patch(
            "tools.delegate_tool._resolve_delegation_credentials",
            return_value={"provider": None, "base_url": None, "api_key": None, "api_mode": None, "model": None},
        ),
        patch("tools.delegate_tool._build_child_agent", side_effect=build_child),
    ):
        first = _decode_tool_result(
            delegate_tool.delegate_task(goal="first sidekick task", sidekick=True, parent_agent=parent)
        )
        # Second call reuses the cached child (proves retention across calls).
        second = _decode_tool_result(
            delegate_tool.delegate_task(goal="second sidekick task", sidekick=True, parent_agent=parent)
        )

    assert first["results"][0]["status"] == "completed"
    assert second["results"][0]["status"] == "completed"
    # The standard child-construction path was used with the sidekick flag.
    assert captured_kwargs.get("persistent_sidekick") is True
    # The cached child retains the standard compression wiring.
    cached = delegate_tool._sidekick_agents[parent.session_id]
    assert cached is child
    assert getattr(cached, "compression_enabled", None) is True
    assert getattr(cached, "context_compressor", None) is not None


def test_fusion_enabled_schema_description_notes_sidekick_sync_exception():
    """Caveat #3: when Fusion is enabled, the delegate schema top-level
    description must not unconditionally claim BOTH modes run in the background
    without noting the sidekick=true synchronous exception.

    This test would have failed before the wording fix because the old
    description stated 'BOTH MODES RUN IN THE BACKGROUND' with no sidekick
    caveat.
    """

    with (
        patch("tools.delegate_tool._load_config", return_value=_delegation_config()),
        patch("hermes_cli.config.load_config", return_value=_full_config(fusion_enabled=True)),
    ):
        enabled_schema = delegate_tool._build_dynamic_schema_overrides()

    desc = enabled_schema["description"]
    # The synchronous sidekick exception is surfaced to the model.
    assert "sidekick" in desc.lower()
    assert "synchronous" in desc.lower() or "synchronously" in desc.lower()

    # Disabled Fusion must NOT carry the sidekick exception note (and must not
    # contain 'sidekick' at all in the top-level description).
    with (
        patch("tools.delegate_tool._load_config", return_value=_delegation_config()),
        patch("hermes_cli.config.load_config", return_value=_full_config(fusion_enabled=False)),
    ):
        disabled_schema = delegate_tool._build_dynamic_schema_overrides()

    disabled_desc = disabled_schema["description"]
    assert "sidekick" not in disabled_desc.lower()
