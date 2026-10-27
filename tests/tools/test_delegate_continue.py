"""delegate_task(action="continue"): a finished child the conversation owns runs again on its own conversation.

The child is rebuilt on the route and toolsets it ran with, its history comes from its saved transcript, and the
follow-up message is its next turn, dispatched as a new background unit through the normal async delegation path.
Only the model is scripted: registry, child build, SessionDB, dispatch and completion delivery are the real ones.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml
import pytest

from hermes_state import SessionDB
from tests.waiting import wait_until
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _empty_completion_queue():
    def drain():
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()
    drain()
    yield
    drain()


class _ScriptedModel:
    """Stands in for AIAgent: records how each child was built and what each turn saw, and persists each turn
    into the child's dedicated SessionDB handle the way a real child's transcript lands there."""

    def __init__(self):
        self.builds, self.turns = [], []

    def __call__(self, **kw):
        from hermes_constants import get_hermes_home

        child = MagicMock()
        child.session_id = kw.get("session_id") or f"child-sess-{len(self.builds)}"
        child.model, child.provider = kw.get("model"), kw.get("provider")
        child._parent_session_id = kw.get("parent_session_id")  # the ownership spine, as AIAgent stamps it
        child.get_activity_summary.return_value = {"api_call_count": 1, "current_tool": None, "last_activity_ts": 0}
        db = kw["session_db"]

        def run_conversation(user_message, conversation_history=None, **_):
            self.turns.append({"session_id": child.session_id, "message": user_message,
                               "history": conversation_history, "home": str(get_hermes_home())})
            db.create_session(child.session_id, "subagent", parent_session_id=kw.get("parent_session_id"))
            reply = f"done: {user_message}"
            db.append_message(child.session_id, "user", user_message)
            db.append_message(child.session_id, "assistant", reply)
            return {"final_response": reply, "completed": True, "api_calls": 1}

        child.run_conversation.side_effect = run_conversation
        self.builds.append({**kw, "home": str(get_hermes_home())})
        return child


def _parent(session_id="parent-sess", db=None):
    parent = MagicMock()
    parent.base_url, parent.api_key, parent.provider = "https://openrouter.ai/api/v1", "parent-key", "openrouter"
    parent.api_mode, parent.model, parent.platform = "chat_completions", "parent-model", "cli"
    parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
    parent.session_id, parent._delegate_depth, parent._interrupt_requested = session_id, 0, False
    parent.enabled_toolsets, parent.disabled_toolsets = ["terminal", "file", "web"], []
    parent._session_db = db if db is not None else SessionDB()
    parent._session_db.create_session(session_id, "cli")
    parent._active_children, parent._active_children_lock = [], threading.Lock()
    parent._print_fn = parent.tool_progress_callback = parent.thinking_callback = None
    return parent


def _completion(delegation_id):
    """The async_delegation completion event for one dispatched unit."""
    def find():
        while not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("type") == "async_delegation" and evt.get("delegation_id") == delegation_id:
                return evt
        return None
    return wait_until(find, what=f"completion of {delegation_id}")


def _spawn(parent, goal):
    from tools.delegate_tool import delegate_task
    out = json.loads(delegate_task(tasks=[{"goal": goal, "context": "repo at /src"}], background=True,
                                   parent_agent=parent))
    assert out["status"] == "dispatched", out
    _completion(out["delegation_id"])
    return out["subagent_ids"][0], out["delegation_id"]


def test_continue_runs_the_finished_child_on_its_transcript_route_and_toolsets():
    from tools.delegate_tool import delegate_task

    model = _ScriptedModel()
    parent = _parent()
    with patch("run_agent.AIAgent", side_effect=model):
        sid, first_id = _spawn(parent, "write the parser")
        parent.model = "switched-model"  # the parent moved on; the child keeps its own route
        out = json.loads(delegate_task(action="continue", subagent_id=sid, message="now add tests for it",
                                       background=True, parent_agent=parent))
        assert out.get("status") == "dispatched", out
        assert out["subagent_ids"] == [sid] and out["delegation_id"] != first_id
        done = _completion(out["delegation_id"])

    first, again = model.builds
    assert again["session_id"] == model.turns[0]["session_id"]  # same conversation, not a new session
    for key in ("model", "provider", "base_url", "enabled_toolsets", "ephemeral_system_prompt"):
        assert again[key] == first[key], key
    assert first["model"] == "parent-model"
    turn = model.turns[1]
    assert turn["message"] == "now add tests for it"
    assert [(m["role"], m["content"]) for m in turn["history"]] == [
        ("user", "write the parser"), ("assistant", "done: write the parser")]
    assert "done: now add tests for it" in json.dumps(done)
    stored = parent._session_db.get_messages_as_conversation(turn["session_id"])
    assert [m["content"] for m in stored][-2:] == ["now add tests for it", "done: now add tests for it"]


def test_an_inherited_route_refuses_once_the_parent_switched_provider():
    from tools.delegate_tool import delegate_task
    from tools.delegate_tool_registry import _recent_subagents

    model = _ScriptedModel()
    parent = _parent()
    with patch("run_agent.AIAgent", side_effect=model):
        sid, _ = _spawn(parent, "write the parser")
        parent.provider, parent.base_url = "custom", "https://second.invalid/v1"
        out = json.loads(delegate_task(action="continue", subagent_id=sid, message="more", background=True,
                                       parent_agent=parent))
        assert "openrouter" in out["error"] and "Spawn a new task" in out["error"], out
        assert len(model.builds) == 1 and "resume" in _recent_subagents[sid]  # nothing built, nothing consumed

        parent.provider, parent.base_url = "openrouter", "https://openrouter.ai/api/v1"
        out = json.loads(delegate_task(action="continue", subagent_id=sid, message="more", background=True,
                                       parent_agent=parent))
        assert out.get("status") == "dispatched", out
        _completion(out["delegation_id"])
    assert (model.builds[1]["provider"], model.builds[1]["base_url"]) == ("openrouter", "https://openrouter.ai/api/v1")


def test_continue_on_a_running_child_points_to_steer():
    from tools.delegate_tool import _register_subagent, _unregister_subagent, delegate_task

    parent = _parent()
    _register_subagent({"subagent_id": "sa-cont-live", "goal": "g", "agent": MagicMock(),
                        "owner_agent_session_id": parent.session_id})
    try:
        out = json.loads(delegate_task(action="continue", subagent_id="sa-cont-live", message="more",
                                       parent_agent=parent))
    finally:
        _unregister_subagent("sa-cont-live")
    assert "still running" in out["error"] and "steer" in out["error"]


def _finished(sid, owner):
    """A finished delegate_task child as the registry retains it."""
    from tools.delegate_tool import _register_subagent, _unregister_subagent
    from tools.delegate_tool_routes import TaskRoute

    child = MagicMock()
    child.session_id = f"{sid}-session"
    child._delegate_resume = {"goal": "g", "context": None, "toolsets": ["file"], "route": TaskRoute({}, {})}
    _register_subagent({"subagent_id": sid, "goal": "g", "agent": child, "owner_agent_session_id": owner})
    _unregister_subagent(sid, agent=child)


def test_non_owner_unknown_and_evicted_ids_are_not_found():
    from tools.delegate_tool import delegate_task
    from tools.delegate_tool_registry import _RECENT_SUBAGENTS_CAP, _recent_subagents

    parent = _parent()
    _finished("sa-cont-foreign", owner="someone-else")
    _finished("sa-cont-evicted", owner=parent.session_id)
    assert "resume" in _recent_subagents["sa-cont-evicted"]
    for i in range(_RECENT_SUBAGENTS_CAP):
        _finished(f"sa-cont-filler-{i}", owner="filler")

    for sid in ("sa-cont-foreign", "sa-cont-unknown", "sa-cont-evicted"):
        out = json.loads(delegate_task(action="continue", subagent_id=sid, message="more", parent_agent=parent))
        assert "No finished subagent" in out["error"], sid


def test_one_shot_sessions_are_not_offered_continue():
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA, _build_dynamic_schema_overrides, oneshot_delegate_function

    full = {**DELEGATE_TASK_SCHEMA, **_build_dynamic_schema_overrides()}
    assert "continue" in full["parameters"]["properties"]["action"]["enum"]
    oneshot = oneshot_delegate_function(full)
    assert "action" not in oneshot["parameters"]["properties"]
    assert "continue" not in json.dumps(oneshot["parameters"])


def test_one_shot_sessions_refuse_continue_without_consuming_it(monkeypatch):
    from tools.delegate_tool import delegate_task
    from tools.delegate_tool_registry import _recent_subagents

    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")  # the finite ``chat -q`` marker
    parent = _parent()
    _finished("sa-cont-oneshot", owner=parent.session_id)
    out = json.loads(delegate_task(action="continue", subagent_id="sa-cont-oneshot", message="more",
                                   parent_agent=parent))
    assert "one-shot" in out["error"], out
    assert "resume" in _recent_subagents["sa-cont-oneshot"]


def _write_profile(home, *, host, model, key):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "base_url": f"https://{host}/v1", "default": model, "key_env": "PROFILE_KEY"},
    }), encoding="utf-8")
    (home / ".env").write_text(f"PROFILE_KEY={key}\n", encoding="utf-8")


def test_profile_routed_child_continues_inside_its_profile_scope(tmp_path, monkeypatch):
    from tools.delegate_tool import delegate_task
    from tools.delegate_tool_routes import _route_to_task_route

    launch = tmp_path / "os-home" / ".hermes"
    coder = launch / "profiles" / "coder"
    _write_profile(launch, host="launch.invalid", model="launch-model", key="launch-key")
    _write_profile(coder, host="coder.invalid", model="coder-model", key="coder-key")
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("PROFILE_KEY", "launch-key")

    model = _ScriptedModel()
    parent = _parent(db=SessionDB(db_path=tmp_path / "parent-db" / "state.db"))
    route = _route_to_task_route({"profile": "coder", "label": "coder"}, parent, {})
    with patch("run_agent.AIAgent", side_effect=model), \
            patch("tools.delegate_tool_routes.resolve_task_routes", return_value=([route], None)):
        sid, _ = _spawn(parent, "work as coder")
    with patch("run_agent.AIAgent", side_effect=model):  # continue never asks the backend again
        out = json.loads(delegate_task(action="continue", subagent_id=sid, message="and the follow-up",
                                       background=True, parent_agent=parent))
        assert out.get("status") == "dispatched", out
        _completion(out["delegation_id"])

    again = model.builds[1]
    assert again["home"] == str(coder) and model.turns[1]["home"] == str(coder)
    assert (again["model"], again["api_key"]) == ("coder-model", "coder-key")
    assert [m["content"] for m in model.turns[1]["history"]] == ["work as coder", "done: work as coder"]
