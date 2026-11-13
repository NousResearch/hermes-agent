"""delegate_task(action="continue"): a finished child the conversation owns runs again on its own conversation.

The child is rebuilt on the route and toolsets it ran with, its history comes from its saved transcript, and the
follow-up message is its next turn, dispatched as a new background unit through the normal async delegation path.
Only the model is scripted: registry, child build, SessionDB, dispatch and completion delivery are the real ones.
"""

import json
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml
import pytest

from hermes_agent.hermes_state import SessionDB
from tests.waiting import wait_until
from hermes_agent.tools.process_registry import process_registry


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
    from hermes_agent.tools.delegate_tool import delegate_task
    out = json.loads(delegate_task(tasks=[{"goal": goal, "context": "repo at /src"}], background=True,
                                   parent_agent=parent))
    assert out["status"] == "dispatched", out
    _completion(out["delegation_id"])
    return out["subagent_ids"][0], out["delegation_id"]


def test_continue_runs_the_finished_child_on_its_transcript_route_and_toolsets():
    from hermes_agent.tools.delegate_tool import delegate_task

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
        # The first turn is main's brief (context first, a cacheable prefix); the follow-up goes out bare.
        ("user", "CONTEXT:\nrepo at /src\n\nTASK:\nwrite the parser"),
        ("assistant", "done: CONTEXT:\nrepo at /src\n\nTASK:\nwrite the parser")]
    assert "done: now add tests for it" in json.dumps(done)
    stored = parent._session_db.get_messages_as_conversation(turn["session_id"])
    assert [m["content"] for m in stored][-2:] == ["now add tests for it", "done: now add tests for it"]


def test_an_inherited_route_refuses_once_the_parent_switched_provider():
    from hermes_agent.tools.delegate_tool import delegate_task
    from hermes_agent.tools.delegate_tool_registry import _recent_subagents

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
    from hermes_agent.tools.delegate_tool import _register_subagent, _unregister_subagent, delegate_task

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
    from hermes_agent.tools.delegate_tool import _register_subagent, _unregister_subagent
    from hermes_agent.tools.delegate_tool_routes import TaskRoute

    child = MagicMock()
    child.session_id = f"{sid}-session"
    child._delegate_resume = {"goal": "g", "context": None, "toolsets": ["file"], "route": TaskRoute({}, {})}
    _register_subagent({"subagent_id": sid, "goal": "g", "agent": child, "owner_agent_session_id": owner})
    _unregister_subagent(sid, agent=child)


def test_non_owner_unknown_and_evicted_ids_are_not_found():
    from hermes_agent.tools.delegate_tool import delegate_task
    from hermes_agent.tools.delegate_tool_registry import _RECENT_SUBAGENTS_CAP, _recent_subagents

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
    from hermes_agent.tools.delegate_tool import DELEGATE_TASK_SCHEMA, _build_dynamic_schema_overrides, oneshot_delegate_function

    full = {**DELEGATE_TASK_SCHEMA, **_build_dynamic_schema_overrides()}
    assert "continue" in full["parameters"]["properties"]["action"]["enum"]
    oneshot = oneshot_delegate_function(full)
    assert "action" not in oneshot["parameters"]["properties"]
    assert "continue" not in json.dumps(oneshot["parameters"])


def test_one_shot_sessions_refuse_continue_without_consuming_it(monkeypatch):
    from hermes_agent.tools.delegate_tool import delegate_task
    from hermes_agent.tools.delegate_tool_registry import _recent_subagents

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
    from hermes_agent.tools.delegate_tool import delegate_task
    from hermes_agent.tools.delegate_tool_routes import _route_to_task_route

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
            patch("hermes_agent.tools.delegate_tool_routes.resolve_task_routes", return_value=([route], None)):
        sid, _ = _spawn(parent, "work as coder")
    with patch("run_agent.AIAgent", side_effect=model):  # continue never asks the backend again
        out = json.loads(delegate_task(action="continue", subagent_id=sid, message="and the follow-up",
                                       background=True, parent_agent=parent))
        assert out.get("status") == "dispatched", out
        _completion(out["delegation_id"])

    again = model.builds[1]
    assert again["home"] == str(coder) and model.turns[1]["home"] == str(coder)
    assert (again["model"], again["api_key"]) == ("coder-model", "coder-key")
    brief = "CONTEXT:\nrepo at /src\n\nTASK:\nwork as coder"
    assert [m["content"] for m in model.turns[1]["history"]] == [brief, f"done: {brief}"]


@pytest.mark.parametrize("background", [False, True], ids=["synchronous", "sync-fallback"])
def test_joined_result_handle_can_continue_the_same_child(background):
    from hermes_agent.tools import delegate_tool  # register the real handler

    registry = delegate_tool.registry
    model, parent = _ScriptedModel(), _parent()
    with patch("run_agent.AIAgent", side_effect=model), \
            patch("hermes_agent.tools.delegate_tool_dispatch._resolve_async_wake_sid", return_value=None):
        out = json.loads(registry.dispatch("delegate_task", {"goal": "write parser", "background": background},
                                           parent_agent=parent))
        if background:
            assert "SYNCHRONOUSLY" in out["note"]
        sid = out["results"][0]["subagent_id"]
        again = json.loads(registry.dispatch("delegate_task", {
            "action": "continue", "subagent_id": sid, "message": "add tests", "background": False,
        }, parent_agent=parent))
    assert again["results"][0]["subagent_id"] == sid
    assert model.turns[1]["session_id"] == model.turns[0]["session_id"]
    assert model.turns[1]["history"][-1]["content"] == "done: write parser"


@pytest.mark.parametrize("caller", ["root", "rebuilt-root", "compressed-root"])
def test_finished_grandchild_keeps_ancestor_continuation_ownership(caller):
    from hermes_agent.tools.delegate_tool import delegate_task

    model, root = _ScriptedModel(), _parent("root-sess")
    middle = _parent("middle-sess", db=root._session_db)
    middle._delegate_parent_ref = weakref.ref(root)
    with patch("run_agent.AIAgent", side_effect=model):
        sid, _ = _spawn(middle, "grandchild work")
        foreign = _parent("unrelated-sess", db=root._session_db)
        refused = json.loads(delegate_task(action="continue", subagent_id=sid, message="foreign work",
                                            background=True, parent_agent=foreign))
        assert "not yours" in refused["error"]
        immediate = json.loads(delegate_task(action="continue", subagent_id=sid, message="immediate follow-up",
                                             background=True, parent_agent=middle))
        assert immediate.get("status") == "dispatched", immediate
        _completion(immediate["delegation_id"])
        parent = root
        if caller in {"rebuilt-root", "compressed-root"}:
            parent = _parent("root-sess", db=root._session_db)
        if caller == "compressed-root":
            root._session_db.create_session("root-rotated", "cli", parent_session_id="root-sess")
            root._session_db.end_session("root-sess", "compression")
            parent.session_id = "root-rotated"
        out = json.loads(delegate_task(action="continue", subagent_id=sid, message="more grandchild work",
                                       background=True, parent_agent=parent))
        assert out.get("status") == "dispatched", out
        _completion(out["delegation_id"])
    assert model.turns[2]["session_id"] == model.turns[0]["session_id"]


def test_spawner_keeps_ownership_after_root_continues_grandchild():
    from hermes_agent.tools.delegate_tool import delegate_task

    model, root = _ScriptedModel(), _parent("root-sess")
    middle = _parent("middle-sess", db=root._session_db)
    middle._delegate_parent_ref = weakref.ref(root)
    foreign = _parent("unrelated-sess", db=root._session_db)
    with patch("run_agent.AIAgent", side_effect=model):
        sid, _ = _spawn(middle, "grandchild work")
        out = json.loads(delegate_task(action="continue", subagent_id=sid, message="root follow-up",
                                       background=True, parent_agent=root))
        assert out.get("status") == "dispatched", out
        _completion(out["delegation_id"])
        refused = json.loads(delegate_task(action="continue", subagent_id=sid, message="foreign work",
                                           background=True, parent_agent=foreign))
        assert "not yours" in refused["error"], refused
        again = json.loads(delegate_task(action="continue", subagent_id=sid, message="middle follow-up",
                                         background=True, parent_agent=middle))
        assert again.get("status") == "dispatched", again
        _completion(again["delegation_id"])


def test_resume_claim_checks_ownership_without_lock_and_only_one_claim_wins(monkeypatch):
    from hermes_agent.tools import delegate_tool_registry as registry
    from hermes_agent.tools.delegate_tool_continue import _claim_resume

    parent, sid = _parent(), "sa-cont-claim-race"
    _finished(sid, owner=parent.session_id)
    spec = registry._recent_subagents[sid]["resume"]
    owns = registry._owns_subagent_record
    checked = threading.Barrier(2, timeout=10)

    def check_ownership(record, caller):
        acquired = registry._active_subagents_lock.acquire(blocking=False)
        assert acquired, "ownership lookup must leave the registry available"
        registry._active_subagents_lock.release()
        result = owns(record, caller)
        checked.wait()
        return result

    monkeypatch.setattr(registry, "_owns_subagent_record", check_ownership)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = [pool.submit(_claim_resume, sid, parent) for _ in range(2)]
        results = [claim.result(timeout=15) for claim in claims]
    assert sum(claimed is spec and error is None for claimed, error in results) == 1
    assert sum(claimed is None and "No finished subagent" in error for claimed, error in results) == 1


def test_resume_claim_refuses_a_record_replaced_during_ownership_check(monkeypatch):
    from hermes_agent.tools import delegate_tool_registry as registry
    from hermes_agent.tools.delegate_tool_continue import _claim_resume

    parent, sid = _parent(), "sa-cont-replaced"
    _finished(sid, owner=parent.session_id)
    record = registry._recent_subagents[sid]
    replacement = {**record, "resume": {**record["resume"]}}

    def replace_record(*_):
        monkeypatch.setitem(registry._recent_subagents, sid, replacement)
        return True

    monkeypatch.setattr(registry, "_owns_subagent_record", replace_record)
    spec, error = _claim_resume(sid, parent)
    assert spec is None and "No finished subagent" in error
    assert "resume" in record and "resume" in replacement


def test_timeout_continue_waits_for_worker_close_and_resource_cleanup(monkeypatch):
    from hermes_agent.tools import delegate_tool

    parent, model = _parent(), _ScriptedModel()
    worker_release, close_release, cleanup_release, next_release = [threading.Event() for _ in range(4)]
    worker_started, close_started, close_done, cleanup_started, cleanup_done, next_started = [
        threading.Event() for _ in range(6)]
    resources = {}

    def build(**kw):
        child = model(**kw)
        original_run = child.run_conversation.side_effect
        first = len(model.builds) == 1

        def run(**args):
            if first:
                result = original_run(**args)
                worker_started.set()
                assert worker_release.wait(30)
                return result
            resources[child.session_id] = "continued task"
            next_started.set()
            assert next_release.wait(30)
            return original_run(**args)

        def close():
            if first:
                close_started.set()
                assert close_release.wait(30)
            parent._session_db.end_session(child.session_id, "completed")
            resources.pop(child.session_id, None)
            if first:
                close_done.set()

        child.run_conversation.side_effect, child.close.side_effect = run, close
        return child

    def shutdown_kernels(session_id):
        if close_done.is_set() and not next_started.is_set():
            cleanup_started.set()
            assert cleanup_release.wait(30)
            resources.pop(session_id, None)
            cleanup_done.set()

    # Force the existing inactivity-timeout path only after the worker reaches its barrier.
    def timeout_first(run, settled, future, timeout):
        if len(model.builds) == 1:
            assert worker_started.wait(10)
            raise TimeoutError()
        assert settled.wait(30)

    monkeypatch.setattr("hermes_agent.tools.delegate_tool_child_run._ChildRun.wait_liveness_aware", timeout_first)
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 1)
    monkeypatch.setattr("hermes_agent.tools.code_kernel.shutdown_kernels_for_delegated_child", shutdown_kernels)

    def follow_up(sid):
        return json.loads(delegate_tool.delegate_task(action="continue", subagent_id=sid, message="follow up",
                                                      background=True, parent_agent=parent))

    attempted = None
    with patch("run_agent.AIAgent", side_effect=build):
        try:
            sid, _ = _spawn(parent, "timed out work")
            attempted = follow_up(sid)
            assert "still stopping" in attempted.get("error", ""), attempted
            assert "Try again shortly; if this persists, spawn a new task." in attempted["error"]
            assert len(model.builds) == 1
            worker_release.set()
            assert close_started.wait(10)
            assert "still stopping" in follow_up(sid)["error"]
            close_release.set()
            assert cleanup_started.wait(10)
            assert "still stopping" in follow_up(sid)["error"]
            cleanup_release.set()
            assert cleanup_done.wait(10)

            def continue_when_ready():
                out = follow_up(sid)
                if "error" in out:
                    assert "still stopping" in out["error"], out
                    return None
                return out

            attempted = wait_until(continue_when_ready, what="resumable after complete old cleanup")
            assert attempted["subagent_ids"] == [sid]
            assert next_started.wait(10)
            session_id = model.turns[0]["session_id"]
            assert parent._session_db.get_session(session_id)["ended_at"] is None
            assert resources[session_id] == "continued task"
        finally:
            for event in (worker_release, close_release, cleanup_release, next_release):
                event.set()
            assert close_done.wait(10)
            if attempted and attempted.get("status") == "dispatched":
                _completion(attempted["delegation_id"])
