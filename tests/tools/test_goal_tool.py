"""goal tool: the agent declares, inspects and closes its own session goal through the registry."""
import json

import pytest

import hermes_cli.goals as goals_mod
import tools.goal_tool  # noqa: F401  (registers)
from tools.registry import registry


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    store = {}
    monkeypatch.setattr(goals_mod, "save_goal", lambda sid, st: store.__setitem__(sid, st.to_json()))
    monkeypatch.setattr(goals_mod, "load_goal", lambda sid: goals_mod.GoalState.from_json(store[sid]) if sid in store else None)
    monkeypatch.setattr(goals_mod, "clear_goal", lambda sid: store.pop(sid, None))
    monkeypatch.setattr(goals_mod, "draft_contract", lambda objective, **kw: None)
    monkeypatch.setattr(goals_mod, "agent_goal_tool_enabled", lambda: True)
    yield store


def _call(session_id="s1", **args):
    out = registry.dispatch("goal", args, session_id=session_id)
    return json.loads(out)


def test_registered_with_toolset_and_gate():
    entry = registry.get_entry("goal")
    assert entry is not None and entry.toolset == "goal"
    assert entry.check_fn() is True


def test_get_empty_then_create_then_get():
    assert _call(action="get") == {"goal": None}
    r = _call(action="create", objective="Cut the v1.2.14 release once the signup test passes")
    assert r["created"] and r["source"] == "agent" and r["status"] == "active"
    assert _call(action="get")["goal"].startswith("Cut the v1.2.14")
    assert "self-set" in goals_mod.GoalManager(session_id="s1").status_line()


def test_create_rejects_short_objective_and_missing_session():
    assert "error" in _call(action="create", objective="fix it")
    assert "error" in json.loads(registry.dispatch("goal", {"action": "get"}))


def test_user_goal_always_wins():
    goals_mod.GoalManager(session_id="s1").set("ship the docs site")
    r = _call(action="create", objective="Do something else that is long enough")
    assert "error" in r and r["goal"] == "ship the docs site"


def test_complete_requires_passing_audit(monkeypatch):
    _call(action="create", objective="Publish the release notes for v1.2.14 to Notion")
    monkeypatch.setattr(goals_mod, "judge_goal", lambda *a, **kw: ("continue", "no page url shown", False, None, False))
    r = _call(action="complete", evidence="I wrote the notes and will publish soon")
    assert "error" in r and r["verdict"] == "continue"
    assert goals_mod.GoalManager(session_id="s1").is_active()
    monkeypatch.setattr(goals_mod, "judge_goal", lambda *a, **kw: ("done", "page exists", False, None, False))
    r = _call(action="complete", evidence="Published: https://notion.so/release-notes-1-2-14, verified via API GET 200")
    assert r["completed"] is True
    assert not goals_mod.GoalManager(session_id="s1").has_goal()


def test_complete_without_evidence_or_goal_errors():
    assert "error" in _call(action="complete", evidence="done")
    _call(action="create", objective="Verify the canary rollout of v1.2.14 is green")
    assert "error" in _call(action="complete", evidence="ok")


def test_blocked_agent_goal_parks_not_nags(monkeypatch):
    _call(action="create", objective="Tag v1.2.14 after the owner approves the release")
    monkeypatch.setattr(goals_mod, "judge_goal", lambda *a, **kw: ("blocked", "needs owner approval", False, None, False))
    mgr = goals_mod.GoalManager(session_id="s1")
    d = mgr.evaluate_after_turn("Waiting on your approval to tag.")
    assert d["status"] == "paused" and "Re-scope" not in d["message"]
    # a parked self-set goal can be replaced by a new declaration
    assert _call(action="create", objective="Verify the canary rollout of v1.2.14 is green")["created"]


def test_goal_guidance_injected_only_when_tool_present():
    from types import SimpleNamespace
    from agent.prompt_builder import GOAL_TOOL_GUIDANCE
    from agent.system_prompt import _tool_guidance_block

    with_tool = _tool_guidance_block(SimpleNamespace(valid_tool_names={"goal", "terminal"}, _kanban_worker_guidance=None)) or ""
    without = _tool_guidance_block(SimpleNamespace(valid_tool_names={"terminal"}, _kanban_worker_guidance=None)) or ""
    assert GOAL_TOOL_GUIDANCE in with_tool and GOAL_TOOL_GUIDANCE not in without
