"""The scheduler's stand-in parent runs on the parent conversation's own route (its /model override), not the profile
default: a child that ran on the session's route resumes, and a stand-in that cannot be built never turns automatic
resumes off."""
import json
from unittest.mock import patch

import pytest

from tests.tools.test_delegate_auto_resume import _settled, _AlwaysLimited, _enable, _repo
from tests.tools.test_delegate_continue import _parent
from tests.tools.test_delegate_resumable import _db_rows

KEY = "agent:main:telegram:dm:42"
SESSION_ROUTE = {"model": "parent-model", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
                 "api_key": "session-key", "api_mode": "chat_completions"}
DEFAULT_ROUTE = {"model": "default-model", "provider": "custom", "base_url": "https://default.invalid/v1",
                 "api_key": "test-key"}


def _limited_child_of_session(tmp_path):
    from hermes_agent.gateway.session_context import clear_session_vars, set_session_vars
    from tools.delegate_tool import delegate_task
    from tools.terminal_tool import record_session_cwd
    _enable()
    parent = _parent()
    parent._current_task_id = "parent-task"
    record_session_cwd("parent-task", str(_repo(tmp_path, "repo")))
    model = _AlwaysLimited()
    tokens = set_session_vars(platform="telegram", chat_id="42", chat_type="dm", session_key=KEY,
                              session_id=parent.session_id)
    try:
        with patch("run_agent.AIAgent", side_effect=model):
            out = json.loads(delegate_task(tasks=[{"goal": "task on the session's route"}], background=True,
                                           parent_agent=parent))
            _settled(out["delegation_id"])
    finally:
        clear_session_vars(tokens)
    return parent, model, out["subagent_ids"][0]


def _runner():
    from hermes_agent.gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.session_store = None
    runner._session_model_overrides[KEY] = dict(SESSION_ROUTE)  # the parent conversation's /model
    return runner


def _start(runner, model, built):
    from hermes_agent.gateway.run_auto_resume import _start_continue

    def factory(**kw):
        if kw.get("platform") == "gateway":
            built.append(kw)
            standin = _parent(kw["session_id"], db=kw["session_db"])
            standin.provider, standin.base_url = kw["provider"], kw["base_url"]
            return standin
        return model(**kw)
    return factory, lambda record, unit, message: _start_continue(runner, record, unit, message)


def test_the_stand_in_parent_keeps_the_sessions_model_override(tmp_path):
    from tools.delegate_tool_auto_resume import tick
    parent, model, sid = _limited_child_of_session(tmp_path)
    built = []
    factory, start = _start(_runner(), model, built)
    with patch("hermes_agent.gateway.run._resolve_runtime_agent_kwargs", return_value=dict(DEFAULT_ROUTE)), \
            patch("run_agent.AIAgent", side_effect=factory):
        resumed = tick(start, capacity=3)
    assert [kw["provider"] for kw in built] == ["openrouter"] and built[0]["model"] == "parent-model"
    assert resumed == [sid], _db_rows(parent)[sid][2].get("auto_note")


def test_the_stand_in_parent_runs_on_the_sessions_own_tier_not_the_runners_last_one(tmp_path):
    """Another conversation's /fast turn left ``runner._service_tier`` at priority; this session has none."""
    from tools.delegate_tool_auto_resume import tick
    _parent_agent, model, sid = _limited_child_of_session(tmp_path)
    runner, built = _runner(), []
    runner._service_tier = "priority"
    factory, start = _start(runner, model, built)
    with patch("hermes_cli.models.resolve_fast_mode_overrides", return_value={"service_tier": "priority"}), \
            patch("run_agent.AIAgent", side_effect=factory):
        assert tick(start, capacity=3) == [sid]
    assert built[0].get("service_tier") is None and "service_tier" not in (built[0]["request_overrides"] or {})


@pytest.mark.parametrize("broken", ["route", "agent"])
def test_a_stand_in_that_cannot_be_built_retries_later_and_never_turns_auto_resume_off(tmp_path, broken):
    from tools.delegate_tool_auto_resume import tick
    parent, model, sid = _limited_child_of_session(tmp_path)
    factory, start = _start(_runner(), model, [])
    target = ("gateway.run.GatewayRunner._resolve_session_agent_runtime" if broken == "route"
              else "gateway.run_auto_resume._stand_in_parent")
    with patch(target, side_effect=RuntimeError("no credentials")), patch("run_agent.AIAgent", side_effect=factory):
        assert tick(start, capacity=3) == []
    state, _pinned, record = _db_rows(parent)[sid]
    assert (state, record.get("auto_off"), record.get("auto_attempts")) == ("resumable", None, 0)


def test_capacity_counts_only_this_profiles_running_units(tmp_path, monkeypatch):
    """One gateway serves many profiles: another profile's running units never use up this one's slots."""
    from agent.runtime_scope import bind_profile_scope
    from hermes_agent.gateway.run_auto_resume import _capacity
    from hermes_constants import hermes_home_key
    from tools import async_delegation
    other, mine = tmp_path / "other", tmp_path / "mine"
    for home, cap in ((other, 2), (mine, 3)):
        home.mkdir()
        (home / "config.yaml").write_text(json.dumps({"delegation": {"max_concurrent_children": cap}}))
    running = {f"u{i}": {"status": "running", "_home": hermes_home_key(other)} for i in range(2)}
    running["m"] = {"status": "running", "_home": hermes_home_key(mine)}
    monkeypatch.setattr(async_delegation, "_records", running)
    with bind_profile_scope(mine, hydrate_secrets=False):
        assert _capacity() == 2
    with bind_profile_scope(other, hydrate_secrets=False):
        assert _capacity() == 0


def test_a_spawn_pause_holds_resumes_without_claiming_or_building_a_stand_in(tmp_path):
    from tools.delegate_tool_auto_resume import tick
    from tools.delegate_tool_registry import set_spawn_paused
    parent, model, sid = _limited_child_of_session(tmp_path)
    built = []
    factory, start = _start(_runner(), model, built)
    set_spawn_paused(True)
    try:
        with patch("run_agent.AIAgent", side_effect=factory):
            assert tick(start, capacity=3) == []
    finally:
        set_spawn_paused(False)
    state, _pinned, record = _db_rows(parent)[sid]
    assert built == [] and (state, record.get("auto_attempts", 0), record.get("auto_off")) == ("resumable", 0, None)
