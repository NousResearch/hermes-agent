"""Sequential local inference must join children without background admission."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import tools.delegate_tool as dt
import tools.delegate_tool_dispatch as dispatch
from tools.registry import registry


def test_registry_sequential_delegation_joins_children_in_order(monkeypatch):
    monkeypatch.setattr(dt, "_load_config", lambda: {"sequential": True, "max_concurrent_children": 2})
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *args: {})
    monkeypatch.setattr(dt, "_oneshot_spawn_budget", lambda *args: None)
    monkeypatch.setattr(dt, "_build_children", lambda tasks, *args, **kwargs:
                        ([(i, task, SimpleNamespace()) for i, task in enumerate(tasks)], None))
    monkeypatch.setattr(dispatch, "_finalize_child_results", lambda *args: None)
    background = Mock(return_value=json.dumps({"status": "dispatched"}))
    monkeypatch.setattr(dispatch, "_dispatch_background", background)
    parent = SimpleNamespace(_delegate_depth=0, _interrupt_requested=False,
                             _delegate_spinner=None, quiet_mode=True)
    first_started, release_first, second_started, returned = (threading.Event() for _ in range(4))
    events, output = [], []

    def child_run(task_index, **kwargs):
        events.append((task_index, "start"))
        if task_index == 0:
            first_started.set()
            assert release_first.wait(10)
        else:
            second_started.set()
        events.append((task_index, "end"))
        return {"task_index": task_index, "status": "completed", "summary": "done",
                "api_calls": 1, "duration_seconds": 0}

    monkeypatch.setattr(dt, "_run_single_child", child_run)

    def drive():
        output.append(json.loads(registry.dispatch("delegate_task", {"tasks": [
            {"goal": "Inspect the first subsystem"}, {"goal": "Inspect the second subsystem"}
        ]}, parent_agent=parent)))
        returned.set()

    worker = threading.Thread(target=drive, daemon=True)
    worker.start()
    try:
        assert first_started.wait(5), output
        assert not returned.is_set()
        assert not second_started.is_set()
        release_first.set()
        assert returned.wait(10), output
        assert events == [(0, "start"), (0, "end"), (1, "start"), (1, "end")]
        assert [r["status"] for r in output[0]["results"]] == ["completed", "completed"]
        background.assert_not_called()
    finally:
        release_first.set()
        worker.join(10)


def test_sequential_policy_follows_real_config_and_preserves_default(tmp_path, monkeypatch):
    from agent.agent_init import _load_tools
    import hermes_cli.config as config
    from agent.secret_scope import set_multiplex_active, set_secret_scope, reset_secret_scope
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    homes = [tmp_path / "a", tmp_path / "b"]
    for home, enabled in zip(homes, (True, False)):
        home.mkdir()
        (home / "config.yaml").write_text(
            "delegation:\n  sequential: " + str(enabled).lower() + "\n", encoding="utf-8")
    background = Mock(return_value='{"status":"dispatched"}')
    aggregate = Mock(return_value={"results": []})
    monkeypatch.setattr(dispatch, "_dispatch_background", background)
    monkeypatch.setattr(dispatch, "_execute_and_aggregate", aggregate)
    from dataclasses import fields
    batch = dispatch._Batch(**{f.name: None for f in fields(dispatch._Batch)})
    batch.max_children = 5
    batch.children = []
    batch.task_list = []
    batch.parent_agent = SimpleNamespace(_delegate_depth=0)
    snapshots = []
    home_token = secret_token = None
    try:
        config._LOAD_CONFIG_CACHE.clear()
        set_multiplex_active(True)
        for home, enabled in ((homes[0], True), (homes[1], False), (homes[0], True)):
            if home_token is not None:
                reset_hermes_home_override(home_token)
                reset_secret_scope(secret_token)
            home_token = set_hermes_home_override(home)
            secret_token = set_secret_scope({}, profile_home=str(home))
            parent = SimpleNamespace(quiet_mode=True, _delegate_depth=0)
            _load_tools(parent, ["delegation"], None)
            definition = next(t for t in parent.tools if t["function"]["name"] == "delegate_task")
            original_tools = json.dumps(parent.tools)
            # Existing sessions keep the advertised policy even if their profile
            # changes; a newly constructed session picks up the new policy.
            (home / "config.yaml").write_text(
                "delegation:\n  sequential: " + str(not enabled).lower() + "\n", encoding="utf-8")
            config._LOAD_CONFIG_CACHE.clear()
            _load_tools(parent, ["delegation"], None)
            assert json.dumps(parent.tools) == original_tools
            batch.parent_agent = parent
            result = json.loads(dispatch._run_batch(batch, background=True))
            assert ("results" in result) == enabled
            description = definition["function"]["description"].lower()
            assert ("sequential" in description) == enabled
            if enabled:
                assert aggregate.call_args.args[0].max_children == 1
            for old, serialized in snapshots:
                assert json.dumps(old) == serialized
            snapshots.append((definition, json.dumps(definition)))
            fresh_parent = SimpleNamespace(quiet_mode=True, _delegate_depth=0)
            _load_tools(fresh_parent, ["delegation"], None)
            fresh_definition = next(t for t in fresh_parent.tools if t["function"]["name"] == "delegate_task")
            assert ("sequential" in fresh_definition["function"]["description"].lower()) == (not enabled)
            batch.parent_agent = fresh_parent
            fresh_result = json.loads(dispatch._run_batch(batch, background=True))
            assert ("results" in fresh_result) == (not enabled)
            assert json.dumps(parent.tools) == original_tools
            (home / "config.yaml").write_text(
                "delegation:\n  sequential: " + str(enabled).lower() + "\n", encoding="utf-8")
            config._LOAD_CONFIG_CACHE.clear()
        assert background.call_count == 3
        assert aggregate.call_count == 3
        assert batch.max_children == 5
    finally:
        if home_token is not None:
            reset_hermes_home_override(home_token)
            reset_secret_scope(secret_token)
        set_multiplex_active(False)
        config._LOAD_CONFIG_CACHE.clear()
