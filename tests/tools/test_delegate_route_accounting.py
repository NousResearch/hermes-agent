"""Budget and actual-child metadata regressions for per-task delegation routes."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.delegate_tool import delegate_task
from tools.delegate_tool_dispatch import _dispatch_unit, _units_of
from tools.async_delegation import _push_completion_event
from tools.process_registry_notifications import format_process_notification
from tests.tools.test_delegate_task_provider_pin import (
    PIN_CREDS, PARENT_CREDS, _fake_resolve, _make_mock_parent, _ok_child,
)


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("invalid", ["provider", "model", "default"])
def test_failed_route_does_not_spend_oneshot_budget(background, invalid):
    parent = _make_mock_parent()
    parent._oneshot_children_spawned = 0
    task = {"goal": "Research a sufficiently detailed topic", "provider": "lmstudio-x121"}
    bad = {"goal": "Research another sufficiently detailed topic", invalid: "unavailable"}
    cfg = {"provider": "unavailable"} if invalid == "default" else {}
    if invalid == "default":
        bad.pop("default")

    def resolve(route, agent):
        if "unavailable" in (route.get("provider"), route.get("model")):
            raise ValueError("route unavailable")
        return _fake_resolve(route, agent)

    with (
        patch("agent.oneshot_footprint.is_single_query_session", return_value=True),
        patch("tools.delegate_tool._get_oneshot_max_children", return_value=2),
        patch("tools.delegate_tool._load_config", return_value=cfg),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=resolve) as resolver,
        patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=lambda **kw: _ok_child()) as build,
        patch("tools.delegate_tool._run_batch", return_value='{"status":"built"}'),
    ):
        assert "route unavailable" in json.loads(delegate_task(
            tasks=[task, bad], parent_agent=parent, background=background,
        ))["error"]
        assert parent._oneshot_children_spawned == 0
        build.assert_not_called()
        # A valid retry can consume the entire original allowance, exactly once.
        resolver.reset_mock()
        assert json.loads(delegate_task(tasks=[task, task], parent_agent=parent, background=background))["status"] == "built"
        assert parent._oneshot_children_spawned == 2
        assert resolver.call_count == 2
        assert build.call_count == 2
        assert "budget" in json.loads(delegate_task(tasks=[task], parent_agent=parent, background=background))["error"].lower()
        assert parent._oneshot_children_spawned == 2
        assert build.call_count == 2


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("routes", [
    ["lmstudio-x121", "lmstudio-x121"], ["lmstudio-x121", "nous"],
    ["lmstudio-x121", None], ["model-only", "model-only"], [None, None],
])
def test_transcript_and_unit_metadata_follow_actual_children(tmp_path, background, routes):
    parent = _make_mock_parent()
    parent._session_db = SimpleNamespace(db_path=str(tmp_path / "state.db"))
    tasks = [{"goal": f"Research detailed topic number {i}",
              **({"model": "task-model"} if p == "model-only" else {"provider": p} if p else {})}
             for i, p in enumerate(routes)]
    children = []
    batches = []

    def build(**kwargs):
        child = _ok_child()
        child.model = kwargs["model"] + ("-canonical" if any(routes) else "")
        child.provider = kwargs["override_provider"]
        children.append(child)
        return child

    def resolve(cfg, agent):
        return dict(_fake_resolve(cfg, agent), **({"model": cfg["model"]} if cfg.get("model") else {}))

    def run(batch, bg):
        assert bg == background
        batches.append(batch)
        return '{"status":"built"}'

    with (
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=resolve),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=build),
        patch("tools.delegate_tool._run_batch", side_effect=run),
    ):
        delegate_task(tasks=tasks, parent_agent=parent, background=background)

    batch = batches[0]
    expected_models = [c.model for c in children]
    expected_providers = [c.provider for c in children]
    model = expected_models[0] if len(set(expected_models)) == 1 else None
    provider = expected_providers[0] if len(set(expected_providers)) == 1 else None
    manifest_path = tmp_path / "cache" / "delegation" / "live" / batch.live_deleg_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["model"] == model
    assert manifest["provider"] == provider
    if any(routes):
        assert [t["model"] for t in manifest["tasks"]] == expected_models
        assert [t["provider"] for t in manifest["tasks"]] == expected_providers
    for secret in [PIN_CREDS["api_key"], PARENT_CREDS["api_key"], "api_key", "base_url"]:
        assert secret not in manifest_path.read_text()
    # Exercise the actual whole-batch and independent/grouped dispatch seams.
    for independent in [False, True]:
        for group in [None, "shared"]:
            for task in batch.task_list:
                task["group"] = group
            with patch("tools.delegate_tool_config._get_independent_completions", return_value=independent):
                units = _units_of(batch)
            for unit in units:
                unit_models = [c.model for _, _, c in unit.children]
                expected = unit_models[0] if len(set(unit_models)) == 1 else None
                with patch("tools.async_delegation.dispatch_async_delegation_batch", return_value={}) as dispatch:
                    _dispatch_unit(unit, "unit-id", None, {})
                assert dispatch.call_args.kwargs["model"] == expected
                # Existing completion persistence/queue/formatter accept null without parent fallback.
                record = {"delegation_id": "unit-id", "is_batch": True, "model": expected,
                          "goals": [t["goal"] for t in batch.task_list]}
                result = {"results": [{"task_index": i, "status": "completed", "model": c.model,
                                       "summary": "ok"} for i, _, c in unit.children]}
                with (
                    patch("tools.async_delegation._persist_completion") as persist,
                    patch("tools.process_registry.process_registry.completion_queue") as queue,
                ):
                    _push_completion_event(record, result, "completed")
                event = queue.put.call_args.args[0]
                assert event["model"] == expected
                assert persist.call_args.args[0]["model"] == expected
                text = format_process_notification(event)
                assert f"Model: {expected or '?'}" in text
                assert parent.model not in text


@pytest.mark.parametrize("background", [False, True])
def test_execution_or_background_fallback_charges_once(background, tmp_path):
    parent = _make_mock_parent()
    parent._oneshot_children_spawned = 0
    parent._session_db = SimpleNamespace(db_path=str(tmp_path / "state.db"))
    child = _ok_child()
    child.model, child.provider = PIN_CREDS["model"], PIN_CREDS["provider"]
    entry = {"task_index": 0, "status": "completed", "model": child.model, "summary": "ok", "api_calls": 1}
    with (
        patch("agent.oneshot_footprint.is_single_query_session", return_value=True),
        patch("tools.delegate_tool._get_oneshot_max_children", return_value=1),
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=_fake_resolve),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=child),
        patch("tools.delegate_tool._run_single_child", return_value=entry) as run_child,
        patch("tools.delegate_tool_dispatch._resolve_async_wake_sid", return_value=None),
    ):
        result = json.loads(delegate_task(
            tasks=[{"goal": "Research a sufficiently detailed topic", "provider": "lmstudio-x121"}],
            parent_agent=parent, background=background,
        ))
    assert result["results"][0]["model"] == child.model
    assert parent._oneshot_children_spawned == 1
    run_child.assert_called_once()


@pytest.mark.parametrize("background", [False, True])
def test_exhausted_unpinned_budget_does_not_create_running_logs(background):
    parent = _make_mock_parent()
    parent._oneshot_children_spawned = 1
    with (
        patch("agent.oneshot_footprint.is_single_query_session", return_value=True),
        patch("tools.delegate_tool._get_oneshot_max_children", return_value=1),
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=_fake_resolve),
        patch("tools.delegation_live_log.create_live_transcripts", return_value=(None, [], [])) as create,
        patch("tools.delegate_tool._announce_batch") as announce,
        patch("tools.delegate_tool._build_child_preserving_parent_tools") as build,
    ):
        result = json.loads(delegate_task(goal="Research a sufficiently detailed topic", parent_agent=parent, background=background))
    assert "budget" in result["error"].lower()
    assert parent._oneshot_children_spawned == 1
    create.assert_not_called()
    announce.assert_not_called()
    build.assert_not_called()
