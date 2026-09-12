"""Grouping is advertised only when the delivery policy consumes it."""

import json
from dataclasses import fields

from tools.delegate_tool import DELEGATE_TASK_SCHEMA, _strip_model_hidden_task_fields
from tools.delegate_tool_dispatch import _Batch, _units_of
from tools.delegate_tool_tasks import _normalize_task_list
from tools.registry import registry


def test_group_schema_tracks_delivery_policy_without_mutating_previous_definitions(monkeypatch):
    from tools import delegate_tool_config

    original = json.dumps(DELEGATE_TASK_SCHEMA)
    snapshots = []
    for config in ({}, {"independent_completions": True}, {"independent_completions": False}):
        monkeypatch.setattr(delegate_tool_config, "_cfg", lambda: config)
        definition = registry.get_definitions({"delegate_task"})[0]
        enabled = config.get("independent_completions", False)
        task = definition["function"]["parameters"]["properties"]["tasks"]["items"]
        assert ("group" in task["properties"]) == enabled
        assert ("group" in definition["function"]["description"]) == enabled
        assert json.dumps(registry.get_definitions({"delegate_task"})[0]) == json.dumps(definition)
        for previous, serialized in snapshots:
            assert json.dumps(previous) == serialized
        snapshots.append((definition, json.dumps(definition)))
    assert json.dumps(DELEGATE_TASK_SCHEMA) == original


def test_legacy_group_replay_remains_accepted_and_delivery_policy_controls_units(monkeypatch):
    from tools import delegate_tool_config

    tasks = [{"goal": "Review first module", "group": "join"}, {"goal": "Review second module", "group": "join"}, {"goal": "Review third module"}]
    assert _strip_model_hidden_task_fields(tasks) is tasks
    normalized, error = _normalize_task_list(None, None, tasks, None, "leaf", 3)
    assert error is None and normalized == tasks
    batch = _Batch(**{field.name: None for field in fields(_Batch)})
    batch.children = [(i, task, None) for i, task in enumerate(tasks)]
    for enabled in (False, True):
        monkeypatch.setattr(delegate_tool_config, "_cfg", lambda: {"independent_completions": enabled})
        units = _units_of(batch)
        if enabled:
            assert [[i for i, _, _ in unit.children] for unit in units] == [[0, 1], [2]]
            assert [unit.group for unit in units] == ["join", None]
        else:
            assert units == [batch]
