"""delegation.allow_model_selection gates the model-facing task route fields.

Off (the default), delegate_task's schema carries no task-level provider /
model / reasoning_effort / service_tier and model-supplied values are dropped;
on, the fields and their guidance are advertised and passed through.
"""

import json

import pytest

from tools.delegate_tool import DELEGATE_TASK_SCHEMA, _strip_model_hidden_task_fields
from tools.registry import registry

ROUTE_FIELDS = {"provider", "model", "reasoning_effort", "service_tier"}


def _definition(monkeypatch, config):
    from tools import delegate_tool_config

    monkeypatch.setattr(delegate_tool_config, "_cfg", lambda: config)
    return registry.get_definitions({"delegate_task"})[0]["function"]


def test_switch_is_off_by_default(monkeypatch):
    from tools import delegate_tool_config

    monkeypatch.setattr(delegate_tool_config, "_cfg", lambda: {})
    assert delegate_tool_config._get_allow_model_selection() is False
    task_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
    assert not set(task_props) & ROUTE_FIELDS


@pytest.mark.parametrize("enabled", [False, True])
def test_schema_and_guidance_follow_the_switch(monkeypatch, enabled):
    original = json.dumps(DELEGATE_TASK_SCHEMA)
    definition = _definition(monkeypatch, {"allow_model_selection": enabled})
    task_props = definition["parameters"]["properties"]["tasks"]["items"]["properties"]
    assert (set(task_props) & ROUTE_FIELDS == ROUTE_FIELDS) is enabled
    assert not set(task_props) & ROUTE_FIELDS or enabled
    assert ("reasoning_effort/service_tier" in definition["description"]) is enabled
    assert "delegation.provider / delegation.model" in definition["description"]
    assert json.dumps(DELEGATE_TASK_SCHEMA) == original


@pytest.mark.parametrize("enabled", [False, True])
def test_model_supplied_route_fields_follow_the_switch(monkeypatch, enabled):
    from tools import delegate_tool_config

    monkeypatch.setattr(delegate_tool_config, "_cfg", lambda: {"allow_model_selection": enabled})
    tasks = [{"goal": "Review this change set", "provider": "openai-codex", "model": "gpt-6-astra",
              "reasoning_effort": "medium", "service_tier": "fast", "acp_command": "x"}]
    stripped = _strip_model_hidden_task_fields(tasks)
    assert (set(stripped[0]) & ROUTE_FIELDS == ROUTE_FIELDS) is enabled
    assert not set(stripped[0]) & ROUTE_FIELDS or enabled
    assert "acp_command" not in stripped[0]
    assert stripped[0]["goal"] == "Review this change set"
    plain = [{"goal": "Review this change set"}]
    assert _strip_model_hidden_task_fields(plain) is plain
