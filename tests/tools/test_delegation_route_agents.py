"""Operator-authored agents narrow core's child surface without changing the parent or model schema."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.tools import test_delegation_profile_route
from tests.tools.test_delegation_route_fields import _Backend, _construct, _parent, _spawn

two_profiles = test_delegation_profile_route.two_profiles


def _construct_agent(**kwargs):
    child = _construct(**kwargs)
    for key in ("enabled_toolsets", "disabled_toolsets", "max_iterations", "ephemeral_system_prompt"):
        setattr(child, key, kwargs[key])
    return child


@pytest.fixture
def parent():
    agent = _parent()
    agent.enabled_toolsets, agent.disabled_toolsets = ["terminal", "file"], ["web"]
    agent.cwd = agent.terminal_cwd = agent._subdirectory_hints = None
    agent._cached_system_prompt = "Parent prompt must stay byte-stable."
    return agent


def _run(routes, parent, **kwargs):
    with patch("tools.delegate_tool._resolve_workspace_hint", return_value=None):
        return _spawn(routes, parent=parent, construct=_construct_agent, **kwargs)


def test_route_instructions_are_child_only_and_report_never_contains_the_text(parent):
    from tools.delegate_tool_child_run import _fabricated_entry

    text = "Inspect the patch and return concrete defects."
    _, (built, plain), (child, sibling) = _run([{"instructions": text}, {}], parent)
    assert text in built["ephemeral_system_prompt"] and text not in plain["ephemeral_system_prompt"]
    assert parent._cached_system_prompt == "Parent prompt must stay byte-stable."
    route = _fabricated_entry(0, "error", "stopped", child)["route"]
    assert route["instructions"] is True and text not in json.dumps(route)
    assert "instructions" not in _fabricated_entry(1, "error", "stopped", sibling)["route"]


@pytest.mark.parametrize("requested, expected", [(["file", "web"], ["file"]), ([], [])])
def test_route_toolsets_only_narrow_after_core_role_and_mcp_resolution(parent, requested, expected):
    from tools.delegate_tool_child_run import _route_report

    parent.enabled_toolsets.append("mcp-probe")
    with patch("tools.delegate_tool._get_orchestrator_enabled", return_value=True):
        _, (built,), (child,) = _run([{"toolsets": requested}], parent)
    assert built["enabled_toolsets"] == expected
    assert "web" in built["disabled_toolsets"]
    assert _route_report(child)["toolsets"] == expected
    assert parent.enabled_toolsets == ["terminal", "file", "mcp-probe"]


def test_route_can_narrow_a_composite_but_cannot_restore_parent_disabled_tools(parent):
    from model_tools import _select_tool_names

    parent.enabled_toolsets, parent.disabled_toolsets = ["hermes-cli"], ["terminal"]
    _, (built,), _ = _run([{"toolsets": ["file", "terminal"]}], parent)
    assert built["enabled_toolsets"] == ["file", "terminal"]
    names = _select_tool_names(built["enabled_toolsets"], built["disabled_toolsets"], quiet_mode=True)
    assert "read_file" in names and "terminal" not in names and "delegate_task" not in names


@pytest.mark.parametrize("requested, effective", [(3, 3), (1000, 7)])
def test_route_iteration_limit_is_capped_and_reported_with_its_effective_value(parent, requested, effective):
    from tools.delegate_tool_child_run import _route_report

    _, (built,), (child,) = _run([{"max_iterations": requested}], parent, cfg={"max_iterations": 7})
    assert built["max_iterations"] == effective
    assert _route_report(child)["max_iterations"] == effective


def test_route_without_agent_keys_keeps_the_default_prompt_and_toolsets(parent):
    from agent.delegation_registry import get_active_provider
    from tools.delegate_tool import _build_child_system_prompt, _resolve_child_toolsets
    from tools.delegate_tool_child_run import _route_report

    _, (built,), (child,) = _run([None], parent, provider=get_active_provider({}))
    assert built["ephemeral_system_prompt"] == _build_child_system_prompt(
        "task number 0", role=child._delegate_role)
    assert (built["enabled_toolsets"], built["disabled_toolsets"]) == _resolve_child_toolsets(
        parent, None, child._delegate_role)
    assert _route_report(child) is None


def test_profile_route_adds_instructions_on_top_of_its_soul_and_narrows_tools(two_profiles, parent):
    from agent.system_prompt import _identity_parts
    from tools.delegate_tool_child_run import _route_report
    from tools.delegate_tool_routes import child_profile_scope

    _, coder = two_profiles
    text = "Review code only; do not implement changes."
    _, (built,), (child,) = _run([{"profile": "coder", "instructions": text,
                                   "toolsets": ["file", "web"], "max_iterations": 100}], parent,
                                cfg={"max_iterations": 5})
    assert built["load_soul_identity"] is True and child._delegate_profile_home == coder
    with child_profile_scope(child):
        identity, loaded = _identity_parts(SimpleNamespace(load_soul_identity=True, skip_context_files=True), None)
    assert loaded and "I am the coder profile." in "\n".join(identity)
    assert text in built["ephemeral_system_prompt"]
    assert built["enabled_toolsets"] == ["file"] and built["max_iterations"] == 5
    report = _route_report(child)
    assert (report["instructions"], report["toolsets"], report["max_iterations"]) == (True, ["file"], 5)
    assert text not in json.dumps(report)


def test_operator_agent_keys_cannot_be_declared_or_supplied_by_the_model(parent):
    from agent.delegation_provider import RESERVED_TASK_PARAMETERS
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA, _build_dynamic_schema_overrides, delegate_task

    keys = {"instructions", "toolsets", "max_iterations"}

    class EchoBackend(_Backend):
        def task_parameters(self, config):
            return {name: {"type": "string"} for name in keys}

        def route_task(self, task, config):
            assert keys.isdisjoint(task)
            return {key: task[key] for key in keys if key in task}

    with (
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("tools.delegate_tool_routes._active_provider", return_value=EchoBackend([])),
        patch("tools.delegate_tool._run_batch", return_value='{"status": "built"}'),
        patch("tools.delegate_tool._resolve_workspace_hint", return_value=None),
        patch("run_agent.AIAgent", side_effect=_construct_agent) as agent,
    ):
        schema = _build_dynamic_schema_overrides()["parameters"]["properties"]
        assert keys <= RESERVED_TASK_PARAMETERS
        assert keys.isdisjoint(schema) and keys.isdisjoint(schema["tasks"]["items"]["properties"])
        assert keys.isdisjoint(DELEGATE_TASK_SCHEMA["parameters"]["properties"])
        result = json.loads(delegate_task(tasks=[{"goal": "Inspect the patch", "instructions": "injected",
                                                 "toolsets": ["web"], "max_iterations": 999}], parent_agent=parent))
    assert result == {"status": "built"}
    built = agent.call_args.kwargs
    assert "injected" not in built["ephemeral_system_prompt"]
    assert built["enabled_toolsets"] == ["terminal", "file"] and built["max_iterations"] == 250


@pytest.mark.parametrize("route", [{"instructions": 4}, {"toolsets": "file"}, {"toolsets": [4]},
                                   {"max_iterations": True}, {"max_iterations": "3"}, {"max_iterations": 0}])
def test_malformed_agent_settings_refuse_only_their_task(parent, route):
    result, built, _ = _run([route, {}], parent)
    assert len(built) == 1 and result["refused"][0]["task_index"] == 0

