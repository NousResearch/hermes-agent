import json

import pytest


WORKFLOW_TOOL_NAMES = {
    "workflow_list",
    "workflow_show",
    "workflow_run",
    "workflow_execution_show",
    "workflow_cancel",
}


@pytest.fixture(autouse=True)
def _isolated_workflow_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_WORKFLOW_CONTEXT", raising=False)

    from model_tools import _clear_tool_defs_cache
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    yield hermes_home
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()


def _enable_workflow_toolset(hermes_home):
    (hermes_home / "config.yaml").write_text("toolsets:\n  - workflow\n", encoding="utf-8")
    from model_tools import _clear_tool_defs_cache
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()


def _tool_names_for_workflow_toolset():
    from model_tools import get_tool_definitions

    tools = get_tool_definitions(
        enabled_toolsets=["workflow"],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    return {tool["function"]["name"] for tool in tools}


def _deploy_demo_workflow():
    from hermes_cli import workflows_db as wfdb
    from hermes_cli.workflows_spec import WorkflowSpec

    spec = WorkflowSpec.model_validate({
        "id": "demo",
        "name": "Demo",
        "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {"start": {"type": "pass", "output": {"ok": True}}},
    })
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, spec, created_by="test")
    return spec


def test_workflow_tools_hidden_without_workflow_toolset():
    assert WORKFLOW_TOOL_NAMES.isdisjoint(_tool_names_for_workflow_toolset())


def test_workflow_tools_visible_when_config_enables_workflow_toolset(_isolated_workflow_home):
    _enable_workflow_toolset(_isolated_workflow_home)

    assert WORKFLOW_TOOL_NAMES.issubset(_tool_names_for_workflow_toolset())


def test_workflow_handlers_return_json_strings(_isolated_workflow_home):
    _enable_workflow_toolset(_isolated_workflow_home)
    spec = _deploy_demo_workflow()

    from hermes_cli import workflows_db as wfdb
    from tools.registry import registry

    with wfdb.connect() as conn:
        execution_id = wfdb.start_execution(
            conn,
            spec.id,
            input_data={},
            trigger_type="manual",
        )

    calls = [
        ("workflow_list", {}),
        ("workflow_show", {"workflow_id": spec.id}),
        ("workflow_run", {"workflow_id": spec.id, "input_json": "{}"}),
        ("workflow_execution_show", {"execution_id": execution_id}),
        ("workflow_cancel", {"execution_id": execution_id}),
    ]
    for name, args in calls:
        result = registry.dispatch(name, args)
        assert isinstance(result, str)
        payload = json.loads(result)
        assert "error" not in payload


def test_workflow_run_creates_execution_with_provided_input(_isolated_workflow_home):
    _enable_workflow_toolset(_isolated_workflow_home)
    spec = _deploy_demo_workflow()

    from hermes_cli import workflows_db as wfdb
    from tools.registry import registry

    result = registry.dispatch(
        "workflow_run",
        {"workflow_id": spec.id, "input_json": '{"x": 1}'},
    )
    payload = json.loads(result)

    assert "error" not in payload
    assert payload["workflow_id"] == spec.id
    assert payload["version"] == spec.version
    assert payload["status"] == "queued"
    assert payload["input"] == {"x": 1}

    with wfdb.connect() as conn:
        execution = wfdb.get_execution(conn, payload["execution_id"])

    assert execution.workflow_id == spec.id
    assert execution.version == spec.version
    assert execution.status == "queued"
    assert execution.input == {"x": 1}
