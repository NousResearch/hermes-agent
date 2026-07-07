import json

import pytest


WORKFLOW_TOOL_NAMES = {
    "workflow_list",
    "workflow_show",
    "workflow_draft",
    "workflow_refine",
    "workflow_validate",
    "workflow_deploy",
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
    monkeypatch.delenv("HERMES_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)

    from model_tools import _clear_tool_defs_cache
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    yield hermes_home
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()


def _write_config(hermes_home, text):
    (hermes_home / "config.yaml").write_text(text, encoding="utf-8")
    from model_tools import _clear_tool_defs_cache
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()


def _enable_workflow_toolset(hermes_home):
    _write_config(hermes_home, "toolsets:\n  - workflow\n")


def _enable_cli_workflow_toolset(hermes_home):
    _write_config(hermes_home, "platform_toolsets:\n  cli:\n    - workflow\n")


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


def test_workflow_tool_visibility_tracks_env_without_manual_cache_invalidation(monkeypatch):
    assert WORKFLOW_TOOL_NAMES.isdisjoint(_tool_names_for_workflow_toolset())

    monkeypatch.setenv("HERMES_WORKFLOW_CONTEXT", "1")
    assert WORKFLOW_TOOL_NAMES.issubset(_tool_names_for_workflow_toolset())

    monkeypatch.delenv("HERMES_WORKFLOW_CONTEXT", raising=False)
    assert WORKFLOW_TOOL_NAMES.isdisjoint(_tool_names_for_workflow_toolset())


def test_workflow_tools_visible_when_config_enables_workflow_toolset(_isolated_workflow_home):
    _enable_workflow_toolset(_isolated_workflow_home)

    assert WORKFLOW_TOOL_NAMES.issubset(_tool_names_for_workflow_toolset())


def test_workflow_tools_visible_when_platform_config_enables_workflow_toolset(
    _isolated_workflow_home,
):
    _enable_cli_workflow_toolset(_isolated_workflow_home)

    assert WORKFLOW_TOOL_NAMES.issubset(_tool_names_for_workflow_toolset())


def test_workflow_tools_ignore_scalar_platform_toolsets(_isolated_workflow_home):
    _write_config(_isolated_workflow_home, "platform_toolsets:\n  cli: notworkflow\n")

    assert WORKFLOW_TOOL_NAMES.isdisjoint(_tool_names_for_workflow_toolset())


def test_workflow_platform_toolsets_check_is_platform_independent(
    _isolated_workflow_home,
    monkeypatch,
):
    _write_config(
        _isolated_workflow_home,
        "platform_toolsets:\n  telegram:\n    - workflow\n",
    )
    monkeypatch.setenv("HERMES_PLATFORM", "cli")
    assert WORKFLOW_TOOL_NAMES.issubset(_tool_names_for_workflow_toolset())

    monkeypatch.setenv("HERMES_PLATFORM", "discord")
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


def test_workflow_validate_tool_accepts_yaml_text(_isolated_workflow_home):
    _enable_workflow_toolset(_isolated_workflow_home)
    from tools.registry import registry

    yaml_text = """
id: validate_tool_demo
name: Validate Tool Demo
version: 1
nodes:
  start:
    type: pass
    output:
      ok: true
""".lstrip()
    payload = json.loads(registry.dispatch("workflow_validate", {"definition_text": yaml_text}))
    assert "error" not in payload
    assert payload["valid"] is True
    assert payload["workflow_id"] == "validate_tool_demo"


def test_workflow_validate_accepts_definition_object(_isolated_workflow_home):
    _enable_workflow_toolset(_isolated_workflow_home)
    from tools.registry import registry

    definition = {
        "id": "validate_object_demo",
        "name": "Validate Object Demo",
        "version": 1,
        "nodes": {"start": {"type": "pass", "output": {"ok": True}}},
    }

    payload = json.loads(registry.dispatch("workflow_validate", {"definition": definition}))

    assert "error" not in payload
    assert payload["workflow_id"] == "validate_object_demo"


def test_workflow_validate_rejects_unsupported_primitives(_isolated_workflow_home):
    _enable_workflow_toolset(_isolated_workflow_home)
    from tools.registry import registry

    definition = {
        "id": "validate_unsupported_demo",
        "name": "Validate Unsupported Demo",
        "version": 1,
        "triggers": [{"type": "manual"}],
        "nodes": {"start": {"type": "send_message", "output": {"text": "hi"}}},
    }

    payload = json.loads(registry.dispatch("workflow_validate", {"definition": definition}))

    assert "error" in payload
    assert "unsupported node type: send_message on node start" in payload["error"]


def test_workflow_draft_tool_returns_validated_spec(_isolated_workflow_home, monkeypatch):
    _enable_workflow_toolset(_isolated_workflow_home)
    import tools.workflow_tools as workflow_tools
    from hermes_cli.workflows_assistant import parse_assistant_payload
    from tools.registry import registry

    def fake_draft(goal):
        assert "code change" in goal
        return parse_assistant_payload({
            "spec": {
                "id": "draft_tool_demo",
                "name": "Draft Tool Demo",
                "version": 1,
                "triggers": [{"type": "manual", "id": "manual"}],
                "nodes": {"start": {"type": "pass", "output": {"ok": True}}},
            },
            "summary": "Drafted.",
            "assumptions": [],
            "questions": [],
            "warnings": [],
            "unsupported_requests": [],
        })

    monkeypatch.setattr(workflow_tools.workflows_assistant, "draft_workflow_with_default_runner", fake_draft)

    payload = json.loads(registry.dispatch("workflow_draft", {"goal": "code change workflow"}))

    assert "error" not in payload
    assert payload["spec"]["id"] == "draft_tool_demo"
    assert payload["summary"] == "Drafted."


def test_workflow_draft_tool_redacts_unexpected_runtime_errors(
    _isolated_workflow_home,
    monkeypatch,
):
    _enable_workflow_toolset(_isolated_workflow_home)
    import tools.workflow_tools as workflow_tools
    from tools.registry import registry

    def fake_draft(_goal):
        raise RuntimeError("secret provider token abc123")

    monkeypatch.setattr(workflow_tools.workflows_assistant, "draft_workflow_with_default_runner", fake_draft)

    result = registry.dispatch("workflow_draft", {"goal": "code change workflow"})
    payload = json.loads(result)

    assert payload["error"].startswith("workflow_draft: workflow assistant failed")
    assert "secret" not in result
    assert "abc123" not in result


def test_workflow_refine_tool_accepts_spec_object(_isolated_workflow_home, monkeypatch):
    _enable_workflow_toolset(_isolated_workflow_home)
    import tools.workflow_tools as workflow_tools
    from hermes_cli.workflows_assistant import parse_assistant_payload
    from tools.registry import registry

    def fake_refine(spec, instruction):
        assert spec.id == "demo"
        assert instruction == "add reviewer"
        return parse_assistant_payload({
            "spec": spec.model_dump(mode="json", by_alias=True),
            "summary": "Refined.",
            "assumptions": [],
            "questions": [],
            "warnings": [],
            "unsupported_requests": [],
        })

    monkeypatch.setattr(workflow_tools.workflows_assistant, "refine_workflow_with_default_runner", fake_refine)

    spec = _deploy_demo_workflow().model_dump(mode="json", by_alias=True)
    payload = json.loads(registry.dispatch("workflow_refine", {"spec": spec, "instruction": "add reviewer"}))

    assert "error" not in payload
    assert payload["summary"] == "Refined."


def test_workflow_refine_tool_redacts_unexpected_runtime_errors(
    _isolated_workflow_home,
    monkeypatch,
):
    _enable_workflow_toolset(_isolated_workflow_home)
    import tools.workflow_tools as workflow_tools
    from tools.registry import registry

    def fake_refine(_spec, _instruction):
        raise RuntimeError("secret provider token abc123")

    monkeypatch.setattr(workflow_tools.workflows_assistant, "refine_workflow_with_default_runner", fake_refine)

    spec = _deploy_demo_workflow().model_dump(mode="json", by_alias=True)
    result = registry.dispatch("workflow_refine", {"spec": spec, "instruction": "add reviewer"})
    payload = json.loads(result)

    assert payload["error"].startswith("workflow_refine: workflow assistant failed")
    assert "secret" not in result
    assert "abc123" not in result


def test_workflow_refine_tool_falls_back_to_workflow_id_when_spec_is_none(
    _isolated_workflow_home,
    monkeypatch,
):
    _enable_workflow_toolset(_isolated_workflow_home)
    import tools.workflow_tools as workflow_tools
    from hermes_cli.workflows_assistant import parse_assistant_payload
    from tools.registry import registry

    seen = {}

    def fake_refine(spec, instruction):
        seen["spec_id"] = spec.id
        assert instruction == "add reviewer"
        return parse_assistant_payload({
            "spec": spec.model_dump(mode="json", by_alias=True),
            "summary": "Refined.",
            "assumptions": [],
            "questions": [],
            "warnings": [],
            "unsupported_requests": [],
        })

    monkeypatch.setattr(workflow_tools.workflows_assistant, "refine_workflow_with_default_runner", fake_refine)
    _deploy_demo_workflow()

    payload = json.loads(registry.dispatch(
        "workflow_refine",
        {"spec": None, "workflow_id": "demo", "instruction": "add reviewer"},
    ))

    assert "error" not in payload
    assert seen["spec_id"] == "demo"


def test_workflow_refine_schema_mentions_spec_or_workflow_id():
    import tools.workflow_tools as workflow_tools

    assert "spec or workflow_id" in workflow_tools._WORKFLOW_REFINE_SCHEMA["description"]


def test_workflow_deploy_tool_accepts_yaml_text(_isolated_workflow_home):
    _enable_workflow_toolset(_isolated_workflow_home)
    from tools.registry import registry

    yaml_text = """
id: deploy_tool_demo
name: Deploy Tool Demo
version: 1
nodes:
  start:
    type: pass
""".lstrip()
    payload = json.loads(registry.dispatch("workflow_deploy", {"definition_text": yaml_text}))
    assert "error" not in payload
    assert payload["workflow_id"] == "deploy_tool_demo"


def test_workflow_deploy_accepts_definition_object(_isolated_workflow_home):
    _enable_workflow_toolset(_isolated_workflow_home)
    from tools.registry import registry

    definition = {
        "id": "deploy_object_demo",
        "name": "Deploy Object Demo",
        "version": 1,
        "nodes": {"start": {"type": "pass"}},
    }

    payload = json.loads(registry.dispatch("workflow_deploy", {"definition": definition}))

    assert "error" not in payload
    assert payload["workflow_id"] == "deploy_object_demo"


def test_workflow_deploy_rejects_unsupported_primitives(_isolated_workflow_home):
    _enable_workflow_toolset(_isolated_workflow_home)
    from tools.registry import registry

    definition = {
        "id": "deploy_unsupported_demo",
        "name": "Deploy Unsupported Demo",
        "version": 1,
        "triggers": [{"type": "manual"}],
        "nodes": {"start": {"type": "send_message", "output": {"text": "hi"}}},
    }

    payload = json.loads(registry.dispatch("workflow_deploy", {"definition": definition}))

    assert "error" in payload
    assert "unsupported node type: send_message on node start" in payload["error"]


def test_workflow_deploy_returns_deployed_version_not_latest(_isolated_workflow_home):
    _enable_workflow_toolset(_isolated_workflow_home)
    from hermes_cli import workflows_db as wfdb
    from tools.registry import registry

    workflow_id = "deploy_version_demo"

    def definition_text(version):
        return f"""
id: {workflow_id}
name: Deploy Version Demo
version: {version}
nodes:
  start:
    type: pass
""".lstrip()

    first = json.loads(registry.dispatch(
        "workflow_deploy",
        {"definition_text": definition_text(2), "created_by": "v2_creator"},
    ))
    assert "error" not in first

    second = json.loads(registry.dispatch(
        "workflow_deploy",
        {"definition_text": definition_text(1), "created_by": "v1_creator"},
    ))

    assert "error" not in second
    assert second["workflow_id"] == workflow_id
    assert second["id"] == workflow_id
    assert second["version"] == 1
    assert second["created_by"] == "v1_creator"

    with wfdb.connect() as conn:
        records = [
            record
            for record in wfdb.list_definitions(conn)
            if record.workflow_id == workflow_id
        ]
    assert {(record.version, record.created_by) for record in records} == {
        (1, "v1_creator"),
        (2, "v2_creator"),
    }


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


def test_workflow_run_accepts_input_object(_isolated_workflow_home):
    _enable_workflow_toolset(_isolated_workflow_home)
    spec = _deploy_demo_workflow()
    from tools.registry import registry

    payload = json.loads(registry.dispatch("workflow_run", {"workflow_id": spec.id, "input": {"x": 2}}))

    assert "error" not in payload
    assert payload["input"] == {"x": 2}


def test_workflow_run_accepts_null_input_as_empty_object(_isolated_workflow_home):
    _enable_workflow_toolset(_isolated_workflow_home)
    spec = _deploy_demo_workflow()
    from tools.registry import registry

    payload = json.loads(registry.dispatch("workflow_run", {"workflow_id": spec.id, "input": None}))

    assert "error" not in payload
    assert payload["input"] == {}


@pytest.mark.parametrize("bad_input", [[], "", 0, False])
def test_workflow_run_rejects_bad_input_object(_isolated_workflow_home, bad_input):
    _enable_workflow_toolset(_isolated_workflow_home)
    spec = _deploy_demo_workflow()
    from tools.registry import registry

    result = registry.dispatch("workflow_run", {"workflow_id": spec.id, "input": bad_input})
    payload = json.loads(result)

    assert "error" in payload
    assert "input must be an object" in payload["error"]


@pytest.mark.parametrize("input_json", ["{", "[]"])
def test_workflow_run_rejects_bad_input_json(_isolated_workflow_home, input_json):
    _enable_workflow_toolset(_isolated_workflow_home)
    spec = _deploy_demo_workflow()

    from tools.registry import registry

    result = registry.dispatch(
        "workflow_run",
        {"workflow_id": spec.id, "input_json": input_json},
    )
    payload = json.loads(result)

    assert "error" in payload
    assert payload["error"].startswith("workflow_run: ")


def test_workflow_cancel_is_idempotent(_isolated_workflow_home):
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

    first = json.loads(registry.dispatch("workflow_cancel", {"execution_id": execution_id}))
    second = json.loads(registry.dispatch("workflow_cancel", {"execution_id": execution_id}))

    assert "error" not in first
    assert first["status"] == "cancelled"
    assert first["cancelled"] is True
    assert "error" not in second
    assert second["status"] == "cancelled"
    assert second["cancelled"] is False

    with wfdb.connect() as conn:
        rows = conn.execute(
            """
            SELECT kind, payload_json FROM workflow_events
             WHERE execution_id = ? AND kind = 'execution_cancelled'
             ORDER BY id
            """,
            (execution_id,),
        ).fetchall()
    assert [row["kind"] for row in rows] == ["execution_cancelled"]
    assert [json.loads(row["payload_json"]) for row in rows] == [
        {"source": "workflow_cancel"}
    ]


def test_shared_cancel_execution_records_source(_isolated_workflow_home):
    spec = _deploy_demo_workflow()

    from hermes_cli import workflows_db as wfdb

    with wfdb.connect() as conn:
        execution_id = wfdb.start_execution(
            conn,
            spec.id,
            input_data={},
            trigger_type="manual",
        )

        execution, cancelled = wfdb.cancel_execution(conn, execution_id, source="unit")
        again, cancelled_again = wfdb.cancel_execution(conn, execution_id, source="unit")
        rows = conn.execute(
            """
            SELECT kind, payload_json FROM workflow_events
             WHERE execution_id = ? AND kind = 'execution_cancelled'
             ORDER BY id
            """,
            (execution_id,),
        ).fetchall()

    assert execution.status == "cancelled"
    assert cancelled is True
    assert again.status == "cancelled"
    assert cancelled_again is False
    assert [json.loads(row["payload_json"]) for row in rows] == [{"source": "unit"}]
