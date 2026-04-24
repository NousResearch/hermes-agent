import json

import pytest

from model_tools import get_tool_definitions, handle_function_call
from tools.registry import registry
from agent.task_store import TaskStatus, TaskStore
from tools.task_tool import TASK_SCHEMA, task_tool


@pytest.fixture
def task_store_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_TASK_STORE_DIR", str(tmp_path / "task-store"))
    return tmp_path / "task-store"


def _allow_mutation_kwargs(**extra):
    return {"can_mutate_tasks": True, "permission_granted": True, **extra}


class TestTaskToolRegistration:
    def test_registry_entry_and_schema(self):
        entry = registry.get_entry("task")
        assert entry is not None
        assert entry.toolset == "task"
        assert entry.schema == TASK_SCHEMA
        assert entry.schema["parameters"]["required"] == ["action"]
        assert "create" in entry.schema["parameters"]["properties"]["action"]["enum"]
        assert "runnable" in entry.schema["parameters"]["properties"]["action"]["enum"]

    def test_task_tool_is_available_in_default_cli_toolset(self):
        names = {
            definition["function"]["name"]
            for definition in get_tool_definitions(enabled_toolsets=["hermes-cli"], quiet_mode=True)
        }
        assert "task" in names


class TestTaskToolValidationAndPermissions:
    def test_create_rejects_invalid_metadata_shape(self, task_store_dir):
        result = json.loads(task_tool(action="create", goal="ship it", metadata="not-a-dict"))

        assert result["error"] == "Invalid task tool arguments"
        assert any("metadata" in detail for detail in result["details"])

    def test_default_direct_calls_do_not_allow_mutation(self, task_store_dir):
        result = json.loads(task_tool(action="create", goal="ship it"))

        assert result["error"] == "Task mutation requires explicit approval"
        assert result["code"] == "task_mutation_not_allowed"

    def test_read_only_agents_cannot_mutate_even_with_permission(self, task_store_dir):
        result = json.loads(
            task_tool(
                action="create",
                goal="ship it",
                read_only_agent=True,
                permission_granted=True,
                can_mutate_tasks=True,
            )
        )

        assert result["error"] == "Read-only agents cannot mutate persistent tasks"
        assert result["code"] == "task_mutation_read_only"


class TestTaskToolOperations:
    def test_create_get_and_list_redact_sensitive_launch_spec_fields(self, task_store_dir):
        created = json.loads(
            task_tool(
                action="create",
                goal="launch",
                metadata={"priority": "high"},
                launch_spec={
                    "command": "delegate",
                    "api_key": "top-secret",
                    "authorization": "Bearer secret",
                    "env": {
                        "OPENAI_API_KEY": "sk-live",
                        "SAFE_FLAG": "1",
                    },
                },
                **_allow_mutation_kwargs(),
            )
        )

        assert created["success"] is True
        task_id = created["task"]["id"]
        assert created["task"]["launch_spec"]["command"] == "delegate"
        assert created["task"]["launch_spec"]["api_key"] == "<redacted>"
        assert created["task"]["launch_spec"]["authorization"] == "<redacted>"
        assert created["task"]["launch_spec"]["env"]["OPENAI_API_KEY"] == "<redacted>"
        assert created["task"]["launch_spec"]["env"]["SAFE_FLAG"] == "1"

        store = TaskStore(root_dir=task_store_dir)
        persisted = store.require_task(task_id)
        assert persisted.launch_spec["api_key"] == "top-secret"
        assert persisted.launch_spec["env"]["OPENAI_API_KEY"] == "sk-live"

        fetched = json.loads(task_tool(action="get", task_id=task_id))
        listed = json.loads(task_tool(action="list"))
        assert fetched["task"]["launch_spec"]["api_key"] == "<redacted>"
        assert listed["tasks"][0]["launch_spec"]["api_key"] == "<redacted>"

    def test_update_metadata_and_dependency_management(self, task_store_dir):
        parent = json.loads(task_tool(action="create", goal="parent", **_allow_mutation_kwargs()))["task"]
        child = json.loads(task_tool(action="create", goal="child", **_allow_mutation_kwargs()))["task"]

        updated = json.loads(
            task_tool(
                action="update_metadata",
                task_id=child["id"],
                metadata={"lane": "A", "priority": 1},
                **_allow_mutation_kwargs(),
            )
        )
        assert updated["task"]["metadata"] == {"lane": "A", "priority": 1}

        dependent = json.loads(
            task_tool(
                action="add_dependency",
                task_id=child["id"],
                dependency_id=parent["id"],
                **_allow_mutation_kwargs(),
            )
        )
        assert dependent["task"]["blockedBy"] == [parent["id"]]

        store = TaskStore(root_dir=task_store_dir)
        assert store.require_task(parent["id"]).blocks == [child["id"]]

        removed = json.loads(
            task_tool(
                action="remove_dependency",
                task_id=child["id"],
                dependency_id=parent["id"],
                **_allow_mutation_kwargs(),
            )
        )
        assert removed["task"]["blockedBy"] == []
        assert store.require_task(parent["id"]).blocks == []

    def test_cancel_retry_reconcile_and_runnable(self, task_store_dir):
        created = json.loads(task_tool(action="create", goal="recoverable", **_allow_mutation_kwargs()))["task"]
        store = TaskStore(root_dir=task_store_dir)
        store.transition_task(created["id"], TaskStatus.queued)
        store.record_result(created["id"], status=TaskStatus.failed, error="boom")
        store.update_continuation(created["id"], status="pending")

        retried = json.loads(task_tool(action="retry", task_id=created["id"], **_allow_mutation_kwargs()))
        assert retried["task"]["execution"]["status"] == "draft"
        assert retried["task"]["execution"]["result"] is None
        assert retried["task"]["summary"] is None

        runnable = json.loads(task_tool(action="runnable"))
        runnable_ids = {task["id"] for task in runnable["tasks"]}
        assert created["id"] in runnable_ids

        active = store.create_task(goal="background")
        store.attach_process(active.id, process_session_id="sess-1", process_command="python worker.py")

        class FakeProcessRegistry:
            def poll(self, session_id):
                assert session_id == "sess-1"
                return {"status": "exited", "exit_code": 0, "output_preview": "done"}

        reconciled = json.loads(
            task_tool(
                action="reconcile",
                task_id=active.id,
                process_registry=FakeProcessRegistry(),
                **_allow_mutation_kwargs(),
            )
        )
        assert reconciled["task"]["execution"]["status"] == "completed"

        cancelled_task = store.create_task(goal="cancel me")
        cancelled = json.loads(
            task_tool(action="cancel", task_id=cancelled_task.id, **_allow_mutation_kwargs())
        )
        assert cancelled["task"]["execution"]["status"] == "cancelled"


class TestTaskToolModelDispatchPermissions:
    def test_handle_function_call_keeps_task_mutations_denied_by_default(self, task_store_dir):
        result = json.loads(handle_function_call("task", {"action": "create", "goal": "ship it"}))

        assert result["error"] == "Task mutation requires explicit approval"
        assert result["code"] == "task_mutation_not_allowed"

    def test_handle_function_call_can_forward_task_mutation_context(self, task_store_dir):
        result = json.loads(
            handle_function_call(
                "task",
                {"action": "create", "goal": "ship it"},
                can_mutate_tasks=True,
                permission_granted=True,
            )
        )

        assert result["success"] is True
        assert result["task"]["goal"] == "ship it"
