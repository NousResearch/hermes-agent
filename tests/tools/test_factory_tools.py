import json
import uuid

import model_tools


def _payload(result: str) -> dict:
    return json.loads(result)


def test_factory_toolset_exposes_factory_tools():
    tools = model_tools.get_tool_definitions(enabled_toolsets=["factory"], quiet_mode=True)
    names = {tool["function"]["name"] for tool in tools}

    assert "factory_project_create" in names
    assert "factory_lane_create" in names
    assert "factory_task_create" in names
    assert "factory_gate_record" in names
    assert "factory_status" in names


def test_factory_tools_create_project_status_and_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    project_id = f"tool-factory-test-{uuid.uuid4().hex[:8]}"

    class FakeBackend:
        def __init__(self):
            self.projects: dict[str, dict] = {}
            self.lanes: list[dict] = []
            self.tasks: list[dict] = []
            self.gates: list[dict] = []

        def create_project(self, name, **kwargs):
            pid = kwargs["project_id"]
            self.projects[pid] = {"project_id": pid, "name": name, "status": "intake"}
            self.lanes = [
                {"lane_id": f"{pid}-zeus", "project_id": pid, "methodology": "zeus_native"},
                {"lane_id": f"{pid}-bmad", "project_id": pid, "methodology": "bmad_hybrid"},
            ]
            return {"project_id": pid, "lanes": self.lanes}

        def create_task(self, project_id, title, **kwargs):
            task_id = f"{project_id}-prepare-implementation-plan"
            task = {"task_id": task_id, "project_id": project_id, "title": title, "status": "todo", **kwargs}
            self.tasks.append(task)
            return {"task_id": task_id, "project_id": project_id, "lane_id": kwargs.get("lane_id")}

        def record_gate(self, project_id, gate_type, status, **kwargs):
            gate = {"gate_id": 1, "project_id": project_id, "gate_type": gate_type, "status": status, **kwargs}
            self.gates.append(gate)
            return {"gate_id": 1, "project_id": project_id, "status": status}

        def status(self, project_id=None):
            projects = list(self.projects.values())
            if project_id:
                projects = [project for project in projects if project["project_id"] == project_id]
            return {
                "db_backend": "fake",
                "projects": projects,
                "lanes": [lane for lane in self.lanes if not project_id or lane["project_id"] == project_id],
                "tasks": [task for task in self.tasks if not project_id or task["project_id"] == project_id],
                "gates": [gate for gate in self.gates if not project_id or gate["project_id"] == project_id],
            }

    fake = FakeBackend()
    from hermes_cli import factory_backend

    monkeypatch.setattr(factory_backend, "get_backend", lambda: fake)

    created = _payload(
        model_tools.handle_function_call(
            "factory_project_create",
            {"name": "Tool Factory Test", "project_id": project_id},
        )
    )
    assert created["ok"] is True
    assert {lane["lane_id"] for lane in created["lanes"]} == {
        f"{project_id}-zeus",
        f"{project_id}-bmad",
    }

    task = _payload(
        model_tools.handle_function_call(
            "factory_task_create",
            {
                "project_id": project_id,
                "lane_id": f"{project_id}-zeus",
                "title": "Prepare implementation plan",
                "owner_agent_id": "implementation-planner",
                "reviewer_agent_id": "quality-reviewer",
                "acceptance_criteria": ["Plan is persisted"],
            },
        )
    )
    assert task["ok"] is True
    assert task["task_id"].startswith(f"{project_id}-prepare-implementation-plan")

    gate = _payload(
        model_tools.handle_function_call(
            "factory_gate_record",
            {
                "project_id": project_id,
                "task_id": task["task_id"],
                "gate_type": "planning",
                "status": "passed",
                "reviewer": "quality-reviewer",
                "notes": "Test evidence accepted",
            },
        )
    )
    assert gate["ok"] is True
    assert gate["status"] == "passed"

    status = _payload(model_tools.handle_function_call("factory_status", {"project_id": project_id}))
    assert status["ok"] is True
    assert [project["project_id"] for project in status["projects"]] == [project_id]
    assert [item["status"] for item in status["gates"]] == ["passed"]
