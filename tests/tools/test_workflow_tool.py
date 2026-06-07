import json


def test_workflow_validate_normalizes_phases(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.workflow_tool import workflow_tool

    result = json.loads(
        workflow_tool(
            action="validate",
            workflow={
                "name": "Audit API",
                "objective": "Find missing auth checks",
                "context": "Repo at /tmp/app",
                "phases": [
                    {
                        "title": "Explore",
                        "type": "fanout",
                        "toolsets": ["terminal", "file"],
                        "tasks": ["Inspect routes", {"goal": "Inspect middleware", "context": "Focus auth"}],
                    },
                    {"title": "Synthesize", "type": "synthesize"},
                ],
            },
        )
    )

    assert result["ok"] is True
    assert result["workflow"]["name"] == "audit-api"
    assert result["workflow"]["phases"][0]["tasks"][0]["toolsets"] == ["terminal", "file"]
    assert result["workflow"]["phases"][1]["tasks"][0]["goal"].startswith("Synthesize")


def test_workflow_run_persists_and_delegates(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = []

    def fake_delegate_task(**kwargs):
        calls.append(kwargs)
        if kwargs.get("tasks"):
            return json.dumps([{"summary": task["goal"]} for task in kwargs["tasks"]])
        return json.dumps({"summary": kwargs["goal"]})

    monkeypatch.setattr("tools.delegate_tool.delegate_task", fake_delegate_task)

    from tools.workflow_tool import workflow_tool

    result = json.loads(
        workflow_tool(
            action="run",
            workflow={
                "name": "Two Phase",
                "objective": "Exercise delegation",
                "phases": [
                    {"title": "Explore", "tasks": ["A", "B"]},
                    {"title": "Gate", "type": "gate", "tasks": [{"goal": "Verify", "toolsets": ["terminal"]}]},
                ],
            },
            parent_agent=object(),
        )
    )

    assert result["ok"] is True
    assert result["workflow_name"] == "two-phase"
    assert len(result["phase_results"]) == 2
    assert calls[0]["tasks"][0]["goal"] == "A"
    assert calls[1]["goal"] == "Verify"
    assert (tmp_path / "workflows" / "runs.db").exists()


def test_workflow_save_and_list(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.workflow_tool import workflow_tool

    saved = json.loads(
        workflow_tool(
            action="save",
            workflow={
                "name": "Reusable Review",
                "objective": "Review branch",
                "phases": [{"title": "Review", "type": "review", "tasks": ["Review diff"]}],
            },
        )
    )
    assert saved["ok"] is True
    assert saved["saved_path"].endswith("reusable-review.json")

    listed = json.loads(workflow_tool(action="list"))
    assert "reusable-review.json" in listed["saved_workflows"]
