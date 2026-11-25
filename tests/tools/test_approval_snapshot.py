"""Approval must cover the exact nested payload dispatched after the prompt."""


def test_mutation_during_approval_does_not_change_dispatched_payload(monkeypatch):
    import copy
    import model_tools
    from hermes_cli import plugins

    payload = {"nested": [{"target": "approved"}]}
    dispatched = []
    monkeypatch.setattr(plugins, "invoke_hook", lambda *a, **kw: [{"action": "approve"}])

    def approve(tool_name, reason, **kwargs):
        assert "approved" in reason
        payload["nested"][0]["target"] = "changed"
        payload["nested"].append({"target": "extra"})
        return {"approved": True}

    monkeypatch.setattr("tools.approval.request_tool_approval", approve)
    monkeypatch.setattr(model_tools.registry, "dispatch", lambda name, args, **kw: dispatched.append(copy.deepcopy(args)) or "ok")
    model_tools.handle_function_call("fixture_snapshot", payload, task_id="test", skip_tool_execution_middleware=True)
    assert dispatched == [{"nested": [{"target": "approved"}]}]
