"""Native/messaging input must reach the caller's actual approval queue."""

import json

import pytest


@pytest.mark.parametrize("choice", ["deny", "once"])
def test_gateway_input_uses_caller_approval_before_backend(monkeypatch, choice):
    from tools import approval
    from tools.approval_context import reset_current_session_key, set_current_session_key
    from tools.computer_use import tool
    from tools.registry import registry
    import tools.computer_use_tool  # noqa: F401 — actual registry registration

    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval.approval_context, "_get_approval_mode", lambda: "manual")
    tool.set_approval_callback(None)
    backend = tool._NoopBackend()
    resolved = []
    def new_backend(permission_mode, **kwargs):
        resolved.append(kwargs)
        return backend
    # Fake only the spawn: the real cache admits the backend, so dispatch runs
    # under the same call-lock/revalidation path as production.
    tool.reset_backend_for_tests()
    monkeypatch.setattr(tool, "_new_backend", new_backend)
    caller = "approval-owner"
    observability_id = "different-conversation-id"
    prompted = []
    foreign = []

    def respond(data):
        assert not resolved and not backend.calls
        prompted.append(data)
        assert not approval.resolve_gateway_approval(
            observability_id, "once", request_id=data["request_id"])
        assert approval.resolve_gateway_approval(caller, choice, request_id=data["request_id"])

    approval.register_gateway_notify(caller, respond)
    approval.register_gateway_notify(observability_id, foreign.append)
    token = set_current_session_key(caller)
    try:
        raw = registry.dispatch(
            "computer_use", {"action": "click", "x": 1, "y": 1, "capture_after": False},
            session_id=observability_id, task_id="ordinary-parent-task")
        assert isinstance(raw, str)
        result = json.loads(raw)
        assert len(prompted) == 1, "input skipped the registered gateway approval queue"
        assert not foreign, "approval was routed by target/observability identity, not caller"
        if choice == "deny":
            assert result.get("error") and not resolved and not backend.calls
        else:
            assert result["ok"]
            assert [name for name, _ in backend.calls] == ["click"]
    finally:
        reset_current_session_key(token)
        tool.reset_backend_for_tests()
        for key in (caller, observability_id):
            approval.unregister_gateway_notify(key)
            approval.clear_session(key)


@pytest.mark.parametrize("case", ["scoped-grant", "timeout", "missing-notifier", "peer-yolo", "off", "capture"])
def test_gateway_input_preserves_approval_boundaries(monkeypatch, case):
    from tools import approval
    from tools.approval_context import reset_current_session_key, set_current_session_key
    from tools.computer_use import tool
    from tools.registry import registry
    import tools.computer_use_tool  # noqa: F401 — actual registry registration

    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval.approval_context, "_get_approval_mode", lambda: "off" if case == "off" else "manual")
    monkeypatch.setattr(approval.approval_context, "_get_approval_timeout", lambda: 0 if case == "timeout" else 2)
    tool.set_approval_callback(None)
    backend = tool._NoopBackend()
    tool.reset_backend_for_tests()
    monkeypatch.setattr(tool, "_new_backend", lambda permission_mode, **kwargs: backend)
    owner, peer = "input-owner", "input-peer"
    prompted = []

    def notify(key, data):
        prompted.append((key, data["pattern_key"]))
        if case != "timeout":
            choice = "session" if case == "scoped-grant" and len(prompted) == 1 else "deny"
            assert approval.resolve_gateway_approval(key, choice, request_id=data["request_id"])

    def dispatch(key, *, action="click", delivery_mode="background"):
        token = set_current_session_key(key)
        try:
            raw = registry.dispatch("computer_use", {
                "action": action, "x": 1, "y": 1, "delivery_mode": delivery_mode,
                "capture_after": False,
            }, session_id="observability-" + key)
            return json.loads(raw) if isinstance(raw, str) else raw
        finally:
            reset_current_session_key(token)

    try:
        if case != "missing-notifier":
            for key in (owner, peer):
                approval.register_gateway_notify(key, lambda data, key=key: notify(key, data))
        if case == "peer-yolo":
            approval.enable_session_yolo(peer)
        if case == "scoped-grant":
            assert dispatch(owner)["ok"]
            assert dispatch(owner)["ok"]
            assert dispatch(owner, delivery_mode="foreground")["error"]
            assert dispatch(peer)["error"]
            assert [key for key, _ in prompted] == [owner, owner, peer]
            assert prompted[0][1] != prompted[1][1]
            assert [name for name, _ in backend.calls] == ["click", "click"]
        else:
            result = dispatch(owner, action="capture" if case == "capture" else "click")
            if case in {"capture", "off"}:
                assert not result.get("error") and not prompted and backend.calls
            else:
                assert result["error"] and not backend.calls
                if case == "timeout":
                    assert result["outcome"] == "timeout" and len(prompted) == 1
                elif case == "missing-notifier":
                    assert result["status"] == "approval_required" and not prompted
                else:
                    assert len(prompted) == 1
    finally:
        tool.reset_backend_for_tests()
        for key in (owner, peer):
            approval.unregister_gateway_notify(key)
            approval.clear_session(key)
