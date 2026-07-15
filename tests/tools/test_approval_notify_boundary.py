from __future__ import annotations


def test_typed_gateway_boundary_failure_reaches_model_without_text_classification(
    monkeypatch,
):
    from tools import approval as mod

    session_key = "typed-owner-escalation-boundary"
    model_message = (
        "BLOCKED: exact Canonical owner escalation was not receipted. "
        "Continue safe read-only work."
    )

    def fail_notify(_data):
        raise mod.ApprovalNotifyBoundaryError(
            "owner_route_back_not_sent",
            model_message,
        )

    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.setattr(
        mod,
        "_get_approval_config",
        lambda: {"mode": "manual", "gateway_timeout": 1},
    )
    mod.register_gateway_notify(session_key, fail_notify)
    token = mod.set_current_session_key(session_key)
    try:
        result = mod.check_all_command_guards("rm -rf .git", "local")
    finally:
        mod.reset_current_session_key(token)
        mod.unregister_gateway_notify(session_key)

    assert result["approved"] is False
    assert result["message"] == model_message
    assert result["outcome"] == "owner_route_back_not_sent"
    assert session_key not in mod._gateway_queues
