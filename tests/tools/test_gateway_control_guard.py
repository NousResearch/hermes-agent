from tools.gateway_control_guard import (
    approve_gateway_control_command,
    check_gateway_control_guard,
    clear_gateway_control_approvals,
)


def teardown_function():
    clear_gateway_control_approvals()


def test_cron_blocks_gateway_lifecycle_command(monkeypatch):
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")

    decision = check_gateway_control_guard(
        "systemctl --user restart hermes-gateway.service",
        "session-a",
    )

    assert decision.is_gateway_control is True
    assert decision.approved is False
    assert "not allowed from cron" in (decision.message or "")


def test_exact_session_approval_allows_matching_gateway_command(monkeypatch):
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    command = "systemctl --user restart hermes-gateway.service"

    blocked = check_gateway_control_guard(command, "session-a")
    approve_gateway_control_command("session-a", command)
    approved = check_gateway_control_guard(command, "session-a")
    other = check_gateway_control_guard(command + " --no-block", "session-a")

    assert blocked.is_gateway_control is True
    assert blocked.approved is False
    assert approved.approved is True
    assert other.approved is False


def test_readonly_gateway_status_is_not_blocked():
    decision = check_gateway_control_guard(
        "systemctl --user status hermes-gateway.service",
        "session-a",
    )

    assert decision.is_gateway_control is False
    assert decision.approved is True
