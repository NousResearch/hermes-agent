"""Company OS approval guard coverage for send_message_tool."""

import json
from types import SimpleNamespace

from tools import send_message_tool as smt


def test_schema_exposes_company_os_approval_id() -> None:
    properties = smt.SEND_MESSAGE_SCHEMA["parameters"]["properties"]

    assert "approval_id" in properties
    assert "Company OS" in properties["approval_id"]["description"]


def test_company_os_guard_requires_approval_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("COMPANY_OS_REQUIRE_APPROVAL", "1")

    result = smt._check_company_os_action_guard({}, "telegram", "12345")

    assert result is not None
    assert "COMPANY_OS_REQUIRE_APPROVAL" in result["error"]


def test_company_os_guard_blocks_when_action_check_fails(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, capture_output, text, timeout):
        calls.append(cmd)
        return SimpleNamespace(
            returncode=3,
            stdout="allowed: false\nreason: approval_pending\n",
            stderr="",
        )

    monkeypatch.setattr(smt.subprocess, "run", fake_run)

    result = smt._check_company_os_action_guard(
        {"approval_id": "approval-20260624-test"},
        "telegram",
        "12345",
    )

    assert result is not None
    assert "approval_pending" in result["error"]
    assert calls
    assert "--approval-id" in calls[0]
    assert "approval-20260624-test" in calls[0]
    assert "--target" in calls[0]
    assert "telegram:12345" in calls[0]
    assert "--record" in calls[0]


def test_company_os_guard_uses_session_route_approval(monkeypatch) -> None:
    from gateway.session_context import clear_company_os_route_context, set_company_os_route_context

    calls = []

    def fake_run(cmd, capture_output, text, timeout):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="allowed: true\n", stderr="")

    monkeypatch.setattr(smt.subprocess, "run", fake_run)
    set_company_os_route_context(approval_id="approval-session-route", hard_gate_required=True)
    try:
        result = smt._check_company_os_action_guard({}, "telegram", "12345")
    finally:
        clear_company_os_route_context()

    assert result is None
    assert calls
    assert "--approval-id" in calls[0]
    assert "approval-session-route" in calls[0]


def test_send_message_blocks_before_platform_send_when_guard_fails(monkeypatch) -> None:
    from gateway.config import Platform

    telegram_cfg = SimpleNamespace(enabled=True, token="***", extra={})
    config = SimpleNamespace(
        platforms={Platform.TELEGRAM: telegram_cfg},
        get_home_channel=lambda _platform: None,
    )
    called = {"send": False}

    async def fake_send_to_platform(*_args, **_kwargs):
        called["send"] = True
        return {"success": True}

    monkeypatch.setattr(
        smt,
        "_check_company_os_action_guard",
        lambda *_args, **_kwargs: {"error": "BLOCKED by Company OS approval guard"},
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    monkeypatch.setattr(smt, "_send_to_platform", fake_send_to_platform)

    result = json.loads(
        smt.send_message_tool(
            {
                "action": "send",
                "target": "telegram:12345",
                "message": "hello",
                "approval_id": "approval-20260624-test",
            }
        )
    )

    assert result["error"] == "BLOCKED by Company OS approval guard"
    assert called["send"] is False
