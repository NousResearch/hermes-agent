"""Regression for #58663: stale cron state must not bypass interactive approval."""

from contextvars import Context

import pytest


@pytest.mark.parametrize("cron_mode", ["deny", "approve"])
@pytest.mark.parametrize("gateway_binding", ["session", "env"])
def test_stale_cron_env_keeps_gateway_approval(monkeypatch, tmp_path, cron_mode, gateway_binding):
    from gateway.session_context import set_session_vars
    from tools import approval, approval_context

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"approvals:\n  mode: manual\n  cron_mode: {cron_mode}\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    for name in ("HERMES_INTERACTIVE", "HERMES_EXEC_ASK", "HERMES_YOLO_MODE"):
        monkeypatch.delenv(name, raising=False)
    if gateway_binding == "env":
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")

    def run():
        if gateway_binding == "session":
            set_session_vars(platform="feishu", chat_id="marker-test")
        assert not approval_context._is_cron_approval_context()
        terminal = approval.check_dangerous_command("rm -rf /tmp/stuff", "local")
        code = approval.check_execute_code_guard("print('hello')", "local")
        for result in (terminal, code):
            assert not result.get("approved")
            assert "without a user present" not in result.get("message", "")
            assert result.get("status") in {"approval_required", "pending_approval"}

    Context().run(run)


@pytest.mark.parametrize("marker", ["1", ""])
def test_explicit_cron_marker_wins_over_gateway_identity(monkeypatch, marker):
    from gateway.session_context import set_session_vars
    from tools.approval_context import _is_cron_approval_context

    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")

    def run():
        set_session_vars(platform="telegram", cron_session=marker)
        assert _is_cron_approval_context() is bool(marker)

    Context().run(run)
