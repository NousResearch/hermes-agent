"""Regression tests for cron approval state isolation in long-lived gateways."""

from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from tools.approval import _is_cron_session, _is_gateway_approval_context


def test_gateway_context_masks_stale_process_cron_flag(monkeypatch):
    """A cron run must not make later Telegram turns look non-interactive."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    tokens = set_session_vars(platform="telegram", cron_session=False)
    try:
        assert _is_cron_session() is False
        assert _is_gateway_approval_context() is True
    finally:
        clear_session_vars(tokens)
        reset_session_vars()


def test_cron_context_is_task_local_without_process_env(monkeypatch):
    """Cron approval mode works without mutating process-global os.environ."""
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    tokens = set_session_vars(platform="", cron_session=True)
    try:
        assert _is_cron_session() is True
        assert _is_gateway_approval_context() is False
    finally:
        clear_session_vars(tokens)
        reset_session_vars()
