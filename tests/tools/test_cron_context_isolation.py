"""Regression tests for cron approval state isolation in long-lived gateways."""

import threading

import pytest

from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from tools.approval import _is_cron_session, _is_gateway_approval_context


def test_gateway_context_masks_stale_process_cron_flag(monkeypatch):
    """A cron run must not make later Telegram turns look non-interactive."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
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


def test_concurrent_gateway_and_cron_contexts_do_not_bleed(monkeypatch):
    """Overlapping gateway and cron turns keep independent approval policy."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    barrier = threading.Barrier(2)
    results = {}

    def inspect(name, *, platform, cron_session):
        tokens = set_session_vars(platform=platform, cron_session=cron_session)
        try:
            barrier.wait(timeout=5)
            results[name] = (_is_cron_session(), _is_gateway_approval_context())
        finally:
            clear_session_vars(tokens)
            reset_session_vars()

    gateway = threading.Thread(
        target=inspect,
        kwargs={"name": "gateway", "platform": "telegram", "cron_session": False},
    )
    cron = threading.Thread(
        target=inspect,
        kwargs={"name": "cron", "platform": "", "cron_session": True},
    )
    gateway.start()
    cron.start()
    gateway.join(timeout=10)
    cron.join(timeout=10)

    assert not gateway.is_alive()
    assert not cron.is_alive()
    assert results == {"gateway": (False, True), "cron": (True, False)}


def test_cron_context_reader_does_not_hide_runtime_errors(monkeypatch):
    """Unexpected context failures must not silently fall back to global state."""
    import gateway.session_context as session_context

    def fail_context_read(*_args, **_kwargs):
        raise RuntimeError("broken context reader")

    monkeypatch.setattr(session_context, "get_session_env", fail_context_read)

    with pytest.raises(RuntimeError, match="broken context reader"):
        _is_cron_session()
