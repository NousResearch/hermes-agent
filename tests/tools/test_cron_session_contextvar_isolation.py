"""Regression tests for CRON-1 — per-task isolation of the cron-session flag.

Before the fix, cron/scheduler set ``os.environ["HERMES_CRON_SESSION"]="1"``
process-globally and never cleared it, so once the in-process scheduler ran a
single job, every subsequent live-chat turn in the same process inherited
cron's dangerous-command approval bypass. The flag now lives in a ContextVar
bound to the cron job's context (with the env var kept as a CLI/test fallback).
"""

import contextvars
import threading

import tools.approval as approval


def test_in_cron_session_default_false(monkeypatch):
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    assert approval._in_cron_session() is False


def test_set_and_reset_cron_session(monkeypatch):
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    token = approval.set_cron_session()
    try:
        assert approval._in_cron_session() is True
    finally:
        approval.reset_cron_session(token)
    assert approval._in_cron_session() is False


def test_env_var_fallback_preserved(monkeypatch):
    """CLI/test callers that set the env var directly still work (back-compat)."""
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    assert approval._in_cron_session() is False
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    assert approval._in_cron_session() is True


def test_flag_confined_to_job_context(monkeypatch):
    """A flag set inside one copy_context() (a cron job) must not leak to the
    context that ran it — the core CRON-1 isolation property."""
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)

    def _job():
        token = approval.set_cron_session()
        try:
            return approval._in_cron_session()
        finally:
            approval.reset_cron_session(token)

    inside = contextvars.copy_context().run(_job)
    assert inside is True
    # The outer context never observes the cron job's flag.
    assert approval._in_cron_session() is False


def test_cron_flag_does_not_leak_to_concurrent_live_thread(monkeypatch):
    """A cron job thread sets the flag; a concurrent live-chat thread running at
    the same time must NOT observe it (this is the bug the fix closes)."""
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    seen = {}
    both_running = threading.Barrier(2)

    def cron_thread():
        token = approval.set_cron_session()
        try:
            both_running.wait(timeout=5)
            seen["cron"] = approval._in_cron_session()
        finally:
            approval.reset_cron_session(token)

    def live_thread():
        both_running.wait(timeout=5)
        seen["live"] = approval._in_cron_session()

    a = threading.Thread(target=cron_thread)
    b = threading.Thread(target=live_thread)
    a.start()
    b.start()
    a.join()
    b.join()

    assert seen["cron"] is True   # the cron thread sees its own flag
    assert seen["live"] is False  # the concurrent live-chat thread does not
