"""Regression tests for #56771: HERMES_CRON_SESSION env var leaks from the
scheduler process into interactive gateway sessions, blocking execute_code
and dangerous commands for users who never ran a cron job in their chat.

The fix replaces the process-global ``os.environ["HERMES_CRON_SESSION"]``
set with a per-job ``ContextVar`` so the cron flag is task-local and cannot
leak into concurrent interactive sessions.
"""

from __future__ import annotations

import os
import threading

import pytest

from tools import approval as A
from tools.approval import check_execute_code_guard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_cron_contextvar():
    """Ensure the cron ContextVar is cleared between tests."""
    try:
        from gateway.session_context import clear_cron_session
        clear_cron_session()
    except ImportError:
        pass
    yield
    try:
        from gateway.session_context import clear_cron_session
        clear_cron_session()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 1. is_cron_session() — contextvar-based cron detection
# ---------------------------------------------------------------------------

class TestCronSessionContextVar:
    def test_false_by_default(self, monkeypatch):
        """Without a cron contextvar or env var, is_cron_session() is False."""
        from gateway.session_context import is_cron_session
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        assert is_cron_session() is False

    def test_true_when_contextvar_set(self, monkeypatch):
        """Setting the cron contextvar makes is_cron_session() True."""
        from gateway.session_context import set_cron_session, is_cron_session
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        set_cron_session(True)
        assert is_cron_session() is True

    def test_contextvar_overrides_leaked_env(self, monkeypatch):
        """Even if HERMES_CRON_SESSION leaked into env, an explicitly-cleared
        contextvar means NOT cron (interactive session in same process)."""
        from gateway.session_context import set_cron_session, is_cron_session
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        set_cron_session(False)
        assert is_cron_session() is False

    def test_env_fallback_when_contextvar_unset(self, monkeypatch):
        """Backward compat: env var still works when contextvar was never set
        (tests, CLI cron that sets the env var directly)."""
        from gateway.session_context import is_cron_session
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        assert is_cron_session() is True


# ---------------------------------------------------------------------------
# 2. Thread isolation — the core fix
# ---------------------------------------------------------------------------

class TestCronContextVarThreadIsolation:
    def test_scheduler_thread_does_not_leak_to_gateway_thread(self, monkeypatch):
        """The cron ContextVar set in the scheduler thread must NOT be visible
        in a concurrent gateway/interactive thread. This is the core mechanism
        that prevents #56771 — os.environ leaks across threads, ContextVars do
        not."""
        from gateway.session_context import set_cron_session, is_cron_session

        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        set_cron_session(True)
        assert is_cron_session() is True  # scheduler thread sees it

        seen: dict = {}

        def gateway_handler():
            # Fresh thread → ContextVar at default (_UNSET) → env fallback
            # → env not set (scheduler no longer sets it) → False
            seen["is_cron"] = is_cron_session()

        t = threading.Thread(target=gateway_handler)
        t.start()
        t.join(timeout=5)

        assert seen["is_cron"] is False

    def test_env_var_does_leak_across_threads(self, monkeypatch):
        """Documents the pre-fix bug: os.environ IS visible across threads."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        seen: dict = {}

        def handler():
            seen["env"] = os.environ.get("HERMES_CRON_SESSION")

        t = threading.Thread(target=handler)
        t.start()
        t.join(timeout=5)

        assert seen["env"] == "1"  # process-global env leaks — this is the bug


# ---------------------------------------------------------------------------
# 3. check_execute_code_guard — cron blocks, interactive doesn't
# ---------------------------------------------------------------------------

class TestExecuteCodeGuardCronContextVar:
    def _setup(self, monkeypatch):
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.setattr(A, "_get_approval_mode", lambda: "manual")
        monkeypatch.setattr(A, "_get_cron_approval_mode", lambda: "deny")

    def test_cron_contextvar_blocks_execute_code(self, monkeypatch):
        """When cron ContextVar is set, execute_code is blocked (real cron job)."""
        from gateway.session_context import set_cron_session
        self._setup(monkeypatch)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        set_cron_session(True)

        result = check_execute_code_guard("print('hi')", "local")
        assert result["approved"] is False
        assert "BLOCKED" in result["message"]

    def test_interactive_not_blocked_without_cron_contextvar(self, monkeypatch):
        """Interactive session (no cron ContextVar, no leaked env) can run
        execute_code."""
        self._setup(monkeypatch)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)

        result = check_execute_code_guard("print('hi')", "local")
        # Headless local non-gateway non-cron → approved (existing contract)
        assert result["approved"] is True

    def test_interactive_thread_not_blocked_after_scheduler_ran(self, monkeypatch):
        """Regression for #56771: after the scheduler thread sets the cron
        ContextVar, a concurrent interactive thread should still be able to
        run execute_code."""
        from gateway.session_context import set_cron_session
        self._setup(monkeypatch)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)

        # Scheduler thread sets the cron flag
        set_cron_session(True)

        seen: dict = {}

        def interactive_session():
            # Fresh thread: ContextVar _UNSET, env not set → not cron
            result = check_execute_code_guard("print('hi')", "local")
            seen["approved"] = result["approved"]

        t = threading.Thread(target=interactive_session)
        t.start()
        t.join(timeout=5)

        assert seen["approved"] is True


# ---------------------------------------------------------------------------
# 4. _is_gateway_approval_context — cron contextvar short-circuits gateway
# ---------------------------------------------------------------------------

class TestGatewayContextCronShortCircuit:
    def test_cron_contextvar_returns_false_for_gateway(self, monkeypatch):
        """When cron ContextVar is set, _is_gateway_approval_context() is False
        even if HERMES_GATEWAY_SESSION is also set (cron takes precedence)."""
        from gateway.session_context import set_cron_session
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        set_cron_session(True)

        assert A._is_gateway_approval_context() is False

    def test_interactive_still_gateway_without_cron(self, monkeypatch):
        """Interactive gateway session (no cron ContextVar) is recognized as
        gateway context."""
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")

        assert A._is_gateway_approval_context() is True
