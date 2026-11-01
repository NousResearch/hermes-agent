"""Inflight exemption for the browser inactivity janitor (tools/browser_session_lifecycle.py).

A browser_exec call's CDP traffic bypasses ``_session_last_activity``; without the inflight
registration the janitor kills a call longer than ``browser.inactivity_timeout`` mid-flight.
Invariants (independently derived, ported from f9203bf53a to the post-refactor session layer):
  1. a session with a RUNNING call is never reaped, however stale its activity clock;
  2. an idle session with no inflight call is still reaped;
  3. scope exit (normal or exceptional) clears the registration and touches activity.
"""

from unittest.mock import patch

import pytest

from tools import browser_session as bs
from tools import browser_session_lifecycle as lifecycle


class TestInflightJanitorExemption:
    def setup_method(self) -> None:
        self.orig_active = bs._active_sessions.copy()
        self.orig_last_activity = bs._session_last_activity.copy()
        self.orig_inflight = set(bs._session_inflight)
        self.orig_owner_homes = bs._session_owner_homes.copy()

    def teardown_method(self) -> None:
        bs._active_sessions.clear()
        bs._active_sessions.update(self.orig_active)
        bs._session_last_activity.clear()
        bs._session_last_activity.update(self.orig_last_activity)
        bs._session_inflight.clear()
        bs._session_inflight.update(self.orig_inflight)
        bs._session_owner_homes.clear()
        bs._session_owner_homes.update(self.orig_owner_homes)

    def test_inflight_session_not_reaped(self) -> None:
        stale = 1.0  # far past BROWSER_SESSION_INACTIVITY_TIMEOUT
        bs._session_last_activity["task-inflight"] = stale
        reaped: list[str] = []

        with lifecycle.inflight_session_scope("task-inflight"), patch.object(
            lifecycle, "_cleanup_one_inactive",
            side_effect=lambda key, elapsed: reaped.append(key)):
            lifecycle.cleanup_inactive_sessions()
            assert "task-inflight" in bs._session_inflight  # 期望: scope 内登记保持

        assert reaped == []  # 期望: RUNNING 会话即使时钟超时也不回收

    def test_idle_session_still_reaped(self) -> None:
        bs._session_last_activity["task-idle"] = 1.0
        reaped: list[str] = []

        with patch.object(lifecycle, "_cleanup_one_inactive",
                          side_effect=lambda key, elapsed: reaped.append(key)):
            lifecycle.cleanup_inactive_sessions()

        assert reaped == ["task-idle"]  # 期望: 无 inflight 登记的闲置会话照常回收

    def test_scope_exit_clears_registration_and_touches(self) -> None:
        with lifecycle.inflight_session_scope("task-done"):
            pass
        assert "task-done" not in bs._session_inflight  # 期望: 退出清登记
        assert bs._session_last_activity["task-done"] > 1.0  # 期望: 退出 touch 活跃时钟

    def test_scope_exit_on_exception_still_clears(self) -> None:
        with pytest.raises(RuntimeError):
            with lifecycle.inflight_session_scope("task-err"):
                raise RuntimeError("call failed mid-flight")
        assert "task-err" not in bs._session_inflight  # 期望: 异常路径也清登记, 无永生泄漏
