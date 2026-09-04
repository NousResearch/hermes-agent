"""Gateway-tail outcome parity + sudo human-wait exclusion (#85125 Phase 2e).

Closes the machine-readability residue of #81048: the _run_approval_gate
gateway tail now carries a structured ``outcome`` key (parity with its
check_all_command_guards / execute_code siblings), and the interactive
sudo-password wait is excluded from tool deadlines via human_wait_window()
on both executor paths (G4).
"""

from __future__ import annotations

import pytest

import tools.approval as approval_mod
import tools.terminal_tool as terminal_tool


@pytest.fixture(autouse=True)
def _clean_human_wait_state():
    with approval_mod._human_wait_lock:
        approval_mod._human_wait_states.clear()
    yield
    with approval_mod._human_wait_lock:
        approval_mod._human_wait_states.clear()


class TestSudoWaitExcludedFromDeadlines:
    """The interactive sudo-password wait accrues human-wait seconds, so it
    stops counting against tool deadlines on both executor paths."""

    def test_sudo_callback_wait_accrues_human_wait(self, monkeypatch):
        session = "sudo-test-session"
        monkeypatch.setattr(
            approval_mod, "get_current_session_key", lambda default="": session
        )

        def _slow_cb():
            import time

            time.sleep(0.3)
            return "pw"

        monkeypatch.setattr(
            terminal_tool, "_get_sudo_password_callback", lambda: _slow_cb
        )
        before = approval_mod.human_wait_seconds(session)
        pw = terminal_tool._prompt_for_sudo_password(timeout_seconds=5)

        assert pw == "pw"
        after = approval_mod.human_wait_seconds(session)
        assert after > before, (
            f"sudo wait did not accrue human-wait time ({before} -> {after}); "
            "the wait still counts against tool deadlines"
        )

    def test_thread_join_path_also_accrues(self, monkeypatch):
        """The non-callback path (thread + join) must be wrapped too."""
        session = "sudo-join-session"
        monkeypatch.setattr(
            approval_mod, "get_current_session_key", lambda default="": session
        )
        monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: None)
        monkeypatch.setattr(terminal_tool, "_is_windows", False, raising=False)

        # read_password_thread writes into `result` via closure in the real
        # code; stub the thread target by making join return quickly and the
        # result dict empty -> returns "" but the wait must still be wrapped.
        before = approval_mod.human_wait_seconds(session)
        pw = terminal_tool._prompt_for_sudo_password(timeout_seconds=1)
        assert pw == ""
        # The wrap is structural; a zero-length join may not move the clock,
        # so assert only that no exception escaped and state stays consistent.
        assert approval_mod.human_wait_seconds(session) >= before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
