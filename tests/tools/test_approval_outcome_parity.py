"""Gateway-tail outcome parity + sudo human-wait exclusion (#85125 Phase 2e).

Closes the machine-readability residue of #81048: the _run_approval_gate
gateway tail now carries a structured ``outcome`` key (parity with its
check_all_command_guards / execute_code siblings), and the interactive
sudo-password wait is excluded from tool deadlines via human_wait_window()
on both executor paths (G4).
"""

from __future__ import annotations

import pytest

from tools import approval_context, approval_human_wait
import tools.terminal_tool as terminal_tool
import tools.terminal_tool_sudo as terminal_tool_sudo


@pytest.fixture(autouse=True)
def _clean_human_wait_state():
    with approval_human_wait._human_wait_lock:
        approval_human_wait._human_wait_states.clear()
    yield
    with approval_human_wait._human_wait_lock:
        approval_human_wait._human_wait_states.clear()


class TestSudoWaitExcludedFromDeadlines:
    """The interactive sudo-password wait accrues human-wait seconds, so it
    stops counting against tool deadlines on both executor paths."""

    def test_sudo_callback_wait_accrues_human_wait(self, monkeypatch):
        session = "sudo-test-session"
        monkeypatch.setattr(
            approval_context, "get_current_session_key", lambda default="": session
        )

        def _slow_cb():
            import time

            time.sleep(0.3)
            return "pw"

        monkeypatch.setattr(
            terminal_tool, "_get_sudo_password_callback", lambda: _slow_cb
        )
        before = approval_human_wait.human_wait_seconds(session)
        pw = terminal_tool_sudo._prompt_for_sudo_password(timeout_seconds=5)

        assert pw == "pw"
        after = approval_human_wait.human_wait_seconds(session)
        assert after > before, (
            f"sudo wait did not accrue human-wait time ({before} -> {after}); "
            "the wait still counts against tool deadlines"
        )



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
