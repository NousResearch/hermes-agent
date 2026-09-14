"""Worker belt: a skill slash that only parks `_pending_input` must not be
ok-replied with the Loading banner. The slash worker has no REPL drain, so
returning that banner lets clients treat the exec as done and drop the prompt.

#107387
"""

from __future__ import annotations

import queue

import pytest


class _SkillQueueCLI:
    def __init__(self):
        self._pending_input = queue.Queue()
        self.console = None

    def process_command(self, cmd):
        print("\n⚡ Loading skill: grilling")
        self._pending_input.put("EXPANDED_SKILL_SCAFFOLD")


def test_slash_worker_refuses_orphaned_skill_queue():
    from tui_gateway import slash_worker

    cli = _SkillQueueCLI()
    # CURRENT MAIN: this succeeds and returns a banner string — THAT is the bug.
    # AFTER FIX: must raise, and must not return the banner as ok output.
    with pytest.raises(Exception) as raised:
        slash_worker._run(cli, "/grilling")
    assert "skill command" in str(raised.value)
    # Drain-then-raise: persistent worker must still serve a later non-skill command.
    assert cli._pending_input.empty()


class _PlainCLI:
    console = None

    def process_command(self, cmd: str) -> None:
        print("ok-plain")


def test_slash_worker_non_skill_still_returns_stdout():
    from tui_gateway import slash_worker

    assert slash_worker._run(_PlainCLI(), "/status") == "ok-plain"


def test_slash_worker_skill_orphan_does_not_poison_next_command():
    from tui_gateway import slash_worker

    class _Both:
        def __init__(self):
            self._pending_input = queue.Queue()
            self.console = None
            self.n = 0

        def process_command(self, cmd):
            self.n += 1
            if self.n == 1:
                print("\n⚡ Loading skill: grilling")
                self._pending_input.put("SCAFFOLD")
            else:
                print("later-ok")

    cli = _Both()
    with pytest.raises(Exception):
        slash_worker._run(cli, "/grilling")
    assert slash_worker._run(cli, "/status") == "later-ok"
