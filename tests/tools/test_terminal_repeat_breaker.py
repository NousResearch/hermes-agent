"""Regression tests: repeated-identical-call circuit breaker for terminal_tool.

Bug (#69256): tools/file_tools.py has a repeated-identical-call breaker for
reads (:1381-1397) and searches (:1857-1895, ``_read_tracker``) that stops the
model looping on a call whose result cannot change. tools/terminal_tool.py had
no equivalent: when a command is hardline-blocked (``tools/approval.py``
``detect_hardline_command`` / ``_hardline_block_result``, ``hardline: True`` --
this covers both the catastrophic-pattern blocklist AND the malformed-payload
guard, ``_MALFORMED_EXEC_DESCRIPTION``), the block is deterministic: an
identical retry has a 0% chance of succeeding. Nothing stopped the model
re-emitting the exact same blocked command until the turn cap, degrading the
session to empty responses.

Design (kept deliberately simple after two failed "consecutive"-streak cuts
that a Sol xhigh review picked apart):

- The breaker counts, per conversation, how many times each exact
  (command, block message) pair has been hardline-blocked -- a plain
  cumulative tally, NOT a consecutive streak. A streak needs a reset on every
  other terminal outcome and the many early-return paths kept leaving stale
  state; a cumulative tally is only ever touched on a hardline block, so
  nothing can corrupt it (``test_cumulative_count_survives_intervening_call``).
- ONE reason-neutral escalation for every hardline reason -- no attempt to
  branch malformed-vs-catastrophic on a message substring (which
  misclassifies executable commands that trip the malformed guard first) and
  no advice that coaches around an unconditional safety block
  (``test_escalation_is_reason_neutral``).
- Keyed on the raw (command, block message); a call with neither task_id nor
  session_id is left untracked; task/session keys are tagged so they can't
  collide; tracker state is bounded on both axes.
"""
import json

import pytest

import tools.terminal_tool as tt
from tools.terminal_tool import terminal_tool


ESCALATION_MARKER = "is deterministic"
# Phrases that would coach a bypass of an unconditional safety block -- the
# escalation must contain NONE of these (Sol HIGH #2).
BYPASS_COACHING = ["without pipes", "own terminal", "redirect output", "command substitution"]

_HARDLINE_CATASTROPHIC = {
    "approved": False,
    "hardline": True,
    "message": "BLOCKED (hardline): recursive delete of root filesystem. This command is on the unconditional blocklist and cannot be executed via the agent. If you genuinely need to run it, run it yourself in a terminal outside the agent.",
}

_HARDLINE_MALFORMED = {
    "approved": False,
    "hardline": True,
    "message": "BLOCKED (hardline): command parser limit or malformed executable payload. This command is on the unconditional blocklist.",
}

_APPROVABLE_BLOCK = {
    "approved": False,
    "status": "blocked",
    "description": "risky but not hardline",
    "message": "BLOCKED: risky command. Use the approval prompt to allow it, or rephrase the command.",
}

_APPROVED = {"approved": True}


def _make_env_config(**overrides):
    config = {
        "env_type": "local",
        "timeout": 180,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }
    config.update(overrides)
    return config


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch):
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config", lambda *a, **k: _make_env_config()
    )
    monkeypatch.setattr("tools.terminal_tool._start_cleanup_thread", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _clean_tracker():
    """The repeat tracker is module-global; start each test from empty."""
    tt._terminal_repeat_tracker.clear()
    yield
    tt._terminal_repeat_tracker.clear()


def _run(monkeypatch, guard_result, command, **kwargs):
    monkeypatch.setattr(
        "tools.terminal_tool._check_all_guards", lambda *a, **k: guard_result
    )
    return json.loads(terminal_tool(command=command, **kwargs))


class TestTerminalRepeatBreaker:
    def test_third_identical_hardline_block_escalates(self, monkeypatch):
        cmd = "rm -rf /"
        r1 = _run(monkeypatch, _HARDLINE_CATASTROPHIC, cmd, task_id="t1")
        r2 = _run(monkeypatch, _HARDLINE_CATASTROPHIC, cmd, task_id="t1")
        r3 = _run(monkeypatch, _HARDLINE_CATASTROPHIC, cmd, task_id="t1")
        assert ESCALATION_MARKER not in r1["error"], r1
        assert ESCALATION_MARKER not in r2["error"], r2
        assert ESCALATION_MARKER in r3["error"], r3
        # The breaker augments the hardline message, never replaces it.
        assert "recursive delete of root filesystem" in r3["error"], r3

    def test_escalation_is_reason_neutral(self, monkeypatch):
        """The single escalation must never coach a bypass -- for a catastrophic
        block that would be dangerous, and the same text is used for the
        malformed-payload block too (no substring classification)."""
        for guard in (_HARDLINE_CATASTROPHIC, _HARDLINE_MALFORMED):
            tt._terminal_repeat_tracker.clear()
            cmd = "grep -P ; rm -rf /"
            r = None
            for _ in range(3):
                r = _run(monkeypatch, guard, cmd, task_id="t-neutral")
            assert ESCALATION_MARKER in r["error"], r
            lowered = r["error"].lower()
            for phrase in BYPASS_COACHING:
                assert phrase not in lowered, (phrase, r["error"])
            assert "different approach" in lowered, r

    def test_cumulative_count_survives_intervening_call(self, monkeypatch):
        """The key property of the cumulative (non-streak) design: an approved
        command between two blocks does NOT reset the tally, and a
        non-hardline block between them doesn't either. Third block of the
        exact command still escalates."""
        cmd = "rm -rf /"
        r1 = _run(monkeypatch, _HARDLINE_CATASTROPHIC, cmd, task_id="t2")
        # intervening approved command (proceeds toward execution)
        _run(monkeypatch, _APPROVED, "echo hi", task_id="t2")
        # intervening non-hardline block
        _run(monkeypatch, _APPROVABLE_BLOCK, "sudo something", task_id="t2")
        r2 = _run(monkeypatch, _HARDLINE_CATASTROPHIC, cmd, task_id="t2")
        r3 = _run(monkeypatch, _HARDLINE_CATASTROPHIC, cmd, task_id="t2")
        assert ESCALATION_MARKER not in r1["error"]
        assert ESCALATION_MARKER not in r2["error"]
        assert ESCALATION_MARKER in r3["error"], r3

    def test_different_command_has_separate_count(self, monkeypatch):
        """Each (command, block) pair is counted independently -- a second
        blocked command doesn't inflate the first one's tally."""
        a = "rm -rf /"
        b = "mkfs /dev/sda"
        _run(monkeypatch, _HARDLINE_CATASTROPHIC, a, task_id="t3")
        rb = _run(monkeypatch, _HARDLINE_CATASTROPHIC, b, task_id="t3")
        _run(monkeypatch, _HARDLINE_CATASTROPHIC, a, task_id="t3")
        ra = _run(monkeypatch, _HARDLINE_CATASTROPHIC, a, task_id="t3")
        assert ESCALATION_MARKER not in rb["error"], rb  # b only hit once
        assert ESCALATION_MARKER in ra["error"], ra       # a hit three times

    def test_non_hardline_block_never_escalates(self, monkeypatch):
        cmd = "some risky command"
        last = None
        for _ in range(6):
            last = _run(monkeypatch, _APPROVABLE_BLOCK, cmd, task_id="t4")
        assert ESCALATION_MARKER not in last["error"], last

    def test_no_task_or_session_id_skips_tracking(self, monkeypatch):
        cmd = "rm -rf /"
        last = None
        for _ in range(5):
            last = _run(monkeypatch, _HARDLINE_CATASTROPHIC, cmd)
        assert ESCALATION_MARKER not in last["error"], last
        assert not tt._terminal_repeat_tracker  # nothing tracked

    def test_session_id_used_when_no_task_id(self, monkeypatch):
        cmd = "rm -rf /"
        r1 = _run(monkeypatch, _HARDLINE_CATASTROPHIC, cmd, session_id="s1")
        r2 = _run(monkeypatch, _HARDLINE_CATASTROPHIC, cmd, session_id="s1")
        r3 = _run(monkeypatch, _HARDLINE_CATASTROPHIC, cmd, session_id="s1")
        assert ESCALATION_MARKER not in r1["error"]
        assert ESCALATION_MARKER in r3["error"], r3

    def test_task_and_session_keys_do_not_collide(self, monkeypatch):
        """A task_id and another conversation's session_id with the same raw
        string must not share a tally (tagged keys, Sol MEDIUM #3)."""
        cmd = "rm -rf /"
        _run(monkeypatch, _HARDLINE_CATASTROPHIC, cmd, task_id="X")
        _run(monkeypatch, _HARDLINE_CATASTROPHIC, cmd, task_id="X")
        # A different conversation whose session_id happens to equal "X":
        r = _run(monkeypatch, _HARDLINE_CATASTROPHIC, cmd, session_id="X")
        assert ESCALATION_MARKER not in r["error"], r  # its own count is 1

    def test_escalated_response_still_denies_execution(self, monkeypatch):
        cmd = "rm -rf /"
        r = None
        for _ in range(3):
            r = _run(monkeypatch, _HARDLINE_CATASTROPHIC, cmd, task_id="t5")
        assert ESCALATION_MARKER in r["error"], r
        assert r["status"] == "blocked", r
        assert r["exit_code"] == -1, r
        assert r["output"] == "", r


class TestTrackerBounds:
    def test_task_keys_bounded_fifo(self):
        tt._terminal_repeat_tracker.clear()
        cap = tt._TERMINAL_REPEAT_TRACKER_MAX_TASKS
        for i in range(cap + 10):
            tt._track_hardline_repeat("t:%d" % i, "cmd", "msg")
        assert len(tt._terminal_repeat_tracker) == cap
        assert "t:0" not in tt._terminal_repeat_tracker
        assert "t:%d" % (cap + 9) in tt._terminal_repeat_tracker

    def test_per_task_keys_bounded_fifo(self):
        tt._terminal_repeat_tracker.clear()
        cap = tt._TERMINAL_REPEAT_MAX_KEYS_PER_TASK
        for i in range(cap + 10):
            tt._track_hardline_repeat("t:solo", "cmd-%d" % i, "msg")
        assert len(tt._terminal_repeat_tracker["t:solo"]) == cap

    def test_none_task_key_is_noop(self):
        tt._terminal_repeat_tracker.clear()
        assert tt._track_hardline_repeat(None, "cmd", "msg") == 0
        assert not tt._terminal_repeat_tracker
