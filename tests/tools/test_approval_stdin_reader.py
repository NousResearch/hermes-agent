"""Tests for the armed stdin reader behind ``_read_choice``.

A timed-out approval prompt used to abandon a thread parked inside
``input()``; the next prompt spawned a second reader and both raced for the
same line, so a late keystroke could be eaten by the orphan and the live
prompt would record a spurious timeout. One reader now owns stdin, armed one
line at a time and only while a prompt is live: between prompts it parks on
the arming condition instead of inside input(), so other input() call sites
keep working. A prompt accepts only lines stamped after its freshness cutoff,
which is taken before the tty typeahead flush.
"""

from __future__ import annotations

import queue
import sys
import threading
import time

import pytest

import tools.approval_prompt as ap


@pytest.fixture
def fake_stdin(monkeypatch):
    """Replace stdin with a feed-controlled fake and reset the reader state.

    ``readline`` blocks until the test feeds a line; feeding ``""`` is EOF.
    The armed reader is shut down at teardown so no thread leaks into the next
    test holding the shared arming condition.
    """
    feed = queue.Queue()
    feed.isatty_value = False
    live = {"on": True}

    class FakeStdin:
        def readline(self, *args, **kwargs):
            # After teardown the fake is dead: return EOF instead of blocking.
            return feed.get() if live["on"] else ""

        def isatty(self):
            return feed.isatty_value

        def fileno(self):
            return 0

    monkeypatch.setattr(sys, "stdin", FakeStdin())
    monkeypatch.setattr(ap, "_stdin_lines", queue.Queue())
    monkeypatch.setattr(ap, "_stdin_eof", False)
    monkeypatch.setattr(ap, "_stdin_want_lines", 0)
    monkeypatch.setattr(ap, "_stdin_shutdown", False)
    monkeypatch.setattr(ap, "_stdin_reader_thread", None)
    try:
        yield feed
    finally:
        live["on"] = False
        ap._stdin_shutdown = True
        with ap._stdin_state:
            ap._stdin_state.notify_all()
        # Release readers blocked in input() ("" reads as EOF) and readers
        # parked on the arming condition, then wait for a clean exit.
        deadline = time.monotonic() + 10
        while _readers() and time.monotonic() < deadline:
            for _ in _readers():
                feed.put("")
            for t in _readers():
                t.join(timeout=1)


def _readers():
    return [t for t in threading.enumerate() if t.name == ap._STDIN_READER_NAME]


def test_read_choice_returns_the_answer(fake_stdin, capsys):
    threading.Timer(0.2, fake_stdin.put, args=("A\n",)).start()
    assert ap._read_choice("approve? ", 5) == "a"
    assert "approve? " in capsys.readouterr().out  # the prompt was displayed


def test_blank_line_is_a_deny_not_eof(fake_stdin):
    threading.Timer(0.2, fake_stdin.put, args=("\n",)).start()
    assert ap._read_choice("approve? ", 5) == ""   # empty answer, deny-shaped
    threading.Timer(0.2, fake_stdin.put, args=("o\n",)).start()
    assert ap._read_choice("approve? ", 5) == "o"  # reader did not park on ""


def test_timed_out_reader_does_not_steal_the_next_answer(fake_stdin):
    # A timed-out prompt arms the reader once; the next prompt re-arms the same
    # reader instead of stacking another thread.
    before = {t.ident for t in _readers()}
    assert ap._read_choice("approve? ", 1) is None
    spawned = {t.ident for t in _readers()} - before
    assert len(spawned) <= 1

    # A line typed while no prompt is live answers the dead prompt: read then,
    # it must not be accepted by the next prompt even though it exists.
    fake_stdin.put("s\n")
    deadline = time.monotonic() + 2
    while ap._stdin_lines.empty() and time.monotonic() < deadline:
        time.sleep(0.01)  # wait for the parked read to deliver the stale line
    threading.Timer(0.3, fake_stdin.put, args=("o\n",)).start()
    assert ap._read_choice("approve? ", 5) == "o"
    # The second prompt added no reader: the same parked reader served it.
    assert ({t.ident for t in _readers()} - before) == spawned


def test_answered_prompt_releases_stdin_for_other_input(fake_stdin):
    """After an answered approval, a plain input() elsewhere must get its line.

    Regression test: the process-lifetime reader used to stay parked inside
    input() between prompts and swallow the line into _stdin_lines, hanging the
    other prompt (e.g. the OAuth authorization-code input) forever.
    """
    threading.Timer(0.2, fake_stdin.put, args=("o\n",)).start()
    assert ap._read_choice("approve? ", 5) == "o"

    got = {}

    def other_feature():
        got["value"] = input("Authorization code: ")

    t = threading.Thread(target=other_feature, daemon=True)
    t.start()
    time.sleep(0.3)  # let the other input() block
    assert t.is_alive()
    fake_stdin.put("sk-my-oauth-token\n")
    t.join(timeout=5)
    assert not t.is_alive(), "other input() hung: the approval reader ate its line"
    assert got["value"] == "sk-my-oauth-token"
    assert ap._stdin_lines.empty(), "the approval reader swallowed a foreign line"


def test_windows_stray_pre_prompt_keystroke_is_rejected(fake_stdin, monkeypatch):
    """A stray pre-prompt 'a' must never become an 'always' grant on Windows.

    Regression test: termios is POSIX-only, so the old flush was a silent no-op
    there and the reader accepted typeahead buffered before the prompt. The
    gated readline waits until the prompt is displayed, so on the old code the
    stray line is deterministically read with a fresh stamp and accepted.
    """
    fake_stdin.isatty_value = True
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "termios", None)  # Windows: no termios

    class FakeMsvcrt:
        @staticmethod
        def kbhit():
            return not fake_stdin.empty()

        @staticmethod
        def getwch():
            return fake_stdin.get_nowait()

    monkeypatch.setitem(sys.modules, "msvcrt", FakeMsvcrt())

    prompt_shown = threading.Event()
    real_stdin, real_stdout = sys.stdin, sys.stdout

    class GatedStdin:
        def readline(self, *args, **kwargs):
            # Wait until the prompt is displayed, then let _read_choice stamp
            # its cutoff first: on the old code the stray line is then read
            # with a fresh stamp and wrongly accepted.
            assert prompt_shown.wait(timeout=10)
            time.sleep(0.5)
            return real_stdin.readline(*args, **kwargs)

        def isatty(self):
            return True

    class StdoutSpy:
        def write(self, s):
            if "approve? " in s:
                prompt_shown.set()
            return real_stdout.write(s)

        def flush(self):
            return real_stdout.flush()

    monkeypatch.setattr(sys, "stdin", GatedStdin())
    monkeypatch.setattr(sys, "stdout", StdoutSpy())

    fake_stdin.put("a\n")  # stray keystroke typed before the prompt existed
    threading.Timer(0.3, fake_stdin.put, args=("o\n",)).start()
    assert ap._read_choice("approve? ", 5) == "o"


def test_tty_typeahead_is_flushed_before_reading(fake_stdin, monkeypatch):
    """Pre-prompt tty typeahead is dropped by the POSIX flush."""
    fake_stdin.isatty_value = True

    class FakeTermios:
        TCIFLUSH = 1

        @staticmethod
        def tcflush(fd, op):
            while not fake_stdin.empty():
                fake_stdin.get_nowait()

    monkeypatch.setitem(sys.modules, "termios", FakeTermios())
    fake_stdin.put("a\n")  # stray keystroke typed before the prompt existed
    threading.Timer(0.3, fake_stdin.put, args=("o\n",)).start()
    assert ap._read_choice("approve? ", 5) == "o"


def test_freshness_cutoff_is_stamped_before_the_flush(fake_stdin, monkeypatch):
    """The cutoff must be stamped before _flush_stdin_pending runs.

    Regression test: the flush used to run before the stamp, leaving a window
    where a pre-prompt keystroke was read with a fresh stamp and accepted.
    """
    events = []
    real_monotonic_ns = time.monotonic_ns

    def spy_monotonic_ns():
        events.append("stamp")
        return real_monotonic_ns()

    real_flush = ap._flush_stdin_pending

    def spy_flush():
        events.append("flush")
        return real_flush()

    monkeypatch.setattr(time, "monotonic_ns", spy_monotonic_ns)
    monkeypatch.setattr(ap, "_flush_stdin_pending", spy_flush)
    threading.Timer(0.2, fake_stdin.put, args=("o\n",)).start()
    assert ap._read_choice("approve? ", 5) == "o"
    assert "flush" in events
    assert events[0] == "stamp", events


def test_stdin_eof_fails_closed_without_rearming(fake_stdin):
    fake_stdin.put("")
    assert ap._read_choice("approve? ", 5) == ""   # EOF reads as an empty deny
    assert ap._read_choice("approve? ", 5) == ""   # dead stdin stays dead, no re-read


def test_stdin_eof_wakes_a_parked_prompt(fake_stdin):
    threading.Timer(0.3, fake_stdin.put, args=("",)).start()
    assert ap._read_choice("approve? ", 5) == ""   # EOF mid-wait, not a timeout


def test_reader_transient_failure_does_not_latch_stdin_dead(fake_stdin, monkeypatch):
    """One bad read must not auto-deny every later approval for the session.

    A transient (non-EOF) read error exits the reader WITHOUT latching stdin
    dead; the next prompt restarts the reader and reads normally.
    """
    real_stdin = sys.stdin
    state = {"fail_once": True}

    class FlakyStdin:
        def readline(self, *args, **kwargs):
            if state["fail_once"]:
                state["fail_once"] = False
                raise RuntimeError("stdin exploded")
            return real_stdin.readline(*args, **kwargs)

        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdin", FlakyStdin())
    assert ap._read_choice("approve? ", 1) is None  # reader died; prompt times out
    assert not ap._stdin_eof  # but stdin is not latched dead
    threading.Timer(0.2, fake_stdin.put, args=("o\n",)).start()
    assert ap._read_choice("approve? ", 5) == "o"  # the next prompt recovers


def test_stdin_offer_drops_oldest_instead_of_blocking(monkeypatch):
    """A full queue must not wedge the reader inside put()."""
    small = queue.Queue(maxsize=2)
    monkeypatch.setattr(ap, "_stdin_lines", small)
    ap._stdin_offer((1, "first"))
    ap._stdin_offer((2, "second"))
    ap._stdin_offer((3, "third"))  # must not block; drops "first"
    assert small.get_nowait() == (2, "second")
    assert small.get_nowait() == (3, "third")


def test_negative_timeout_fails_closed_as_timeout(fake_stdin):
    assert ap._read_choice("approve? ", -5) is None
