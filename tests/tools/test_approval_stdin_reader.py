"""Tests for the shared stdin reader behind ``_read_choice``.

A timed-out approval prompt used to abandon a thread parked inside
``input()``; the next prompt spawned a second reader and both raced for the
same line, so a late keystroke could be eaten by the orphan and the live
prompt would record a spurious timeout. One persistent reader now owns
stdin, and a prompt accepts only lines stamped after it displayed.
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

    ``readline`` blocks until the test feeds a line; feeding ``""`` is EOF
    and ends the reader thread.
    """
    # Reader threads leaked by earlier tests (e.g. a builtins.input mock that
    # returned instantly) are parked on the pre-swap queue objects; the fresh
    # _stdin_lines below orphans them, so they cannot consume this test's feed.
    feed = queue.Queue()
    live = {"on": True}

    class FakeStdin:
        def readline(self, *args, **kwargs):
            # A reader orphaned from an earlier test can still land here; dead
            # after teardown so it sees EOF instead of eating a later test's feed.
            return feed.get() if live["on"] else ""

        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdin", FakeStdin())
    monkeypatch.setattr(ap, "_stdin_reader_started", False)
    monkeypatch.setattr(ap, "_stdin_eof", False)
    monkeypatch.setattr(ap, "_stdin_lines", queue.Queue())
    try:
        yield feed
    finally:
        live["on"] = False
        feed.put("")  # EOF any parked read, then wait for exit
        for t in _readers():
            t.join(timeout=2)


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
    # An earlier test can leave a reader parked mid-input (e.g. a hanging
    # builtins.input mock); the invariant is that this test spawns at most one
    # shared reader and the timeout does not stack another.
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
    # The second prompt added no reader: the one parked read kept serving it.
    assert ({t.ident for t in _readers()} - before) == spawned


def test_stdin_eof_fails_closed_without_rearming(fake_stdin):
    fake_stdin.put("")
    assert ap._read_choice("approve? ", 5) == ""   # EOF reads as an empty deny
    assert ap._read_choice("approve? ", 5) == ""   # dead stdin stays dead, no re-read


def test_stdin_eof_wakes_a_parked_prompt(fake_stdin):
    threading.Timer(0.3, fake_stdin.put, args=("",)).start()
    assert ap._read_choice("approve? ", 5) == ""   # EOF mid-wait, not a timeout


def test_reader_failure_parks_permanently(fake_stdin, monkeypatch):
    class ExplodingStdin:
        def readline(self, *args, **kwargs):
            raise RuntimeError("stdin exploded")

        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdin", ExplodingStdin())
    assert ap._read_choice("approve? ", 5) == ""   # fail closed fast
    assert ap._read_choice("approve? ", 5) == ""   # and stay parked


def test_negative_timeout_fails_closed_as_timeout(fake_stdin):
    assert ap._read_choice("approve? ", -5) is None
