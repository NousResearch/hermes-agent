"""The delegation preamble must not label a task's RUNTIME as its age.

A result can sit in the durable ledger for hours (the owner process is gone, or the session
went idle) and is only re-injected at the next drain. The "Dispatched:" line used to render
``completed_at - dispatched_at`` — the task's runtime — with the word "ago", so a three-hour-old
result read as though it had just landed.
"""
import time

from tools.process_registry_notifications import _format_age, _preamble


def _dispatched_line(evt, completed_at):
    return next(l for l in _preamble(evt, "TITLE", "INTRO", completed_at, with_goal=False)
                if l.startswith("Dispatched:"))


def test_preamble_reports_true_age_and_keeps_runtime_labelled():
    now = time.time()
    dispatched = now - 11_209      # ~3h07m before the render
    completed = dispatched + 4     # the task itself finished 4s after dispatch
    line = _dispatched_line({"dispatched_at": dispatched}, completed)

    assert "(4s ago" not in line, f"runtime still rendered as age: {line}"
    assert "3h" in line, f"true age missing: {line}"
    assert "ran 4s" in line, f"runtime not reported: {line}"


def test_preamble_unchanged_for_a_fresh_result():
    now = time.time()
    dispatched, completed = now - 6, now - 2
    line = _dispatched_line({"dispatched_at": dispatched}, completed)
    assert "ago, ran 4s" in line, line


def test_preamble_skips_the_line_without_a_dispatch_time():
    assert not [l for l in _preamble({}, "T", "I", time.time(), with_goal=False)
                if l.startswith("Dispatched:")]


def test_format_age_units():
    assert _format_age(45) == "45s"
    assert _format_age(18 * 60) == "18m"
    assert _format_age(3 * 3600 + 7 * 60) == "3h7m"
