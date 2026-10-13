"""A background process's output past the rolling buffer is gone; every view must say so.

``output_buffer`` keeps the last ``max_output_chars``. Before, ``process(action="log")`` served
that tail as "the full output" starting mid-line, the completion notice promised the log "has the
full output", and ``output_cut`` counted only the kept buffer.
"""
import time

import tools.process_registry as module
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_notifications import format_process_notification


def _session(sid: str, **fields) -> ProcessSession:
    return ProcessSession(id=sid, command="cmd", task_id="task", started_at=time.time(), **fields)


def _overflowed(registry: ProcessRegistry, lines: int = 40, keep: int = 100) -> ProcessSession:
    session = _session("proc_overflow", max_output_chars=keep)
    for i in range(lines):
        session.append_output(f"line {i}\n")
    registry._running[session.id] = session
    return session


def test_log_after_overflow_says_what_was_dropped_and_starts_on_a_whole_line():
    registry = ProcessRegistry()
    session = _overflowed(registry)
    printed = sum(len(f"line {i}\n") for i in range(40))

    result = registry.read_log(session.id, offset=0, limit=3)

    assert result["truncated"] is True
    assert result["dropped_chars"] == printed - 100
    assert "were not kept" in result["note"]
    first = result["output"].splitlines()[0]
    assert first.startswith("line ") and first[5:].isdigit(), result["output"]
    assert result["total_lines"] == len(registry.read_log(session.id, offset=0, limit=1000)["output"].splitlines())


def test_log_that_never_overflowed_is_reported_as_before():
    registry = ProcessRegistry()
    session = _overflowed(registry, lines=5, keep=1000)

    result = registry.read_log(session.id, offset=0, limit=10)

    assert result["output"].splitlines() == [f"line {i}" for i in range(5)]
    assert "truncated" not in result and "dropped_chars" not in result


def test_completion_counts_dropped_output_and_the_notice_does_not_promise_the_full_log():
    registry = ProcessRegistry()
    session = _overflowed(registry)
    session.completion_output_chars = 20
    printed = sum(len(f"line {i}\n") for i in range(40))

    snapshot = module.completion_output(session)

    assert snapshot["output_cut"] == printed - 20
    assert snapshot["output_dropped"] == printed - 100
    notice = format_process_notification(
        {"type": "completion", "session_id": session.id, "command": "cmd", "exit_code": 0, **snapshot})
    assert "has the full output" not in notice
    assert f"the first {printed - 100} were not kept" in notice


def test_completion_without_overflow_keeps_the_full_output_wording():
    registry = ProcessRegistry()
    session = _overflowed(registry, lines=40, keep=10_000)
    session.completion_output_chars = 20

    snapshot = module.completion_output(session)

    assert "output_dropped" not in snapshot
    notice = format_process_notification(
        {"type": "completion", "session_id": session.id, "command": "cmd", "exit_code": 0, **snapshot})
    assert "has the full output" in notice
