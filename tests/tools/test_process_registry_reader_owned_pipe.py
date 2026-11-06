"""A finished session must not close a pipe its live reader still owns (#142193).

On Windows ``_reader_selectable`` is never set (``select()`` needs a POSIX fd), so a direct child
that exits while a descendant still holds its stdout pipe leaves the reader blocked in ``read()``.
``_reconcile_local_exit`` then publishes the session through ``_move_to_finished`` →
``_publish_finished`` → ``_release_finished_handles``, and closing the stream from the reconciling
thread waits on the buffer lock that blocked read holds: the one-shot run hung at exit for hours
with ``tail -f`` from Git Bash holding the pipe, and its completion was never delivered.

The close still has to happen — the reader does it from its own thread when its read ends, and the
prune path does it once the reader is gone. These cases pin the three-way rule.
"""
import threading
from typing import Any, cast

from tools import process_registry as pr
from tools.process_registry import ProcessRegistry, ProcessSession


class _Stream:
    """A Popen pipe stand-in that records which thread closed it."""

    def __init__(self) -> None:
        self.closed_by: list[str] = []

    def close(self) -> None:
        self.closed_by.append(threading.current_thread().name)


class _Proc:
    def __init__(self) -> None:
        self.stdout = _Stream()
        self.stderr = None  # local spawns merge stderr into stdout
        self.stdin = _Stream()


def _session(proc: _Proc, reader: threading.Thread | None = None) -> ProcessSession:
    session = ProcessSession(id='proc-1', command='tail -f x', task_id='task-1', started_at=0.0)
    session.process = cast(Any, proc)
    session._reader_thread = reader
    return session


def _live_reader(name: str = 'reader') -> tuple[threading.Thread, threading.Event]:
    """A thread that is alive (blocked in ``read()``, like a descendant-held pipe) until released."""
    hold = threading.Event()
    thread = threading.Thread(target=hold.wait, name=name, daemon=True)
    thread.start()
    return thread, hold


def test_a_windows_session_keeps_the_pipe_a_live_reader_still_reads(monkeypatch):
    monkeypatch.setattr(pr, '_IS_WINDOWS', True)
    proc = _Proc()
    reader, hold = _live_reader()
    session = _session(proc, reader)
    try:
        ProcessRegistry()._release_finished_handles(session)
    finally:
        hold.set()
        reader.join(5)
    assert proc.stdout.closed_by == [], 'closing the pipe would wait on the reader that holds it'
    assert proc.stdin.closed_by == ['MainThread'], 'the reader never reads stdin, so it closes here'


def test_the_reader_thread_itself_still_releases_the_pipe(monkeypatch):
    """The reader's own teardown (``_finish_exited`` → ``_move_to_finished``) is not the blocked case:
    it is the thread holding the lock, so it closes the stream there and the FD is not leaked."""
    monkeypatch.setattr(pr, '_IS_WINDOWS', True)
    proc = _Proc()
    registry, done = ProcessRegistry(), threading.Event()

    def read_to_eof() -> None:
        registry._release_finished_handles(session)
        done.set()

    thread = threading.Thread(target=read_to_eof, name='reader')
    session = _session(proc, thread)
    thread.start()
    assert done.wait(5), 'the reader did not finish its own release'
    thread.join(5)
    assert proc.stdout.closed_by == ['reader']


def test_a_posix_session_closes_the_stream_so_its_reader_exits(monkeypatch):
    """Unchanged on POSIX, where the close raises into the reader's blocking read and ends its loop."""
    monkeypatch.setattr(pr, '_IS_WINDOWS', False)
    proc = _Proc()
    reader, hold = _live_reader()
    session = _session(proc, reader)
    try:
        ProcessRegistry()._release_finished_handles(session)
    finally:
        hold.set()
        reader.join(5)
    assert proc.stdout.closed_by == ['MainThread']
