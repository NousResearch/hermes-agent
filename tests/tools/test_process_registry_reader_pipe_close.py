"""The one-shot exit must not hang on a pipe its live reader still holds (#142193).

On Windows ``_reader_selectable`` is never set (``select()`` needs a POSIX fd), so a direct child
that exits while a descendant still holds its stdout pipe leaves the reader blocked in ``read()``.
``_reconcile_local_exit`` then publishes the session through ``_move_to_finished`` ->
``_publish_finished`` -> ``_release_finished_handles``, and closing the stream from the
reconciling thread waits on the buffer lock the blocked read holds: the one-shot run hung at exit
for hours, and its completion was never delivered.

These cases drive the REAL Windows path - a real process tree, a real pipe, a real reader thread -
so nothing about the host is faked: the sibling test covers the release contract with stubs, this
one proves the symptom is gone.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time

import pytest


def _wait_until(predicate, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


@pytest.mark.platforms("windows")
@pytest.mark.live_system_guard_bypass  # the orphan is reparented to init: signalling it is the point
def test_the_one_shot_linger_completes_with_a_reader_owned_pipe(tmp_path, monkeypatch):
    """The reported symptom: a live descendant holds the pipe, and the linger must still finish."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    import tools.process_registry as pr

    registry = pr.ProcessRegistry()
    # The direct child exits at once and leaves a descendant holding the stdout pipe:
    # the reader is blocked in read() with nothing left to read.
    child = (
        "import subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "print(p.pid, flush=True)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", child],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace",
    )
    session = registry.adopt_local(proc, command="reader-owned-pipe", cwd=str(tmp_path))
    reader = session._reader_thread
    grandchild_pid = None
    try:
        assert _wait_until(lambda: session.process.poll() is not None), "the direct child did not exit"
        assert _wait_until(lambda: session.output_buffer.strip()), "the reader captured nothing"
        grandchild_pid = int(session.output_buffer.strip().split()[0])
        assert reader is not None and reader.is_alive(), "the reader must still hold the pipe"

        returned = threading.Event()
        result = {}

        def _linger():
            result["value"] = registry.wait_for_pending_completions(timeout=20, poll_interval=0.2)
            returned.set()

        threading.Thread(target=_linger, daemon=True).start()
        assert returned.wait(25), (
            "the one-shot linger blocked closing the reader-owned pipe (#142193)")
        assert session.id in result["value"]["completed"], "the linger must complete the session"
        assert session._completion_event.is_set(), "the completion must be published"
        # The pipe is still the reader's - closing it from the finisher is what deadlocked.
        assert not session.process.stdout.closed
    finally:
        if grandchild_pid:
            with contextlib.suppress(OSError, ProcessLookupError):
                os.kill(grandchild_pid, signal.SIGTERM)
        if reader is not None:
            deadline = time.monotonic() + 30
            while reader.is_alive() and time.monotonic() < deadline:
                time.sleep(0.1)
