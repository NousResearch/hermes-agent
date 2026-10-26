"""Windows execution tests for the trace writer's win32-specific paths.

These tests are marked ``platforms("windows")`` so the Windows CI lane
(the ``tests-os.yml`` matrix, which selects ``-m "platforms and not
integration"``) imports and runs this file on a real Windows host; every
other lane skips it. They exist because the writer's two win32-only code
paths — the ``O_BINARY`` flag (text-mode newline translation) and the
process-wide append lock — carry no Windows-native coverage otherwise:
the shared tests in ``tests/test_trace_writer.py`` are marked
``platforms("posix")`` where they depend on POSIX semantics, so nothing
exercises ``append_trace`` on Windows without this file.

On POSIX hosts the conftest's ``platforms`` gate skips each test with the
standard "host not covered by platforms() spec" reason.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from trace_writer import append_trace, iter_records, list_segments

from tests.test_trace_writer import DAY_ONE, make_record

pytestmark = pytest.mark.platforms("windows")


def test_append_writes_lf_only_no_crlf_inflation(tmp_path):
    """Every line must land as exactly one LF byte, not CR LF.

    Without ``O_BINARY``, ``os.open`` on Windows opens files in text mode
    and each ``b"\\n"`` is translated to ``b"\\r\\n"`` on disk: JSONL lines
    gain a stray byte, ``st_size`` inflates by one per record (the 1066 vs
    1065 review finding), and readers on any platform see CRLF-corrupted
    records. Assert byte-exact file contents, not just parseability.
    """

    root = tmp_path / "traces"
    record = make_record(1)
    segment = append_trace(record, root, now=DAY_ONE)
    raw = segment.read_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\r\n")
    assert b"\r" not in raw
    # st_size == exact serialized length: no per-line inflation.
    line_len = len(
        json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
    )
    assert segment.stat().st_size == line_len
    assert segment.stat().st_size == len(raw)
    # And the record round-trips.
    [(_, decoded)] = list(iter_records(segment))
    assert decoded == record


def test_threaded_appends_serialized_no_loss(tmp_path):
    """Concurrent threaded appends must not lose records on Windows.

    Windows makes no ``O_APPEND`` atomicity guarantee, so the writer
    serializes appends under a process-wide ``threading.Lock`` on win32.
    Without the lock, 8 threads × 25 appends lose records silently (the
    review's reproduction: 200 sent → 102 parseable + 13 torn lines).
    With it, every record lands and every line parses.
    """

    root = tmp_path / "traces"
    n_threads, per_thread = 8, 25
    expected = set(range(n_threads * per_thread))

    def worker(offset: int):
        for i in range(per_thread):
            append_trace(make_record(offset + i), root, now=DAY_ONE)

    threads = [
        threading.Thread(target=worker, args=(t * per_thread,))
        for t in range(n_threads)
    ]
    deadline = time.monotonic() + 60.0
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=max(0.0, deadline - time.monotonic()))
        assert not t.is_alive(), "appender thread hung — append lock deadlock"

    found: set[int] = set()
    for segment in list_segments(root):
        for _, rec in iter_records(segment):
            found.add(rec["run_id"])
    assert found == expected, f"lost {len(expected - found)} of {len(expected)} records"