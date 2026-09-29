"""Tests for trace_writer — the append-only storage layer for trace schema v1.

Covers the guarantees specs/trace-schema.md (v1.0.0) fixes for the writer:

- one compact JSON line per record, single ``O_APPEND`` write + ``fsync``;
- UTC date rotation (``traces-YYYY-MM-DD.jsonl``) and the 512 MB size-cap
  rollover to ``.NNN`` suffixes;
- never overwrite / never truncate an existing file, and fail loudly on a
  target that is not a regular appendable file;
- concurrent appends from multiple processes and threads interleave only at
  line granularity;
- a kill -9 mid-append leaves every prior record intact, and the reader
  tolerates the torn final line (skip + warn, never crash);
- records the writer accepts validate against the normative JSON schema,
  and records round-trip byte-exactly through the reader.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import stat
import subprocess
import sys
import threading
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import trace_writer
from trace_writer import (
    SEGMENT_CAP_BYTES,
    SCHEMA_VERSION,
    TornRecordWarning,
    TraceWriteError,
    append_trace,
    default_trace_root,
    iter_records,
    list_segments,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "specs" / "trace-v1.schema.json"

# 2026-09-29 12:00:00 UTC and 2026-09-30 00:00:05 UTC — fixed anchors so date
# rotation across a midnight boundary is exercised deterministically.
DAY_ONE = 1_790_683_200
DAY_TWO = 1_790_726_405


def _schema_validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def make_record(run_id: int, *, trace_id: str | None = None) -> dict:
    """A complete record that validates against specs/trace-v1.schema.json."""

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "session_id": f"20260929_test_{run_id:04d}",
        "profile_slug": "viiy-coder",
        "board_slug": "specialized",
        "task_id": "t_99bf72be",
        "trace_id": trace_id or str(uuid.uuid4()),
        "model": "kimi-k3",
        "provider": "ollama-cloud",
        "harness_version": {
            "prompt_builder_sha": "89cba9d8459191276f195ba73e183673d7709f58",
            "config_sha256": "a" * 64,
        },
        "messages": [
            {"role": "system", "content": "sys", "tool_call_id": None, "name": None, "index": 0},
            {"role": "user", "content": f"task {run_id}", "tool_call_id": None, "name": None, "index": 1},
        ],
        "tool_calls": [
            {
                "id": f"call_{run_id:08x}",
                "name": "kanban_complete",
                "arguments": {"summary": "done"},
                "result": "ok",
                "exit_code": None,
                "duration_ms": 12,
                "is_terminal": True,
            }
        ],
        "outcome": "completed",
        "terminal_call": "kanban_complete",
        "exit_code": 0,
        "error_class": None,
        "killswitch": None,
        "safety_tier_hit": None,
        "token_counts": {"input": 10, "output": 5, "cache_read": 0, "cache_write": 0, "total": 15},
        "cost_usd": 0.001,
        "token_counts_complete": True,
        "started_at": DAY_ONE,
        "ended_at": DAY_ONE + 60,
        "wall_clock_seconds": 60,
    }


def _read_lines(segment: Path) -> list[bytes]:
    return segment.read_bytes().splitlines()


# ---------------------------------------------------------------------------
# Basic append semantics
# ---------------------------------------------------------------------------


def test_append_creates_root_and_segment_with_restrictive_perms(tmp_path):
    root = tmp_path / "nested" / "traces"
    segment = append_trace(make_record(1), root, now=DAY_ONE)
    assert segment == root / "traces-2026-09-29.jsonl"
    assert segment.is_file()
    assert stat.S_IMODE(segment.stat().st_mode) == 0o600
    assert stat.S_IMODE(root.stat().st_mode) & 0o077 == 0


def test_append_writes_one_compact_line_per_record(tmp_path):
    root = tmp_path / "traces"
    first = append_trace(make_record(1), root, now=DAY_ONE)
    second = append_trace(make_record(2), root, now=DAY_ONE)
    assert first == second
    lines = _read_lines(first)
    assert len(lines) == 2
    for line in lines:
        text = line.decode("utf-8")
        assert "\n" not in text.rstrip("\n")
        assert ": " not in text  # compact separators, per the spec
    assert (json.loads(lines[0].decode("utf-8"))["run_id"], json.loads(lines[1].decode("utf-8"))["run_id"]) == (1, 2)


def test_append_line_ends_with_newline(tmp_path):
    segment = append_trace(make_record(1), tmp_path / "traces", now=DAY_ONE)
    assert segment.read_bytes().endswith(b"\n")


def test_append_is_append_only_across_calls(tmp_path):
    root = tmp_path / "traces"
    append_trace(make_record(1), root, now=DAY_ONE)
    append_trace(make_record(2), root, now=DAY_ONE)
    appended = append_trace(make_record(3), root, now=DAY_ONE)
    runs = [rec["run_id"] for _, rec in iter_records(appended)]
    assert runs == [1, 2, 3]


def test_append_validates_against_normative_schema(tmp_path):
    """Records the writer accepts must pass specs/trace-v1.schema.json."""

    validator = _schema_validator()
    segment = append_trace(make_record(42), tmp_path / "traces", now=DAY_ONE)
    records = [rec for _, rec in iter_records(segment)]
    assert len(records) == 1
    validator.validate(records[0])


def test_rejects_non_dict_record(tmp_path):
    with pytest.raises(TraceWriteError):
        append_trace(["not", "a", "dict"], tmp_path / "traces")  # type: ignore[arg-type]
    assert list_segments(tmp_path / "traces") == []


def test_rejects_unserializable_record_without_touching_disk(tmp_path):
    root = tmp_path / "traces"
    good = append_trace(make_record(1), root, now=DAY_ONE)
    before = good.read_bytes()
    with pytest.raises(TraceWriteError):
        append_trace({"bad": object()}, root, now=DAY_ONE)  # type: ignore[dict-item]
    assert good.read_bytes() == before  # serialize-first: failure never reaches disk


def test_round_trip_is_byte_exact(tmp_path):
    record = make_record(7)
    record["messages"][1]["content"] = "unicode — é漢字 ✓"
    segment = append_trace(record, tmp_path / "traces", now=DAY_ONE)
    [[_, loaded]] = iter_records(segment)
    assert loaded == record


def test_round_trip_preserves_redaction_marker_shape(tmp_path):
    marker = {"redacted": True, "reason": "credential", "sha256": "b" * 64}
    record = make_record(8)
    record["tool_calls"][0]["arguments"] = marker
    segment = append_trace(record, tmp_path / "traces", now=DAY_ONE)
    [[_, loaded]] = iter_records(segment)
    assert loaded["tool_calls"][0]["arguments"] == marker


def test_default_root_is_cross_profile_sessions_traces(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    assert default_trace_root() == tmp_path / ".hermes" / "sessions" / "traces"


# ---------------------------------------------------------------------------
# Never overwrite / fail loudly
# ---------------------------------------------------------------------------


def test_refuses_symlink_target(tmp_path):
    root = tmp_path / "traces"
    root.mkdir()
    target = root / "traces-2026-09-29.jsonl"
    real = tmp_path / "elsewhere"
    real.write_bytes(b'{"precious": true}\n')
    target.symlink_to(real)
    with pytest.raises(TraceWriteError):
        append_trace(make_record(1), root, now=DAY_ONE)
    assert real.read_bytes() == b'{"precious": true}\n'  # symlink target untouched


def test_refuses_directory_target(tmp_path):
    root = tmp_path / "traces"
    (root / "traces-2026-09-29.jsonl").mkdir(parents=True)
    with pytest.raises(TraceWriteError):
        append_trace(make_record(1), root, now=DAY_ONE)


def test_refuses_fifo_target(tmp_path):
    root = tmp_path / "traces"
    root.mkdir()
    os.mkfifo(root / "traces-2026-09-29.jsonl")
    with pytest.raises(TraceWriteError):
        append_trace(make_record(1), root, now=DAY_ONE)


def test_refuses_non_directory_root(tmp_path):
    not_a_dir = tmp_path / "traces"
    not_a_dir.write_text("x", encoding="utf-8")
    with pytest.raises(TraceWriteError):
        append_trace(make_record(1), not_a_dir, now=DAY_ONE)


def test_never_truncates_existing_segment(tmp_path):
    root = tmp_path / "traces"
    segment = append_trace(make_record(1), root, now=DAY_ONE)
    before = segment.read_bytes()
    for run_id in (2, 3, 4, 5):
        append_trace(make_record(run_id), root, now=DAY_ONE)
    after = segment.read_bytes()
    assert after.startswith(before)
    assert len(after) > len(before)


# ---------------------------------------------------------------------------
# Rotation: date boundary and size cap
# ---------------------------------------------------------------------------


def test_date_rotation_uses_utc_date_of_write(tmp_path):
    root = tmp_path / "traces"
    first = append_trace(make_record(1), root, now=DAY_ONE)
    second = append_trace(make_record(2), root, now=DAY_TWO)
    assert first.name == "traces-2026-09-29.jsonl"
    assert second.name == "traces-2026-09-30.jsonl"
    assert list_segments(root) == sorted([first, second])


def test_date_rotation_near_midnight_boundary(tmp_path):
    root = tmp_path / "traces"
    before_midnight = datetime(2026, 9, 29, 23, 59, 59, tzinfo=timezone.utc).timestamp()
    after_midnight = datetime(2026, 9, 30, 0, 0, 1, tzinfo=timezone.utc).timestamp()
    seg_a = append_trace(make_record(1), root, now=before_midnight)
    seg_b = append_trace(make_record(2), root, now=after_midnight)
    assert seg_a.name != seg_b.name


def test_size_cap_rolls_to_suffix_segments(tmp_path):
    root = tmp_path / "traces"
    record = make_record(1)
    line_len = len(json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8")) + 1
    cap = 2 * line_len + 1  # fits exactly two lines, third rolls over
    seg1 = append_trace(make_record(1), root, now=DAY_ONE, max_segment_bytes=cap)
    seg2 = append_trace(make_record(2), root, now=DAY_ONE, max_segment_bytes=cap)
    seg3 = append_trace(make_record(3), root, now=DAY_ONE, max_segment_bytes=cap)
    assert seg1.name == "traces-2026-09-29.jsonl"
    assert seg2 == seg1
    assert seg3.name == "traces-2026-09-29.002.jsonl"
    # And when .002 also fills, the suffix increments monotonically.
    seg4 = append_trace(make_record(4), root, now=DAY_ONE, max_segment_bytes=cap)
    assert seg4 == seg3
    seg5 = append_trace(make_record(5), root, now=DAY_ONE, max_segment_bytes=cap)
    assert seg5.name == "traces-2026-09-29.003.jsonl"
    for seg, expected in ((seg1, [1, 2]), (seg3, [3, 4]), (seg5, [5])):
        assert [rec["run_id"] for _, rec in iter_records(seg)] == expected


def test_size_cap_suffix_skips_existing_non_regular_path(tmp_path):
    root = tmp_path / "traces"
    record = make_record(1)
    line_len = len(json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8")) + 1
    primary = append_trace(make_record(1), root, now=DAY_ONE, max_segment_bytes=line_len)
    assert primary.stat().st_size == line_len  # exactly full
    (root / "traces-2026-09-29.002.jsonl").mkdir()  # squatting directory in the suffix slot
    with pytest.raises(TraceWriteError):
        append_trace(make_record(2), root, now=DAY_ONE, max_segment_bytes=line_len)


def test_saturated_suffixes_fail_loudly(tmp_path):
    root = tmp_path / "traces"
    record = make_record(1)
    line_len = len(json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8")) + 1
    append_trace(make_record(1), root, now=DAY_ONE, max_segment_bytes=line_len)
    for suffix in range(2, 1000):  # fill every remaining slot as a saturated regular file
        (root / f"traces-2026-09-29.{suffix:03d}.jsonl").write_bytes(b"x" * line_len)
    with pytest.raises(TraceWriteError, match="exhausted"):
        append_trace(make_record(2), root, now=DAY_ONE, max_segment_bytes=line_len)


# ---------------------------------------------------------------------------
# Concurrent append: threads and processes
# ---------------------------------------------------------------------------


def _assert_no_line_interleaving(root: Path, expected_run_ids: set[int]):
    found: set[int] = set()
    for segment in list_segments(root):
        for line in _read_lines(segment):
            decoded = json.loads(line.decode("utf-8"))  # any interleaving breaks this
            assert decoded["run_id"] not in found
            found.add(decoded["run_id"])
    assert found == expected_run_ids


def test_concurrent_threads_no_interleaving(tmp_path):
    root = tmp_path / "traces"
    n_threads, per_thread = 8, 25

    def worker(offset: int):
        for i in range(per_thread):
            append_trace(make_record(offset + i), root, now=DAY_ONE)

    threads = [threading.Thread(target=worker, args=(t * per_thread,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    _assert_no_line_interleaving(root, set(range(n_threads * per_thread)))


def _process_appender(root: str, base: int, count: int) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    for i in range(count):
        append_trace(make_record(base + i), root, now=DAY_ONE)


def test_concurrent_processes_no_interleaving(tmp_path):
    root = tmp_path / "traces"
    n_procs, per_proc = 6, 20
    ctx = multiprocessing.get_context("fork")
    procs = [
        ctx.Process(target=_process_appender, args=(str(root), p * per_proc, per_proc))
        for p in range(n_procs)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"appender exited {p.exitcode}"
    _assert_no_line_interleaving(root, set(range(n_procs * per_proc)))


# ---------------------------------------------------------------------------
# kill -9 mid-append: prior records intact, torn tail tolerated
# ---------------------------------------------------------------------------


def test_sigkill_mid_append_prior_records_survive(tmp_path):
    root = tmp_path / "traces"
    n_good = 20
    for run_id in range(n_good):
        append_trace(make_record(run_id), root, now=DAY_ONE)
    segment = root / "traces-2026-09-29.jsonl"
    before = segment.stat().st_size

    # A child opens the segment O_APPEND and writes a partial line (no fsync,
    # then kill -9) — the exact torn-tail the spec tells readers to tolerate.
    child_src = (
        "import os, sys, time\n"
        f"fd = os.open({str(segment)!r}, os.O_WRONLY | os.O_APPEND)\n"
        "os.write(fd, b'{\"run_id\": 999, \"parti')  # torn final line, no newline\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", child_src])
    deadline = time.monotonic() + 10
    while segment.stat().st_size == before and time.monotonic() < deadline:
        time.sleep(0.01)
    assert segment.stat().st_size > before, "child never wrote"
    proc.kill()
    proc.wait()
    assert proc.returncode == -9

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TornRecordWarning)
        runs = [rec["run_id"] for _, rec in iter_records(segment)]
    assert runs == list(range(n_good))  # every prior record intact, in order


def test_reader_skips_torn_tail_and_reports_it(tmp_path):
    root = tmp_path / "traces"
    segment = append_trace(make_record(1), root, now=DAY_ONE)
    append_trace(make_record(2), root, now=DAY_ONE)
    with open(segment, "ab") as handle:  # simulate the torn tail a kill -9 leaves
        handle.write(b'{"run_id": 3, "s')

    torn: list[TornRecordWarning] = []
    with pytest.warns(TornRecordWarning):
        rows = list(iter_records(segment, on_torn=torn.append))
    assert [rec["run_id"] for _, rec in rows] == [1, 2]
    assert [row[0] for row in rows] == [1, 2]  # line numbers are 1-based

    assert len(torn) == 1
    assert torn[0].line_no == 3
    assert torn[0].path == segment
    assert torn[0].raw_preview.startswith(b'{"run_id": 3')


def test_reader_skips_malformed_utf8_tail_without_crashing(tmp_path):
    root = tmp_path / "traces"
    segment = append_trace(make_record(1), root, now=DAY_ONE)
    with open(segment, "ab") as handle:
        handle.write(b'{"run_id": 2, "x": "\xff\xfe"}')  # invalid UTF-8 bytes
    with pytest.warns(TornRecordWarning):
        rows = list(iter_records(segment))
    assert [rec["run_id"] for _, rec in rows] == [1]


def test_reader_skips_non_object_json_line(tmp_path):
    root = tmp_path / "traces"
    segment = append_trace(make_record(1), root, now=DAY_ONE)
    with open(segment, "ab") as handle:
        handle.write(b'[1, 2, 3]\n')
    with pytest.warns(TornRecordWarning):
        rows = list(iter_records(segment))
    assert len(rows) == 1


def test_reader_ignores_blank_lines(tmp_path):
    root = tmp_path / "traces"
    segment = append_trace(make_record(1), root, now=DAY_ONE)
    with open(segment, "ab") as handle:
        handle.write(b"\n   \n")
    append_trace(make_record(2), root, now=DAY_ONE)
    rows = list(iter_records(segment))  # no warnings raised for blanks
    assert [rec["run_id"] for _, rec in rows] == [1, 2]


# ---------------------------------------------------------------------------
# list_segments
# ---------------------------------------------------------------------------


def test_list_segments_skips_non_regular_and_non_segment_names(tmp_path):
    root = tmp_path / "traces"
    root.mkdir()
    a = root / "traces-2026-09-29.jsonl"
    b = root / "traces-2026-09-29.002.jsonl"
    a.write_bytes(b"{}\n")
    b.write_bytes(b"{}\n")
    (root / "traces-2026-09-30.jsonl").mkdir()  # directory with a segment name
    (root / "notes.txt").write_text("nope", encoding="utf-8")
    link = root / "traces-2026-09-28.jsonl"
    link.symlink_to(a)
    segs = list_segments(root)
    assert set(segs) == {a, b}
    assert segs == sorted(segs)


def test_list_segments_missing_root_is_empty(tmp_path):
    assert list_segments(tmp_path / "does-not-exist") == []
