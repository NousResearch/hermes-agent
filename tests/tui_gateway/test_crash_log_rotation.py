"""Regression coverage for unbounded ``tui_gateway_crash.log`` growth (#136624).

The crash log is best-effort forensics, but every writer appended without a
size cap: recurring benign events (restart/SIGHUP cycles land a
``gateway received`` + ``gateway exit`` pair each time, per-thread panics hit
the hooks, per-turn dispatcher exceptions land a traceback) grew the file to
15 MB / 778 records on one host. All three Python writers must rotate to a
single ``.1`` generation before appending once the file is over the cap,
bounding the total at 2x the cap. Rotation runs under a cross-writer
``O_CREAT|O_EXCL`` lock (also honored by the Node parent) so a stale rotator
cannot rename a small fresh log over the retained generation, and every
record is truncated to the cap at the shared append boundary so the bound
survives oversized tracebacks.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

from tui_gateway import entry, server


def test_record_crash_rotates_oversized_log(tmp_path, monkeypatch, capsys):
    crash_log = tmp_path / "tui_gateway_crash.log"
    crash_log.write_bytes(b"x" * 4096)
    monkeypatch.setattr(server, "_CRASH_LOG", str(crash_log))
    monkeypatch.setattr(server, "_CRASH_LOG_MAX_BYTES", 1024)

    server._record_crash("unhandled exception", ValueError, ValueError("boom"), None)

    rotated = tmp_path / "tui_gateway_crash.log.1"
    assert rotated.read_bytes() == b"x" * 4096
    assert "=== unhandled exception" in crash_log.read_text(encoding="utf-8")
    assert crash_log.stat().st_size < 4096


def test_record_crash_keeps_a_small_log_in_place(tmp_path, monkeypatch, capsys):
    crash_log = tmp_path / "tui_gateway_crash.log"
    crash_log.write_bytes(b"x" * 512)
    monkeypatch.setattr(server, "_CRASH_LOG", str(crash_log))
    monkeypatch.setattr(server, "_CRASH_LOG_MAX_BYTES", 1024)

    server._record_crash("unhandled exception", ValueError, ValueError("boom"), None)

    assert not (tmp_path / "tui_gateway_crash.log.1").exists()
    assert crash_log.read_bytes().startswith(b"x" * 512)
    assert b"=== unhandled exception" in crash_log.read_bytes()


def test_record_crash_without_existing_log_does_not_rotate(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(server, "_CRASH_LOG", str(tmp_path / "tui_gateway_crash.log"))

    server._record_crash("unhandled exception", ValueError, ValueError("boom"), None)

    assert (tmp_path / "tui_gateway_crash.log").exists()
    assert not (tmp_path / "tui_gateway_crash.log.1").exists()


def test_append_crash_log_rotates_oversized_log(tmp_path, monkeypatch):
    crash_log = tmp_path / "tui_gateway_crash.log"
    crash_log.write_bytes(b"y" * 4096)
    # _append_crash_record resolves _CRASH_LOG against server's namespace, so the
    # temporary file only has to be patched there.
    monkeypatch.setattr(server, "_CRASH_LOG", str(crash_log))
    monkeypatch.setattr(server, "_CRASH_LOG_MAX_BYTES", 1024)

    entry._append_crash_log("gateway exit · 2026-10-11 · reason=signal")

    rotated = tmp_path / "tui_gateway_crash.log.1"
    assert rotated.read_bytes() == b"y" * 4096
    assert "=== gateway exit" in crash_log.read_text(encoding="utf-8")


def test_append_crash_log_keeps_a_small_log_in_place(tmp_path, monkeypatch):
    crash_log = tmp_path / "tui_gateway_crash.log"
    crash_log.write_bytes(b"y" * 512)
    monkeypatch.setattr(server, "_CRASH_LOG", str(crash_log))
    monkeypatch.setattr(server, "_CRASH_LOG_MAX_BYTES", 1024)

    entry._append_crash_log("gateway exit · 2026-10-11 · reason=signal")

    assert not (tmp_path / "tui_gateway_crash.log.1").exists()
    assert crash_log.read_text(encoding="utf-8").startswith("y" * 512)
    assert "=== gateway exit" in crash_log.read_text(encoding="utf-8")


def test_recover_turn_exception_rotates_oversized_log(tmp_path, monkeypatch):
    crash_log = tmp_path / "tui_gateway_crash.log"
    crash_log.write_bytes(b"z" * 4096)
    monkeypatch.setattr(server, "_CRASH_LOG", str(crash_log))
    monkeypatch.setattr(server, "_CRASH_LOG_MAX_BYTES", 1024)
    # The recover path keeps going after the crash-log append: no-op the
    # history restore and the terminal emit so the test only observes the log.
    monkeypatch.setattr(server, "_restore_agent_history_after_turn_error", lambda *a, **k: None)
    monkeypatch.setattr(server, "_emit_terminal_turn_error", lambda *a, **k: None)
    st = SimpleNamespace(agent=None, terminal_callback=None, receipt_attempted=True,
                         receipt_committed=False, prompt_text="hi", error_retained=False,
                         error_detail=None, marker_key=None)

    server._recover_turn_exception("sid-test", {}, st, RuntimeError("boom"))

    rotated = tmp_path / "tui_gateway_crash.log.1"
    assert rotated.read_bytes() == b"z" * 4096
    text = crash_log.read_text(encoding="utf-8")
    assert "=== turn-dispatcher exception" in text
    assert "sid=sid-test" in text


def test_recover_turn_exception_keeps_a_small_log_in_place(tmp_path, monkeypatch):
    crash_log = tmp_path / "tui_gateway_crash.log"
    crash_log.write_bytes(b"z" * 512)
    monkeypatch.setattr(server, "_CRASH_LOG", str(crash_log))
    monkeypatch.setattr(server, "_CRASH_LOG_MAX_BYTES", 1024)
    monkeypatch.setattr(server, "_restore_agent_history_after_turn_error", lambda *a, **k: None)
    monkeypatch.setattr(server, "_emit_terminal_turn_error", lambda *a, **k: None)
    st = SimpleNamespace(agent=None, terminal_callback=None, receipt_attempted=True,
                         receipt_committed=False, prompt_text="hi", error_retained=False,
                         error_detail=None, marker_key=None)

    server._recover_turn_exception("sid-test", {}, st, RuntimeError("boom"))

    assert not (tmp_path / "tui_gateway_crash.log.1").exists()
    text = crash_log.read_text(encoding="utf-8")
    assert text.startswith("z" * 512)
    assert "=== turn-dispatcher exception" in text


def _lock_path(crash_log) -> str:
    return f"{crash_log}.rotate.lock"


def test_rotate_skips_while_another_writer_holds_a_fresh_lock(tmp_path, monkeypatch):
    # A rotator running on a stale stat must not rename the (small, fresh) active
    # log over the retained .1 generation: while another writer holds a fresh
    # lock, rotation is skipped entirely and both files stay untouched.
    crash_log = tmp_path / "tui_gateway_crash.log"
    crash_log.write_bytes(b"x" * 4096)
    retained = tmp_path / "tui_gateway_crash.log.1"
    retained.write_bytes(b"previous generation")
    monkeypatch.setattr(server, "_CRASH_LOG", str(crash_log))
    monkeypatch.setattr(server, "_CRASH_LOG_MAX_BYTES", 1024)
    lock = _lock_path(crash_log)
    os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))

    server._rotate_crash_log()

    assert retained.read_bytes() == b"previous generation"
    assert crash_log.read_bytes() == b"x" * 4096
    assert os.path.exists(lock)  # the other writer's lock is left alone


def test_rotate_breaks_a_stale_lock_and_proceeds(tmp_path, monkeypatch):
    # A lock older than the stale timeout belongs to a killed writer; rotation
    # must not be wedged forever by it.
    crash_log = tmp_path / "tui_gateway_crash.log"
    crash_log.write_bytes(b"x" * 4096)
    monkeypatch.setattr(server, "_CRASH_LOG", str(crash_log))
    monkeypatch.setattr(server, "_CRASH_LOG_MAX_BYTES", 1024)
    lock = _lock_path(crash_log)
    os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    old = time.time() - server._CRASH_ROTATE_LOCK_STALE_S - 60
    os.utime(lock, (old, old))

    server._rotate_crash_log()

    assert (tmp_path / "tui_gateway_crash.log.1").read_bytes() == b"x" * 4096
    assert not os.path.exists(lock)  # broken and cleaned up


def test_rotate_releases_the_lock_after_rotating(tmp_path, monkeypatch):
    crash_log = tmp_path / "tui_gateway_crash.log"
    crash_log.write_bytes(b"x" * 4096)
    monkeypatch.setattr(server, "_CRASH_LOG", str(crash_log))
    monkeypatch.setattr(server, "_CRASH_LOG_MAX_BYTES", 1024)

    server._rotate_crash_log()

    assert (tmp_path / "tui_gateway_crash.log.1").read_bytes() == b"x" * 4096
    assert not os.path.exists(_lock_path(crash_log))


def test_record_crash_preserves_retained_generation_while_lock_held(tmp_path, monkeypatch, capsys):
    # End to end through a real writer: with a concurrent rotator holding the
    # lock, the append must not destroy the retained generation.
    crash_log = tmp_path / "tui_gateway_crash.log"
    crash_log.write_bytes(b"x" * 4096)
    retained = tmp_path / "tui_gateway_crash.log.1"
    retained.write_bytes(b"previous generation")
    monkeypatch.setattr(server, "_CRASH_LOG", str(crash_log))
    monkeypatch.setattr(server, "_CRASH_LOG_MAX_BYTES", 1024)
    os.close(os.open(_lock_path(crash_log), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))

    server._record_crash("unhandled exception", ValueError, ValueError("boom"), None)

    assert retained.read_bytes() == b"previous generation"
    assert crash_log.read_bytes().startswith(b"x" * 4096)
    assert b"=== unhandled exception" in crash_log.read_bytes()


def test_oversized_record_is_truncated_to_the_cap(tmp_path, monkeypatch):
    # One record must not re-admit an oversized traceback whole into a fresh
    # generation: the shared append boundary truncates it, keeping head, tail
    # and an explicit marker, in UTF-8 bytes.
    crash_log = tmp_path / "tui_gateway_crash.log"
    monkeypatch.setattr(server, "_CRASH_LOG", str(crash_log))
    monkeypatch.setattr(server, "_CRASH_LOG_MAX_BYTES", 1024)

    record = "HEAD " + "x" * (3 * 1024 * 1024) + " TAIL"
    server._append_crash_record(record)

    text = crash_log.read_text(encoding="utf-8")
    assert crash_log.stat().st_size <= 1024
    assert text.startswith("HEAD")
    assert text.rstrip().endswith("TAIL")
    assert "[truncated" in text


def test_append_rotates_when_the_incoming_record_would_exceed_the_cap(tmp_path, monkeypatch):
    # Rotation must consider the incoming record too: a fresh log accepts at
    # most one full record, and an almost-full log rotates instead of growing
    # past the cap.
    crash_log = tmp_path / "tui_gateway_crash.log"
    crash_log.write_bytes(b"y" * 900)
    monkeypatch.setattr(server, "_CRASH_LOG", str(crash_log))
    monkeypatch.setattr(server, "_CRASH_LOG_MAX_BYTES", 1024)

    server._append_crash_record("R" * 200)

    rotated = tmp_path / "tui_gateway_crash.log.1"
    assert rotated.read_bytes() == b"y" * 900
    assert crash_log.read_bytes() == b"R" * 200
    assert crash_log.stat().st_size <= 1024
