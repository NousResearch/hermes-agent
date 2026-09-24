"""Tests for the #99869 review rework: markers, journal, periodic, durable writes."""

import json
import os
import signal as _signal
import time

import pytest

import tools.checkpoint_manager as cm
from tools.checkpoint_manager import (
    CheckpointManager,
    build_interruption_note,
    clear_interrupted_marker,
    durable_atomic_write,
    install_termination_handlers,
    list_interrupted_markers,
    read_interrupted_marker,
    write_interrupted_marker,
)


@pytest.fixture()
def ckpt_base(tmp_path, monkeypatch):
    base = tmp_path / "cps"
    monkeypatch.setattr(cm, "CHECKPOINT_BASE", base)
    return base


@pytest.fixture()
def proj(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    (d / "f.txt").write_text("one\n")
    # Project-root marker so get_working_dir_for_path stops here instead of
    # climbing into the real filesystem above tmp_path.
    (d / "pyproject.toml").write_text("[project]\n")
    return d


# =========================================================================
# Session-terminate markers
# =========================================================================

class TestInterruptedMarkers:
    def test_write_read_clear_roundtrip(self, tmp_path):
        assert read_interrupted_marker("sess-1", base=tmp_path) is None
        marker = write_interrupted_marker(
            "sess-1", last_action="write_file:x", reason="auth_error", base=tmp_path
        )
        assert marker is not None and marker.exists()
        data = read_interrupted_marker("sess-1", base=tmp_path)
        assert data["session_id"] == "sess-1"
        assert data["last_action"] == "write_file:x"
        assert data["reason"] == "auth_error"
        assert data["timestamp"] > 0
        assert clear_interrupted_marker("sess-1", base=tmp_path) is True
        assert read_interrupted_marker("sess-1", base=tmp_path) is None
        assert clear_interrupted_marker("sess-1", base=tmp_path) is False

    def test_list_orders_newest_first(self, tmp_path):
        write_interrupted_marker("old", reason="a", base=tmp_path)
        time.sleep(0.02)
        write_interrupted_marker("new", reason="b", base=tmp_path)
        ids = [m["session_id"] for m in list_interrupted_markers(base=tmp_path)]
        assert ids == ["new", "old"]

    def test_session_id_sanitized_to_safe_filename(self, tmp_path):
        marker = write_interrupted_marker("cron-script:../../evil", base=tmp_path)
        assert marker is not None
        assert "/" not in marker.name and "\\" not in marker.name
        assert marker.resolve().parent == (tmp_path / "sessions").resolve()

    def test_write_prunes_markers_older_than_7d(self, tmp_path):
        stale = tmp_path / "sessions"
        stale.mkdir(parents=True)
        old_file = stale / "ancient.interrupted"
        old_file.write_text(
            '{"session_id": "ancient", "timestamp": 1}', encoding="utf-8"
        )
        ancient = time.time() - 8 * 86400
        os.utime(old_file, (ancient, ancient))
        write_interrupted_marker("fresh", base=tmp_path)
        ids = [m["session_id"] for m in list_interrupted_markers(base=tmp_path)]
        assert "ancient" not in ids
        assert "fresh" in ids


# =========================================================================
# Periodic checkpoints
# =========================================================================

class TestMaybePeriodicCheckpoint:
    def test_disabled_manager_never_fires(self, tmp_path, ckpt_base):
        m = CheckpointManager(enabled=False)
        assert m.maybe_periodic_checkpoint(str(tmp_path)) is False
        assert m.maybe_periodic_checkpoint(str(tmp_path)) is False
        assert m._periodic_counter == 0

    def test_fires_every_interval(self, proj, ckpt_base):
        m = CheckpointManager(enabled=True, checkpoint_interval=2)
        assert m.maybe_periodic_checkpoint(str(proj)) is False
        assert m.maybe_periodic_checkpoint(str(proj)) is True

    def test_rate_limited_by_last_periodic_ts(self, proj, ckpt_base):
        m = CheckpointManager(enabled=True, checkpoint_interval=1)
        assert m.maybe_periodic_checkpoint(str(proj)) is True
        # Counter gate passes (interval=1) but the fresh timestamp blocks.
        assert m.maybe_periodic_checkpoint(str(proj)) is False
        # Backdate the clock -> fires again (with fresh content to snapshot).
        key = str(cm._normalize_path(str(proj)))
        m._last_periodic_ts[key] = time.time() - cm._PERIODIC_MIN_INTERVAL_S - 1
        (proj / "f.txt").write_text("changed\n")
        assert m.maybe_periodic_checkpoint(str(proj)) is True


# =========================================================================
# Mutation journal (always on, even when snapshots are disabled)
# =========================================================================

class TestMutationJournal:
    def test_records_when_snapshots_disabled(self, proj, ckpt_base):
        import hashlib as _hashlib
        target = proj / "notes.md"
        target.write_text("v1\n")
        v1bytes = target.read_bytes()
        m = CheckpointManager(enabled=False)
        m.note_pending_mutation(str(target), tool="write_file")
        target.write_text("v2\n")
        v2bytes = target.read_bytes()
        assert m.record_mutation(str(target), tool="write_file") is True
        entries = m.read_journal(str(proj))
        assert len(entries) == 1
        assert entries[0]["tool"] == "write_file"
        assert entries[0]["before"] == _hashlib.sha256(v1bytes).hexdigest()
        assert entries[0]["after"] == _hashlib.sha256(v2bytes).hexdigest()
        assert entries[0]["path"].endswith("notes.md")

    def test_before_is_none_without_preflight(self, proj, ckpt_base):
        m = CheckpointManager(enabled=False)
        assert m.record_mutation(str(proj / "f.txt"), tool="patch") is True
        entries = m.read_journal(str(proj))
        assert entries[0]["before"] is None
        assert entries[0]["after"] is not None

    def test_trims_to_cap(self, proj, ckpt_base, monkeypatch):
        monkeypatch.setattr(cm, "_JOURNAL_MAX_LINES", 4)
        m = CheckpointManager(enabled=False)
        # Enough entries to trip the size-gated trim (~200B each).
        for _ in range(20):
            assert m.record_mutation(str(proj / "f.txt"), tool="write_file") is True
        assert len(m.read_journal(str(proj), limit=100)) == 4


# =========================================================================
# Interruption note (continuation-prompt injection)
# =========================================================================

class TestInterruptionNote:
    def test_none_without_markers(self, tmp_path):
        assert build_interruption_note(base=tmp_path) is None

    def test_none_when_only_own_marker(self, tmp_path):
        write_interrupted_marker("mine", reason="in_flight", base=tmp_path)
        assert build_interruption_note(base=tmp_path, exclude_session_id="mine") is None

    def test_note_carries_reason_checkpoint_and_journal(self, tmp_path, proj, ckpt_base):
        mgr = CheckpointManager(enabled=True)
        assert mgr.ensure_checkpoint(str(proj), "baseline") is True
        mgr.note_pending_mutation(str(proj / "f.txt"), tool="write_file")
        (proj / "f.txt").write_text("two\n")
        assert mgr.record_mutation(str(proj / "f.txt"), tool="write_file") is True
        write_interrupted_marker(
            "dead-beef", reason="sigterm", last_action="write_file:f.txt", base=tmp_path
        )
        note = build_interruption_note(
            working_dir=str(proj), exclude_session_id="live", base=tmp_path
        )
        assert note is not None
        assert "dead-beef" in note
        assert "sigterm" in note
        assert "f.txt" in note
        assert "Last checkpoint" in note

    def test_stale_markers_ignored(self, tmp_path):
        marker = write_interrupted_marker("old", reason="sigterm", base=tmp_path)
        data = json.loads(marker.read_text(encoding="utf-8"))
        data["timestamp"] = time.time() - 2 * 86400
        marker.write_text(json.dumps(data), encoding="utf-8")
        assert build_interruption_note(base=tmp_path) is None


# =========================================================================
# Durable atomic writes + termination handlers
# =========================================================================

class TestDurableAtomicWrite:
    def test_str_and_bytes_roundtrip(self, tmp_path):
        p = tmp_path / "sub" / "out.txt"
        durable_atomic_write(str(p), "hello\n")
        assert p.read_text(encoding="utf-8") == "hello\n"
        durable_atomic_write(str(p), "hi\n".encode("utf-8"))
        assert p.read_text(encoding="utf-8") == "hi\n"
        assert list(tmp_path.rglob("*.tmp.*")) == []


class TestTerminationHandlers:
    def test_install_once_per_session(self):
        sid = "test-handler-session-xyz"
        old_term = _signal.getsignal(_signal.SIGTERM)
        old_int = _signal.getsignal(_signal.SIGINT)
        try:
            assert install_termination_handlers(sid) is True
            assert install_termination_handlers(sid) is False
        finally:
            _signal.signal(_signal.SIGTERM, old_term)
            _signal.signal(_signal.SIGINT, old_int)
            cm._installed_termination_sessions.discard(sid)
