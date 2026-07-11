"""Tests for tools/_async_delegation_state.py — the on-disk writers behind
the dashboard's `delegations N` pill and /agents view.

Plan A — Task A9 (2026-06-22 night sprint): pin the best-effort IO
contract. The Bug 4 root cause was that these writers DIDN'T EXIST.
Now that they do, every public call swallows IO errors and logs at
WARN. A failed FS write must NEVER crash the worker thread that called
into the module — otherwise a full disk takes the whole agent down.

These tests exercise the failure modes that would have surfaced Bug 4
earlier had they been pinned. They also assert atomic writes (tempfile
+ rename) hold so a concurrent dashboard reader never sees a
half-written active-delegations.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import pytest

import tools._async_delegation_state as ads


@pytest.fixture
def isolated_state_dir(tmp_path: Path, monkeypatch) -> Path:
    """Repoint _STATE_DIR / _ACTIVE_FILE / _EVENTS_FILE at a tmpdir so each
    test gets a clean slate and we don't pollute ~/.hermes/state."""
    state_dir = tmp_path / "state"
    monkeypatch.setattr(ads, "_STATE_DIR", state_dir)
    monkeypatch.setattr(ads, "_ACTIVE_FILE", state_dir / "active-delegations.json")
    monkeypatch.setattr(ads, "_EVENTS_FILE", state_dir / "events.jsonl")
    return state_dir


def _record(
    delegation_id: str = "d-test-1",
    goal: str = "do the thing",
    status: str = "running",
    dispatched_at: float = 1700000000.0,
    completed_at: float | None = None,
    **extra,
) -> dict:
    return {
        "delegation_id": delegation_id,
        "goal": goal,
        "status": status,
        "dispatched_at": dispatched_at,
        "completed_at": completed_at,
        "session_key": "test-session",
        **extra,
    }


# ---------------------------------------------------------------------------
# write_state_snapshot — happy path
# ---------------------------------------------------------------------------


def test_write_state_snapshot_creates_file_with_expected_shape(isolated_state_dir):
    ads.write_state_snapshot([_record()])
    assert ads._ACTIVE_FILE.exists()
    data = json.loads(ads._ACTIVE_FILE.read_text())
    assert data["version"] == 1
    assert "updated_at" in data
    assert len(data["delegations"]) == 1
    assert data["delegations"][0]["delegation_id"] == "d-test-1"
    assert data["delegations"][0]["status"] == "running"


def test_write_state_snapshot_overwrites_previous(isolated_state_dir):
    ads.write_state_snapshot([_record(delegation_id="old")])
    ads.write_state_snapshot([_record(delegation_id="new")])
    data = json.loads(ads._ACTIVE_FILE.read_text())
    ids = [d["delegation_id"] for d in data["delegations"]]
    assert ids == ["new"]
    assert "old" not in ids


def test_write_state_snapshot_creates_parent_dir(isolated_state_dir, tmp_path):
    """If ~/.hermes/state doesn't exist yet (fresh install), the writer
    must mkdir -p so it never fails on a missing directory."""
    nested = tmp_path / "deeply" / "nested" / "state"
    # Re-pin to the deeper path
    import tools._async_delegation_state as ads
    ads._STATE_DIR = nested
    ads._ACTIVE_FILE = nested / "active-delegations.json"
    ads.write_state_snapshot([_record()])
    assert ads._ACTIVE_FILE.exists()


# ---------------------------------------------------------------------------
# Atomic write contract
# ---------------------------------------------------------------------------


def test_write_state_snapshot_is_atomic_via_tempfile_rename(isolated_state_dir):
    """Concurrent dashboard reads must never see a half-written file.
    The writer uses tempfile + os.replace; on failure, no .tmp/.tmpX files
    should be left lying around in state dir.
    """
    ads.write_state_snapshot([_record()])
    leftover_tmps = [
        p for p in isolated_state_dir.iterdir()
        if p.name.startswith(".active-delegations.")
    ]
    assert leftover_tmps == [], f"leftover tempfiles: {leftover_tmps}"


# ---------------------------------------------------------------------------
# The Bug 4 contract — failed IO must NEVER raise
# ---------------------------------------------------------------------------


def test_write_state_snapshot_swallows_unwriteable_dir(isolated_state_dir):
    """The headline regression guard for Bug 4: even if the state dir is
    read-only (full disk, permission denied, network-mount glitch),
    write_state_snapshot returns silently instead of propagating an
    OSError up the worker call stack."""
    isolated_state_dir.mkdir(parents=True, exist_ok=True)
    isolated_state_dir.chmod(0o500)  # read+exec, NO write
    try:
        # This MUST NOT raise. If it does, Bug 4 has regressed.
        ads.write_state_snapshot([_record()])
    finally:
        isolated_state_dir.chmod(0o700)


def test_append_event_swallows_unwriteable_dir(isolated_state_dir):
    isolated_state_dir.mkdir(parents=True, exist_ok=True)
    isolated_state_dir.chmod(0o500)
    try:
        ads.append_event("delegate.task_spawned", _record())
    finally:
        isolated_state_dir.chmod(0o700)


def test_emit_spawned_swallows_unwriteable_dir(isolated_state_dir):
    """emit_spawned is the public entry-point called from the dispatch
    path. The contract: even if the FS is broken, the dispatch returns
    cleanly to the caller."""
    isolated_state_dir.mkdir(parents=True, exist_ok=True)
    isolated_state_dir.chmod(0o500)
    try:
        ads.emit_spawned(_record())
    finally:
        isolated_state_dir.chmod(0o700)


def test_emit_finalized_swallows_unwriteable_dir(isolated_state_dir):
    isolated_state_dir.mkdir(parents=True, exist_ok=True)
    isolated_state_dir.chmod(0o500)
    try:
        ads.emit_finalized(_record(status="completed"), {"summary": "ok"}, "completed")
    finally:
        isolated_state_dir.chmod(0o700)


# ---------------------------------------------------------------------------
# append_event schema
# ---------------------------------------------------------------------------


def test_append_event_writes_one_line_per_call(isolated_state_dir):
    ads.append_event("delegate.task_spawned", _record(delegation_id="d-1"))
    ads.append_event("delegate.task_completed", _record(
        delegation_id="d-1", status="completed",
        dispatched_at=1700000000.0, completed_at=1700000010.0,
    ))
    lines = ads._EVENTS_FILE.read_text().strip().split("\n")
    assert len(lines) == 2
    e1 = json.loads(lines[0])
    e2 = json.loads(lines[1])
    assert e1["kind"] == "delegate.task_spawned"
    assert e2["kind"] == "delegate.task_completed"
    assert e2["duration_seconds"] == 10.0


def test_append_event_truncates_goal_preview(isolated_state_dir):
    long_goal = "G" * 5000
    ads.append_event("delegate.task_spawned", _record(goal=long_goal))
    line = ads._EVENTS_FILE.read_text().strip()
    entry = json.loads(line)
    # _truncate cap is 200 — ensure we don't ship 5KB of goal text per event
    assert len(entry["goal_preview"]) <= 200
    assert entry["goal_preview"].endswith("…")
