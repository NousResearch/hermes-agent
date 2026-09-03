"""Snapshot artifact ownership, stale cleanup, and inode admission."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.environments.local import LocalEnvironment
from tools.environments.snapshot_lifecycle import (
    FAIL_CLOSED,
    DEFER,
    RUN,
    InodeHeadroom,
    SnapshotLifecycleSettings,
    cleanup_owned_artifacts,
    decide_inode_admission,
    prepare_owned_artifacts,
    reap_stale_owned_artifacts,
)


def _settings(**overrides) -> SnapshotLifecycleSettings:
    values = {
        "ttl_seconds": 86_400,
        "min_free_inode_ratio": 0.15,
        "critical_free_inode_ratio": 0.10,
    }
    values.update(overrides)
    return SnapshotLifecycleSettings(**values)


@pytest.mark.parametrize(
    ("ratio", "outcome"),
    [
        (0.150001, RUN),
        (0.15, DEFER),
        (0.12, DEFER),
        (0.10, DEFER),
        (0.099999, FAIL_CLOSED),
        (0.0, FAIL_CLOSED),
        (None, FAIL_CLOSED),
    ],
)
def test_inode_admission_boundaries(ratio, outcome):
    result = decide_inode_admission(ratio, _settings())
    assert result.outcome == outcome
    assert result.free_inode_ratio == ratio


def test_invalid_thresholds_fail_closed():
    result = decide_inode_admission(
        0.5,
        _settings(min_free_inode_ratio=0.05, critical_free_inode_ratio=0.10),
    )
    assert result.outcome == FAIL_CLOSED
    assert result.reason == "INVALID_THRESHOLDS"


def test_low_ratio_with_ample_absolute_headroom_runs():
    result = decide_inode_admission(0.0276, _settings(), free_inodes=1_600_000)
    assert result.outcome == RUN
    assert result.reason == "INODES_AVAILABLE"


def test_critically_low_absolute_headroom_fails_even_with_high_ratio():
    result = decide_inode_admission(0.80, _settings(), free_inodes=999)
    assert result.outcome == FAIL_CLOSED
    assert result.reason == "INODES_CRITICAL"


def test_low_absolute_headroom_defers_snapshot():
    result = decide_inode_admission(0.80, _settings(), free_inodes=5_000)
    assert result.outcome == DEFER
    assert result.reason == "INODE_PRESSURE"


def test_absolute_headroom_can_admit_when_ratio_is_unavailable():
    result = decide_inode_admission(None, _settings(), free_inodes=100_000)
    assert result.outcome == RUN


def test_invalid_absolute_thresholds_fail_closed():
    result = decide_inode_admission(
        0.80,
        _settings(min_free_inodes=1_000, critical_free_inodes=1_000),
        free_inodes=100_000,
    )
    assert result.outcome == FAIL_CLOSED
    assert result.reason == "INVALID_THRESHOLDS"


@pytest.mark.parametrize("ttl", [float("nan"), float("inf"), -1.0])
def test_invalid_ttl_fails_closed(ttl):
    result = decide_inode_admission(0.5, _settings(ttl_seconds=ttl))
    assert result.outcome == FAIL_CLOSED
    assert result.reason == "INVALID_THRESHOLDS"


def test_prepare_writes_private_owner_marker(tmp_path):
    owned = prepare_owned_artifacts(
        tmp_path,
        "a1b2c3d4e5f6",
        now=1000.0,
        pid=4321,
        uid=os.getuid() if hasattr(os, "getuid") else None,
        hostname="test-host",
    )
    marker = Path(owned.marker_path)
    assert marker.exists()
    assert marker.stat().st_mode & 0o777 == 0o600
    payload = json.loads(marker.read_text())
    assert payload["session_id"] == "a1b2c3d4e5f6"
    assert payload["pid"] == 4321
    assert Path(owned.snapshot_path).parent == tmp_path
    assert Path(owned.cwd_path).parent == tmp_path


def test_symlink_temp_root_resolves_to_stable_real_directory(tmp_path):
    real_root = tmp_path / "real"
    real_root.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real_root, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    owned = prepare_owned_artifacts(
        alias,
        "a1b2c3d4e5f6",
        hostname="test-host",
    )

    assert Path(owned.marker_path).parent == real_root
    assert Path(owned.snapshot_path).parent == real_root
    cleanup_owned_artifacts(owned)


def test_cleanup_removes_only_exact_owned_artifacts(tmp_path):
    owned = prepare_owned_artifacts(
        tmp_path,
        "a1b2c3d4e5f6",
        now=1000.0,
        pid=4321,
        uid=os.getuid() if hasattr(os, "getuid") else None,
        hostname="test-host",
    )
    Path(owned.snapshot_path).write_text("snapshot")
    Path(owned.cwd_path).write_text("cwd")
    temp = Path(owned.snapshot_path + ".tmp.deadbeef")
    temp.write_text("partial")
    foreign = tmp_path / "hermes-snap-ffffffffffff.sh"
    foreign.write_text("foreign")

    removed = cleanup_owned_artifacts(owned)

    assert set(removed) == {
        owned.snapshot_path,
        owned.cwd_path,
        str(temp),
        owned.marker_path,
    }
    assert foreign.exists()


def test_stale_reaper_requires_valid_marker_ttl_and_dead_pid(tmp_path):
    stale = prepare_owned_artifacts(
        tmp_path,
        "a1b2c3d4e5f6",
        now=1000.0,
        pid=4321,
        uid=os.getuid() if hasattr(os, "getuid") else None,
        hostname="test-host",
    )
    Path(stale.snapshot_path).write_text("snapshot")

    active = prepare_owned_artifacts(
        tmp_path,
        "111111111111",
        now=1000.0,
        pid=9999,
        uid=os.getuid() if hasattr(os, "getuid") else None,
        hostname="test-host",
    )
    Path(active.snapshot_path).write_text("active")

    reaped = reap_stale_owned_artifacts(
        tmp_path,
        _settings(ttl_seconds=100),
        now=1200.0,
        uid=os.getuid() if hasattr(os, "getuid") else None,
        hostname="test-host",
        pid_alive=lambda pid: pid == 9999,
    )

    assert stale.session_id in reaped
    assert not Path(stale.marker_path).exists()
    assert Path(active.marker_path).exists()
    assert Path(active.snapshot_path).exists()


def test_stale_reaper_refuses_lookalike_foreign_and_symlink_markers(tmp_path):
    lookalike_snapshot = tmp_path / "hermes-snap-222222222222.sh"
    lookalike_snapshot.write_text("keep")
    malformed = tmp_path / "hermes-session-222222222222.owner.json"
    malformed.write_text("not-json")

    foreign = prepare_owned_artifacts(
        tmp_path,
        "333333333333",
        now=1000.0,
        pid=1234,
        uid=(os.getuid() + 1) if hasattr(os, "getuid") else 999,
        hostname="other-host",
    )
    Path(foreign.snapshot_path).write_text("keep")

    real_marker = tmp_path / "real-marker"
    real_marker.write_text("{}")
    symlink_marker = tmp_path / "hermes-session-444444444444.owner.json"
    try:
        symlink_marker.symlink_to(real_marker)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    reaped = reap_stale_owned_artifacts(
        tmp_path,
        _settings(ttl_seconds=100),
        now=1200.0,
        uid=os.getuid() if hasattr(os, "getuid") else None,
        hostname="test-host",
        pid_alive=lambda _pid: False,
    )

    assert reaped == []
    assert lookalike_snapshot.exists()
    assert Path(foreign.snapshot_path).exists()
    assert symlink_marker.is_symlink()


def test_local_environment_run_band_owns_and_cleans_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(LocalEnvironment, "get_temp_dir", lambda _self: str(tmp_path))
    monkeypatch.setattr(
        "tools.environments.local.measure_inode_headroom",
        lambda _path: InodeHeadroom(0.20, 100_000),
    )

    def fake_init(self):
        Path(self._snapshot_path).write_text("snapshot")
        self._snapshot_ready = True

    monkeypatch.setattr(LocalEnvironment, "init_session", fake_init)
    env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
    owned = env._owned_snapshot_artifacts
    assert owned is not None
    marker = Path(owned.marker_path)
    assert marker.exists()
    assert Path(env._snapshot_path).exists()
    assert env._snapshot_inode_admission.outcome == RUN

    env.cleanup()

    assert not marker.exists()
    assert not Path(env._snapshot_path).exists()


def test_local_environment_defer_band_uses_no_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(LocalEnvironment, "get_temp_dir", lambda _self: str(tmp_path))
    monkeypatch.setattr(
        "tools.environments.local.measure_inode_headroom",
        lambda _path: InodeHeadroom(0.12, 5_000),
    )
    called = []
    monkeypatch.setattr(LocalEnvironment, "init_session", lambda _self: called.append(True))

    env = LocalEnvironment(cwd=str(tmp_path), timeout=10)

    assert called == []
    assert env._snapshot_inode_admission.outcome == DEFER
    assert env._owned_snapshot_artifacts is None
    assert not list(tmp_path.glob("hermes-session-*.owner.json"))


def test_local_environment_critical_band_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(LocalEnvironment, "get_temp_dir", lambda _self: str(tmp_path))
    monkeypatch.setattr(
        "tools.environments.local.measure_inode_headroom",
        lambda _path: InodeHeadroom(0.05, 500),
    )
    called = []
    monkeypatch.setattr(LocalEnvironment, "init_session", lambda _self: called.append(True))

    with pytest.raises(RuntimeError, match="INODES_CRITICAL"):
        LocalEnvironment(cwd=str(tmp_path), timeout=10)

    assert called == []
