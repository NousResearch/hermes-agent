from __future__ import annotations

import pytest

from workstation.soak import run_workstation_soak


def test_workstation_soak_reconnects_persistent_worker_and_journals(tmp_path):
    report = run_workstation_soak(duration_seconds=10, max_iterations=2, root=tmp_path / "soak")

    assert report.result.failures == 0
    assert report.result.errors == []
    assert report.result.completed_iterations == 2
    assert report.result.timed_out is False
    assert report.journal_events == 24
    assert report.session_count == 3
    assert report.process_restarts == 2
    assert report.memory_snapshots == 6
    assert report.max_live_memory_records <= report.session_count * 8
    assert report.migrated_sessions == 3
    assert report.model_changes == 3
    assert report.cold_reloads == 3
    assert report.reconstructed_workers == 3
    assert report.retained_root is True


def test_workstation_soak_rejects_negative_bounds():
    with pytest.raises(ValueError, match="non-negative"):
        run_workstation_soak(duration_seconds=-1, max_iterations=1)
    with pytest.raises(ValueError, match="non-negative"):
        run_workstation_soak(duration_seconds=1, max_iterations=-1)
