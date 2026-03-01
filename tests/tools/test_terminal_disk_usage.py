"""Tests for get_active_environments_info disk usage calculation."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tools.terminal_tool import get_active_environments_info

# 1 MiB of data so the rounded MB value is clearly distinguishable
_1MB = b"x" * (1024 * 1024)


@pytest.fixture()
def fake_scratch(tmp_path):
    """Create fake hermes scratch directories with known sizes."""
    # Task A: 1 MiB
    task_a_dir = tmp_path / "hermes-sandbox-aaaaaaaa"
    task_a_dir.mkdir()
    (task_a_dir / "data.bin").write_bytes(_1MB)

    # Task B: 1 MiB
    task_b_dir = tmp_path / "hermes-sandbox-bbbbbbbb"
    task_b_dir.mkdir()
    (task_b_dir / "data.bin").write_bytes(_1MB)

    return tmp_path


class TestDiskUsageGlob:
    def test_only_counts_matching_task_dirs(self, fake_scratch):
        """Each task should only count its own directories, not all hermes-* dirs."""
        fake_envs = {
            "aaaaaaaa-1111-2222-3333-444444444444": MagicMock(),
        }

        with (
            patch("tools.terminal_tool._active_environments", fake_envs),
            patch("tools.terminal_tool._get_scratch_dir", return_value=fake_scratch),
        ):
            info = get_active_environments_info()

        # Task A only: ~1.0 MB. With the bug (hardcoded hermes-*),
        # it would also count task B -> ~2.0 MB.
        assert info["total_disk_usage_mb"] == pytest.approx(1.0, abs=0.1)

    def test_multiple_tasks_no_double_counting(self, fake_scratch):
        """With 2 active tasks, each should count only its own dirs."""
        fake_envs = {
            "aaaaaaaa-1111-2222-3333-444444444444": MagicMock(),
            "bbbbbbbb-5555-6666-7777-888888888888": MagicMock(),
        }

        with (
            patch("tools.terminal_tool._active_environments", fake_envs),
            patch("tools.terminal_tool._get_scratch_dir", return_value=fake_scratch),
        ):
            info = get_active_environments_info()

        # Should be ~2.0 MB total (1 MB per task).
        # With the bug, each task globs everything -> ~4.0 MB.
        assert info["total_disk_usage_mb"] == pytest.approx(2.0, abs=0.1)
