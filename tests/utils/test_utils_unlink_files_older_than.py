"""``utils.unlink_files_older_than``: the shared age-based file sweep."""

import os
import time
from pathlib import Path

from utils import unlink_files_older_than


def _aged(path, age_seconds: float, now: float):
    path.write_text("x", encoding="utf-8")
    os.utime(path, (now - age_seconds, now - age_seconds))
    return path


def test_removes_only_matching_files_older_than_the_age(tmp_path):
    now = time.time()
    old = _aged(tmp_path / "old.json", 120, now)
    fresh = _aged(tmp_path / "fresh.json", 10, now)
    other = _aged(tmp_path / "old.txt", 120, now)
    (tmp_path / "olddir.json").mkdir()
    os.utime(tmp_path / "olddir.json", (now - 120, now - 120))

    assert unlink_files_older_than(tmp_path, "*.json", 60, now=now) == 1

    assert not old.exists()
    assert fresh.exists() and other.exists() and (tmp_path / "olddir.json").is_dir()


def test_missing_directory_is_zero_not_an_error(tmp_path):
    assert unlink_files_older_than(tmp_path / "absent", "*", 0) == 0


def test_unlink_failure_skips_that_file_and_keeps_sweeping(tmp_path, monkeypatch):
    now = time.time()
    stuck = _aged(tmp_path / "a.json", 120, now)
    later = _aged(tmp_path / "b.json", 120, now)
    real_unlink = Path.unlink

    def unlink(self, *args, **kwargs):
        if self.name == "a.json":
            raise PermissionError("locked")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter([self / "a.json", self / "b.json"]))  # fail first

    assert unlink_files_older_than(tmp_path, "*.json", 60, now=now) == 1
    assert stuck.exists() and not later.exists()
