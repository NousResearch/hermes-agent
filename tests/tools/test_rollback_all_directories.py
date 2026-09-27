"""Regression tests for the /rollback all-directories fallback (#10505).

Bare /rollback searched only TERMINAL_CWD's project; checkpoints created
under a different session cwd were invisible ("No checkpoints found for
/home/user" while checkpoints existed seconds earlier). Reapply of PR
#10633 by @nightq onto the v2 single-store layout.
"""

import json

import tools.checkpoint_manager as cm
from tools.checkpoint_manager import CheckpointManager, format_checkpoint_list


def _make_store_with_project(tmp_path, workdir: str):
    store = tmp_path / "store"
    (store / cm._PROJECTS_DIRNAME).mkdir(parents=True)
    (store / "HEAD").write_text("ref: refs/heads/main\n")
    dir_hash = cm._project_hash(workdir)
    meta = {"workdir": workdir, "created_at": 1, "last_touch": 2}
    (store / cm._PROJECTS_DIRNAME / f"{dir_hash}.json").write_text(json.dumps(meta))
    return store


class TestListAllCheckpoints:
    def test_aggregates_projects_and_labels_workdir(self, tmp_path, monkeypatch):
        workdir = str(tmp_path / "proj")
        store = _make_store_with_project(tmp_path, workdir)
        monkeypatch.setattr(cm, "CHECKPOINT_BASE", tmp_path)
        monkeypatch.setattr(cm, "_store_path", lambda base=None: store)

        mgr = CheckpointManager(enabled=True)
        fake_entries = [
            {"hash": "a" * 40, "short_hash": "aaaaaaa", "timestamp": "2026-08-19T10:00:00",
             "reason": "before write_file", "files_changed": 1, "insertions": 2, "deletions": 0},
        ]
        monkeypatch.setattr(
            CheckpointManager, "list_checkpoints", lambda self, wd: list(fake_entries)
        )

        results = mgr.list_all_checkpoints()
        assert len(results) == 1
        assert results[0]["workdir"] == workdir

    def test_empty_store_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cm, "CHECKPOINT_BASE", tmp_path)
        monkeypatch.setattr(cm, "_store_path", lambda base=None: tmp_path / "missing")
        assert CheckpointManager(enabled=True).list_all_checkpoints() == []


class TestFormatAllDirectories:
    def test_workdir_label_shown_in_all_directories_view(self):
        cps = [{
            "hash": "b" * 40, "short_hash": "bbbbbbb",
            "timestamp": "2026-08-19T10:00:00", "reason": "before patch",
            "files_changed": 0, "insertions": 0, "deletions": 0,
            "workdir": "/tmp/qa-repo",
        }]
        out = format_checkpoint_list(cps, "all directories")
        assert "[qa-repo]" in out

    def test_single_directory_view_unchanged(self):
        cps = [{
            "hash": "c" * 40, "short_hash": "ccccccc",
            "timestamp": "2026-08-19T10:00:00", "reason": "before patch",
            "files_changed": 0, "insertions": 0, "deletions": 0,
        }]
        out = format_checkpoint_list(cps, "/tmp/qa-repo")
        assert "[qa-repo]" not in out
        assert "ccccccc" in out
