"""Tests for tools/checkpoint_manager.py — CheckpointManager (v2 single-store)."""

import json
import logging
import os
import subprocess
import time
import pytest
from pathlib import Path
from unittest.mock import patch

from tools.checkpoint_manager import (
    CheckpointManager,
    _shadow_repo_path,
    _init_shadow_repo,
    _init_store,
    _run_git,
    _git_env,
    _dir_file_count,
    _project_hash,
    _store_path,
    _ref_name,
    _project_meta_path,
    _touch_project,
    format_checkpoint_list,
    DEFAULT_EXCLUDES,
    CHECKPOINT_BASE,
    prune_checkpoints,
    maybe_auto_prune_checkpoints,
    store_status,
    clear_all,
    clear_legacy,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture()
def work_dir(tmp_path):
    d = tmp_path / "project"
    d.mkdir()
    (d / "main.py").write_text("print('hello')\n")
    (d / "README.md").write_text("# Project\n")
    return d


@pytest.fixture()
def checkpoint_base(tmp_path):
    """Isolated checkpoint base — never writes to ~/.hermes/."""
    return tmp_path / "checkpoints"


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("HOMEDRIVE", raising=False)
    monkeypatch.delenv("HOMEPATH", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


@pytest.fixture()
def mgr(work_dir, checkpoint_base, monkeypatch):
    monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
    return CheckpointManager(enabled=True, max_snapshots=50)


@pytest.fixture()
def disabled_mgr(checkpoint_base, monkeypatch):
    monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
    return CheckpointManager(enabled=False)


# =========================================================================
# Store path + project hash
# =========================================================================

class TestStorePath:
    def test_store_is_single_shared_path(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        # All projects resolve to the same store.
        p1 = _shadow_repo_path(str(work_dir))
        p2 = _shadow_repo_path(str(work_dir.parent / "other"))
        assert p1 == p2 == _store_path(checkpoint_base)

    def test_project_hash_deterministic(self, work_dir):
        assert _project_hash(str(work_dir)) == _project_hash(str(work_dir))

    def test_project_hash_differs_per_dir(self, tmp_path):
        assert _project_hash(str(tmp_path / "a")) != _project_hash(str(tmp_path / "b"))

    def test_tilde_and_expanded_home_share_project_hash(
        self, fake_home, checkpoint_base, monkeypatch,
    ):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        project = fake_home / "project"
        project.mkdir()
        tilde = f"~/{project.name}"
        assert _project_hash(tilde) == _project_hash(str(project))


# =========================================================================
# Store init + legacy migration
# =========================================================================

class TestStoreInit:
    def test_creates_git_store(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        store = _store_path(checkpoint_base)
        err = _init_store(store, str(work_dir))
        assert err is None
        assert (store / "HEAD").exists()
        assert (store / "objects").exists()
        assert (store / "info" / "exclude").exists()
        assert "node_modules/" in (store / "info" / "exclude").read_text()

    def test_no_git_in_project_dir(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        store = _store_path(checkpoint_base)
        _init_store(store, str(work_dir))
        assert not (work_dir / ".git").exists()

    def test_init_idempotent(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        store = _store_path(checkpoint_base)
        assert _init_store(store, str(work_dir)) is None
        assert _init_store(store, str(work_dir)) is None

    def test_bc_init_shadow_repo_shim(self, work_dir, checkpoint_base, monkeypatch):
        """Backward-compatible helper still works for old callers/tests."""
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        store = _shadow_repo_path(str(work_dir))
        err = _init_shadow_repo(store, str(work_dir))
        assert err is None
        assert (store / "HEAD").exists()
        assert (store / "HERMES_WORKDIR").exists()

    def test_legacy_migration_archives_prev2_repos(
        self, checkpoint_base, work_dir,
    ):
        """Pre-v2 per-project shadow repos get moved into legacy-<ts>/."""
        base = checkpoint_base
        base.mkdir(parents=True)
        # Simulate a pre-v2 repo directly under base
        fake_repo = base / "deadbeefcafebabe"
        fake_repo.mkdir()
        (fake_repo / "HEAD").write_text("ref: refs/heads/main\n")
        (fake_repo / "HERMES_WORKDIR").write_text(str(work_dir) + "\n")
        (fake_repo / "objects").mkdir()

        # Init store — should migrate the fake pre-v2 repo
        store = _store_path(base)
        err = _init_store(store, str(work_dir))
        assert err is None

        assert not fake_repo.exists()
        legacies = [p for p in base.iterdir() if p.name.startswith("legacy-")]
        assert len(legacies) == 1
        assert (legacies[0] / fake_repo.name).exists()
        assert (legacies[0] / fake_repo.name / "HEAD").exists()


# =========================================================================
# CheckpointManager — disabled
# =========================================================================

class TestDisabledManager:
    def test_ensure_checkpoint_returns_false(self, disabled_mgr, work_dir):
        assert disabled_mgr.ensure_checkpoint(str(work_dir)) is False

    def test_new_turn_works(self, disabled_mgr):
        disabled_mgr.new_turn()


# =========================================================================
# CheckpointManager — taking checkpoints
# =========================================================================

class TestTakeCheckpoint:
    def test_first_checkpoint(self, mgr, work_dir):
        result = mgr.ensure_checkpoint(str(work_dir), "initial")
        assert result is True

    def test_dedup_same_turn(self, mgr, work_dir):
        r1 = mgr.ensure_checkpoint(str(work_dir), "first")
        r2 = mgr.ensure_checkpoint(str(work_dir), "second")
        assert r1 is True
        assert r2 is False  # dedup'd

    def test_new_turn_resets_dedup(self, mgr, work_dir):
        assert mgr.ensure_checkpoint(str(work_dir), "turn 1") is True
        mgr.new_turn()
        (work_dir / "main.py").write_text("print('modified')\n")
        assert mgr.ensure_checkpoint(str(work_dir), "turn 2") is True

    def test_no_changes_skips_commit(self, mgr, work_dir):
        mgr.ensure_checkpoint(str(work_dir), "initial")
        mgr.new_turn()
        assert mgr.ensure_checkpoint(str(work_dir), "no changes") is False

    def test_skip_root_dir(self, mgr):
        assert mgr.ensure_checkpoint("/", "root") is False

    def test_skip_home_dir(self, mgr):
        assert mgr.ensure_checkpoint(str(Path.home()), "home") is False

    def test_multiple_projects_share_store(self, mgr, tmp_path):
        """Two projects commit to the SAME shared store (dedup wins)."""
        a = tmp_path / "proj-a"
        a.mkdir()
        (a / "f.py").write_text("a\n")
        b = tmp_path / "proj-b"
        b.mkdir()
        (b / "g.py").write_text("b\n")

        assert mgr.ensure_checkpoint(str(a), "a") is True
        mgr.new_turn()
        assert mgr.ensure_checkpoint(str(b), "b") is True

        # Only one "store" directory exists.
        bases = list(Path(mgr._checkpointed_dirs).__iter__()) if False else None
        from tools.checkpoint_manager import CHECKPOINT_BASE as BASE
        # Exactly one store dir + two project metas
        assert (BASE / "store" / "HEAD").exists()
        assert (BASE / "store" / "projects" / f"{_project_hash(str(a))}.json").exists()
        assert (BASE / "store" / "projects" / f"{_project_hash(str(b))}.json").exists()


# =========================================================================
# CheckpointManager — listing
# =========================================================================

class TestListCheckpoints:
    def test_empty_when_no_checkpoints(self, mgr, work_dir):
        assert mgr.list_checkpoints(str(work_dir)) == []

    def test_list_after_take(self, mgr, work_dir):
        mgr.ensure_checkpoint(str(work_dir), "test checkpoint")
        result = mgr.list_checkpoints(str(work_dir))
        assert len(result) == 1
        assert result[0]["reason"] == "test checkpoint"
        assert "hash" in result[0]
        assert "short_hash" in result[0]
        assert "timestamp" in result[0]

    def test_multiple_checkpoints_ordered(self, mgr, work_dir):
        mgr.ensure_checkpoint(str(work_dir), "first")
        mgr.new_turn()
        (work_dir / "main.py").write_text("v2\n")
        mgr.ensure_checkpoint(str(work_dir), "second")
        mgr.new_turn()
        (work_dir / "main.py").write_text("v3\n")
        mgr.ensure_checkpoint(str(work_dir), "third")

        result = mgr.list_checkpoints(str(work_dir))
        assert len(result) == 3
        assert result[0]["reason"] == "third"
        assert result[2]["reason"] == "first"

    def test_list_isolated_per_project(self, mgr, tmp_path):
        """Listing one project doesn't leak checkpoints from another."""
        a = tmp_path / "a"
        a.mkdir()
        (a / "f").write_text("A\n")
        b = tmp_path / "b"
        b.mkdir()
        (b / "g").write_text("B\n")

        mgr.ensure_checkpoint(str(a), "A-1")
        mgr.new_turn()
        mgr.ensure_checkpoint(str(b), "B-1")

        assert [c["reason"] for c in mgr.list_checkpoints(str(a))] == ["A-1"]
        assert [c["reason"] for c in mgr.list_checkpoints(str(b))] == ["B-1"]

    def test_tilde_path_lists_same_checkpoints(self, checkpoint_base, fake_home, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        m = CheckpointManager(enabled=True, max_snapshots=50)
        project = fake_home / "project"
        project.mkdir()
        (project / "main.py").write_text("v1\n")
        assert m.ensure_checkpoint(f"~/{project.name}", "initial") is True
        listed = m.list_checkpoints(str(project))
        assert len(listed) == 1
        assert listed[0]["reason"] == "initial"


# =========================================================================
# Pruning: max_snapshots actually enforced (v2 fix)
# =========================================================================

class TestRealPruning:
    def test_max_snapshots_trims_history(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        # Tiny cap to test enforcement.
        m = CheckpointManager(enabled=True, max_snapshots=3)

        for i in range(6):
            (work_dir / "main.py").write_text(f"v{i}\n")
            m.new_turn()
            m.ensure_checkpoint(str(work_dir), f"step-{i}")

        cps = m.list_checkpoints(str(work_dir))
        assert len(cps) == 3
        reasons = [c["reason"] for c in cps]
        # Newest first — step-5, step-4, step-3
        assert reasons[0] == "step-5"
        assert reasons[-1] == "step-3"

    def test_max_file_size_mb_skips_large_files(
        self, tmp_path, checkpoint_base, monkeypatch,
    ):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        wd = tmp_path / "proj"
        wd.mkdir()
        (wd / "small.py").write_text("tiny\n")
        big = wd / "weights.bin"
        big.write_bytes(b"\0" * (2 * 1024 * 1024))  # 2 MB

        m = CheckpointManager(enabled=True, max_snapshots=5, max_file_size_mb=1)
        assert m.ensure_checkpoint(str(wd), "initial") is True

        store = _store_path(checkpoint_base)
        ok, files, _ = _run_git(
            ["ls-tree", "-r", "--name-only", _ref_name(_project_hash(str(wd)))],
            store, str(wd),
        )
        assert ok
        names = set(files.splitlines())
        assert "small.py" in names
        assert "weights.bin" not in names  # filtered by size cap


# =========================================================================
# CheckpointManager — restoring
# =========================================================================

class TestRestore:
    def test_restore_to_previous(self, mgr, work_dir):
        (work_dir / "main.py").write_text("original\n")
        mgr.ensure_checkpoint(str(work_dir), "original state")
        mgr.new_turn()

        (work_dir / "main.py").write_text("modified\n")

        cps = mgr.list_checkpoints(str(work_dir))
        assert len(cps) == 1

        result = mgr.restore(str(work_dir), cps[0]["hash"])
        assert result["success"] is True
        assert (work_dir / "main.py").read_text() == "original\n"

    def test_restore_invalid_hash(self, mgr, work_dir):
        mgr.ensure_checkpoint(str(work_dir), "initial")
        result = mgr.restore(str(work_dir), "deadbeef1234")
        assert result["success"] is False

    def test_restore_no_checkpoints(self, mgr, work_dir):
        result = mgr.restore(str(work_dir), "abc123")
        assert result["success"] is False

    def test_restore_creates_pre_rollback_snapshot(self, mgr, work_dir):
        (work_dir / "main.py").write_text("v1\n")
        mgr.ensure_checkpoint(str(work_dir), "v1")
        mgr.new_turn()

        (work_dir / "main.py").write_text("v2\n")
        cps = mgr.list_checkpoints(str(work_dir))
        mgr.restore(str(work_dir), cps[0]["hash"])

        all_cps = mgr.list_checkpoints(str(work_dir))
        assert len(all_cps) >= 2
        assert "pre-rollback" in all_cps[0]["reason"]

    def test_tilde_path_supports_diff_and_restore_flow(
        self, checkpoint_base, fake_home, monkeypatch,
    ):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        m = CheckpointManager(enabled=True, max_snapshots=50)
        project = fake_home / "project"
        project.mkdir()
        file_path = project / "main.py"
        file_path.write_text("original\n")

        tilde = f"~/{project.name}"
        assert m.ensure_checkpoint(tilde, "initial") is True
        m.new_turn()

        file_path.write_text("changed\n")
        cps = m.list_checkpoints(str(project))
        diff_result = m.diff(tilde, cps[0]["hash"])
        assert diff_result["success"] is True
        assert "main.py" in diff_result["diff"]

        restore_result = m.restore(tilde, cps[0]["hash"])
        assert restore_result["success"] is True
        assert file_path.read_text() == "original\n"


# =========================================================================
# CheckpointManager — working dir resolution
# =========================================================================

class TestWorkingDirResolution:
    def test_resolves_git_project_root(self, tmp_path):
        m = CheckpointManager(enabled=True)
        project = tmp_path / "myproject"
        project.mkdir()
        (project / ".git").mkdir()
        subdir = project / "src"
        subdir.mkdir()
        filepath = subdir / "main.py"
        filepath.write_text("x\n")

        assert m.get_working_dir_for_path(str(filepath)) == str(project)

    def test_resolves_pyproject_root(self, tmp_path):
        m = CheckpointManager(enabled=True)
        project = tmp_path / "pyproj"
        project.mkdir()
        (project / "pyproject.toml").write_text("[project]\n")
        subdir = project / "src"
        subdir.mkdir()
        assert m.get_working_dir_for_path(str(subdir / "file.py")) == str(project)

    def test_falls_back_to_parent(self, tmp_path, monkeypatch):
        m = CheckpointManager(enabled=True)
        filepath = tmp_path / "random" / "file.py"
        filepath.parent.mkdir(parents=True)
        filepath.write_text("x\n")

        import pathlib as _pl
        _real_exists = _pl.Path.exists

        def _guarded_exists(self):
            s = str(self)
            stop = str(tmp_path)
            if not s.startswith(stop) and any(
                s.endswith("/" + m) or s == "/" + m
                for m in (".git", "pyproject.toml", "package.json",
                          "Cargo.toml", "go.mod", "Makefile", "pom.xml",
                          ".hg", "Gemfile")
            ):
                return False
            return _real_exists(self)

        monkeypatch.setattr(_pl.Path, "exists", _guarded_exists)
        assert m.get_working_dir_for_path(str(filepath)) == str(filepath.parent)

    def test_resolves_tilde_path_to_project_root(self, fake_home):
        m = CheckpointManager(enabled=True)
        project = fake_home / "myproject"
        project.mkdir()
        (project / "pyproject.toml").write_text("[project]\n")
        subdir = project / "src"
        subdir.mkdir()
        filepath = subdir / "main.py"
        filepath.write_text("x\n")

        assert m.get_working_dir_for_path(
            f"~/{project.name}/src/main.py"
        ) == str(project)


# =========================================================================
# Git env isolation
# =========================================================================

class TestGitEnvIsolation:
    def test_sets_git_dir(self, tmp_path):
        store = tmp_path / "store"
        env = _git_env(store, str(tmp_path / "work"))
        assert env["GIT_DIR"] == str(store)

    def test_sets_work_tree(self, tmp_path):
        store = tmp_path / "store"
        work = tmp_path / "work"
        env = _git_env(store, str(work))
        assert env["GIT_WORK_TREE"] == str(work.resolve())

    def test_clears_index_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GIT_INDEX_FILE", "/some/index")
        env = _git_env(tmp_path / "store", str(tmp_path))
        assert "GIT_INDEX_FILE" not in env

    def test_sets_index_file_when_provided(self, tmp_path):
        env = _git_env(
            tmp_path / "store", str(tmp_path),
            index_file=tmp_path / "store" / "indexes" / "abc",
        )
        assert env["GIT_INDEX_FILE"].endswith("indexes/abc")

    def test_expands_tilde_in_work_tree(self, fake_home, tmp_path):
        work = fake_home / "work"
        work.mkdir()
        env = _git_env(tmp_path / "store", f"~/{work.name}")
        assert env["GIT_WORK_TREE"] == str(work.resolve())


# =========================================================================
# format_checkpoint_list
# =========================================================================

class TestFormatCheckpointList:
    def test_empty_list(self):
        assert "No checkpoints" in format_checkpoint_list([], "/some/dir")

    def test_formats_entries(self):
        cps = [
            {"hash": "abc123", "short_hash": "abc1",
             "timestamp": "2026-03-09T21:15:00-07:00",
             "reason": "before write_file"},
            {"hash": "def456", "short_hash": "def4",
             "timestamp": "2026-03-09T21:10:00-07:00",
             "reason": "before patch"},
        ]
        result = format_checkpoint_list(cps, "/home/user/project")
        assert "abc1" in result
        assert "def4" in result
        assert "before write_file" in result
        assert "/rollback" in result


# =========================================================================
# Dir size / file count guards
# =========================================================================

class TestDirFileCount:
    def test_counts_files(self, work_dir):
        assert _dir_file_count(str(work_dir)) >= 2

    def test_nonexistent_dir(self, tmp_path):
        assert _dir_file_count(str(tmp_path / "nonexistent")) == 0


# =========================================================================
# Error resilience
# =========================================================================

class TestErrorResilience:
    def test_no_git_installed(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        m = CheckpointManager(enabled=True)
        monkeypatch.setattr("shutil.which", lambda x: None)
        m._git_available = None
        assert m.ensure_checkpoint(str(work_dir), "test") is False

    def test_run_git_allows_expected_nonzero_without_error_log(
        self, tmp_path, caplog,
    ):
        work = tmp_path / "work"
        work.mkdir()
        completed = subprocess.CompletedProcess(
            args=["git", "diff", "--cached", "--quiet"],
            returncode=1, stdout="", stderr="",
        )
        with patch("tools.checkpoint_manager.subprocess.run", return_value=completed):
            with caplog.at_level(logging.ERROR, logger="tools.checkpoint_manager"):
                ok, stdout, stderr = _run_git(
                    ["diff", "--cached", "--quiet"],
                    tmp_path / "store", str(work),
                    allowed_returncodes={1},
                )
        assert ok is False
        assert stdout == ""
        assert not caplog.records

    def test_run_git_invalid_working_dir_reports_path_error(self, tmp_path, caplog):
        missing = tmp_path / "missing"
        with caplog.at_level(logging.ERROR, logger="tools.checkpoint_manager"):
            ok, _, stderr = _run_git(
                ["status"], tmp_path / "store", str(missing),
            )
        assert ok is False
        assert "working directory not found" in stderr
        assert not any(
            "Git executable not found" in r.getMessage() for r in caplog.records
        )

    def test_run_git_missing_git_reports_git_not_found(
        self, tmp_path, monkeypatch, caplog,
    ):
        work = tmp_path / "work"
        work.mkdir()

        def raise_missing_git(*args, **kwargs):
            raise FileNotFoundError(2, "No such file or directory", "git")

        monkeypatch.setattr("tools.checkpoint_manager.subprocess.run", raise_missing_git)
        with caplog.at_level(logging.ERROR, logger="tools.checkpoint_manager"):
            ok, _, stderr = _run_git(
                ["status"], tmp_path / "store", str(work),
            )
        assert ok is False
        assert stderr == "git not found"
        assert any(
            "Git executable not found" in r.getMessage() for r in caplog.records
        )

    def test_checkpoint_failure_does_not_raise(self, mgr, work_dir, monkeypatch):
        def broken_run_git(*args, **kwargs):
            raise OSError("git exploded")
        monkeypatch.setattr("tools.checkpoint_manager._run_git", broken_run_git)
        assert mgr.ensure_checkpoint(str(work_dir), "test") is False


class TestTouchProjectMalformedMeta:
    """_touch_project must not raise when the project metadata file is corrupted.

    The try/except in _touch_project only catches ``(OSError, ValueError)``.
    When ``json.load`` succeeds but returns a non-dict (e.g. a list ``[]``,
    ``null``, or a scalar), the subsequent ``meta["workdir"] = ...`` raises
    ``TypeError: list indices must be integers…``.  This TypeError propagates
    uncaught out of ``_touch_project`` and up through ``_take`` into
    ``ensure_checkpoint``, where it is swallowed by the broad ``except
    Exception`` safety net — but the effect is that the checkpoint is silently
    skipped for the entire session.

    Fix: add ``if not isinstance(meta, dict): meta = {}`` after parsing,
    mirroring the same guard already present in ``_list_projects``.
    """

    @pytest.mark.parametrize("payload", ["[]", "null", "42", '"oops"'])
    def test_non_dict_meta_does_not_raise(self, tmp_path, payload):
        store = tmp_path / "store"
        workdir = str(tmp_path / "project")
        _init_store(store, workdir)

        dir_hash = _project_hash(workdir)
        meta_path = _project_meta_path(store, dir_hash)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(payload, encoding="utf-8")

        # Must not raise TypeError
        _touch_project(store, workdir)

        # Metadata file should now be a valid dict with last_touch updated
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "last_touch" in data
        assert "workdir" in data


# =========================================================================
# Security / input validation
# =========================================================================

class TestSecurity:
    def test_restore_rejects_argument_injection(self, mgr, work_dir):
        mgr.ensure_checkpoint(str(work_dir), "initial")
        result = mgr.restore(str(work_dir), "--patch")
        assert result["success"] is False
        assert "Invalid commit hash" in result["error"]
        assert "must not start with '-'" in result["error"]

        result = mgr.restore(str(work_dir), "-p")
        assert result["success"] is False
        assert "Invalid commit hash" in result["error"]

    def test_restore_rejects_invalid_hex_chars(self, mgr, work_dir):
        mgr.ensure_checkpoint(str(work_dir), "initial")
        result = mgr.restore(str(work_dir), "abc; rm -rf /")
        assert result["success"] is False
        assert "expected 4-64 hex characters" in result["error"]

        result = mgr.diff(str(work_dir), "abc&def")
        assert result["success"] is False
        assert "expected 4-64 hex characters" in result["error"]

    def test_restore_rejects_path_traversal(self, mgr, work_dir):
        mgr.ensure_checkpoint(str(work_dir), "initial")
        cps = mgr.list_checkpoints(str(work_dir))
        target_hash = cps[0]["hash"]

        result = mgr.restore(str(work_dir), target_hash, file_path="/etc/passwd")
        assert result["success"] is False
        assert "got absolute path" in result["error"]

        result = mgr.restore(str(work_dir), target_hash, file_path="../outside_file.txt")
        assert result["success"] is False
        assert "escapes the working directory" in result["error"]

    def test_restore_accepts_valid_file_path(self, mgr, work_dir):
        mgr.ensure_checkpoint(str(work_dir), "initial")
        cps = mgr.list_checkpoints(str(work_dir))
        target_hash = cps[0]["hash"]

        result = mgr.restore(str(work_dir), target_hash, file_path="main.py")
        assert result["success"] is True

        (work_dir / "subdir").mkdir()
        (work_dir / "subdir" / "test.txt").write_text("hello")
        mgr.new_turn()
        mgr.ensure_checkpoint(str(work_dir), "second")
        cps = mgr.list_checkpoints(str(work_dir))
        result = mgr.restore(str(work_dir), cps[0]["hash"], file_path="subdir/test.txt")
        assert result["success"] is True


# =========================================================================
# GPG / global git config isolation
# =========================================================================

class TestGpgAndGlobalConfigIsolation:
    def test_git_env_isolates_global_and_system_config(self, tmp_path):
        env = _git_env(tmp_path / "store", str(tmp_path))
        assert env["GIT_CONFIG_GLOBAL"] == os.devnull
        assert env["GIT_CONFIG_SYSTEM"] == os.devnull
        assert env["GIT_CONFIG_NOSYSTEM"] == "1"

    def test_init_sets_commit_gpgsign_false(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        store = _store_path(checkpoint_base)
        _init_store(store, str(work_dir))
        result = subprocess.run(
            ["git", "config", "--file", str(store / "config"),
             "--get", "commit.gpgsign"],
            capture_output=True, text=True,
        )
        assert result.stdout.strip() == "false"

    def test_init_sets_tag_gpgsign_false(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        store = _store_path(checkpoint_base)
        _init_store(store, str(work_dir))
        result = subprocess.run(
            ["git", "config", "--file", str(store / "config"),
             "--get", "tag.gpgSign"],
            capture_output=True, text=True,
        )
        assert result.stdout.strip() == "false"

    def test_checkpoint_works_with_global_gpgsign_and_broken_gpg(
        self, work_dir, checkpoint_base, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        (fake_home / ".gitconfig").write_text(
            "[user]\n    email = real@user.com\n    name = Real User\n"
            "[commit]\n    gpgsign = true\n"
            "[tag]\n    gpgSign = true\n"
            "[gpg]\n    program = /nonexistent/fake-gpg-binary\n"
        )
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.delenv("GPG_TTY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)

        m = CheckpointManager(enabled=True)
        assert m.ensure_checkpoint(str(work_dir), reason="with-global-gpgsign") is True
        assert len(m.list_checkpoints(str(work_dir))) == 1


# =========================================================================
# prune_checkpoints + maybe_auto_prune_checkpoints
# =========================================================================

def _seed_legacy_repo(base: Path, name: str, workdir: Path, mtime: float = None) -> Path:
    """Create a minimal pre-v2 shadow repo directly under base."""
    shadow = base / name
    shadow.mkdir(parents=True)
    (shadow / "HEAD").write_text("ref: refs/heads/main\n")
    (shadow / "HERMES_WORKDIR").write_text(str(workdir) + "\n")
    (shadow / "info").mkdir()
    (shadow / "info" / "exclude").write_text("node_modules/\n")
    if mtime is not None:
        for p in shadow.rglob("*"):
            os.utime(p, (mtime, mtime))
        os.utime(shadow, (mtime, mtime))
    return shadow


def _seed_v2_project(base: Path, workdir: Path, last_touch: float = None) -> str:
    """Register a v2 project in the shared store (no commits, just metadata)."""
    store = _store_path(base)
    _init_store(store, str(workdir if workdir.exists() else base))
    dir_hash = _project_hash(str(workdir))
    meta = {
        "workdir": str(workdir.resolve()) if workdir.exists() else str(workdir),
        "created_at": (last_touch or time.time()),
        "last_touch": (last_touch or time.time()),
    }
    mp = _project_meta_path(store, dir_hash)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(meta))
    return dir_hash


class TestPruneCheckpointsLegacy:
    """Backwards-compat: prune still handles pre-v2 per-project shadow repos."""

    def test_deletes_orphan_when_workdir_missing(self, tmp_path):
        base = tmp_path / "checkpoints"
        alive_work = tmp_path / "alive"
        alive_work.mkdir()
        alive_repo = _seed_legacy_repo(base, "aaaa" * 4, alive_work)
        orphan_repo = _seed_legacy_repo(base, "bbbb" * 4, tmp_path / "was-deleted")

        result = prune_checkpoints(retention_days=0, checkpoint_base=base)

        assert result["scanned"] == 2
        assert result["deleted_orphan"] == 1
        assert result["deleted_stale"] == 0
        assert alive_repo.exists()
        assert not orphan_repo.exists()

    def test_deletes_stale_by_mtime(self, tmp_path):
        base = tmp_path / "checkpoints"
        work = tmp_path / "work"
        work.mkdir()
        fresh_repo = _seed_legacy_repo(base, "cccc" * 4, work)
        stale_work = tmp_path / "stale_work"
        stale_work.mkdir()
        old = time.time() - 60 * 86400
        stale_repo = _seed_legacy_repo(base, "dddd" * 4, stale_work, mtime=old)

        result = prune_checkpoints(
            retention_days=30, delete_orphans=False, checkpoint_base=base,
        )
        assert result["deleted_stale"] == 1
        assert fresh_repo.exists()
        assert not stale_repo.exists()

    def test_delete_orphans_disabled_keeps_orphans(self, tmp_path):
        base = tmp_path / "checkpoints"
        orphan = _seed_legacy_repo(base, "ffff" * 4, tmp_path / "gone")

        result = prune_checkpoints(
            retention_days=0, delete_orphans=False, checkpoint_base=base,
        )
        assert result["deleted_orphan"] == 0
        assert orphan.exists()

    def test_skips_non_shadow_dirs(self, tmp_path):
        base = tmp_path / "checkpoints"
        base.mkdir()
        (base / "garbage-dir").mkdir()
        (base / "garbage-dir" / "random.txt").write_text("hi")

        result = prune_checkpoints(retention_days=0, checkpoint_base=base)
        assert result["scanned"] == 0
        assert (base / "garbage-dir").exists()

    def test_base_missing_returns_empty_counts(self, tmp_path):
        result = prune_checkpoints(checkpoint_base=tmp_path / "does-not-exist")
        assert result["scanned"] == 0
        assert result["deleted_orphan"] == 0


class TestPruneCheckpointsV2:
    """v2 pruning walks the shared store's projects/ metadata."""

    def test_deletes_orphan_project_entry(self, tmp_path, monkeypatch):
        base = tmp_path / "checkpoints"
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", base)

        alive = tmp_path / "alive"
        alive.mkdir()
        (alive / "f").write_text("a")
        gone = tmp_path / "was-gone"
        gone.mkdir()
        (gone / "g").write_text("b")

        m = CheckpointManager(enabled=True)
        assert m.ensure_checkpoint(str(alive), "alive") is True
        m.new_turn()
        assert m.ensure_checkpoint(str(gone), "gone") is True

        # Simulate deletion of "gone"
        import shutil as _shutil
        _shutil.rmtree(gone)

        result = prune_checkpoints(retention_days=0, checkpoint_base=base)

        assert result["deleted_orphan"] >= 1
        # Alive project survives
        alive_hash = _project_hash(str(alive))
        assert (base / "store" / "projects" / f"{alive_hash}.json").exists()
        # Gone project metadata wiped
        gone_hash = _project_hash(str(gone))
        assert not (base / "store" / "projects" / f"{gone_hash}.json").exists()

    def test_deletes_stale_project_by_last_touch(self, tmp_path, monkeypatch):
        base = tmp_path / "checkpoints"
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", base)

        fresh = tmp_path / "fresh"
        fresh.mkdir()
        (fresh / "f").write_text("f")
        stale = tmp_path / "stale"
        stale.mkdir()
        (stale / "s").write_text("s")

        m = CheckpointManager(enabled=True)
        m.ensure_checkpoint(str(fresh), "fresh")
        m.new_turn()
        m.ensure_checkpoint(str(stale), "stale")

        # Backdate stale's last_touch to 60 days ago
        stale_hash = _project_hash(str(stale))
        meta_path = base / "store" / "projects" / f"{stale_hash}.json"
        meta = json.loads(meta_path.read_text())
        meta["last_touch"] = time.time() - 60 * 86400
        meta_path.write_text(json.dumps(meta))

        result = prune_checkpoints(
            retention_days=30, delete_orphans=False, checkpoint_base=base,
        )

        assert result["deleted_stale"] >= 1
        fresh_hash = _project_hash(str(fresh))
        assert (base / "store" / "projects" / f"{fresh_hash}.json").exists()
        assert not meta_path.exists()

    def test_legacy_archive_dirs_also_pruned(self, tmp_path, monkeypatch):
        """legacy-<ts>/ dirs older than retention_days get wiped."""
        base = tmp_path / "checkpoints"
        base.mkdir()
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", base)

        old_legacy = base / "legacy-20200101-000000"
        old_legacy.mkdir()
        (old_legacy / "junk").write_bytes(b"x" * 1000)
        old = time.time() - 60 * 86400
        for p in old_legacy.rglob("*"):
            os.utime(p, (old, old))
        os.utime(old_legacy, (old, old))

        result = prune_checkpoints(retention_days=7, checkpoint_base=base)
        assert result["deleted_stale"] >= 1
        assert not old_legacy.exists()


class TestMaybeAutoPruneCheckpoints:
    def test_first_call_prunes_and_writes_marker(self, tmp_path):
        base = tmp_path / "checkpoints"
        _seed_legacy_repo(base, "0000" * 4, tmp_path / "gone")

        out = maybe_auto_prune_checkpoints(checkpoint_base=base)
        assert out["skipped"] is False
        assert out["result"]["deleted_orphan"] == 1
        assert (base / ".last_prune").exists()

    def test_second_call_within_interval_skips(self, tmp_path):
        base = tmp_path / "checkpoints"
        _seed_legacy_repo(base, "1111" * 4, tmp_path / "gone")

        first = maybe_auto_prune_checkpoints(
            checkpoint_base=base, min_interval_hours=24,
        )
        assert first["skipped"] is False

        _seed_legacy_repo(base, "2222" * 4, tmp_path / "also-gone")
        second = maybe_auto_prune_checkpoints(
            checkpoint_base=base, min_interval_hours=24,
        )
        assert second["skipped"] is True
        assert (base / ("2222" * 4)).exists()

    def test_corrupt_marker_treated_as_no_prior_run(self, tmp_path):
        base = tmp_path / "checkpoints"
        base.mkdir()
        (base / ".last_prune").write_text("not-a-timestamp")
        _seed_legacy_repo(base, "3333" * 4, tmp_path / "gone")

        out = maybe_auto_prune_checkpoints(checkpoint_base=base)
        assert out["skipped"] is False
        assert out["result"]["deleted_orphan"] == 1

    def test_missing_base_no_raise(self, tmp_path):
        out = maybe_auto_prune_checkpoints(
            checkpoint_base=tmp_path / "does-not-exist",
        )
        assert out["skipped"] is False
        assert out["result"]["scanned"] == 0


# =========================================================================
# store_status / clear_all / clear_legacy
# =========================================================================

class TestStoreStatus:
    def test_empty_base(self, tmp_path, monkeypatch):
        base = tmp_path / "checkpoints"
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", base)
        info = store_status()
        assert info["project_count"] == 0
        assert info["total_size_bytes"] == 0

    def test_reports_projects_and_legacy(self, tmp_path, monkeypatch, work_dir):
        base = tmp_path / "checkpoints"
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", base)

        m = CheckpointManager(enabled=True)
        m.ensure_checkpoint(str(work_dir), "initial")

        # Add a legacy archive dir manually
        legacy = base / "legacy-20200101-000000"
        legacy.mkdir()
        (legacy / "junk").write_bytes(b"x" * 100)

        info = store_status()
        assert info["project_count"] == 1
        assert info["projects"][0]["workdir"] == str(work_dir.resolve())
        assert info["projects"][0]["commits"] >= 1
        assert info["projects"][0]["exists"] is True
        assert len(info["legacy_archives"]) == 1
        assert info["legacy_archives"][0]["size_bytes"] >= 100


class TestClearFunctions:
    def test_clear_all_wipes_base(self, tmp_path, monkeypatch, work_dir):
        base = tmp_path / "checkpoints"
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", base)
        m = CheckpointManager(enabled=True)
        m.ensure_checkpoint(str(work_dir), "initial")
        assert base.exists()

        result = clear_all()
        assert result["deleted"] is True
        assert result["bytes_freed"] > 0
        assert not base.exists()

    def test_clear_legacy_only_removes_legacy_dirs(
        self, tmp_path, monkeypatch, work_dir,
    ):
        base = tmp_path / "checkpoints"
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", base)
        m = CheckpointManager(enabled=True)
        m.ensure_checkpoint(str(work_dir), "initial")

        legacy = base / "legacy-20200101-000000"
        legacy.mkdir()
        (legacy / "junk").write_bytes(b"x" * 1000)

        result = clear_legacy()
        assert result["deleted"] == 1
        assert result["bytes_freed"] >= 1000
        assert not legacy.exists()
        # Store preserved
        assert (base / "store" / "HEAD").exists()

    def test_clear_all_on_missing_base_is_noop(self, tmp_path, monkeypatch):
        base = tmp_path / "does-not-exist"
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", base)
        result = clear_all()
        assert result["deleted"] is False
        assert result["bytes_freed"] == 0
