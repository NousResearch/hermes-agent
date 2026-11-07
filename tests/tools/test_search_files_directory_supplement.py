"""Regression: search_files directory supplement must stay bounded and skip concrete filenames.

P0 fix for wall-time outliers where ``_search_directories_with_find`` ran
unbounded ``find | sort -rn`` on large roots (e.g. ~/.hermes, ~/src) even for
exact filename lookups such as ``config.yaml``.
"""

from __future__ import annotations

import shutil
import time

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.file_operations_search import (
    _directory_supplement_cap,
    _should_supplement_directories,
)


class TestShouldSupplementDirectories:
    def test_concrete_filenames_skip(self):
        # Heuristic uses the pre-wrap search pattern.
        assert _should_supplement_directories("config.yaml") is False
        assert _should_supplement_directories("SOUL.md") is False
        assert _should_supplement_directories("agent.log") is False

    def test_directoryish_and_globs_keep(self):
        assert _should_supplement_directories("vault") is True
        assert _should_supplement_directories("*") is True
        assert _should_supplement_directories("*.py") is True
        assert _should_supplement_directories("*dir") is True
        assert _should_supplement_directories("cache") is True

    def test_cap_oversamples_but_hard_limits(self):
        assert _directory_supplement_cap(1) == 20
        assert _directory_supplement_cap(50) == 1000
        assert _directory_supplement_cap(200) == 1000


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
class TestConcreteFilenameSkipsDirectoryFind:
    def test_exact_filename_does_not_invoke_directory_find(self, tmp_path, monkeypatch):
        root = tmp_path / "repo"
        root.mkdir()
        target = root / "config.yaml"
        target.write_text("x: 1\n")
        (root / "nested").mkdir()
        (root / "nested" / "config.yaml").write_text("y: 2\n")

        ops = ShellFileOperations(LocalEnvironment("/"))
        calls = []

        def boom(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("directory supplement must be skipped for concrete filenames")

        monkeypatch.setattr(ops, "_search_directories_with_find", boom)
        result = ops.search("config.yaml", path=str(root), target="files", limit=50)

        assert result.error is None
        assert calls == []
        assert str(target) in result.files
        assert str(root / "nested" / "config.yaml") in result.files

    def test_bare_directory_name_still_supplements(self, tmp_path):
        root = tmp_path / "repo"
        empty = root / "vault"
        empty.mkdir(parents=True)

        ops = ShellFileOperations(LocalEnvironment("/"))
        result = ops.search("vault", path=str(root), target="files")

        assert result.error is None
        assert str(empty) in result.files


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
class TestBoundedDirectorySupplement:
    def test_directory_find_uses_head_not_sort(self, tmp_path, monkeypatch):
        root = tmp_path / "repo"
        (root / "vault").mkdir(parents=True)

        ops = ShellFileOperations(LocalEnvironment("/"))
        find_cmds = []

        real_exec = ops._exec

        def wrap(cmd, **kwargs):
            if isinstance(cmd, str) and cmd.lstrip().startswith("find "):
                find_cmds.append(cmd)
            return real_exec(cmd, **kwargs)

        monkeypatch.setattr(ops, "_exec", wrap)
        dirs, _reason = ops._search_directories_with_find("*vault", str(root), fetch_limit=5)

        assert str(root / "vault") in dirs
        assert find_cmds, "expected a find invocation"
        assert all("sort -rn" not in cmd for cmd in find_cmds)
        assert all("| head -n" in cmd for cmd in find_cmds)
        assert all("-print" in cmd for cmd in find_cmds)

    def test_hidden_directories_still_excluded(self, tmp_path):
        root = tmp_path / "repo"
        visible = root / "visible" / "cache"
        hidden = root / ".hidden" / "cache"
        visible.mkdir(parents=True)
        hidden.mkdir(parents=True)

        ops = ShellFileOperations(LocalEnvironment("/"))
        result = ops.search("cache", path=str(root), target="files")

        assert result.error is None
        assert str(visible) in result.files
        assert str(hidden) not in result.files

    def test_paths_with_spaces(self, tmp_path):
        root = tmp_path / "my repo"
        named = root / "cool dir"
        named.mkdir(parents=True)
        hit = named / "notes.md"
        hit.write_text("hi")

        ops = ShellFileOperations(LocalEnvironment("/"))
        result = ops.search("notes.md", path=str(root), target="files")

        assert result.error is None
        assert str(hit) in result.files

    def test_unknown_filename_skips_directory_find(self, tmp_path, monkeypatch):
        root = tmp_path / "repo"
        root.mkdir()
        (root / "a.txt").write_text("x")

        ops = ShellFileOperations(LocalEnvironment("/"))
        calls = []

        def boom(*args, **kwargs):
            calls.append(1)
            raise AssertionError("directory supplement must not run for unknown concrete names")

        monkeypatch.setattr(ops, "_search_directories_with_find", boom)
        result = ops.search("zz_missing_9f3a2b.qxyz", path=str(root), target="files")

        assert result.error is None
        assert result.files == []
        assert calls == []

    def test_content_search_unchanged(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        f = root / "mod.py"
        f.write_text("def _search_directories_with_find():\n    pass\n")

        ops = ShellFileOperations(LocalEnvironment("/"))
        result = ops.search(
            "_search_directories_with_find", path=str(root), target="content"
        )

        assert result.error is None
        assert any(
            "_search_directories_with_find" in (m.content or "")
            for m in (result.matches or [])
        )

    def test_large_root_exact_filename_skips_directory_find(self, tmp_path, monkeypatch):
        """Synthetic large tree: concrete filename must not invoke directory walk."""
        root = tmp_path / "big"
        root.mkdir()
        for i in range(400):
            d = root / f"pkg_{i:04d}"
            d.mkdir()
            (d / "mod.py").write_text("x")
        target = root / "config.yaml"
        target.write_text("ok\n")

        ops = ShellFileOperations(LocalEnvironment("/"))
        calls = []

        def boom(*args, **kwargs):
            calls.append(1)
            raise AssertionError("directory supplement must be skipped")

        monkeypatch.setattr(ops, "_search_directories_with_find", boom)
        t0 = time.perf_counter()
        result = ops.search("config.yaml", path=str(root), target="files", limit=20)
        elapsed = time.perf_counter() - t0

        assert result.error is None
        assert str(target) in result.files
        assert calls == []
        # Soft latency guard for cold LocalEnvironment startup on large trees.
        assert elapsed < 10.0

    def test_no_repeated_unbounded_traversal(self, tmp_path, monkeypatch):
        root = tmp_path / "repo"
        (root / "vault").mkdir(parents=True)
        ops = ShellFileOperations(LocalEnvironment("/"))
        calls = []
        real = ops._search_directories_with_find

        def counted(pattern, path, fetch_limit):
            calls.append((pattern, path, fetch_limit))
            return real(pattern, path, fetch_limit)

        monkeypatch.setattr(ops, "_search_directories_with_find", counted)
        ops.search("vault", path=str(root), target="files")
        assert len(calls) == 1


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
class TestPartialAndTimeoutSurface:
    def test_directory_cap_sets_limit_reason(self, tmp_path, monkeypatch):
        root = tmp_path / "repo"
        for i in range(30):
            (root / f"dir_{i:02d}").mkdir(parents=True)

        ops = ShellFileOperations(LocalEnvironment("/"))
        from tools import file_operations_search as fos

        monkeypatch.setattr(fos, "_directory_supplement_cap", lambda fetch_limit: 5)
        dirs, reason = ops._search_directories_with_find("*", str(root), fetch_limit=2)
        assert reason == "directory_supplement_cap"
        assert len(dirs) <= 2
