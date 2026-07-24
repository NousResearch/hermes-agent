"""Regression tests for #67185 — write_file doubled-path fix.

When a model emits a relative path that textually mirrors the working
directory (e.g. ``home/user/dev/notes/x.md`` — an absolute path missing its
leading ``/``), ``_resolve_path_for_task`` silently creates a doubled path
like ``/home/user/dev/home/user/dev/notes/x.md``.  The fix detects this
pattern and strips the duplicated prefix so the write lands where the model
intended.  A heuristic warning also fires for bare-absolute inputs.
"""

import os
from pathlib import Path, PurePosixPath

import pytest

import tools.file_tools as ft
import tools.terminal_tool as terminal_tool


@pytest.fixture
def _workspace(tmp_path, monkeypatch):
    """Set up a workspace with known structure and record its cwd."""
    workspace = tmp_path / "home" / "user" / "dev"
    workspace.mkdir(parents=True)
    (workspace / "notes").mkdir()
    (workspace / "notes" / "existing.md").write_text("hello\n")
    monkeypatch.chdir(tmp_path)
    terminal_tool.record_session_cwd("default", str(workspace))
    monkeypatch.setattr(terminal_tool, "_session_cwd", {"default": str(workspace)})
    return workspace


# ── _strip_doubled_base_prefix unit tests ──────────────────────────────────


class TestStripDoubledBasePrefix:
    def test_strips_full_doubled_prefix(self):
        base = Path("/home/user/dev")
        resolved = Path("/home/user/dev/home/user/dev/notes/file.md")
        result = ft._strip_doubled_base_prefix(base, resolved)
        assert result == Path("/home/user/dev/notes/file.md")

    def test_strips_partial_doubled_prefix(self):
        base = Path("/home/user/dev")
        resolved = Path("/home/user/dev/home/user/file.md")
        result = ft._strip_doubled_base_prefix(base, resolved)
        assert result == Path("/home/user/dev/file.md")

    def test_no_strip_for_normal_relative_path(self):
        base = Path("/home/user/dev")
        resolved = Path("/home/user/dev/src/main.py")
        result = ft._strip_doubled_base_prefix(base, resolved)
        assert result is None

    def test_no_strip_for_absolute_input(self):
        base = Path("/home/user/dev")
        resolved = Path("/etc/hosts")
        result = ft._strip_doubled_base_prefix(base, resolved)
        assert result is None

    def test_no_strip_for_single_segment_overlap(self):
        base = Path("/home/user/dev")
        resolved = Path("/home/other/file.md")
        result = ft._strip_doubled_base_prefix(base, resolved)
        assert result is None

    def test_strips_when_input_is_base_itself(self):
        base = Path("/home/user/dev")
        resolved = Path("/home/user/dev/home/user/dev")
        result = ft._strip_doubled_base_prefix(base, resolved)
        assert result == Path("/home/user/dev")

    def test_no_strip_when_resolved_shorter_than_base(self):
        base = Path("/home/user/dev/very/long/path")
        resolved = Path("/home/user/dev")
        result = ft._strip_doubled_base_prefix(base, resolved)
        assert result is None

    @pytest.mark.skipif(
        os.name == "nt",
        reason="PurePosixPath str uses / but WindowsPath str uses \\",
    )
    def test_posix_pure_path_base(self):
        base = PurePosixPath("/home/user/dev")
        resolved = Path("/home/user/dev/home/user/dev/notes/file.md")
        result = ft._strip_doubled_base_prefix(base, resolved)
        assert result == Path("/home/user/dev/notes/file.md")


# ── _resolve_path_for_task integration tests ───────────────────────────────


class TestResolvePathDoubledPrefix:
    def test_cwd_shaped_relative_path_strips_prefix(self, _workspace):
        """The core bug: home/user/dev/notes/file.md must resolve correctly."""
        resolved = ft._resolve_path_for_task(
            "home/user/dev/notes/file.md", task_id="default"
        )
        assert resolved == (_workspace / "notes" / "file.md")

    def test_cwd_shaped_deep_path_strips_prefix(self, _workspace):
        """Deeper nesting: home/user/dev/a/b/c/file.py."""
        resolved = ft._resolve_path_for_task(
            "home/user/dev/a/b/c/file.py", task_id="default"
        )
        assert resolved == (_workspace / "a" / "b" / "c" / "file.py")

    def test_normal_relative_path_unchanged(self, _workspace):
        """A normal relative path like src/main.py must not be altered."""
        ( _workspace / "src").mkdir()
        resolved = ft._resolve_path_for_task("src/main.py", task_id="default")
        assert resolved == (_workspace / "src" / "main.py")

    def test_absolute_path_unchanged(self, _workspace):
        """An absolute input path is never re-anchored."""
        abs_path = str(_workspace / "notes" / "existing.md")
        resolved = ft._resolve_path_for_task(abs_path, task_id="default")
        assert resolved == Path(abs_path).resolve()

    def test_tilde_path_not_affected(self, _workspace):
        """Tilde expansion happens before the doubled-prefix check."""
        # ~ expansion is handled by _expand_tilde; after expansion the
        # result should be treated as absolute and pass through.
        # This test just verifies tilde paths don't crash.
        resolved = ft._resolve_path_for_task("~/notes/file.md", task_id="default")
        # Should resolve under HOME, not be mangled by the prefix stripper.
        assert resolved.is_absolute()


# ── _path_resolution_warning bare-absolute heuristic tests ─────────────────


class TestBareAbsoluteWarning:
    def test_warns_for_bare_absolute_home_path(self, _workspace):
        """home/user/dev/file.md should trigger the bare-absolute warning."""
        resolved = _workspace / "file.md"
        warn = ft._path_resolution_warning(
            "home/user/dev/file.md", resolved, task_id="default"
        )
        assert warn is not None
        assert "missing" in warn.lower() or "leading /" in warn

    def test_warns_for_bare_absolute_tmp_path(self, _workspace):
        """tmp/foo/bar.py should trigger the bare-absolute warning."""
        resolved = Path("/tmp/foo/bar.py")
        warn = ft._path_resolution_warning("tmp/foo/bar.py", resolved, task_id="default")
        assert warn is not None

    def test_no_warning_for_normal_relative_path(self, _workspace):
        """src/main.py should NOT trigger the bare-absolute warning."""
        warn = ft._path_resolution_warning(
            "src/main.py", _workspace / "src" / "main.py", task_id="default"
        )
        # Should be None (inside workspace) — no bare-absolute heuristic fires.
        assert warn is None

    def test_no_warning_for_absolute_input(self, _workspace):
        """An absolute path should never trigger any warning."""
        warn = ft._path_resolution_warning(
            str(_workspace / "file.md"),
            _workspace / "file.md",
            task_id="default",
        )
        assert warn is None
