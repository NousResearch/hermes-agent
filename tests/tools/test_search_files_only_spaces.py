"""Regression tests for search_files files_only mode with spaces in paths.

Issue #91698: ``search_files`` with ``output_mode="files_only"`` silently
dropped any file whose path contained a space because the diagnostic
classifier regex (``_SEARCH_OUTPUT_RE``) forbids whitespace in bare-path
lines.  The fix makes ``_split_tool_diagnostics`` mode-aware: in
``files_only`` mode every non-diagnostic-prefixed line is treated as a
path, spaces included.
"""

import json
import os
import subprocess
from unittest.mock import MagicMock

import pytest

from tools.file_operations import ShellFileOperations, _split_tool_diagnostics


@pytest.fixture
def spaced_dir(tmp_path):
    """Create a directory tree with spaces in paths."""
    d = tmp_path / "Obsidian Vault"
    d.mkdir()
    (d / "file with space.md").write_text("needle found here\n")
    (d / "plain.md").write_text("needle also here\n")
    (d / "sub dir").mkdir()
    (d / "sub dir" / "nested note.md").write_text("needle deep inside\n")
    return d


class TestSplitToolDiagnosticsFilesOnly:
    """Unit tests for _split_tool_diagnostics with output_mode='files_only'."""

    def test_files_only_keeps_paths_with_spaces(self):
        output = "/tmp/Obsidian Vault/notes.md\n/tmp/plain.py\n"
        diagnostics, payload = _split_tool_diagnostics(output, output_mode="files_only")
        assert "Obsidian Vault/notes.md" in payload
        assert "plain.py" in payload
        assert diagnostics == ""

    def test_files_only_keeps_diagnostic_prefixed_lines_out(self):
        output = "rg: /tmp/secret: Permission denied\n/tmp/plain.py\n"
        diagnostics, payload = _split_tool_diagnostics(output, output_mode="files_only")
        assert "Permission denied" in diagnostics
        assert "plain.py" in payload

    def test_files_only_filters_indented_caret_lines(self):
        """rg regex-parse-error block emits indented caret lines that are not paths."""
        output = "rg: regex parse error:\n  ^\n/tmp/plain.py\n"
        diagnostics, payload = _split_tool_diagnostics(output, output_mode="files_only")
        assert "regex parse error" in diagnostics
        assert "^" in diagnostics
        assert "plain.py" in payload

    def test_files_only_filters_error_lines(self):
        output = "error: something went wrong\n/tmp/plain.py\n"
        diagnostics, payload = _split_tool_diagnostics(output, output_mode="files_only")
        assert "error: something went wrong" in diagnostics
        assert "plain.py" in payload

    def test_content_mode_still_rejects_spaces_in_bare_paths(self):
        """In content mode the old behavior is preserved — bare paths with
        spaces are classified as diagnostics because they don't match the
        match-line shape (path:line:content)."""
        output = "/tmp/Obsidian Vault/notes.md\n/tmp/plain.py:1:needle\n"
        diagnostics, payload = _split_tool_diagnostics(output, output_mode="content")
        # The spaced bare path is a diagnostic in content mode (no line number)
        assert "Obsidian Vault/notes.md" in diagnostics
        # The match line with line number is payload
        assert "plain.py:1:needle" in payload

    def test_count_mode_still_rejects_spaces_in_bare_paths(self):
        output = "/tmp/Obsidian Vault/notes.md:3\n/tmp/plain.py:1\n"
        diagnostics, payload = _split_tool_diagnostics(output, output_mode="count")
        # Count lines have path:count shape; the spaced path matches the
        # first alternative (path:digit) so it's payload.
        assert "Obsidian Vault/notes.md:3" in payload
        assert "plain.py:1" in payload


class TestSearchFilesOnlyWithSpaces:
    """Integration tests through ShellFileOperations.search with real subprocess."""

    def _make_real_env(self, cwd: str) -> MagicMock:
        env = MagicMock()
        env.cwd = cwd

        def execute(command, **kwargs):
            completed = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
                cwd=cwd,
            )
            return {
                "output": completed.stdout + completed.stderr,
                "returncode": completed.returncode,
            }

        env.execute = execute
        return env

    def test_files_only_returns_spaced_paths(self, spaced_dir):
        """files_only mode must return files whose paths contain spaces."""
        env = self._make_real_env(str(spaced_dir.parent))
        ops = ShellFileOperations(env)
        result = ops.search("needle", path=str(spaced_dir), output_mode="files_only")
        assert result.error is None
        assert result.total_count == 3, f"Expected 3 files, got {result.total_count}: {result.files}"
        # All three spaced paths must be present
        files = set(result.files)
        assert any("file with space.md" in f for f in files)
        assert any("plain.md" in f for f in files)
        assert any("nested note.md" in f for f in files)

    def test_files_only_with_all_spaced_paths(self, tmp_path):
        """When every match lives under a space-containing directory, the
        search must not return total_count=0."""
        d = tmp_path / "My Vault"
        d.mkdir()
        (d / "note one.md").write_text("unique_needle_token\n")
        (d / "note two.md").write_text("unique_needle_token\n")
        env = self._make_real_env(str(tmp_path))
        ops = ShellFileOperations(env)
        result = ops.search("unique_needle_token", path=str(d), output_mode="files_only")
        assert result.error is None
        assert result.total_count == 2, f"Expected 2 files, got {result.total_count}: {result.files}"

    def test_content_mode_still_works_with_spaces(self, spaced_dir):
        """content mode was never broken — verify it still works."""
        env = self._make_real_env(str(spaced_dir.parent))
        ops = ShellFileOperations(env)
        result = ops.search("needle", path=str(spaced_dir), output_mode="content")
        assert result.error is None
        assert result.total_count == 3

    def test_count_mode_still_works_with_spaces(self, spaced_dir):
        """count mode was never broken — verify it still works."""
        env = self._make_real_env(str(spaced_dir.parent))
        ops = ShellFileOperations(env)
        result = ops.search("needle", path=str(spaced_dir), output_mode="count")
        assert result.error is None
        assert result.total_count == 3