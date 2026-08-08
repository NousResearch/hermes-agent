"""Tests for read_file() total_lines correctness.

Bug #3907: wc -l counts newline characters, not lines.
A file without a trailing newline was undercounted by 1.

Fix: use grep -c '' which counts lines regardless of trailing newline.
"""

import subprocess
from pathlib import Path

import pytest

from tools.file_operations import ShellFileOperations


# ---------------------------------------------------------------------------
# Minimal local terminal env that shells out via subprocess
# ---------------------------------------------------------------------------

class _LocalEnv:
    """Thin wrapper so ShellFileOperations can run real shell commands."""

    def __init__(self, cwd: str):
        self.cwd = cwd

    def execute(self, command: str, cwd: str = None, **kwargs) -> dict:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd or self.cwd,
            capture_output=True,
            text=True,
        )
        return {"output": result.stdout, "returncode": result.returncode}


def _make_ops(tmp_path: Path) -> ShellFileOperations:
    return ShellFileOperations(_LocalEnv(str(tmp_path)), cwd=str(tmp_path))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_total_lines_no_trailing_newline(tmp_path):
    """File without trailing newline must report the correct line count."""
    f = tmp_path / "no_newline.txt"
    f.write_bytes(b"line1\nline2\nline3")  # 3 lines, no trailing \n

    ops = _make_ops(tmp_path)
    result = ops.read_file(str(f))

    assert result.total_lines == 3


def test_total_lines_with_trailing_newline(tmp_path):
    """File with trailing newline must still report the correct line count."""
    f = tmp_path / "with_newline.txt"
    f.write_bytes(b"line1\nline2\nline3\n")  # 3 lines, with trailing \n

    ops = _make_ops(tmp_path)
    result = ops.read_file(str(f))

    assert result.total_lines == 3


def test_total_lines_empty_file(tmp_path):
    """Empty file must report 0 lines."""
    f = tmp_path / "empty.txt"
    f.write_bytes(b"")

    ops = _make_ops(tmp_path)
    result = ops.read_file(str(f))

    assert result.total_lines == 0


def test_total_lines_single_line_no_newline(tmp_path):
    """Single line without trailing newline must report 1."""
    f = tmp_path / "single.txt"
    f.write_bytes(b"hello")

    ops = _make_ops(tmp_path)
    result = ops.read_file(str(f))

    assert result.total_lines == 1


def test_total_lines_single_line_with_newline(tmp_path):
    """Single line with trailing newline must report 1."""
    f = tmp_path / "single_nl.txt"
    f.write_bytes(b"hello\n")

    ops = _make_ops(tmp_path)
    result = ops.read_file(str(f))

    assert result.total_lines == 1


def test_total_lines_unicode_no_trailing_newline(tmp_path):
    """Unicode content without trailing newline must count correctly."""
    f = tmp_path / "unicode.txt"
    f.write_bytes("merhaba\ndünya\nhermes".encode("utf-8"))

    ops = _make_ops(tmp_path)
    result = ops.read_file(str(f))

    assert result.total_lines == 3


def test_content_line_count_matches_total_lines(tmp_path):
    """Reported total_lines must match the actual number of lines in content."""
    f = tmp_path / "match.txt"
    lines = [f"line{i}" for i in range(1, 11)]
    f.write_bytes("\n".join(lines).encode())  # no trailing newline, 10 lines

    ops = _make_ops(tmp_path)
    result = ops.read_file(str(f))

    content_line_count = len(result.content.splitlines())
    assert result.total_lines == 10
    assert content_line_count == result.total_lines


def test_truncated_flag_correct_with_no_trailing_newline(tmp_path):
    """truncated flag and hint must use the corrected total_lines."""
    f = tmp_path / "long.txt"
    # 10 lines, no trailing newline
    lines = [f"line{i}" for i in range(1, 11)]
    f.write_bytes("\n".join(lines).encode())

    ops = _make_ops(tmp_path)
    # Read only first 5 lines
    result = ops.read_file(str(f), offset=1, limit=5)

    assert result.total_lines == 10
    assert result.truncated is True
    assert "10" in result.hint
