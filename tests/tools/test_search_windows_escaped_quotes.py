"""Native Windows search preserves regex patterns containing escaped quotes (#142111)."""

import shutil

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


@pytest.mark.platforms("windows")
@pytest.mark.parametrize(
    ("content", "pattern", "expected_count"),
    [
        ('value = x"y\n', 'x"y', 1),
        ('value = x"y\n', r"x\"y", 1),
        (r"value = x\"y" + "\n", r'x\\"y', 1),
        ('line1\nvalue = x"y\nline2\n', '(?x)x\\"y\n', 1),
        ('lookup = "hello"\n', r"\"hello\"", 1),
    ],
    ids=(
        "bare-quote",
        "escaped-quote",
        "literal-bs-quote",
        "multiline-extended",
        "surrounding-escaped-quotes",
    ),
)
def test_native_windows_search_escaped_quotes(
    tmp_path, content, pattern, expected_count
):
    """On native Windows, patterns with backslash-quote must match content accurately."""
    if shutil.which("rg") is None:
        pytest.skip("native Windows ripgrep is not installed")

    (tmp_path / "sample.txt").write_text(content, encoding="utf-8")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))

    result = ops.search(pattern, path=str(tmp_path / "sample.txt"))

    assert result.error is None
    assert result.total_count == expected_count
    assert len(result.matches) == expected_count
