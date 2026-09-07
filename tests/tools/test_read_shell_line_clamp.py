"""Shell-pipeline per-line clamp in ShellFileOperations.read_file.

A file with one pathological line (e.g. a 400MB minified bundle on a single
line) used to cross the exec transport in full: ``sed -n`` emitted the whole
line and Python only clamped it afterwards. read_file now pipes through
``cut -b1-{4*max_line_length+1}`` so the shell bounds each line before the
bytes reach Python.

Byte-vs-char note: GNU ``cut -c`` is byte-based, so the clamp can split a
multibyte UTF-8 codepoint at the boundary. The transport decodes with
errors="replace", turning a split into U+FFFD; the budget of
4*max_line_length+1 bytes guarantees at least max_line_length+1 decoded chars
survive for any over-long line, so the Python clamp always fires, always
appends "... [truncated]", and always removes any boundary U+FFFD (it can
only appear past char max_line_length).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.tool_output_limits import get_max_line_length


@pytest.fixture
def ops():
    return ShellFileOperations(LocalEnvironment())


def test_monster_line_clamped_in_shell(tmp_path, ops):
    """A single multi-MB line comes back clamped, with the truncated suffix."""
    max_len = get_max_line_length()
    monster = tmp_path / "monster.txt"
    with open(monster, "w") as f:
        f.write("x" * 5_000_000)
        f.write("\n")
        f.write("after\n")

    result = ops.read_file(str(monster))
    assert result.error is None
    lines = result.content.split("\n")
    assert lines[0] == "1|" + "x" * max_len + "... [truncated]"
    assert lines[1] == "2|after"
    # Sanity: the whole result stays tiny relative to the 5MB line.
    assert len(result.content) < max_len + 200


def test_offset_past_monster_returns_normal_lines(tmp_path, ops):
    monster = tmp_path / "monster.txt"
    with open(monster, "w") as f:
        f.write("y" * 1_000_000 + "\n")
        for i in range(5):
            f.write(f"normal line {i}\n")

    result = ops.read_file(str(monster), offset=2)
    assert result.error is None
    assert result.content.split("\n")[:5] == [
        f"{i + 2}|normal line {i}" for i in range(5)
    ]


def test_normal_multiline_read_unchanged(tmp_path, ops):
    p = tmp_path / "plain.txt"
    p.write_text("alpha\nbeta\ngamma\n")
    result = ops.read_file(str(p))
    assert result.error is None
    assert result.content.startswith("1|alpha\n2|beta\n3|gamma")


def test_no_trailing_newline_preserved(tmp_path, ops):
    """cut newline-terminates its output; read_file must strip the artifact."""
    p = tmp_path / "nonl.txt"
    p.write_text("a\nb")
    result = ops.read_file(str(p))
    assert result.content == "1|a\n2|b"


def test_utf8_multibyte_boundary_no_mojibake(tmp_path, ops):
    """Byte clamp may split a codepoint; no U+FFFD may reach the result."""
    max_len = get_max_line_length()
    p = tmp_path / "mb.txt"
    with open(p, "w", encoding="utf-8") as f:
        # 2-byte chars sized so the byte clamp (4*max_len+1) lands mid-char.
        f.write("é" * (2 * max_len + 1) + "\n")
        f.write("second\n")

    result = ops.read_file(str(p))
    assert result.error is None
    lines = result.content.split("\n")
    assert lines[0] == "1|" + "é" * max_len + "... [truncated]"
    assert "\ufffd" not in result.content
    assert lines[1] == "2|second"


def test_multibyte_line_over_limit_still_marked_truncated(tmp_path, ops):
    """A multibyte line just over the char limit must still get the suffix.

    This is the case a max_line_length+1 BYTE clamp would silently break:
    2001 chars of 'é' are 4002 bytes, so a 2001-byte clamp would deliver
    ~1000 chars and the Python clamp would never fire.
    """
    max_len = get_max_line_length()
    p = tmp_path / "just_over.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write("é" * (max_len + 1) + "\n")

    result = ops.read_file(str(p))
    assert result.error is None
    first = result.content.split("\n")[0]
    assert first == "1|" + "é" * max_len + "... [truncated]"


def test_read_file_raw_not_clamped(tmp_path, ops):
    """read_file_raw is documented as no-per-line-truncation — verify."""
    max_len = get_max_line_length()
    p = tmp_path / "long_raw.txt"
    long_line = "z" * (10 * max_len)
    p.write_text(long_line + "\n")
    result = ops.read_file_raw(str(p))
    assert result.error is None
    assert result.content == long_line + "\n"
