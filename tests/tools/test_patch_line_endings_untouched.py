"""An edit rewrites only what it edits: patch (replace mode and V4A) gives the lines it produces
the file's dominant ending, and every other byte — a CRLF past the 4 KB detection window of an LF
file, an LF-only line in a CRLF file, a lone-CR progress line — stays exactly as it was."""

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations

_FILLER = "".join(f"line {i}\n" for i in range(700))  # > 4 KB: the ending is detected from here

SHAPES = {
    # LF-dominant; CRLF + lone CR only after the detection window (HTTP fixture, \r progress).
    "lf": (b"status: pending\n" + _FILLER.encode()
           + b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n50%\r100%\n", b"\n"),
    # CRLF-dominant with an LF-only line and a lone CR.
    "crlf": (b"status: pending\r\nb\r\nkeep\n\nlf-only\nprog 1\r2\r\n", b"\r\n"),
}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mode", ["replace", "v4a"])
def test_patch_rewrites_only_the_edited_line(tmp_path, shape, mode):
    original, ending = SHAPES[shape]
    target = tmp_path / "data.txt"
    target.write_bytes(original)
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))

    if mode == "replace":
        result = ops.patch_replace(str(target), "status: pending", "status: done")
    else:
        result = ops.patch_v4a(
            f"*** Begin Patch\n*** Update File: {target}\n-status: pending\n+status: done\n*** End Patch")

    assert result.success, result.error
    first_eol = original.index(ending) + len(ending)
    assert target.read_bytes() == b"status: done" + ending + original[first_eol:]


def test_v4a_context_lines_keep_their_bytes_among_repeated_lines(tmp_path):
    """A V4A context line is one the hunk leaves unchanged, so it keeps its own ending even when
    repeated text would let a line diff align it with a produced line. The hunk turns the second
    ``c`` into ``b`` and the final ``a`` into ``c``: the produced lines take the file's CRLF, and
    the context lines ``c`` and ``b`` keep their LF."""
    target = tmp_path / "data.txt"
    target.write_bytes(b"c\nc\nb\na\r\n")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))

    result = ops.patch_v4a(
        f"*** Begin Patch\n*** Update File: {target}\n c\n-c\n+b\n b\n-a\n+c\n*** End Patch")

    assert result.success, result.error
    assert target.read_bytes() == b"c\n" + b"b\r\n" + b"b\n" + b"c\r\n"


@pytest.mark.parametrize("before, after", [
    # CRLF file, last line terminated: the insertion goes after the existing CRLF.
    (b"alpha\r\nbravo\r\n", b"alpha\r\nbravo\r\nAPPENDED\r\n"),
    # CRLF file, last line unterminated: the break the insertion adds takes the file's CRLF.
    (b"alpha\r\nbravo", b"alpha\r\nbravo\r\nAPPENDED\r\n"),
    # LF file, last line unterminated.
    (b"alpha\nbravo", b"alpha\nbravo\nAPPENDED\n"),
    # CRLF file whose last line ends in a bare LF: that LF is the file's own byte and stays.
    (b"alpha\r\nbravo\n", b"alpha\r\nbravo\nAPPENDED\r\n"),
])
def test_v4a_eof_append_line_break_matches_the_file(tmp_path, before, after):
    """An addition-only hunk with no @@ hint appends at EOF. The break before the appended text is
    the last line's own terminator when it has one, else the file's ending — never a bare LF the
    file did not have."""
    target = tmp_path / "data.txt"
    target.write_bytes(before)
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))

    result = ops.patch_v4a(f"*** Begin Patch\n*** Update File: {target}\n+APPENDED\n*** End Patch")

    assert result.success, result.error
    assert target.read_bytes() == after
