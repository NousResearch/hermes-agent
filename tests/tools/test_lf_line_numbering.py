"""Edit diffs and "did you mean" hints number lines the way read_file does: ``\\n`` only.

``str.splitlines`` also breaks on form feed, U+2028 and friends, so a patch diff (and the
CLI/ACP renderers of it) or a no-match hint pointed one line further down per such
character than the ``N|`` gutter the model just read.
"""

import re

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.fuzzy_match import find_closest_lines

# Form feed (common in C/CPython sources) and U+2028 in a string literal, before the edit site.
CONTENT = "import os\n\x0c\ndef f():\n    s = 'a\u2028b'\n    return 1\n"


def test_patch_diff_hunk_lines_match_read_file_numbering(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text(CONTENT, encoding="utf-8")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))

    read_line = next(
        int(row.split("|", 1)[0])
        for row in ops.read_file(str(target), 1, 50).content.split("\n")
        if row.endswith("return 1")
    )
    result = ops.patch_replace(str(target), "    return 1\n", "    return 2\n")

    assert result.success, result.error
    hunk = re.search(r"^@@ -(\d+),(\d+) ", result.diff, re.MULTILINE)
    assert hunk, result.diff
    old_start, old_len = map(int, hunk.groups())
    body = [ln for ln in result.diff.split("\n")[3:] if ln[:1] in (" ", "-")]
    assert old_start + body.index("-    return 1") == read_line == 5
    assert old_len == len(body)


def test_no_match_hint_numbers_lines_like_read_file():
    hint = find_closest_lines("    return 9", CONTENT, context_lines=0, max_results=1)
    assert hint == "   5|     return 1"
