"""read_file names how to reach the rest of a line it clamped.

read_file pages by LINE, so ``offset`` can never reach the part of a long line cut at
``tool_output.max_line_length``. The hint must hand the model a command that, run as given,
returns exactly the next slice of that line, whatever the line holds.
"""

import json
import os
import re
import subprocess

import pytest

from tools.file_tools import read_file_tool
from tools.tool_output_limits import get_max_line_length


def _hint_command(result):
    match = re.search(r"terminal: (.+)$", result.get("_hint", ""))
    assert match, result
    return match.group(1)


@pytest.mark.platforms("posix")  # runs the hint's shell pipeline (tail | head) under bash
@pytest.mark.parametrize(("name", "alphabet", "home_relative"), [
    ("it's a bundle.min.js", "abcdefghij", False),  # quote in the name
    ("wide.txt", "界世漢字かなカナ한글😀", False),      # characters, not bytes
    ("home.min.js", "abcdefghij", True),             # a ~/ path the backend expands
    ("keys.jsonl", "abcdefghij", False),             # a masked secret in the visible prefix
    ("bom.txt", "abcdefghij", False),                # U+FEFF opening line 2 is text, kept as read_file keeps it
])
def test_clipped_line_hint_command_returns_the_next_slice(tmp_path, monkeypatch, name, alphabet, home_relative):
    monkeypatch.setenv("HOME", str(tmp_path))
    width = get_max_line_length()
    line = "".join(alphabet[(i // 997) % len(alphabet)] for i in range(3 * width))
    if name == "keys.jsonl":
        line = 'token="ghp_' + "A" * 36 + '" ' + line
    if name == "bom.txt":
        line = "\ufeff" + line
    target = tmp_path / name
    # line 1 ends in the marker text but is short: it must not be taken for a clamped line
    target.write_text("note... [truncated]\n" + line + "\nlast\n", encoding="utf-8")

    result = json.loads(read_file_tool(f"~/{name}" if home_relative else str(target), task_id="clip-hint"))

    assert result["truncated_lines"] is True
    assert result["_hint"].startswith("Line(s) 2 were cut")
    env = {"HOME": str(tmp_path), "PATH": os.environ["PATH"]}
    out = subprocess.run(["bash", "-c", _hint_command(result)], capture_output=True, encoding="utf-8", text=True,
                         check=True, env=env).stdout
    assert out.rstrip("\n") == line[width:2 * width]


def test_no_slice_command_without_a_clamped_line(tmp_path):
    plain = tmp_path / "plain.txt"
    plain.write_text("one\ntwo\n")
    assert "_hint" not in json.loads(read_file_tool(str(plain), task_id="clip-hint-none"))
    transcoded = tmp_path / "utf16.txt"  # on-disk bytes are not the text the model saw
    transcoded.write_text("x" * (2 * get_max_line_length()) + "\n", encoding="utf-16")
    result = json.loads(read_file_tool(str(transcoded), task_id="clip-hint-utf16"))
    assert "terminal:" not in result.get("_hint", "")
