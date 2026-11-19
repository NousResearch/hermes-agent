"""read_file names how to reach the rest of a line it clamped.

read_file pages by LINE, so ``offset`` can never reach the part of a long line cut at
``tool_output.max_line_length``. The hint must hand the model a command that, run as given,
returns exactly the next slice of that line.
"""

import json
import re
import subprocess

from tools.file_tools import read_file_tool
from tools.tool_output_limits import get_max_line_length


def test_clipped_line_hint_command_returns_the_next_slice(tmp_path):
    width = get_max_line_length()
    line = "".join(chr(ord("a") + (i // 997) % 26) for i in range(3 * width))
    target = tmp_path / "it's a bundle.min.js"  # quote in the name: the command must still run
    target.write_text("short\n" + line + "\nlast\n")

    result = json.loads(read_file_tool(str(target), task_id="clip-hint"))

    assert result["truncated_lines"] is True
    match = re.search(r"terminal: (.+)$", result["_hint"])
    assert match, result["_hint"]
    command = match.group(1)
    out = subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=True).stdout
    assert out.rstrip("\n") == line[width:2 * width]


def test_no_hint_when_nothing_was_clamped(tmp_path):
    target = tmp_path / "plain.txt"
    target.write_text("one\ntwo\n")
    assert "_hint" not in json.loads(read_file_tool(str(target), task_id="clip-hint-none"))
