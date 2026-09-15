"""Regression for #93749: programmatic reads must stay raw on repeat."""

from __future__ import annotations

import json

from tools import file_tools


def test_programmatic_read_returns_raw_content_on_repeated_calls(tmp_path):
    target = tmp_path / "sample.txt"
    raw = "alpha\n1|literal\nomega\n"
    target.write_text(raw, encoding="utf-8")
    programmatic = getattr(file_tools, "read_file_programmatic_tool", None)
    if programmatic is not None:
        first = json.loads(programmatic(str(target), task_id="sandbox-read"))
        second = json.loads(programmatic(str(target), task_id="sandbox-read"))
    else:
        first = json.loads(file_tools.read_file_tool(str(target), task_id="sandbox-read"))
        second = json.loads(file_tools.read_file_tool(str(target), task_id="sandbox-read"))

    assert first["success"] is True
    assert first["content"] == raw
    assert second == first
