"""The read_file display option preserves content across real read paths (#4581)."""

import json

import pytest

from tools.file_tools import _handle_read_file


@pytest.mark.parametrize("document", [False, True], ids=["text", "notebook"])
def test_handler_switches_numbering_without_losing_content(tmp_path, document):
    path = tmp_path / ("sample.ipynb" if document else "sample.txt")
    if document:
        path.write_text(json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["1|literal\n", "second\n"]}
        ]}), encoding="utf-8")
    else:
        path.write_text("1|literal\nsecond\nlast\n", encoding="utf-8")
    args = {"path": str(path), "offset": 1, "limit": 50}
    numbered = json.loads(_handle_read_file(args, task_id="format-switch"))
    raw = json.loads(_handle_read_file({**args, "add_line_numbers": False}, task_id="format-switch"))
    assert "error" not in numbered and "error" not in raw
    assert "1|literal" in raw["content"]
    assert "1|literal" in numbered["content"]
    expected = "\n".join(line.split("|", 1)[1] for line in numbered["content"].split("\n"))
    assert raw["content"] == expected
    page = json.loads(_handle_read_file({**args, "offset": 2, "limit": 1, "add_line_numbers": False}, task_id="page"))
    assert page["content"] == expected.split("\n")[1]
    eof = json.loads(_handle_read_file({**args, "offset": 99, "add_line_numbers": False}, task_id="eof"))
    assert eof["content"] == ""


@pytest.mark.parametrize("document", [False, True], ids=["text", "notebook"])
def test_unnumbered_reads_keep_line_and_output_budgets(tmp_path, monkeypatch, document):
    from tools import file_tools
    from tools.tool_output_limits import get_max_line_length

    path = tmp_path / ("budget.ipynb" if document else "budget.txt")
    long_line = "x" * (get_max_line_length() + 20)
    if document:
        path.write_text(json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [long_line]}
        ]}), encoding="utf-8")
    else:
        path.write_text(long_line, encoding="utf-8")
    response = json.loads(_handle_read_file({"path": str(path), "add_line_numbers": False}, task_id="clamp"))
    assert response["truncated_lines"]
    assert "... [truncated]" in response["content"]
    monkeypatch.setattr(file_tools, "_get_max_read_chars", lambda: 50)
    limited = json.loads(_handle_read_file({"path": str(path), "add_line_numbers": False}, task_id="budget"))
    assert limited["truncated"] and limited["truncated_by"] == "bytes"
    assert len(limited["content"]) <= 50
