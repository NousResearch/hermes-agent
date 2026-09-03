from __future__ import annotations

import json
import sys

import pytest

from tools import code_execution_tool
from tools import file_tools


@pytest.mark.skipif(sys.platform == "win32", reason="UDS not available on Windows")
def test_execute_code_read_file_returns_stable_raw_shape(tmp_path):
    target = tmp_path / "sandbox.txt"
    target.write_text("alpha\n1|literal\nomega\n", encoding="utf-8")
    code = f"""
import json
from hermes_tools import read_file

first = read_file({str(target)!r})
second = read_file({str(target)!r})
print(json.dumps([first, second], sort_keys=True))
"""

    result = json.loads(
        code_execution_tool.execute_code(
            code=code,
            task_id=f"sandbox-{tmp_path.name}",
            enabled_tools=["read_file"],
        )
    )
    reads = json.loads(result["output"].strip())

    assert result["status"] == "success"
    assert result["tool_calls_made"] == 2
    assert reads[0]["success"] is True
    assert reads[0]["content"] == "alpha\n1|literal\nomega\n"
    assert reads[1] == reads[0]


def test_programmatic_read_returns_raw_content_on_repeated_calls(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("alpha\n1|literal\nomega\n", encoding="utf-8")
    first = json.loads(
        file_tools.read_file_programmatic_tool(str(target), task_id="sandbox-read")
    )
    second = json.loads(
        file_tools.read_file_programmatic_tool(str(target), task_id="sandbox-read")
    )

    assert first["success"] is True
    assert first["content"] == "alpha\n1|literal\nomega\n"
    assert second == first


def test_programmatic_read_failure_keeps_stable_content_key(tmp_path):
    result = json.loads(
        file_tools.read_file_programmatic_tool(
            str(tmp_path / "missing.txt"), task_id="sandbox-read"
        )
    )

    assert result["success"] is False
    assert result["content"] == ""
    assert result["error"]


def test_programmatic_read_preserves_explicit_failure(monkeypatch):
    monkeypatch.setattr(
        file_tools,
        "read_file_tool",
        lambda **_kwargs: '{"success": false, "note": "blocked"}',
    )

    result = json.loads(file_tools.read_file_programmatic_tool("blocked.pipe"))

    assert result == {
        "success": False,
        "note": "blocked",
        "content": "",
    }


def test_programmatic_read_preserves_pagination_without_display_gutters(tmp_path):
    target = tmp_path / "pages.txt"
    target.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    result = json.loads(
        file_tools.read_file_programmatic_tool(
            str(target), offset=2, limit=2, task_id="sandbox-page"
        )
    )

    assert result["content"] == "two\nthree\n"
    assert result["total_lines"] == 4
    assert result["truncated"] is True


def test_programmatic_read_does_not_change_chat_dedup_contract(tmp_path):
    target = tmp_path / "chat.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    task_id = f"chat-{tmp_path.name}"

    programmatic = json.loads(
        file_tools.read_file_programmatic_tool(str(target), task_id=task_id)
    )
    chat_first = json.loads(file_tools.read_file_tool(str(target), task_id=task_id))
    chat_second = json.loads(file_tools.read_file_tool(str(target), task_id=task_id))

    assert programmatic["content"] == "alpha\nbeta\n"
    assert chat_first["content"] == "1|alpha\n2|beta\n3|"
    assert chat_second["status"] == "unchanged"
    assert "content" not in chat_second


def test_sandbox_dispatch_uses_standard_dispatcher(monkeypatch):
    captured = {}

    def fake_handle_function_call(tool_name, tool_args, task_id=None):
        captured.update(
            tool_name=tool_name,
            tool_args=tool_args,
            task_id=task_id,
            programmatic=file_tools._programmatic_read.get(),
        )
        return '{"success": true, "content": "raw"}'

    monkeypatch.setattr(
        "model_tools.handle_function_call",
        fake_handle_function_call,
    )

    result = code_execution_tool._dispatch_sandbox_tool_call(
        "read_file",
        {"path": "notes.md", "offset": 4, "limit": 7},
        task_id="task-1",
    )

    assert json.loads(result)["content"] == "raw"
    assert captured == {
        "tool_name": "read_file",
        "tool_args": {"path": "notes.md", "offset": 4, "limit": 7},
        "task_id": "task-1",
        "programmatic": True,
    }
    assert file_tools._programmatic_read.get() is False


def test_programmatic_read_and_chat_raw_paths_are_byte_identical(tmp_path):
    """Pin the two content construction paths to identical raw output.

    The early structured-document branch builds its page natively (no
    gutter, no per-line truncation), while the file_ops branch either adds
    numbers natively or — with line_numbers=False — applies the same clamp
    without a gutter. Both must produce byte-identical raw content for the
    same window so the "stable" programmatic contract cannot drift from
    the chat path.
    """
    target = tmp_path / "paths.txt"
    long_line = "x" * 3000  # exceeds the default 2000-char per-line clamp
    target.write_text(f"alpha\n{long_line}\ngamma\n", encoding="utf-8")

    programmatic = json.loads(
        file_tools.read_file_programmatic_tool(str(target), task_id="pin-prog")
    )
    chat_raw = json.loads(
        file_tools.read_file_tool(str(target), task_id="pin-chat", line_numbers=False)
    )

    assert programmatic["content"] == chat_raw["content"]
    assert programmatic["content"].splitlines()[1] != long_line  # clamped identically


def test_programmatic_early_document_branch_matches_file_ops_path_byte_for_byte(
    tmp_path,
):
    """Pin the early structured-document branch against the file_ops path.

    For an extractable document (``.ipynb``), ``read_file_tool`` builds the
    page natively in its early branch — no gutter, no per-line clamp. The
    same window (offset=2, limit=1) read through a real
    ``ShellFileOperations.read_file(..., line_numbers=False)`` call must
    produce byte-identical content, so the "stable" programmatic contract
    cannot drift between the two construction paths.
    """
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    notebook = {
        "cells": [
            {"cell_type": "code", "source": ["alpha\n", "beta\n"], "outputs": []},
            {"cell_type": "markdown", "source": ["tail line\n"]},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = tmp_path / "pins.ipynb"
    target.write_text(json.dumps(notebook), encoding="utf-8")

    # Chat path: early structured-document branch builds natively (no gutter).
    chat = json.loads(
        file_tools.read_file_tool(
            str(target), offset=2, limit=1, task_id="pin-doc-chat", line_numbers=False
        )
    )
    assert chat.get("extracted_document") is True

    # Programmatic path: same early branch, same window, no gutter.
    programmatic = json.loads(
        file_tools.read_file_programmatic_tool(
            str(target), offset=2, limit=1, task_id="pin-doc-prog"
        )
    )

    # file_ops path: mirror the extracted text into a plain-text file and
    # run the REAL backend read on the identical window with
    # line_numbers=False. Both paths share canonical newline semantics:
    # a raw page ends with "\n" (sed/cut always newline-terminate; the
    # early branch appends it to its joined page).
    from tools.read_extract import extract_document_text

    # Line 2 of the extraction is the first source line ("alpha"); line 1
    # is the "# ── Code cell N ──" header the extractor adds.
    extracted_line2 = extract_document_text(str(target)).splitlines()[1]
    assert extracted_line2 == "alpha"  # sanity: non-empty page content
    mirror = tmp_path / "mirror.txt"
    mirror.write_text(f"sentinel\n{extracted_line2}\ntail\n", encoding="utf-8")

    file_ops = ShellFileOperations(LocalEnvironment())
    raw = file_ops.read_file(str(mirror), offset=2, limit=1, line_numbers=False)

    assert not raw.error
    expected_page = f"{extracted_line2}\n"
    # Byte-identical page content across all three construction paths.
    assert programmatic["content"] == chat["content"] == raw.content == expected_page
