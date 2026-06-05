"""The write_file tool accepts file_path/file_content as aliases for path/content.

skill_manage(action="write_file") names the same two arguments file_path/
file_content, while the standalone write_file tool uses path/content. Models
routinely carry one tool's naming across to the other, which previously rejected
a fully-formed call (the content was present, just under the wrong key) and cost
a wasted turn. These tests pin the alias behaviour. See issue #39964.
"""

import json
from unittest.mock import patch

import tools.file_tools as file_tools


def _capturing_write_file_tool():
    captured = {}

    def _stub(**kwargs):
        captured.update(kwargs)
        return json.dumps({"ok": True})

    return captured, _stub


def test_write_file_accepts_file_content_alias():
    """The reported failure: {path, file_content} must succeed."""
    captured, stub = _capturing_write_file_tool()
    with patch.object(file_tools, "write_file_tool", stub):
        out = file_tools._handle_write_file(
            {"path": "/tmp/x.txt", "file_content": "hello"}, task_id="t"
        )
    assert json.loads(out) == {"ok": True}
    assert captured["path"] == "/tmp/x.txt"
    assert captured["content"] == "hello"


def test_write_file_accepts_file_path_alias():
    captured, stub = _capturing_write_file_tool()
    with patch.object(file_tools, "write_file_tool", stub):
        file_tools._handle_write_file(
            {"file_path": "/tmp/y.txt", "content": "data"}, task_id="t"
        )
    assert captured["path"] == "/tmp/y.txt"
    assert captured["content"] == "data"


def test_write_file_canonical_names_win_over_aliases():
    captured, stub = _capturing_write_file_tool()
    with patch.object(file_tools, "write_file_tool", stub):
        file_tools._handle_write_file(
            {
                "path": "/canon", "file_path": "/alias",
                "content": "canon", "file_content": "alias",
            },
            task_id="t",
        )
    assert captured["path"] == "/canon"
    assert captured["content"] == "canon"


def test_write_file_empty_string_content_preserved():
    """Empty content is a valid write (truncate); the alias path must not
    misread '' as missing and fall back to file_content."""
    captured, stub = _capturing_write_file_tool()
    with patch.object(file_tools, "write_file_tool", stub):
        file_tools._handle_write_file(
            {"path": "/tmp/empty.txt", "content": "", "file_content": "SHOULD_NOT_WIN"},
            task_id="t",
        )
    assert captured["content"] == ""


def test_write_file_truly_missing_content_still_errors():
    out = file_tools._handle_write_file({"path": "/tmp/x.txt"}, task_id="t")
    assert "missing required field 'content'" in out


def test_write_file_truly_missing_path_still_errors():
    out = file_tools._handle_write_file({"content": "data"}, task_id="t")
    assert "missing required field 'path'" in out
