"""Cross-tool parameter-name aliases for file-writing tools.

The standalone ``write_file`` tool names its arguments ``path`` / ``content``,
while ``skill_manage(action="write_file")`` names the same two arguments
``file_path`` / ``file_content``. Models routinely carry one tool's naming
across to the other, which previously rejected a fully-formed call (the content
was present under the wrong key) and cost a wasted turn. These tests pin the
alias behaviour so the synonyms are accepted interchangeably.
"""

import json
from unittest.mock import patch

import tools.file_tools as file_tools
import tools.skill_manager_tool as smt
from tools.registry import registry


# ---------------------------------------------------------------------------
# write_file tool: accept file_path / file_content as aliases
# ---------------------------------------------------------------------------

def _capturing_write_file_tool():
    captured = {}

    def _stub(**kwargs):
        captured.update(kwargs)
        return json.dumps({"ok": True})

    return captured, _stub


def test_write_file_accepts_file_content_alias():
    """The exact failure from the field: {path, file_content} must succeed."""
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
    misread '' as missing and fall back."""
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


# ---------------------------------------------------------------------------
# skill_manage write_file action: accept path / content as aliases
# ---------------------------------------------------------------------------

def test_skill_manage_write_file_accepts_content_alias():
    """file_content omitted, `content` supplied -> falls back to content."""
    captured = {}

    def _stub(name, file_path, file_content):
        captured.update(name=name, file_path=file_path, file_content=file_content)
        return {"success": True, "message": "ok"}

    with patch.object(smt, "_write_file", _stub):
        smt.skill_manage(
            action="write_file", name="myskill",
            content="BODY", file_path="references/a.md", file_content=None,
        )
    assert captured["file_path"] == "references/a.md"
    assert captured["file_content"] == "BODY"


def test_skill_manage_write_file_accepts_path_alias_via_handler():
    """The registry handler maps the model's `path` -> file_path."""
    captured = {}

    def _stub(name, file_path, file_content):
        captured.update(name=name, file_path=file_path, file_content=file_content)
        return {"success": True, "message": "ok"}

    entry = registry.get_entry("skill_manage")
    assert entry is not None, "skill_manage not registered"
    with patch.object(smt, "_write_file", _stub):
        entry.handler({
            "action": "write_file", "name": "myskill",
            "path": "references/b.md", "content": "DATA",
        })
    assert captured["file_path"] == "references/b.md"
    assert captured["file_content"] == "DATA"


def test_skill_manage_write_file_canonical_file_names_still_work():
    captured = {}

    def _stub(name, file_path, file_content):
        captured.update(name=name, file_path=file_path, file_content=file_content)
        return {"success": True, "message": "ok"}

    entry = registry.get_entry("skill_manage")
    with patch.object(smt, "_write_file", _stub):
        entry.handler({
            "action": "write_file", "name": "myskill",
            "file_path": "references/c.md", "file_content": "CANON",
        })
    assert captured["file_path"] == "references/c.md"
    assert captured["file_content"] == "CANON"
