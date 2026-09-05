"""read_file optional Markdown outline mode (#103374).

Covers the opt-in ``mode="outline"`` behavior: Markdown heading extraction
(ATX and Setext levels, source line numbers), empty outlines for Markdown
without headings, the clear note for non-Markdown paths, bounded output with
an explicit continuation offset, and the unchanged default read mode.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.file_tools import READ_FILE_SCHEMA, read_file_tool
from tools.markdown_outline import markdown_outline


def _mock_raw_read(content: str, file_size: int = 1234, error: str = None):
    """Patch _get_file_ops so read_file_raw serves *content* (or *error*)."""
    ops = MagicMock()
    result = MagicMock()
    result.content = content if error is None else ""
    to_dict = {"file_size": file_size}
    if error is not None:
        to_dict = {"error": error, "file_size": file_size}
    else:
        to_dict["content"] = content
    result.to_dict.return_value = to_dict
    ops.read_file_raw.return_value = result
    patcher = patch("tools.file_tools._get_file_ops", return_value=ops)
    patcher.start()
    return ops, patcher


class TestMarkdownOutlineScanner:
    def test_atx_levels_and_closing_hash_sequence(self):
        entries = markdown_outline("# Title ###\n## Sub\n### Deep\nbody\n###### H6\n")
        assert entries == [
            {"line": 1, "level": 1, "heading": "Title"},
            {"line": 2, "level": 2, "heading": "Sub"},
            {"line": 3, "level": 3, "heading": "Deep"},
            {"line": 5, "level": 6, "heading": "H6"},
        ]

    def test_setext_headings(self):
        entries = markdown_outline("Setext H1\n=======\nSetext H2\n-------\n")
        assert entries == [
            {"line": 1, "level": 1, "heading": "Setext H1"},
            {"line": 3, "level": 2, "heading": "Setext H2"},
        ]

    def test_ignores_heading_like_lines_inside_fences(self):
        content = (
            "# Real\n"
            "```\n"
            "# Not a heading\n"
            "```\n"
            "~~~python\n"
            "## Also not\n"
            "~~~\n"
            "# Real 2\n"
        )
        entries = markdown_outline(content)
        assert [e["heading"] for e in entries] == ["Real", "Real 2"]
        assert [e["line"] for e in entries] == [1, 8]

    def test_duplicate_headings_kept_as_separate_entries(self):
        entries = markdown_outline("# Same\nbody\n# Same\n# Same\n")
        assert [e["line"] for e in entries] == [1, 3, 4]
        assert all(e["heading"] == "Same" for e in entries)

    def test_crlf_line_numbers(self):
        entries = markdown_outline("# A\r\nbody\r\n## B\r\n")
        assert [e["line"] for e in entries] == [1, 3]

    def test_no_headings(self):
        assert markdown_outline("just text\nno structure\n") == []
        assert markdown_outline("") == []


class TestReadFileOutlineTool:
    def test_markdown_with_multiple_levels(self, tmp_path):
        md = (
            "# Title\n"
            "intro\n"
            "## Background\n"
            "### Details\n"
            "## Scope\n"
        )
        ops, patcher = _mock_raw_read(md)
        try:
            result = json.loads(read_file_tool("/tmp/outline_multi.md", mode="outline"))
        finally:
            patcher.stop()
        assert result["mode"] == "outline"
        assert result["total_headings"] == 4
        assert result["truncated"] is False
        assert result["outline"] == [
            {"line": 1, "level": 1, "heading": "Title"},
            {"line": 3, "level": 2, "heading": "Background"},
            {"line": 4, "level": 3, "heading": "Details"},
            {"line": 5, "level": 2, "heading": "Scope"},
        ]
        ops.read_file_raw.assert_called_once_with("/tmp/outline_multi.md")
        ops.read_file.assert_not_called()

    def test_markdown_without_headings(self, tmp_path):
        ops, patcher = _mock_raw_read("plain paragraph\nno headings here\n")
        try:
            result = json.loads(read_file_tool("/tmp/outline_nohead.md", mode="outline"))
        finally:
            patcher.stop()
        assert result["outline"] == []
        assert result["total_headings"] == 0
        assert result["truncated"] is False

    def test_non_markdown_file_returns_clear_note(self, tmp_path):
        ops, patcher = _mock_raw_read("print('hello')\n")
        try:
            result = json.loads(read_file_tool("/tmp/outline_prog.py", mode="outline"))
        finally:
            patcher.stop()
        assert result["outline"] == []
        assert "Markdown" in result["note"]
        assert ".py" in result["note"]
        ops.read_file_raw.assert_not_called()
        ops.read_file.assert_not_called()

    def test_default_mode_is_unchanged(self, tmp_path):
        ops = MagicMock()
        result_obj = MagicMock()
        result_obj.content = "line1\nline2"
        result_obj.to_dict.return_value = {"content": "line1\nline2", "total_lines": 2}
        ops.read_file.return_value = result_obj
        with patch("tools.file_tools._get_file_ops", return_value=ops):
            result = json.loads(read_file_tool("/tmp/outline_default.txt"))
        assert result["content"] == "line1\nline2"
        assert "mode" not in result
        ops.read_file.assert_called_once_with("/tmp/outline_default.txt", 1, 2000)
        ops.read_file_raw.assert_not_called()

    def test_explicit_mode_read_is_unchanged(self, tmp_path):
        ops = MagicMock()
        result_obj = MagicMock()
        result_obj.content = "line1\nline2"
        result_obj.to_dict.return_value = {"content": "line1\nline2", "total_lines": 2}
        ops.read_file.return_value = result_obj
        with patch("tools.file_tools._get_file_ops", return_value=ops):
            result = json.loads(read_file_tool("/tmp/outline_read.txt", mode="read"))
        assert result["content"] == "line1\nline2"
        ops.read_file.assert_called_once_with("/tmp/outline_read.txt", 1, 2000)

    def test_outline_truncated_with_continuation_offset(self):
        md = "\n".join(f"# Heading {i}" for i in range(505))
        ops, patcher = _mock_raw_read(md)
        try:
            first = json.loads(read_file_tool("/tmp/outline_big.md", mode="outline"))
            second = json.loads(read_file_tool("/tmp/outline_big.md", mode="outline", offset=501))
        finally:
            patcher.stop()
        assert first["total_headings"] == 505
        assert len(first["outline"]) == 500
        assert first["truncated"] is True
        assert "offset=501" in first["_hint"]
        assert len(second["outline"]) == 5
        assert second["truncated"] is False

    def test_unknown_mode_returns_error(self):
        result = json.loads(read_file_tool("/tmp/outline_unknown.md", mode="table"))
        assert "unknown mode" in result["error"]

    def test_binary_markdown_returns_backend_error(self):
        ops, patcher = _mock_raw_read("", error="Binary file (PNG image data, 1.2 KB)")
        try:
            result = json.loads(read_file_tool("/tmp/outline_binary.md", mode="outline"))
        finally:
            patcher.stop()
        assert "error" in result

    def test_schema_exposes_opt_in_mode_default_read(self):
        mode = READ_FILE_SCHEMA["parameters"]["properties"]["mode"]
        assert mode["type"] == "string"
        assert mode["enum"] == ["read", "outline"]
        assert mode["default"] == "read"
        assert "path" in READ_FILE_SCHEMA["parameters"]["required"]
