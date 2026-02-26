"""Tests for the file tools module (schema and handler wiring).

These tests verify the tool schemas and handler wiring without
requiring a running terminal environment. The actual file operations
(ShellFileOperations) depend on a terminal backend, so we mock
_get_file_ops to test the handler logic in isolation.
"""

import json
from unittest.mock import MagicMock, patch

from tools.file_tools import (
    FILE_TOOLS,
    READ_FILE_SCHEMA,
    WRITE_FILE_SCHEMA,
    PATCH_SCHEMA,
    SEARCH_FILES_SCHEMA,
)


class TestSchemas:
    def test_read_file_schema(self):
        assert READ_FILE_SCHEMA["name"] == "read_file"
        props = READ_FILE_SCHEMA["parameters"]["properties"]
        assert "path" in props
        assert "offset" in props
        assert "limit" in props

    def test_write_file_schema(self):
        assert WRITE_FILE_SCHEMA["name"] == "write_file"
        assert "path" in WRITE_FILE_SCHEMA["parameters"]["properties"]
        assert "content" in WRITE_FILE_SCHEMA["parameters"]["properties"]

    def test_patch_schema(self):
        assert PATCH_SCHEMA["name"] == "patch"
        props = PATCH_SCHEMA["parameters"]["properties"]
        assert "mode" in props
        assert "old_string" in props
        assert "new_string" in props

    def test_search_files_schema(self):
        assert SEARCH_FILES_SCHEMA["name"] == "search_files"
        props = SEARCH_FILES_SCHEMA["parameters"]["properties"]
        assert "pattern" in props
        assert "target" in props


class TestFileToolsList:
    def test_file_tools_has_expected_entries(self):
        names = {t["name"] for t in FILE_TOOLS}
        assert names == {"read_file", "write_file", "patch", "search_files"}


class TestReadFileHandler:
    @patch("tools.file_tools._get_file_ops")
    def test_read_file_returns_json(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"content": "hello", "total_lines": 1}
        mock_ops.read_file.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import read_file_tool

        result = json.loads(read_file_tool("/tmp/test.txt"))
        assert result["content"] == "hello"
        mock_ops.read_file.assert_called_once_with("/tmp/test.txt", 1, 500)


class TestWriteFileHandler:
    @patch("tools.file_tools._get_file_ops")
    def test_write_file_returns_json(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "path": "/tmp/test.txt"}
        mock_ops.write_file.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import write_file_tool

        result = json.loads(write_file_tool("/tmp/test.txt", "content"))
        assert result["status"] == "ok"
        mock_ops.write_file.assert_called_once_with("/tmp/test.txt", "content")


class TestPatchHandler:
    @patch("tools.file_tools._get_file_ops")
    def test_replace_mode_missing_path_errors(self, mock_get):
        from tools.file_tools import patch_tool

        result = json.loads(patch_tool(mode="replace", path=None, old_string="a", new_string="b"))
        assert "error" in result

    @patch("tools.file_tools._get_file_ops")
    def test_unknown_mode_errors(self, mock_get):
        from tools.file_tools import patch_tool

        result = json.loads(patch_tool(mode="unknown"))
        assert "error" in result
