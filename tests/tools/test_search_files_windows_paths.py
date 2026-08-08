"""Windows path normalization regressions for search_files."""

import json
import sys
from unittest.mock import patch

import pytest

from tools.file_operations import SearchResult
from tools.file_operations import ShellFileOperations
from tools.file_tools import search_tool
from tools.environments.local import LocalEnvironment


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific regression")
def test_search_tool_normalizes_msys_drive_path_before_backend(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    native = str(root)
    drive, tail = native.split(":", 1)
    msys_path = f"/{drive.lower()}{tail.replace(chr(92), '/')}"

    with patch("tools.file_tools._get_file_ops") as get_file_ops:
        get_file_ops.return_value.search.return_value = SearchResult(total_count=0)

        result = json.loads(
            search_tool("needle", path=msys_path, task_id="windows-msys-search")
        )

    assert result["total_count"] == 0
    dispatched_path = get_file_ops.return_value.search.call_args.kwargs["path"]
    assert dispatched_path == native


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific regression")
def test_native_rg_path_quoting_is_local_backend_only():
    local_env = object.__new__(LocalEnvironment)
    local_ops = ShellFileOperations(local_env, cwd="C:/workspace")
    assert local_ops._escape_native_tool_path(r"C:\workspace\src") == "'C:/workspace/src'"

    class RemoteEnvironment:
        cwd = "/workspace"

    remote_ops = ShellFileOperations(RemoteEnvironment())
    assert remote_ops._escape_native_tool_path("/c/container-data") == "'/c/container-data'"
