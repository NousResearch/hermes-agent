"""Regression tests for backend-visible canonical spillover paths."""

import stat
from pathlib import PurePosixPath
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.parametrize("backend", ["ssh", "daytona", "vercel_sandbox"])
def test_tilde_backend_canonical_spillover_enforces_expiry(backend, monkeypatch):
    from tools.file_tools import _expand_tilde, read_file_tool

    monkeypatch.setenv("TERMINAL_ENV", backend)
    requested = "~/.hermes/cache/spillover/tool-results/expired.txt"
    resolved = _expand_tilde(requested)
    env = MagicMock()
    file_ops = MagicMock(env=env)

    with (
        patch("tools.file_tools._get_file_ops", return_value=file_ops),
        patch("tools.file_tools._file_ops_uses_host_paths", return_value=False),
        patch(
            "tools.file_tools._resolve_path_for_task",
            return_value=PurePosixPath(resolved),
        ),
        patch(
            "tools.file_tools.os.lstat",
            return_value=MagicMock(st_mode=stat.S_IFDIR),
        ),
        patch(
            "tools.tool_result_storage._expire_remote_spillover_on_access",
            return_value=True,
        ) as expire,
    ):
        result = read_file_tool(requested)

    assert "expired after 24 hours" in result
    expire.assert_called_once_with(resolved, env)
    file_ops.read_file.assert_not_called()
