"""Test that _build_browser_env merges Hermes node dir into PATH."""

import os
from unittest.mock import patch
from tools.browser_tool import _build_browser_env, _merge_browser_path


def test_build_browser_env_merges_browser_path(tmp_path):
    """Verify _build_browser_env() merges Hermes node dirs into the subprocess PATH (Closes #97186)."""
    fake_node_bin = tmp_path / "hermes_node" / "bin"
    fake_node_bin.mkdir(parents=True, exist_ok=True)
    fake_bin_str = str(fake_node_bin)

    with patch("tools.browser_tool._browser_candidate_path_dirs", return_value=[fake_bin_str]):
        with patch.dict("os.environ", {"PATH": "existing_bin_path"}, clear=False):
            env = _build_browser_env()
            assert "PATH" in env
            parts = env["PATH"].split(os.pathsep)
            assert fake_bin_str in parts
            assert parts[0] == fake_bin_str
            assert "existing_bin_path" in parts


def test_build_browser_env_path_deduplication(tmp_path):
    """Verify _build_browser_env() is idempotent and does not duplicate existing PATH entries."""
    fake_node_bin = tmp_path / "hermes_node" / "bin"
    fake_node_bin.mkdir(parents=True, exist_ok=True)
    fake_bin_str = str(fake_node_bin)

    with patch("tools.browser_tool._browser_candidate_path_dirs", return_value=[fake_bin_str]):
        initial_path = os.pathsep.join([fake_bin_str, "other_bin"])
        with patch.dict("os.environ", {"PATH": initial_path}, clear=False):
            env = _build_browser_env()
            parts = env["PATH"].split(os.pathsep)
            assert parts.count(fake_bin_str) == 1
