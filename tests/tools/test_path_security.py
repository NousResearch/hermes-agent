"""Unit tests for has_traversal_component (tools/path_security.py).

has_traversal_component is a pure, I/O-free guard used by skills_tool,
tts_tool, file_tools, and skill_manager_tool to reject path-traversal
attempts before expensive resolution. These tests pin its contract: it
flags ".." only as a whole path component, never as a substring, and
treats "." / "..." as non-traversal.

All inputs use forward slashes and relative paths so pathlib parses them
identically on POSIX and Windows.
"""

import pytest

from tools.path_security import has_traversal_component


class TestHasTraversalComponent:
    @pytest.mark.parametrize("path_str", [
        "a/../b", "../etc/passwd", "..", "foo/bar/..", "../../secret",
    ])
    def test_traversal_component_detected(self, path_str):
        assert has_traversal_component(path_str) is True

    @pytest.mark.parametrize("path_str", [
        "a/b/c", "safe/path.txt", "config.yaml", "dir/sub/file",
    ])
    def test_clean_paths_pass(self, path_str):
        assert has_traversal_component(path_str) is False

    @pytest.mark.parametrize("path_str", ["a/b..c/d", "file..txt", "ab..cd"])
    def test_double_dot_substring_is_not_traversal(self, path_str):
        # ".." must match a whole path component, not appear as a substring.
        assert has_traversal_component(path_str) is False

    @pytest.mark.parametrize("path_str", ["...", "a/.../b"])
    def test_triple_dot_is_not_traversal(self, path_str):
        assert has_traversal_component(path_str) is False

    def test_current_dir_is_not_traversal(self):
        assert has_traversal_component("./a") is False

    def test_empty_string_is_not_traversal(self):
        assert has_traversal_component("") is False
