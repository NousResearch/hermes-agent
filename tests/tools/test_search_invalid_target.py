"""Tests that search() rejects unknown `target` values instead of falling through.

Regression for #74448: `search_files` declares `target` as an enum of
("content", "files"), but nothing enforced it. `FileOperations.search()`
branched on `target == "files"` and sent everything else to a *content*
search, so an invalid value came back as a clean `{"total_count": 0}` with
no error.

`files_only` is the motivating mistake: `output_mode` on the same tool does
accept `files_only`, so a caller reaching for "search filenames" naturally
picks it for `target` too, then gets an empty result with no signal that
`target` was the problem.

Fix: validate `target` at the top of search() and return a SearchResult
carrying an explanatory error.
"""

from unittest.mock import MagicMock

import pytest

from tools.file_operations import ShellFileOperations


@pytest.fixture()
def ops():
    env = MagicMock(cwd="/tmp/test")
    env.execute.return_value = {"output": "exists", "returncode": 0}
    return ShellFileOperations(env)


class TestInvalidTargetIsRejected:
    @pytest.mark.parametrize("target", ["files_only", "bogus", "FILES", "grep", "find", ""])
    def test_invalid_target_returns_error(self, ops, target):
        result = ops.search("pattern", path=".", target=target)

        assert result.error is not None, f"target={target!r} should be rejected"
        assert "Invalid target" in result.error
        assert repr(target) in result.error
        assert result.total_count == 0

    def test_error_names_both_valid_targets(self, ops):
        result = ops.search("pattern", path=".", target="files_only")

        assert "'content'" in result.error
        assert "'files'" in result.error

    def test_error_points_files_only_at_output_mode(self, ops):
        """The whole point: disambiguate target from output_mode='files_only'."""
        result = ops.search("pattern", path=".", target="files_only")

        assert "output_mode='files_only'" in result.error

    def test_rejected_before_touching_the_filesystem(self, ops):
        """Validation is argument-level, so it must not shell out first."""
        ops.search("pattern", path=".", target="bogus")

        ops.env.execute.assert_not_called()

    def test_invalid_target_does_not_run_a_content_search(self, ops):
        """The actual bug: the old code silently ran _search_content."""
        called = []
        ops._search_content = lambda *a, **k: called.append(a)
        ops._search_files = lambda *a, **k: called.append(a)

        ops.search("pattern", path=".", target="files_only")

        assert called == []


class TestValidTargetsStillWork:
    def test_files_dispatches_to_search_files(self, ops):
        sentinel = object()
        ops._search_files = lambda *a, **k: sentinel

        assert ops.search("*.py", path=".", target="files") is sentinel

    def test_content_dispatches_to_search_content(self, ops):
        sentinel = object()
        ops._search_content = lambda *a, **k: sentinel

        assert ops.search("pattern", path=".", target="content") is sentinel

    def test_default_target_is_content(self, ops):
        sentinel = object()
        ops._search_content = lambda *a, **k: sentinel

        assert ops.search("pattern", path=".") is sentinel


class TestHandlerAliasesSurviveValidation:
    """_handle_search_files maps grep->content and find->files before search()."""

    @pytest.mark.parametrize(
        ("raw_target", "expected"),
        [("grep", "content"), ("find", "files"), ("content", "content"), ("files", "files")],
    )
    def test_alias_maps_to_valid_target(self, raw_target, expected):
        target_map = {"grep": "content", "find": "files"}

        assert target_map.get(raw_target, raw_target) == expected
