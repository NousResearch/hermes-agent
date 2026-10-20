"""Multi-path content search paginates once globally, not per root.

Regression: ``_try_multi_path_search`` applied the caller's offset/limit to
every root (skipping offset×N roots' worth, truncating each root to limit
before the global slice) and dropped already-collected roots on a later
root's error.
"""

from unittest.mock import MagicMock, patch

from tools.file_operations import ShellFileOperations
from tools.file_operations_common import SearchMatch, SearchResult


def _capturing_ops():
    env = MagicMock()
    env.cwd = "/tmp/test"
    return ShellFileOperations(env)


def _match_result(tag, n, start=0):
    return SearchResult(
        matches=[SearchMatch(path=f"/{tag}/f{i}.py", line_number=1, content=f"hit {i}")
                 for i in range(start, start + n)],
        total_count=n,
    )


def _probe(self, path):
    # The combined multi-path string probes missing; each part exists.
    return "not_found" if " " in path else "exists"


def _run_multi(results, offset=0, limit=50):
    ops = _capturing_ops()
    with (
        patch.object(ShellFileOperations, "_expand_path", side_effect=lambda p: p),
        patch.object(ShellFileOperations, "_path_exists_probe", autospec=True,
                     side_effect=_probe),
        patch.object(ShellFileOperations, "_search_content", side_effect=results),
        patch.object(ShellFileOperations, "_effective_macos_search_exclusions", return_value=[]),
    ):
        return ops.search("needle", path="/r1 /r2", limit=limit, offset=offset)


class TestMultipathPagination:
    def test_offset_applies_once_globally(self):
        first = _match_result("r1", 10)
        second = _match_result("r2", 10)
        result = _run_multi([first, second], offset=10, limit=10)
        assert [m.path for m in result.matches] == [f"/r2/f{i}.py" for i in range(10)]

    def test_limit_applies_once_globally(self):
        first = _match_result("r1", 10)
        second = _match_result("r2", 10)
        result = _run_multi([first, second], offset=0, limit=5)
        assert [m.path for m in result.matches] == [f"/r1/f{i}.py" for i in range(5)]

    def test_roots_share_one_fetch_budget(self):
        seen = {}

        def fake_content(self, pattern, path, file_glob, limit, offset, *rest):
            seen[path] = (limit, offset)
            return SearchResult()

        ops = _capturing_ops()
        with (
            patch.object(ShellFileOperations, "_expand_path", side_effect=lambda p: p),
            patch.object(ShellFileOperations, "_path_exists_probe", autospec=True,
                         side_effect=_probe),
            patch.object(ShellFileOperations, "_search_content", autospec=True,
                         side_effect=fake_content),
            patch.object(ShellFileOperations, "_effective_macos_search_exclusions",
                         return_value=[]),
        ):
            ops.search("needle", path="/r1 /r2", limit=7, offset=3)
        # Each root fetched from zero with the shared limit+offset budget.
        assert seen == {"/r1": (10, 0), "/r2": (10, 0)}

    def test_partial_root_error_preserves_collected_matches(self):
        good = _match_result("r1", 3)
        bad = SearchResult(error="boom")
        result = _run_multi([good, bad])
        assert [m.path for m in result.matches] == [f"/r1/f{i}.py" for i in range(3)]
        assert result.error is None
        assert "boom" in (result.warning or "")

    def test_all_roots_failing_still_errors(self):
        result = _run_multi([SearchResult(error="boom"), SearchResult(error="boom")])
        assert result.error is not None
        assert "boom" in result.error
        assert result.matches == []
