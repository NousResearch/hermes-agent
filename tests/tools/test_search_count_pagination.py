"""Count-mode search pages the per-file table like every other mode.

Regression: ``output_mode="count"`` ignored offset/limit (offset a no-op,
the head-capped fetch reported as complete), while files_only/content slice
``[offset:offset+limit]`` with a truncated flag.
"""
import pytest

from tools.file_operations_common import ExecuteResult
from tools.file_operations_search import _parse_search_output


def _count_result(stdout, limit=2, offset=1, exit_code=0):
    return _parse_search_output(
        ExecuteResult(stdout=stdout, exit_code=exit_code),
        "count", limit, offset, 0)


class TestCountPagination:
    STDOUT = "e.py:5\na.py:1\nc.py:3\nb.py:2\nd.py:4\n"

    def test_page_is_sorted_and_sliced(self):
        result = _count_result(self.STDOUT)
        assert result.counts == {"b.py": 2, "c.py": 3}
        assert result.total_count == 15

    def test_truncated_when_window_cuts(self):
        result = _count_result(self.STDOUT)
        assert result.truncated is True

    def test_full_window_not_truncated(self):
        result = _parse_search_output(
            ExecuteResult(stdout="a.py:1\nb.py:2\n", exit_code=0),
            "count", 10, 0, 0)
        assert result.counts == {"a.py": 1, "b.py": 2}
        assert result.total_count == 3
        assert result.truncated is False


class TestCountPaginationEdgeCases:

    def test_empty_stdout(self):
        """Empty output must produce empty counts, not crash."""
        result = _count_result("", limit=10, offset=0)
        assert result.counts == {}
        assert result.total_count == 0
        assert result.truncated is False
        assert result.error is None

    def test_offset_beyond_all_entries(self):
        """Offset past the last entry must produce an empty page."""
        result = _count_result("a.py:1\nb.py:2\n", limit=10, offset=100)
        assert result.counts == {}
        assert result.total_count == 3
        assert result.truncated is False

    def test_limit_zero(self):
        """limit=0 must produce an empty page, not crash."""
        result = _count_result("a.py:1\nb.py:2\n", limit=0, offset=0)
        assert result.counts == {}
        assert result.total_count == 3
        assert result.truncated is True  # 2 > 0+0

    def test_non_numeric_count_skipped(self):
        """A line with a non-numeric count must be silently skipped."""
        result = _count_result("a.py:1\nb.py:xyz\nc.py:3\n", limit=10, offset=0)
        assert result.counts == {"a.py": 1, "c.py": 3}
        assert result.total_count == 4

    def test_duplicate_path_last_wins(self):
        """Duplicate paths: the last occurrence overwrites (not sum)."""
        result = _count_result("a.py:1\na.py:99\n", limit=10, offset=0)
        assert result.counts == {"a.py": 99}
        assert result.total_count == 99

    def test_exit_code_2_with_no_payload_returns_error(self):
        """Exit code 2 with no usable output must surface as error."""
        result = _count_result("", limit=10, offset=0, exit_code=2)
        assert result.error is not None
        assert "Search failed" in result.error

    def test_exit_code_2_with_payload_still_parses(self):
        """Exit code 2 with usable payload must still parse (partial error)."""
        result = _count_result("a.py:1\n", limit=10, offset=0, exit_code=2)
        assert result.error is None
        assert result.counts == {"a.py": 1}
