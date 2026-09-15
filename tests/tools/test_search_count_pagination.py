"""Count-mode search pages the per-file table like every other mode.

Regression: ``output_mode="count"`` ignored offset/limit (offset a no-op,
the head-capped fetch reported as complete), while files_only/content slice
``[offset:offset+limit]`` with a truncated flag.
"""

from tools.file_operations_common import ExecuteResult
from tools.file_operations_search import _parse_search_output


def _count_result(stdout):
    return _parse_search_output(
        ExecuteResult(stdout=stdout, exit_code=0),
        "count", 2, 1, 0)


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
