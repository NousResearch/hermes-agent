"""Updated file contents are progress, even when the requested region is unchanged."""

import json

from agent.tool_result_classification import GUARDRAIL_REFUSAL_KEY
from tools.file_tools import clear_file_ops_cache
from tools.registry import registry


def test_changed_file_reads_remain_available_but_unchanged_loops_are_bounded(tmp_path):
    path = tmp_path / "progress.txt"
    task = "external-file-progress"

    def read():
        return json.loads(registry.dispatch("read_file", {"path": str(path)}, task_id=task))

    try:
        for version in range(6):
            # Another actor updates the file; no tool in this task resets its counter.
            path.write_text(f"Progress version {version}\n", encoding="utf-8")
            result = read()
            assert "error" not in result, result
            assert f"Progress version {version}" in result["content"]
        assert any(read().get(GUARDRAIL_REFUSAL_KEY) for _ in range(4)), (
            "unchanged repeat reads must still trigger the no-progress guard")
    finally:
        clear_file_ops_cache(task)


def test_changed_search_results_reset_the_no_progress_streak(tmp_path):
    path = tmp_path / "progress.txt"
    task = "external-search-progress"

    def search():
        return json.loads(registry.dispatch(
            "search_files", {"path": str(path), "pattern": "Progress"}, task_id=task))

    try:
        for version in range(6):
            path.write_text(f"Progress version {version}\n", encoding="utf-8")
            result = search()
            assert "error" not in result, result
            assert f"Progress version {version}" in json.dumps(result)
        assert any(search().get(GUARDRAIL_REFUSAL_KEY) for _ in range(4)), (
            "unchanged repeat searches must still trigger the no-progress guard")
    finally:
        clear_file_ops_cache(task)
