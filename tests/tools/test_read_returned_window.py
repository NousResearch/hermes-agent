"""Returned-window identity must distinguish repetition from actual progress."""
import json

import pytest

from agent.tool_result_classification import GUARDRAIL_REFUSAL_KEY
from tools.file_tools import clear_file_ops_cache
from tools.file_tools_read_tracking import notify_other_tool_call
from tools.registry import registry


@pytest.mark.parametrize("budget", [None, 40])
def test_limit_jitter_cannot_disguise_identical_returned_windows(tmp_path, monkeypatch, budget):
    path = tmp_path / "notes.txt"
    text = "one\ntwo\nthree\n" if budget is None else "".join(
        f"row {i:03d} payload\n" for i in range(100))
    path.write_text(text, encoding="utf-8")
    if budget is not None:
        monkeypatch.setattr("tools.file_tools._get_max_read_chars", lambda: budget)
    task = "returned-window-repetition"
    try:
        results = [json.loads(registry.dispatch(
            "read_file", {"path": str(path), "limit": limit}, task_id=task
        )) for limit in (10, 20, 30, 40)]
        assert all(r.get("content") for r in results[:3]), results
        assert results[0]["content"] == results[2]["content"]
        if budget is not None:
            assert results[0]["truncated_by"] == "bytes"
            assert results[0]["next_offset"] < 10
        assert results[-1].get(GUARDRAIL_REFUSAL_KEY), results[-1]
        if budget is None:
            written = json.loads(registry.dispatch("write_file", {
                "path": str(path), "content": "replacement\n",
            }, task_id=task))
            assert written.get("verified"), written
            assert path.read_text(encoding="utf-8") == "replacement\n"
    finally:
        clear_file_ops_cache(task)


@pytest.mark.parametrize("scenario", [
    "changed", "expanding", "empty", "past_eof", "reset", "failed", "clamped",
])
def test_progress_and_non_region_results_do_not_form_a_false_streak(tmp_path, monkeypatch, scenario):
    path = tmp_path / "notes.txt"
    text = "" if scenario == "empty" else "x" * 1000 + "\n" if scenario == "clamped" else "one\n"
    if scenario == "expanding":
        text = "".join(f"row {i}\n" for i in range(20))
    path.write_text(text, encoding="utf-8")
    task = "returned-window-" + scenario
    count = 3 if scenario == "failed" else 4
    try:
        for i in range(count):
            limit = i + 1 if scenario == "expanding" else (i + 1) * 10
            if scenario == "changed":
                path.write_text(f"version {i}\n", encoding="utf-8")
            if scenario == "clamped":
                budget = (i + 1) * 20
                monkeypatch.setattr("tools.file_tools._get_max_read_chars", lambda: budget)
            if scenario == "reset" and i == 3:
                notify_other_tool_call(task)
            if scenario == "failed" and i == 2:
                missing = json.loads(registry.dispatch(
                    "read_file", {"path": str(tmp_path / "absent.txt")}, task_id=task))
                assert missing.get("error"), missing
            result = json.loads(registry.dispatch("read_file", {
                "path": str(path), "limit": limit,
                "offset": 10 if scenario in ("empty", "past_eof") else 1,
            }, task_id=task))
            assert not result.get(GUARDRAIL_REFUSAL_KEY), result
            if scenario == "clamped":
                assert result.get("truncated_lines"), result
                assert len(result["content"]) == budget
            elif scenario == "changed":
                assert f"version {i}" in result["content"]
            elif scenario == "expanding":
                assert f"row {i}" in result["content"]
            elif scenario not in ("empty", "past_eof"):
                assert "one" in result.get("content", ""), result
    finally:
        clear_file_ops_cache(task)
