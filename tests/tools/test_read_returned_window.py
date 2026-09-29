"""Requested-limit jitter must not disguise unchanged local-file rereads."""
import json

import pytest

from agent.tool_result_classification import GUARDRAIL_REFUSAL_KEY
from tools.file_tools import clear_file_ops_cache
from tools.file_tools_read_tracking import notify_other_tool_call
from tools.registry import registry


@pytest.mark.parametrize("text", ["", "one\n"])
def test_empty_or_past_eof_read_does_not_mint_a_region(tmp_path, text):
    path = tmp_path / "empty.txt"
    path.write_text(text, encoding="utf-8")
    task = "returned-window-empty"
    try:
        for limit in (10, 20, 30, 40):
            result = json.loads(registry.dispatch(
                "read_file", {"path": str(path), "offset": 10, "limit": limit}, task_id=task))
            assert not result.get(GUARDRAIL_REFUSAL_KEY), result
    finally:
        clear_file_ops_cache(task)


def test_other_tool_breaks_returned_window_streak(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("one\n", encoding="utf-8")
    task = "returned-window-reset"
    try:
        for limit in (10, 20, 30):
            registry.dispatch("read_file", {"path": str(path), "limit": limit}, task_id=task)
        notify_other_tool_call(task)
        result = json.loads(registry.dispatch(
            "read_file", {"path": str(path), "limit": 40}, task_id=task))
        assert "one" in result.get("content", ""), result
    finally:
        clear_file_ops_cache(task)


def test_oversized_limits_do_not_restart_unchanged_read_streak(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    task = "returned-window-loop"
    try:
        results = [json.loads(registry.dispatch(
            "read_file", {"path": str(path), "limit": limit}, task_id=task
        )) for limit in (10, 20, 30, 40)]
        assert all("three" in r.get("content", "") for r in results[:3]), results
        assert results[-1].get(GUARDRAIL_REFUSAL_KEY), results[-1]
    finally:
        clear_file_ops_cache(task)


def test_changed_file_with_same_returned_window_is_progress(tmp_path):
    path = tmp_path / "notes.txt"
    task = "returned-window-changing"
    try:
        for i, limit in enumerate((10, 20, 30, 40, 50, 60)):
            path.write_text(f"version {i}\n", encoding="utf-8")
            result = json.loads(registry.dispatch(
                "read_file", {"path": str(path), "limit": limit}, task_id=task))
            assert "error" not in result, result
            assert f"version {i}" in result["content"]
    finally:
        clear_file_ops_cache(task)


def test_byte_budget_windows_count_actual_returned_lines(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.file_tools._get_max_read_chars", lambda: 40)
    path = tmp_path / "budget.txt"
    path.write_text("".join(f"row {i:03d} payload\n" for i in range(100)), encoding="utf-8")
    task = "returned-window-budget"
    try:
        results = [json.loads(registry.dispatch(
            "read_file", {"path": str(path), "limit": limit}, task_id=task
        )) for limit in (10, 20, 30, 40)]
        assert results[0]["truncated_by"] == "bytes"
        assert results[0]["next_offset"] < 10
        assert results[-1].get(GUARDRAIL_REFUSAL_KEY), results[-1]
    finally:
        clear_file_ops_cache(task)


def test_growing_budget_on_clamped_first_line_remains_progress(tmp_path, monkeypatch):
    path = tmp_path / "long-line.txt"
    path.write_text("x" * 1000 + "\n", encoding="utf-8")
    task = "returned-window-clamped"
    try:
        for limit, budget in zip((10, 20, 30, 40), (20, 40, 60, 80)):
            monkeypatch.setattr("tools.file_tools._get_max_read_chars", lambda: budget)
            result = json.loads(registry.dispatch(
                "read_file", {"path": str(path), "limit": limit}, task_id=task))
            assert result.get("truncated_lines"), result
            assert len(result["content"]) == budget
            assert not result.get(GUARDRAIL_REFUSAL_KEY), result
    finally:
        clear_file_ops_cache(task)


def test_failed_read_does_not_advance_returned_window_streak(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("one\n", encoding="utf-8")
    task = "returned-window-error"
    try:
        for limit in (10, 20):
            registry.dispatch("read_file", {"path": str(path), "limit": limit}, task_id=task)
        missing = json.loads(registry.dispatch(
            "read_file", {"path": str(tmp_path / "absent.txt")}, task_id=task))
        assert missing.get("error"), missing
        third = json.loads(registry.dispatch(
            "read_file", {"path": str(path), "limit": 30}, task_id=task))
        assert "one" in third.get("content", ""), third
    finally:
        clear_file_ops_cache(task)


def test_expanding_actual_windows_keep_returning_content(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("".join(f"row {i}\n" for i in range(20)), encoding="utf-8")
    task = "returned-window-expanding"
    try:
        for limit in (1, 2, 3, 4, 5, 6):
            result = json.loads(registry.dispatch(
                "read_file", {"path": str(path), "limit": limit}, task_id=task))
            assert "error" not in result, result
            assert f"row {limit - 1}" in result["content"]
    finally:
        clear_file_ops_cache(task)
