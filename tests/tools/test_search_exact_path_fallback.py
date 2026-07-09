from __future__ import annotations

import json


def test_exact_file_search_files_returns_single_existing_file(tmp_path):
    from tools.file_tools import search_tool

    target = tmp_path / "notes.txt"
    target.write_text("alpha\nneedle\n", encoding="utf-8")

    result = json.loads(
        search_tool("*.txt", target="files", path=str(target), limit=10)
    )

    assert result["total_count"] == 1
    assert result["files"] == [str(target.resolve())]


def test_exact_file_content_search_honors_offset_limit_and_truncation(tmp_path):
    from tools.file_tools import search_tool

    target = tmp_path / "notes.txt"
    target.write_text("needle one\nskip\nneedle two\nneedle three\n", encoding="utf-8")

    result = json.loads(
        search_tool("needle", target="content", path=str(target), limit=1, offset=1)
    )

    assert result["total_count"] == 3
    assert result["truncated"] is True
    assert result["matches"] == [
        {"path": str(target.resolve()), "line": 3, "content": "needle two"}
    ]


def test_exact_file_content_search_honors_file_glob(tmp_path):
    from tools.file_tools import search_tool

    target = tmp_path / "notes.txt"
    target.write_text("needle\n", encoding="utf-8")

    result = json.loads(
        search_tool(
            "needle",
            target="content",
            path=str(target),
            file_glob="*.py",
            limit=10,
        )
    )

    assert result["total_count"] == 0
    assert "matches" not in result


def test_usage_guard_caps_still_apply_to_exact_file_fallback(tmp_path):
    from hermes_cli.usage_guard import (
        activate_usage_guard_after_warning,
        reset_usage_guard_for_tests,
    )
    from tools.file_tools import search_tool

    target = tmp_path / "notes.txt"
    target.write_text("needle\n", encoding="utf-8")
    try:
        activate_usage_guard_after_warning(task_id="exact-search", reason="test")

        denied = json.loads(
            search_tool(
                "needle",
                target="content",
                path=str(target),
                limit=50,
                task_id="exact-search",
            )
        )
        allowed = json.loads(
            search_tool(
                "needle",
                target="content",
                path=str(target),
                limit=1,
                task_id="exact-search",
            )
        )

        assert denied["usage_guard"] == "active"
        assert "limit <=" in denied["error"]
        assert allowed["total_count"] == 1
        assert allowed["matches"][0]["content"] == "needle"
    finally:
        reset_usage_guard_for_tests()
