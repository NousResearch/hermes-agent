"""Regression for #147722 through the registered tool and real local files."""

import json

import pytest

import tools.file_tools  # registers search_files
from tools.registry import registry


@pytest.mark.parametrize("offset, expected_lines", [(0, [1, 2, 3]), (1, [3, 4, 5])])
def test_search_pages_share_context_without_counting_it(tmp_path, offset, expected_lines):
    document = tmp_path / "notes.txt"
    document.write_text("before\nHERON first\nshared context\nHERON second\nafter\n", encoding="utf-8")

    result = json.loads(registry.dispatch("search_files", {
        "pattern": "HERON", "path": str(document), "context": 1,
        "limit": 1, "offset": offset,
    }, task_id=f"context-page-{offset}"))

    assert "error" not in result
    assert result["total_count"] == 2
    assert [row["line"] for row in result["matches"]] == expected_lines
    assert [row.get("is_context", False) for row in result["matches"]] == [True, False, True]
    assert result.get("truncated", False) is (offset == 0)
