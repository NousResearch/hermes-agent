"""Literal no-match feedback preserves exact string semantics, including backslashes."""

import json

import pytest

import tools.file_tools  # Register the real file handlers.
from tools.registry import registry


@pytest.mark.parametrize("pattern,count", [
    (r"item\[tag\]", 0),
    ("item[tag]", 1),
    (r"store\folder", 1),
])
def test_literal_search_preserves_input_and_guides_only_on_no_matches(tmp_path, pattern, count):
    (tmp_path / "notes.txt").write_text("item[tag]\nstore\\folder\n")
    result = json.loads(registry.dispatch(
        "search_files", {"pattern": pattern, "path": str(tmp_path), "literal": True},
        task_id="literal-guidance"))
    assert result["total_count"] == count
    if count == 0:
        assert result["_hint"] == f"No matches for the exact text: {pattern}"
    else:
        assert "_hint" not in result
        assert pattern in result["matches"][0]["content"]


@pytest.mark.parametrize("target,literal,missing", [
    ("content", False, False),
    ("files", False, False),
    ("files", True, False),
    ("content", True, True),
])
def test_other_search_modes_and_errors_do_not_get_literal_guidance(tmp_path, target, literal, missing):
    (tmp_path / "notes.txt").write_text("ordinary text\n")
    root = tmp_path / "missing" if missing else tmp_path
    result = json.loads(registry.dispatch(
        "search_files", {"pattern": "absent", "path": str(root), "target": target, "literal": literal},
        task_id="literal-guidance-other"))
    assert result["total_count"] == 0
    assert "_hint" not in result
    assert bool(result.get("error")) == (missing or (target == "files" and literal))
