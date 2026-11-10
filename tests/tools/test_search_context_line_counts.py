"""search_files counts and paginates real matches; context=N lines ride along, marked."""

import json
from pathlib import Path

import pytest

from tools.file_tools import search_tool


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    d = tmp_path / "proj"
    d.mkdir()
    # A hit on every fourth line, so context=1 groups never touch.
    (d / "many.txt").write_text("".join("NEEDLE\nfiller\nfiller\nfiller\n" for _ in range(100)), encoding="utf-8")
    (d / "one.md").write_text("line one\nline two\nCodeword: HERON-41.\nline four\nline five\n", encoding="utf-8")
    return d


def test_context_lines_are_marked_and_not_counted(proj):
    r = json.loads(search_tool("HERON", path=str(proj), context=1, task_id="t-ctx"))
    assert r["total_count"] == 1
    assert [(m["line"], m.get("is_context", False)) for m in r["matches"]] == [(2, True), (3, False), (4, True)]


def test_limit_and_offset_count_matches_and_carry_their_context(proj):
    r = json.loads(search_tool("NEEDLE", path=str(proj / "many.txt"), context=1, limit=1, offset=2, task_id="t-ctx"))
    # Third hit is line 9; one match per page however many context rows surround it.
    assert [(m["line"], m.get("is_context", False)) for m in r["matches"]] == [(8, True), (9, False), (10, True)]


def test_truncation_survives_large_context_on_many_matches(proj):
    # Each hit spends 2*context+2 output rows, so a row-counted fetch would run out before the page ends.
    r = json.loads(search_tool("NEEDLE", path=str(proj / "many.txt"), context=30, limit=40, task_id="t-ctx"))
    assert r["truncated"] is True
    assert r["total_count"] > 40
    assert "matches_text" in r
    rows = [line.strip() for line in r["matches_text"].splitlines()[1:]]
    assert sum(": " in row for row in rows) == 40


def test_dense_block_marks_context_rows(proj):
    r = json.loads(search_tool("NEEDLE", path=str(proj / "many.txt"), context=1, limit=3, task_id="t-ctx"))
    assert r["matches_text"].splitlines()[1:5] == ["  1: NEEDLE", "  2- filler", "  4- filler", "  5: NEEDLE"]
    assert "context lines" in r["matches_format"]


def test_context_free_search_keeps_its_shape(proj):
    r = json.loads(search_tool("HERON", path=str(proj), task_id="t-ctx"))
    assert r["total_count"] == 1
    assert len(r["matches"]) == 1
    row = r["matches"][0]
    assert Path(row["path"]) == proj / "one.md"
    assert {key: value for key, value in row.items() if key != "path"} == {
        "line": 3, "content": "Codeword: HERON-41."}
    assert "context lines" not in json.dumps(r)


def test_multi_root_limit_counts_matches(proj, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "two.md").write_text("before\nHERON again\nafter\n", encoding="utf-8")
    r = json.loads(search_tool("HERON", path=f"{proj / 'one.md'},{other / 'two.md'}", context=1, limit=2, task_id="t-ctx"))
    assert r["total_count"] == 2
    rows = r["matches_text"].splitlines()
    assert sum(": " in row and row.startswith("  ") for row in rows) == 2
    assert sum("- " in row and row.startswith("  ") for row in rows) == 4
