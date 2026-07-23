"""Tests for the memory tool's regex ``patch`` action (issue #35995).

``patch`` locates a span with a regex and rewrites only that span inside the
matching entry, tolerating the whitespace/wording drift that breaks the
exact-substring ``replace`` action.
"""

import json

import pytest

from tools.memory_tool import MemoryStore, memory_tool, MEMORY_SCHEMA


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    s = MemoryStore(memory_char_limit=500, user_char_limit=300)
    s.load_from_disk()
    return s


class TestPatchStore:
    def test_rewrites_matched_span_preserving_rest(self, store):
        store.add("memory", "Project uses Python 3.11 with FastAPI")
        result = store.patch("memory", r"Python 3\.11", "Python 3.12")
        assert result["success"] is True
        assert store.memory_entries == ["Project uses Python 3.12 with FastAPI"]

    def test_tolerates_whitespace_drift(self, store):
        # Exact-match replace would miss this because of the doubled spaces.
        store.add("memory", "VPS  rules:   no   inbound  ports")
        result = store.patch("memory", r"no\s+inbound\s+ports", "ports 80/443 only")
        assert result["success"] is True
        assert "ports 80/443 only" in store.memory_entries[0]

    def test_case_insensitive_match(self, store):
        store.add("memory", "User prefers DARK mode")
        result = store.patch("memory", r"dark mode", "light mode")
        assert result["success"] is True
        assert store.memory_entries == ["User prefers light mode"]

    def test_only_first_occurrence_replaced(self, store):
        store.add("memory", "todo todo todo")
        store.patch("memory", r"todo", "done")
        assert store.memory_entries == ["done todo todo"]

    def test_no_match_returns_fuzzy_candidates(self, store):
        store.add("memory", "Deploy target is the staging cluster")
        store.add("memory", "CI runs on GitHub Actions")
        result = store.patch("memory", r"production database url", "x")
        assert result["success"] is False
        assert "No entry matched" in result["error"]
        assert len(result["candidates"]) == 2
        # Candidates are ranked by descending confidence.
        confidences = [c["confidence"] for c in result["candidates"]]
        assert confidences == sorted(confidences, reverse=True)
        assert all("snippet" in c for c in result["candidates"])

    def test_invalid_regex_is_reported(self, store):
        store.add("memory", "anything")
        result = store.patch("memory", r"unbalanced(group", "x")
        assert result["success"] is False
        assert "Invalid regex" in result["error"]

    def test_empty_pattern_rejected(self, store):
        result = store.patch("memory", "   ", "x")
        assert result["success"] is False
        assert "pattern cannot be empty" in result["error"]

    def test_empty_replacement_rejected(self, store):
        store.add("memory", "keep me")
        result = store.patch("memory", r"keep", "   ")
        assert result["success"] is False
        assert "new_content cannot be empty" in result["error"]

    def test_replacement_text_is_literal_not_a_template(self, store):
        # A bare backslash group ref in the replacement must not raise or be
        # interpreted as a backreference — it lands verbatim.
        store.add("memory", "token = abc")
        result = store.patch("memory", r"abc", r"value_\1_end")
        assert result["success"] is True
        assert store.memory_entries == [r"token = value_\1_end"]

    def test_char_limit_blocks_oversized_patch(self, store):
        store.add("memory", "short note")
        big = "x" * 600  # exceeds the 500-char limit
        result = store.patch("memory", r"short note", big)
        assert result["success"] is False
        assert "would put memory" in result["error"]
        assert result["current_entries"] == ["short note"]

    def test_ambiguous_distinct_matches_require_tightening(self, store):
        store.add("memory", "alpha shared token")
        store.add("memory", "beta shared token")
        result = store.patch("memory", r"shared token", "rotated")
        assert result["success"] is False
        assert "matched multiple entries" in result["error"]
        assert len(result["matches"]) == 2

    def test_patch_succeeds_after_duplicates_collapse_on_load(self, store, tmp_path):
        # Duplicate entries on disk are collapsed by the dedup-on-reload path,
        # so a pattern that would otherwise match "multiple" entries resolves
        # to a single one and patches cleanly.
        (tmp_path / "MEMORY.md").write_text("dup line\n§\ndup line\n", encoding="utf-8")
        store.load_from_disk()
        result = store.patch("memory", r"dup line", "deduped")
        assert result["success"] is True
        assert store.memory_entries == ["deduped"]


class TestPatchDispatch:
    def test_memory_tool_patch_roundtrip(self, store):
        store.add("memory", "endpoint at /v1/old")
        out = memory_tool(action="patch", target="memory",
                          pattern=r"/v1/old", content="/v2/new", store=store)
        payload = json.loads(out)
        assert payload["success"] is True
        assert store.memory_entries == ["endpoint at /v2/new"]

    def test_patch_requires_pattern(self, store):
        out = memory_tool(action="patch", target="memory", content="x", store=store)
        payload = json.loads(out)
        assert payload["success"] is False
        assert "pattern is required" in payload["error"]

    def test_patch_requires_content(self, store):
        out = memory_tool(action="patch", target="memory", pattern=r"x", store=store)
        payload = json.loads(out)
        assert payload["success"] is False
        assert "content is required" in payload["error"]


class TestPatchSchema:
    def test_schema_advertises_patch(self):
        params = MEMORY_SCHEMA["parameters"]["properties"]
        assert "patch" in params["action"]["enum"]
        assert "pattern" in params
        assert "patch" in MEMORY_SCHEMA["description"]
