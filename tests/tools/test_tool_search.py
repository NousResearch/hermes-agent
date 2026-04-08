"""Tests for tool_search meta-tool and keyword matching."""

import json
import pytest

from tools.registry import ToolRegistry
from tools.tool_search import (
    _match_score,
    search_tools,
    get_tool_details,
    register_tool_search,
    _TOOLSET_NAME,
)


def _make_schema(name, desc=""):
    return {
        "name": name,
        "description": desc or f"A {name} tool",
        "parameters": {"type": "object", "properties": {}},
    }


def _dummy_handler(args, **kw):
    return json.dumps({"ok": True})


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch):
    """Replace the module-level singleton with a fresh registry per test."""
    fresh = ToolRegistry()
    import tools.tool_search as ts_mod
    monkeypatch.setattr(ts_mod, "registry", fresh)
    return fresh


# ── _match_score ─────────────────────────────────────────────────────

class TestMatchScore:
    def test_exact_name_match(self):
        assert _match_score("read_file", "read_file", "Read a file") == 1.0

    def test_query_substring_of_name(self):
        score = _match_score("read", "read_file", "Read a file from disk")
        assert score >= 0.7

    def test_name_substring_of_query(self):
        score = _match_score("read_file contents", "read_file", "")
        assert score >= 0.7

    def test_word_overlap_in_name(self):
        score = _match_score("search files", "search_files", "Search for files")
        assert score >= 0.8

    def test_word_overlap_in_description_only(self):
        score = _match_score("shell", "terminal", "Run a shell command")
        assert 0.3 < score < 0.8

    def test_no_match_returns_zero(self):
        assert _match_score("database", "web_search", "Search the web") == 0.0

    def test_case_insensitive(self):
        score = _match_score("READ_FILE", "read_file", "Read a file")
        assert score == 1.0

    def test_hyphen_splitting(self):
        score = _match_score("web search", "web-search", "Search the web")
        assert score >= 0.8


# ── search_tools ─────────────────────────────────────────────────────

class TestSearchTools:
    def test_empty_query_returns_error(self, _fresh_registry):
        result = json.loads(search_tools({"query": ""}))
        assert "error" in result
        assert result["matched_count"] == 0

    def test_finds_matching_tools(self, _fresh_registry):
        _fresh_registry.register(
            name="read_file", toolset="file", schema=_make_schema("read_file", "Read a file from disk"),
            handler=_dummy_handler, description="Read a file from disk",
        )
        _fresh_registry.register(
            name="write_file", toolset="file", schema=_make_schema("write_file", "Write to a file"),
            handler=_dummy_handler, description="Write to a file",
        )
        _fresh_registry.register(
            name="web_search", toolset="web", schema=_make_schema("web_search", "Search the internet"),
            handler=_dummy_handler, description="Search the internet",
        )

        result = json.loads(search_tools({"query": "file"}))
        assert result["matched_count"] >= 1
        names = {t["function"]["name"] for t in result["tools"]}
        assert "read_file" in names or "write_file" in names

    def test_returns_full_openai_schemas(self, _fresh_registry):
        _fresh_registry.register(
            name="terminal", toolset="core", schema=_make_schema("terminal", "Run shell commands"),
            handler=_dummy_handler, description="Run shell commands",
        )
        result = json.loads(search_tools({"query": "terminal"}))
        assert result["matched_count"] == 1
        schema = result["tools"][0]
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "parameters" in schema["function"]

    def test_excludes_meta_tools(self, _fresh_registry):
        register_tool_search()
        _fresh_registry.register(
            name="read_file", toolset="file", schema=_make_schema("read_file"),
            handler=_dummy_handler, description="Read a file",
        )
        result = json.loads(search_tools({"query": "tool search"}))
        names = {t["function"]["name"] for t in result["tools"]}
        assert "tool_search" not in names
        assert "tool_details" not in names

    def test_max_results_cap(self, _fresh_registry):
        for i in range(20):
            name = f"file_tool_{i}"
            _fresh_registry.register(
                name=name, toolset="file", schema=_make_schema(name, f"File operation {i}"),
                handler=_dummy_handler, description=f"File operation {i}",
            )
        result = json.loads(search_tools({"query": "file"}))
        assert result["matched_count"] <= 5

    def test_hint_present(self, _fresh_registry):
        _fresh_registry.register(
            name="test_tool", toolset="t", schema=_make_schema("test_tool"),
            handler=_dummy_handler, description="A test tool",
        )
        result = json.loads(search_tools({"query": "test"}))
        assert "hint" in result


# ── get_tool_details ─────────────────────────────────────────────────

class TestGetToolDetails:
    def test_empty_name_returns_error(self):
        result = json.loads(get_tool_details({"name": ""}))
        assert "error" in result

    def test_returns_schema_for_existing_tool(self, _fresh_registry):
        _fresh_registry.register(
            name="patch", toolset="file", schema=_make_schema("patch", "Apply a patch"),
            handler=_dummy_handler, description="Apply a patch",
        )
        result = json.loads(get_tool_details({"name": "patch"}))
        assert result["matched_count"] == 1
        assert result["tools"][0]["function"]["name"] == "patch"

    def test_returns_error_for_missing_tool(self):
        result = json.loads(get_tool_details({"name": "nonexistent_tool"}))
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_suggests_tool_search_on_miss(self):
        result = json.loads(get_tool_details({"name": "nope"}))
        assert "tool_search" in result["error"]


# ── register_tool_search ─────────────────────────────────────────────

class TestRegisterToolSearch:
    def test_registers_both_meta_tools(self, _fresh_registry):
        register_tool_search()
        assert _fresh_registry._tools.get("tool_search") is not None
        assert _fresh_registry._tools.get("tool_details") is not None

    def test_idempotent(self, _fresh_registry):
        register_tool_search()
        register_tool_search()
        assert _fresh_registry._tools.get("tool_search") is not None

    def test_meta_tools_in_correct_toolset(self, _fresh_registry):
        register_tool_search()
        assert _fresh_registry._tools["tool_search"].toolset == _TOOLSET_NAME
        assert _fresh_registry._tools["tool_details"].toolset == _TOOLSET_NAME
