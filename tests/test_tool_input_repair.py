"""Tests for the tool-input repair layer (model_tools.repair_tool_args)."""

import pytest
from model_tools import (
    repair_tool_args,
    _relational_repair,
    _record_repairs,
    get_repair_stats,
    _relational_note,
    coerce_tool_args,
)


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _ensure_tools_discovered():
    """Make sure the registry is populated before any test runs."""
    from model_tools import discover_builtin_tools
    discover_builtin_tools()


@pytest.fixture(autouse=True)
def _clear_repair_stats():
    """Start each test with a clean stats slate."""
    from model_tools import _repair_stats
    _repair_stats.clear()


# ── Test: strip null ─────────────────────────────────────────────

def test_strip_null_non_nullable_optional():
    """Null is stripped from optional fields that don't allow null."""
    args = {"pattern": "foo", "target": None}
    fixed, repairs = repair_tool_args("search_files", dict(args))
    assert "target" not in fixed
    assert fixed["pattern"] == "foo"
    assert any("target:strip_null" in r for r in repairs)


def test_strip_null_from_boolean():
    """Null is stripped from optional boolean fields."""
    args = {"command": "ls", "background": None}
    fixed, repairs = repair_tool_args("terminal", dict(args))
    assert "background" not in fixed
    assert fixed["command"] == "ls"


def test_strip_null_required_field_preserved():
    """Null is NOT stripped from required fields (model intent unknown)."""
    args = {"path": None}
    fixed, repairs = repair_tool_args("read_file", dict(args))
    # Required fields with None are left alone — caller should catch this
    assert "path" in fixed


def test_strip_null_unknown_tool():
    """Unknown tool returns args unchanged."""
    args = {"x": None}
    fixed, repairs = repair_tool_args("nonexistent_tool", dict(args))
    assert args == fixed
    assert repairs == []


# ── Test: autolink unwrap ────────────────────────────────────────

def test_autolink_unwrap_degenerate():
    """Degenerate autolink [text](http://text) → text."""
    fixed, repairs = repair_tool_args("write_file",
                                      {"path": "[notes.md](http://notes.md)", "content": "hi"})
    assert fixed["path"] == "notes.md"
    assert fixed["content"] == "hi"
    assert any("path:autolink_unwrap" in r for r in repairs)


def test_autolink_preserves_real_markdown():
    """Real markdown links are preserved."""
    fixed, repairs = repair_tool_args("write_file",
                                      {"path": "[click here](https://x.com)", "content": "..."})
    assert fixed["path"] == "[click here](https://x.com)"
    assert "path:autolink_unwrap" not in repairs


def test_autolink_with_extension():
    """Autolink with .md extension is unwrapped."""
    fixed, repairs = repair_tool_args("write_file",
                                      {"path": "[readme.md](http://readme.md)", "content": "..."})
    assert fixed["path"] == "readme.md"


def test_autolink_on_read_file_path():
    """Autolink works on read_file path fields."""
    fixed, repairs = repair_tool_args("read_file",
                                      {"path": "[notes.md](http://notes.md)"})
    assert fixed["path"] == "notes.md"


def test_autolink_no_false_positive_on_normal_paths():
    """Normal file paths are untouched."""
    fixed, repairs = repair_tool_args("write_file",
                                      {"path": "/tmp/test.txt", "content": "hi"})
    assert fixed["path"] == "/tmp/test.txt"
    assert repairs == []


# ── Test: valid inputs untouched ─────────────────────────────────

def test_valid_inputs_untouched_read_file():
    """Correct read_file args produce zero repairs."""
    _, repairs = repair_tool_args("read_file",
                                  {"path": "/etc/hosts", "offset": 1, "limit": 500})
    assert repairs == []


def test_valid_inputs_untouched_web_search():
    _, repairs = repair_tool_args("web_search",
                                  {"query": "python asyncio", "limit": 3})
    assert repairs == []


def test_valid_inputs_untouched_terminal():
    _, repairs = repair_tool_args("terminal", {"command": "ls -la"})
    assert repairs == []


def test_valid_inputs_untouched_web_extract():
    _, repairs = repair_tool_args("web_extract",
                                  {"urls": ["https://a.com", "https://b.com"]})
    assert repairs == []


def test_valid_inputs_untouched_write_file():
    _, repairs = repair_tool_args("write_file",
                                  {"path": "/tmp/test.txt", "content": "hello world"})
    assert repairs == []


# ── Test: bare value wrapping (via coerce_tool_args, already upstream) ──

def test_coerce_bare_string_to_array():
    """Bare string for array field is wrapped into a list."""
    fixed = coerce_tool_args("web_extract", {"urls": "https://single-url.com"})
    assert fixed["urls"] == ["https://single-url.com"]


def test_coerce_json_string_to_array():
    """JSON-encoded array string is parsed, not double-wrapped."""
    fixed = coerce_tool_args("web_extract",
                             {"urls": '["https://a.com", "https://b.com"]'})
    assert fixed["urls"] == ["https://a.com", "https://b.com"]


def test_coerce_singleton_dict_to_array():
    """Single dict is wrapped when schema expects array."""
    fixed = coerce_tool_args("todo",
                             {"todos": {"id": "1", "content": "test", "status": "pending"}})
    assert fixed["todos"] == [{"id": "1", "content": "test", "status": "pending"}]


# ── Test: relational repairs ─────────────────────────────────────

def test_relational_limit_only_defaults_offset():
    """Providing only limit defaults offset to 1."""
    args = {"path": "/etc/hosts", "limit": 30}
    fixed, repairs = _relational_repair("read_file", dict(args))
    assert fixed["offset"] == 1
    assert fixed["limit"] == 30
    assert "offset:defaulted_to_1" in repairs


def test_relational_offset_only_defaults_limit():
    """Providing only offset defaults limit to 500."""
    args = {"path": "/etc/hosts", "offset": 50}
    fixed, repairs = _relational_repair("read_file", dict(args))
    assert fixed["limit"] == 500
    assert fixed["offset"] == 50
    assert "limit:defaulted_to_500" in repairs


def test_relational_both_provided_no_defaults():
    """When both offset and limit are present, nothing changes."""
    args = {"path": "/etc/hosts", "offset": 10, "limit": 20}
    fixed, repairs = _relational_repair("read_file", dict(args))
    assert "offset" not in repairs
    assert "limit" not in repairs


def test_relational_neither_provided_no_defaults():
    """When neither is present, nothing changes (tool defaults apply later)."""
    args = {"path": "/etc/hosts"}
    fixed, repairs = _relational_repair("read_file", dict(args))
    assert "offset" not in fixed
    assert "limit" not in fixed
    assert repairs == []


def test_relational_skips_non_read_file():
    """Relational repair is a no-op for non-read_file tools."""
    args = {"command": "ls", "timeout": 30}
    fixed, repairs = _relational_repair("terminal", dict(args))
    assert fixed == args
    assert repairs == []


# ── Test: relational notes ───────────────────────────────────────

def test_relational_note_offset():
    note = _relational_note(["offset:defaulted_to_1"])
    assert note is not None
    assert "offset" in note.lower()
    assert "limit" in note.lower()


def test_relational_note_limit():
    note = _relational_note(["limit:defaulted_to_500"])
    assert note is not None
    assert "limit" in note.lower()
    assert "offset" in note.lower()


def test_relational_note_none_when_no_relational_repairs():
    assert _relational_note(["urls:strip_null"]) is None
    assert _relational_note([]) is None


# ── Test: telemetry ──────────────────────────────────────────────

def test_record_repairs_increments():
    _record_repairs("read_file", ["offset:defaulted_to_1", "limit:defaulted_to_500"])
    _record_repairs("read_file", ["offset:defaulted_to_1"])
    stats = get_repair_stats()
    assert stats["read_file:offset:defaulted_to_1"] == 2
    assert stats["read_file:limit:defaulted_to_500"] == 1


def test_record_repairs_separate_tools():
    _record_repairs("terminal", ["command:strip_null"])
    _record_repairs("search_files", ["target:strip_null"])
    stats = get_repair_stats()
    assert "terminal:command:strip_null" in stats
    assert "search_files:target:strip_null" in stats


def test_get_repair_stats_returns_copy():
    _record_repairs("patch", ["path:strip_null"])
    s1 = get_repair_stats()
    s1["extra"] = 999  # Mutate the returned dict
    s2 = get_repair_stats()
    assert "extra" not in s2


# ── Test: edge cases ─────────────────────────────────────────────

def test_empty_args_dict():
    _, repairs = repair_tool_args("read_file", {})
    assert repairs == []


def test_unknown_field_passes_through():
    args = {"path": "/etc/hosts", "unknown_field": "blah"}
    fixed, repairs = repair_tool_args("read_file", dict(args))
    assert fixed["unknown_field"] == "blah"
    assert repairs == []


def test_repair_order_strip_null_before_autolink():
    """Null is stripped before autolink processing."""
    args = {"path": "[notes.md](http://notes.md)", "target": None}
    fixed, repairs = repair_tool_args("search_files", dict(args))
    assert "target" not in fixed
    # Autolink still fires on the path field
    assert any("path:autolink_unwrap" in r for r in repairs)
    assert any("target:strip_null" in r for r in repairs)
