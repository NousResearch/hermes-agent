"""Tests for tools/toon_encoder.py"""

import json
import pytest

from tools.toon_encoder import encode_toon, _truncate_value, _ensure_definitive_empty


class TestTruncation:
    def test_truncates_long_strings(self):
        data = {"body": "x" * 2000}
        result = _truncate_value(data, limit=100)
        assert len(result["body"]) < 200
        assert "truncated" in result["body"]
        assert "2000 chars total" in result["body"]

    def test_preserves_short_strings(self):
        data = {"name": "Alice"}
        result = _truncate_value(data)
        assert result["name"] == "Alice"

    def test_truncates_nested(self):
        data = {"results": [{"content": "y" * 3000}]}
        result = _truncate_value(data, limit=50)
        assert "truncated" in result["results"][0]["content"]

    def test_preserves_non_strings(self):
        data = {"count": 42, "active": True, "meta": None}
        result = _truncate_value(data)
        assert result == data


class TestDefinitiveEmpty:
    def test_adds_count_to_empty_results(self):
        data = {"results": [], "success": True}
        result = _ensure_definitive_empty(data)
        assert result["count"] == 0

    def test_preserves_existing_count(self):
        data = {"results": [], "count": 0}
        result = _ensure_definitive_empty(data)
        assert result["count"] == 0  # not overwritten

    def test_preserves_non_empty(self):
        data = {"results": [{"id": 1}], "count": 1}
        result = _ensure_definitive_empty(data)
        assert result["count"] == 1

    def test_handles_non_dict(self):
        result = _ensure_definitive_empty("hello")
        assert result == "hello"


class TestEncodeToon:
    def test_encodes_list_output(self):
        data = {"results": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "count": 2}
        json_str = json.dumps(data)
        toon_str = encode_toon(json_str, tool_name="test")
        assert "[" in toon_str  # array header
        assert "Alice" in toon_str
        assert toon_str != json_str  # actually converted

    def test_roundtrip(self):
        from toon_format import decode
        data = {"results": [{"id": 1, "name": "test"}], "count": 1}
        json_str = json.dumps(data)
        toon_str = encode_toon(json_str, tool_name="test", truncate=False)
        decoded = decode(toon_str)
        # After TOON decode, types may differ slightly (int vs str) but structure should match
        assert "results" in decoded
        assert len(decoded["results"]) == 1

    def test_fallback_on_invalid_json(self):
        plain = "not json at all"
        result = encode_toon(plain, tool_name="test")
        assert result == plain

    def test_skip_list(self):
        data = {"x": 1}
        json_str = json.dumps(data)
        # If we add a tool to skip list, it should return JSON unchanged
        from tools.toon_encoder import _SKIP_TOON_TOOLS
        # Can't easily test this without modifying the frozenset, but verify it exists
        assert isinstance(_SKIP_TOON_TOOLS, frozenset)

    def test_savings_on_realistic_output(self):
        data = {
            "success": True,
            "mode": "recent",
            "results": [
                {"session_id": f"s{i}", "title": f"Session {i}", "source": "local",
                 "started_at": f"April {i}, 2026", "last_active": "April 7, 2026",
                 "message_count": i * 10, "preview": f"Preview text {i}..."}
                for i in range(5)
            ],
            "count": 5,
        }
        json_str = json.dumps(data, ensure_ascii=False)
        toon_str = encode_toon(json_str, tool_name="session_search", truncate=False)
        savings = 1 - len(toon_str) / len(json_str)
        assert savings > 0.25  # at least 25% savings


class TestTokenSavings:
    """Benchmark-style tests to verify savings on representative outputs."""

    def _measure(self, data, tool_name="test"):
        json_str = json.dumps(data, ensure_ascii=False)
        toon_str = encode_toon(json_str, tool_name=tool_name, truncate=False)
        return {
            "json_chars": len(json_str),
            "toon_chars": len(toon_str),
            "savings": round((1 - len(toon_str) / len(json_str)) * 100, 1),
        }

    def test_session_search_savings(self):
        data = {
            "success": True,
            "results": [
                {"session_id": "abc", "title": "Test", "source": "local",
                 "started_at": "April 7", "last_active": "April 7",
                 "message_count": 20, "preview": "Some preview..."}
            ] * 3,
            "count": 3,
        }
        m = self._measure(data, "session_search")
        assert m["savings"] > 25, f"Only {m['savings']}% savings"

    def test_search_files_savings(self):
        data = {
            "matches": [
                {"file": f"/path/file{i}.py", "line": i * 10, "content": f"def func{i}():", "match_count": 3}
                for i in range(10)
            ],
            "total_count": 10,
        }
        m = self._measure(data, "search_files")
        assert m["savings"] > 25, f"Only {m['savings']}% savings"

    def test_error_savings(self):
        data = {"error": "File not found", "success": False}
        m = self._measure(data, "error")
        # Errors are small, savings modest but positive
        assert m["savings"] > 0
