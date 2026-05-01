"""Tests for tools/line_hash.py — hashline utilities."""

import pytest
from tools.line_hash import (
    compute_line_hash,
    format_hash_lines,
    strip_hash_prefix,
    parse_anchor,
    parse_anchor_range,
    build_anchor_map,
    suggest_similar_anchors,
    AnchorInfo,
)


# =========================================================================
# compute_line_hash
# =========================================================================

class TestComputeLineHash:
    def test_deterministic(self):
        assert compute_line_hash("hello") == compute_line_hash("hello")

    def test_different_inputs_differ(self):
        assert compute_line_hash("abc") != compute_line_hash("xyz")

    def test_length_is_four(self):
        assert len(compute_line_hash("anything")) == 4

    def test_only_base36_chars(self):
        valid = set("0123456789abcdefghijklmnopqrstuvwxyz")
        for ch in compute_line_hash("test"):
            assert ch in valid

    def test_empty_string(self):
        h = compute_line_hash("")
        assert len(h) == 4

    def test_unicode(self):
        h = compute_line_hash("你好世界")
        assert len(h) == 4

    def test_whitespace_sensitive(self):
        assert compute_line_hash("  x") != compute_line_hash(" x")


# =========================================================================
# format_hash_lines
# =========================================================================

class TestFormatHashLines:
    def test_basic_format(self):
        lines = ["aaa", "bbb"]
        out = format_hash_lines(lines)
        rows = out.split("\n")
        assert len(rows) == 2
        assert rows[0].endswith("|aaa")
        assert rows[1].endswith("|bbb")

    def test_offset(self):
        lines = ["x"]
        out = format_hash_lines(lines, offset=9)
        assert out.startswith("10:")

    def test_collision_disambiguation(self):
        lines = ["dup", "dup"]
        out = format_hash_lines(lines)
        rows = out.split("\n")
        assert "#1" in rows[0]
        assert "#2" in rows[1]

    def test_no_suffix_for_unique_hashes(self):
        lines = ["aaa", "bbb", "ccc"]
        out = format_hash_lines(lines)
        for row in out.split("\n"):
            assert "#" not in row.split("|")[0]


# =========================================================================
# strip_hash_prefix
# =========================================================================

class TestStripHashPrefix:
    def test_strips_prefix(self):
        assert strip_hash_prefix("1:k7m2|hello()") == "hello()"

    def test_no_pipe_returns_original(self):
        assert strip_hash_prefix("no pipe here") == "no pipe here"

    def test_preserves_pipe_in_content(self):
        assert strip_hash_prefix("1:abc|a|b") == "a|b"


# =========================================================================
# parse_anchor
# =========================================================================

class TestParseAnchor:
    def test_simple_anchor(self):
        info = parse_anchor("k7m2")
        assert info is not None
        assert info.hash == "k7m2"
        assert info.disambig == 0

    def test_disambiguated_anchor(self):
        info = parse_anchor("k7m2#1")
        assert info is not None
        assert info.hash == "k7m2"
        assert info.disambig == 1

    def test_invalid_too_short(self):
        assert parse_anchor("ab") is None

    def test_invalid_uppercase(self):
        assert parse_anchor("K7M2") is None

    def test_invalid_disambig_zero(self):
        assert parse_anchor("k7m2#0") is None


# =========================================================================
# parse_anchor_range
# =========================================================================

class TestParseAnchorRange:
    def test_range(self):
        assert parse_anchor_range("k7m2:a9f1") == ("k7m2", "a9f1")

    def test_single_anchor(self):
        assert parse_anchor_range("k7m2") == ("k7m2", "k7m2")

    def test_empty_string(self):
        assert parse_anchor_range("") is None

    def test_empty_part(self):
        assert parse_anchor_range("k7m2:") is None


# =========================================================================
# build_anchor_map
# =========================================================================

class TestBuildAnchorMap:
    def test_unique_lines(self):
        lines = ["aaa", "bbb", "ccc"]
        amap = build_anchor_map(lines)
        assert len(amap) == 3
        assert set(amap.values()) == {1, 2, 3}

    def test_duplicate_lines_disambiguated(self):
        lines = ["dup", "dup"]
        amap = build_anchor_map(lines)
        assert len(amap) == 2
        tags = sorted(amap.keys())
        assert "#1" in tags[0]
        assert "#2" in tags[1]

    def test_offset(self):
        lines = ["x"]
        amap = build_anchor_map(lines, offset=100)
        assert list(amap.values()) == [101]


# =========================================================================
# suggest_similar_anchors
# =========================================================================

class TestSuggestSimilarAnchors:
    def test_returns_suggestions(self):
        lines = ["def hello():", "def world():"]
        amap = build_anchor_map(lines)
        h = compute_line_hash("def hello():")
        fake = h[:3] + ("z" if h[3] != "z" else "a")
        suggestions = suggest_similar_anchors(fake, amap, lines)
        assert len(suggestions) > 0

    def test_no_suggestions_for_gibberish(self):
        lines = ["aaa"]
        amap = build_anchor_map(lines)
        suggestions = suggest_similar_anchors("zzzz", amap, lines)
        assert suggestions == []
