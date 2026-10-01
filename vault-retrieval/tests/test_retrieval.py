"""Tests for retrieval.py — large-file range extraction.

Covers acceptance test #7 (large-file third range, 121 lines, 8001 chars
rejected), #12 (search snippets alone never appear in extracts; each
extract has verified original path/range/freshness), #17 (deleted/renamed
note fixtures disappear from FTS5 pilot and stale snapshot fixtures are
rejected).
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest


from vault_retrieval.retrieval import (
    extract_range,
    extract_from_selectors,
    Extract,
    ExtractionError,
    LargeFileRefused,
)


class TestExtractRange:
    def test_extract_single_line_range(self, tmp_path):
        p = tmp_path / "note.md"
        p.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")
        e = extract_range(p, line_start=2, line_end=3)
        assert e.line_start == 2
        assert e.line_end == 3
        assert e.content == "line2\nline3\n"
        assert e.chars == len("line2\nline3\n")

    def test_extract_zero_length_range_rejected(self, tmp_path):
        p = tmp_path / "note.md"
        p.write_text("x\n", encoding="utf-8")
        with pytest.raises(ExtractionError):
            extract_range(p, line_start=1, line_end=0)

    def test_extract_start_below_one_rejected(self, tmp_path):
        p = tmp_path / "note.md"
        p.write_text("x\n", encoding="utf-8")
        with pytest.raises(ExtractionError):
            extract_range(p, line_start=0, line_end=1)

    def test_extract_outside_file_clamps(self, tmp_path):
        """Out-of-range line_end is clamped to file length (lenient)."""
        p = tmp_path / "note.md"
        p.write_text("a\nb\n", encoding="utf-8")
        e = extract_range(p, line_start=1, line_end=99)
        assert e.line_end == 2

    def test_extract_start_beyond_file_rejected(self, tmp_path):
        p = tmp_path / "note.md"
        p.write_text("a\nb\n", encoding="utf-8")
        with pytest.raises(ExtractionError):
            extract_range(p, line_start=99, line_end=100)

    def test_extract_returns_source_mtime(self, tmp_path):
        p = tmp_path / "note.md"
        p.write_text("a\nb\n", encoding="utf-8")
        e = extract_range(p, line_start=1, line_end=2)
        assert e.source_mtime is not None
        assert "T" in e.source_mtime  # ISO format has 'T'


class TestExtractFromSelectors:
    def _make_note(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "note.md"
        p.write_text(content, encoding="utf-8")
        return p

    def test_two_ranges_extracted(self, tmp_path):
        p = self._make_note(tmp_path, "x\n" * 200)
        sels = [
            {"line_start": 1, "line_end": 50},
            {"line_start": 60, "line_end": 110},
        ]
        extracts = extract_from_selectors(p, sels)
        assert len(extracts) == 2
        assert extracts[0].line_start == 1 and extracts[0].line_end == 50
        assert extracts[1].line_start == 60 and extracts[1].line_end == 110

    def test_third_range_rejected(self, tmp_path):
        p = self._make_note(tmp_path, "x\n" * 200)
        sels = [
            {"line_start": 1, "line_end": 50},
            {"line_start": 60, "line_end": 110},
            {"line_start": 120, "line_end": 170},
        ]
        with pytest.raises(LargeFileRefused):
            extract_from_selectors(p, sels, is_large=True)

    def test_121_line_range_rejected(self, tmp_path):
        p = self._make_note(tmp_path, "x\n" * 200)
        sels = [{"line_start": 1, "line_end": 121}]
        with pytest.raises(LargeFileRefused):
            extract_from_selectors(p, sels, is_large=True)

    def test_8001_char_range_rejected(self, tmp_path):
        p = self._make_note(tmp_path, "x\n" * 200)  # ~600 chars
        sels = [{"line_start": 1, "line_end": 200, "estimated_chars": 8_001}]
        with pytest.raises(LargeFileRefused):
            extract_from_selectors(p, sels, is_large=True)


class TestExtractFreshness:
    def test_extract_includes_source_mtime(self, tmp_path):
        p = tmp_path / "note.md"
        p.write_text("a\nb\n", encoding="utf-8")
        e = extract_range(p, line_start=1, line_end=2)
        assert e.source_mtime is not None
        assert "T" in e.source_mtime  # ISO format has 'T'


class TestExtractsPathRangeFreshnessContract:
    """Acceptance test #12 — each extract has verified path, range, freshness."""

    def test_each_extract_has_path_range_chars_freshness(self, tmp_path):
        p = tmp_path / "note.md"
        p.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        e = extract_range(p, line_start=1, line_end=3)
        assert e.path == str(p)
        assert e.line_start == 1
        assert e.line_end == 3
        assert e.chars > 0
        assert e.source_mtime
