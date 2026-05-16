"""Tests for tools/memory_format.py — CLI and markdown platform formatters."""

import pytest

from tools.memory_format import (
    format_memory_cli,
    format_memory_markdown,
)


def _readout(mem_entries=None, usr_entries=None, mem_pct=0, usr_pct=0):
    return {
        "memory": {
            "entries": mem_entries or [],
            "char_count": 100,
            "char_limit": 1000,
            "pct": mem_pct,
        },
        "user": {
            "entries": usr_entries or [],
            "char_count": 50,
            "char_limit": 500,
            "pct": usr_pct,
        },
    }


class TestFormatMemoryCli:
    def test_all_target_renders_both(self):
        out = format_memory_cli(_readout(["a"], ["b"]), "all")
        assert "MEMORY.md" in out
        assert "USER.md" in out
        assert "1. a" in out
        assert "1. b" in out

    def test_memory_target_only(self):
        out = format_memory_cli(_readout(["zebra"], ["yak"]), "memory")
        assert "MEMORY.md" in out
        assert "USER.md" not in out
        assert "zebra" in out
        assert "yak" not in out

    def test_user_target_only(self):
        out = format_memory_cli(_readout(["a"], ["b"]), "user")
        assert "USER.md" in out
        assert "MEMORY.md" not in out
        assert "1. b" in out

    def test_empty_renders_placeholder(self):
        out = format_memory_cli(_readout(), "all")
        assert "(empty)" in out

    def test_warns_above_threshold(self):
        out = format_memory_cli(_readout(["a"], mem_pct=95), "memory")
        assert "close to cap" in out

    def test_no_warn_below_threshold(self):
        out = format_memory_cli(_readout(["a"], mem_pct=50), "memory")
        assert "close to cap" not in out

    def test_truncates_long_entries(self):
        long_entry = "x" * 500
        out = format_memory_cli(_readout([long_entry]), "memory")
        assert "…" in out
        assert "x" * 500 not in out

    def test_uses_rich_markup(self):
        out = format_memory_cli(_readout(["a"]), "memory")
        assert "[bold]" in out
        assert "[bold cyan]" in out


class TestFormatMemoryMarkdown:
    def test_all_target(self):
        out = format_memory_markdown(_readout(["a"], ["b"]), "all")
        assert "**MEMORY.md**" in out
        assert "**USER.md**" in out

    def test_memory_target_only(self):
        out = format_memory_markdown(_readout(["zebra"], ["yak"]), "memory")
        assert "**MEMORY.md**" in out
        assert "USER.md" not in out
        assert "yak" not in out

    def test_empty_uses_italic_placeholder(self):
        out = format_memory_markdown(_readout(), "all")
        assert "_(empty)_" in out

    def test_warn_emoji(self):
        out = format_memory_markdown(_readout(["a"], mem_pct=90), "memory")
        assert "⚠️" in out

    def test_no_rich_markup(self):
        out = format_memory_markdown(_readout(["a"]), "memory")
        assert "[bold]" not in out
