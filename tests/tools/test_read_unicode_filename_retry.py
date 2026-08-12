"""Tests for unicode-equivalent filename retry + near-miss suggestions.

NFC/NFD, narrow no-break space, and curly quotes render identically in a
terminal — a model that retypes a visually-correct path can never discover
the byte mismatch. The repair is the tool's job (single unambiguous match
only). Visible differences stay with did-you-mean suggestions.
"""

import json
import unicodedata

import pytest

from tools.file_tools import read_file_tool

HOSTILE = unicodedata.normalize(
    "NFD", "Meeting\u202fnotes\u2019 re\u0301sume\u0301 3.04\u202fPM.txt"
)
CLEAN = "Meeting notes\u2019 r\u00e9sum\u00e9 3.04 PM.txt"  # NFC + plain spaces


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / HOSTILE).write_text("- rotate the keys\n")
    (tmp_path / "AGENTS.md").write_text("npm run build:prod\n")
    return tmp_path


class TestUnicodeVariantRepair:
    def test_nfc_plain_space_spelling_repairs(self, ws):
        result = json.loads(read_file_tool(str(ws / "notes" / CLEAN)))
        assert "rotate the keys" in result.get("content", "")
        assert "unicode-equivalent" in result.get("hint", "")

    def test_exact_spelling_no_note(self, ws):
        result = json.loads(read_file_tool(str(ws / "notes" / HOSTILE)))
        assert "rotate the keys" in result.get("content", "")
        assert "unicode-equivalent" not in (result.get("hint") or "")

    def test_visible_difference_not_repaired(self, ws):
        # Straight quote + accent-less = visibly different: suggest, don't repair
        result = json.loads(
            read_file_tool(str(ws / "notes" / "Meeting notes' resume 3.04 PM.txt"))
        )
        assert result.get("error"), "visible diff must stay a not-found"
        assert "unicode-equivalent" not in (result.get("hint") or "")
        assert result.get("similar_files")

    def test_ambiguous_twins_not_repaired(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        (tmp_path / "caf\u00e9.txt").write_text("nfc\n")
        (tmp_path / "cafe\u0301.txt").write_text("nfd\n")
        # Both canonicalize to café.txt; a third spelling must not guess.
        result = json.loads(read_file_tool(str(tmp_path / "CAFE.txt")))
        assert "unicode-equivalent" not in (result.get("hint") or "")

    def test_plain_missing_file_unchanged(self, ws):
        result = json.loads(read_file_tool(str(ws / "missing.txt")))
        assert "not found" in result.get("error", "").lower()


class TestNearMissSuggestion:
    def test_agent_md_suggests_agents_md(self, ws):
        result = json.loads(read_file_tool(str(ws / "AGENT.md")))
        sims = result.get("similar_files") or []
        assert any("AGENTS.md" in s for s in sims), sims

    def test_unrelated_name_no_suggestion_of_agents(self, ws):
        result = json.loads(read_file_tool(str(ws / "zzz_qqq.bin")))
        sims = result.get("similar_files") or []
        assert not any("AGENTS.md" in s for s in sims)
