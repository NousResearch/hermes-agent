"""Tests for the past-EOF and empty-file notes in read_file.

An empty content string is ambiguous from inside the model (empty file?
bad offset? broken tool?) — the tool names the dead end and the recovery.
"""

import json

from tools.file_tools import read_file_tool


class TestPastEofNote:
    def test_offset_beyond_eof_names_recovery(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        p = tmp_path / "r.txt"
        p.write_text("\n".join(f"l{i}" for i in range(1, 51)) + "\n")
        result = json.loads(read_file_tool(str(p), offset=900, limit=50))
        hint = result.get("hint") or ""
        assert "beyond the end" in hint
        assert "50" in hint  # states actual line count
        assert not result.get("error"), "a fact about the file is not an error"

    def test_offset_at_last_line_still_reads(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        p = tmp_path / "r.txt"
        p.write_text("a\nb\nc\n")
        result = json.loads(read_file_tool(str(p), offset=3, limit=10))
        assert "c" in result.get("content", "")
        assert "beyond" not in (result.get("hint") or "")

    def test_empty_file_says_so(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        p = tmp_path / "e.txt"
        p.write_text("")
        result = json.loads(read_file_tool(str(p)))
        assert "empty" in (result.get("hint") or "").lower()
        assert not result.get("error")

    def test_normal_pagination_hint_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        p = tmp_path / "r.txt"
        p.write_text("\n".join(f"l{i}" for i in range(1, 300)) + "\n")
        result = json.loads(read_file_tool(str(p), offset=1, limit=100))
        assert "offset=101" in (result.get("hint") or "")
