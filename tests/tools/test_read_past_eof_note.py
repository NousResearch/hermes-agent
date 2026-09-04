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

    def test_no_trailing_newline_last_line_still_reads(self, tmp_path, monkeypatch):
        # wc -l counts newlines, so "a\nb\nc" reports 2 lines. Asking for
        # line 3 looks past EOF to that count, but the line exists and the
        # read must return it instead of claiming the file ends at line 2.
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        p = tmp_path / "r.txt"
        p.write_text("a\nb\nc")
        result = json.loads(read_file_tool(str(p), offset=3, limit=10))
        assert "c" in result.get("content", "")
        assert "beyond" not in (result.get("hint") or "")

    def test_unparseable_size_probe_not_reported_empty(self, tmp_path, monkeypatch):
        # file_size falls back to 0 when the wc probe output is unparseable
        # (transport junk with exit 0). A non-empty file must not be
        # reported as "File is empty" in that case: the read already
        # produced its content.
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        p = tmp_path / "r.txt"
        p.write_text("real content\n")
        from tools import file_tools
        ops = file_tools._get_file_ops("default")
        real_exec = ops._exec

        def noisy_exec(command, **kwargs):
            res = real_exec(command, **kwargs)
            if "wc -c <" in command:
                res.stdout = "shell profile junk\r\n" + res.stdout
            return res

        monkeypatch.setattr(ops, "_exec", noisy_exec)
        result = json.loads(read_file_tool(str(p)))
        assert "real content" in result.get("content", "")
        assert "empty" not in (result.get("hint") or "").lower()
