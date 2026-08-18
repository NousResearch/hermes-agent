"""Tests for the stat-based special-file guard in read_file_tool.

The name blocklist (_is_blocked_device) catches /dev/* and /proc/* aliases;
_special_file_kind catches the CLASS — any FIFO/socket/device anywhere.
Without it, read_file on a workspace FIFO blocks until the exec timeout.
"""

import json
import os
import socket

import pytest

from tools.file_tools import _special_file_kind, read_file_tool


class TestSpecialFileKind:
    def test_regular_file(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("hi")
        assert _special_file_kind(p) is None

    def test_directory(self, tmp_path):
        assert _special_file_kind(tmp_path) is None

    def test_missing_path(self, tmp_path):
        assert _special_file_kind(tmp_path / "nope") is None

    def test_fifo(self, tmp_path):
        fifo = tmp_path / "p.pipe"
        os.mkfifo(fifo)
        assert "FIFO" in (_special_file_kind(fifo) or "")

    def test_socket(self, tmp_path):
        sock_path = tmp_path / "s.sock"
        s = socket.socket(socket.AF_UNIX)
        try:
            s.bind(str(sock_path))
            assert "socket" in (_special_file_kind(sock_path) or "")
        finally:
            s.close()

    def test_symlink_to_fifo_followed(self, tmp_path):
        fifo = tmp_path / "p.pipe"
        os.mkfifo(fifo)
        link = tmp_path / "innocent.txt"
        link.symlink_to(fifo)
        assert "FIFO" in (_special_file_kind(link) or "")

    def test_char_device(self):
        if not os.path.exists("/dev/null"):
            pytest.skip("no /dev/null")
        assert "character device" in (_special_file_kind("/dev/null") or "")


class TestReadFileToolFifoGuard:
    def test_fifo_read_returns_note_instantly(self, tmp_path, monkeypatch):
        import time

        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        fifo = tmp_path / "live.pipe"
        os.mkfifo(fifo)
        t0 = time.monotonic()
        result = json.loads(read_file_tool(str(fifo)))
        assert time.monotonic() - t0 < 5, "guard must not block on the FIFO"
        assert result["success"] is False
        assert "FIFO" in result["note"]
        assert "no read was attempted" in result["note"]

    def test_regular_file_unaffected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        f = tmp_path / "ok.txt"
        f.write_text("alpha\nbeta\n")
        result = json.loads(read_file_tool(str(f)))
        assert result.get("success", True) is not False
        assert "alpha" in result.get("content", "")
