"""Regression tests for lossy UTF-8 decoding of TUI gateway subprocess output (#53137).

On Windows with a non-UTF-8 system locale (e.g. GBK on Chinese Windows), text-mode
subprocess reads defaulted to the locale encoding, and a child emitting bytes invalid in
that encoding raised UnicodeDecodeError inside the RPC handler / gateway thread.
#53137 decodes every captured child stream as UTF-8 with ``errors="replace"``.

These tests drive the real RPC handlers against a real child process that writes
invalid UTF-8, so they fail on any host if the lossy decode is dropped (strict UTF-8
raises on ``b"\\xff\\xfe"`` just like GBK does on the reported bytes).

# Test pattern adapted from @devorun's PR #52700 (salvage convention).
"""

from __future__ import annotations

import sys

import pytest

from tui_gateway import server

@pytest.fixture()
def bad_bytes_cmd(tmp_path):
    """A shell command running a child that writes bytes invalid in UTF-8 (and in most
    locale codepages), then text. A script file rather than ``-c`` so the shell.exec
    approval gate lets it through; quoted paths work under /bin/sh and cmd.exe."""
    script = tmp_path / "emit_bad_bytes.py"
    script.write_text(
        "import sys\nsys.stdout.buffer.write(bytes([255, 254]) + b' tail-ok')\nsys.stdout.flush()\n",
        encoding="utf-8",
    )
    return f'"{sys.executable}" "{script}"'


def test_shell_exec_survives_non_utf8_child_output(bad_bytes_cmd):
    """``!cmd`` via shell.exec returns the output (with replacement chars) instead of a
    decode failure surfaced as a 5003 error."""
    resp = server.handle_request(
        {"id": "enc", "method": "shell.exec", "params": {"command": bad_bytes_cmd}}
    )

    assert "error" not in resp, resp
    assert resp["result"]["code"] == 0
    assert "tail-ok" in resp["result"]["stdout"]
    assert "\ufffd" in resp["result"]["stdout"]


def test_quick_command_exec_survives_non_utf8_child_output(monkeypatch, bad_bytes_cmd):
    """A ``type: exec`` quick command runs subprocess.run outside any try block, so a
    strict decode would raise straight out of command.dispatch."""
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"quick_commands": {"badbytes": {"type": "exec", "command": bad_bytes_cmd}}},
    )

    resp = server.handle_request(
        {"id": "enc", "method": "command.dispatch", "params": {"name": "badbytes"}}
    )

    assert "error" not in resp, resp
    assert resp["result"]["type"] == "exec"
    assert "tail-ok" in resp["result"]["output"]
