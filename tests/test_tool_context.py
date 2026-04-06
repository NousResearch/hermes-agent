"""Regression tests for environments.tool_context file-transfer helpers."""

import base64
import shlex
from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest.mock import MagicMock

from environments.tool_context import ToolContext


def test_upload_file_quotes_remote_paths(tmp_path, monkeypatch):
    """upload_file() must quote remote shell operands with spaces/quotes."""
    local_file = tmp_path / "payload.bin"
    local_file.write_bytes(b"hello world")

    commands = []
    ctx = ToolContext("task-123")
    monkeypatch.setattr(
        ctx,
        "terminal",
        lambda command, timeout=0: commands.append((command, timeout)) or {"exit_code": 0, "output": ""},
    )

    remote_path = "/tmp/space dir/quote's/file.bin"
    result = ctx.upload_file(str(local_file), remote_path)

    assert result["exit_code"] == 0
    assert commands[0][0] == f"mkdir -p -- {shlex.quote(str(PurePosixPath(remote_path).parent))}"
    assert commands[1][0].endswith(f"> {shlex.quote(remote_path)}")


def test_download_file_quotes_remote_paths(tmp_path, monkeypatch):
    """download_file() must quote remote shell operands before base64."""
    payload = b"downloaded bytes"
    encoded = base64.b64encode(payload).decode("ascii")

    commands = []
    ctx = ToolContext("task-123")
    monkeypatch.setattr(
        ctx,
        "terminal",
        lambda command, timeout=0: commands.append((command, timeout)) or {"exit_code": 0, "output": encoded},
    )

    local_path = tmp_path / "result.bin"
    remote_path = "/tmp/space dir/quote's/file.bin"
    result = ctx.download_file(remote_path, str(local_path))

    assert result == {"success": True, "bytes": len(payload)}
    assert local_path.read_bytes() == payload
    assert commands == [
        (f"base64 < {shlex.quote(remote_path)} 2>/dev/null", 30),
    ]


def test_download_dir_avoids_prefix_confusion(tmp_path, monkeypatch):
    """Sibling paths sharing a prefix must not be treated as descendants."""
    downloads = []
    commands = []
    ctx = ToolContext("task-123")

    def fake_terminal(command, timeout=0):
        commands.append((command, timeout))
        return {
            "exit_code": 0,
            "output": "/tmp/run/output.txt\n/tmp/run-extra/secret.txt\n",
        }

    monkeypatch.setattr(ctx, "terminal", fake_terminal)
    monkeypatch.setattr(
        ctx,
        "download_file",
        lambda remote, local: downloads.append((remote, local)) or {"success": True},
    )

    results = ctx.download_dir("/tmp/run", str(tmp_path))

    assert results == [{"success": True}, {"success": True}]
    assert commands == [(f"find {shlex.quote('/tmp/run')} -type f 2>/dev/null", 15)]
    assert downloads == [
        ("/tmp/run/output.txt", str(tmp_path / "output.txt")),
        ("/tmp/run-extra/secret.txt", str(tmp_path / "secret.txt")),
    ]


def test_upload_dir_normalizes_windows_relative_paths(tmp_path, monkeypatch):
    """upload_dir() should convert host Windows separators to POSIX remote paths."""
    uploads = []
    ctx = ToolContext("task-123")
    monkeypatch.setattr(
        ctx,
        "upload_file",
        lambda local, remote: uploads.append((local, remote)) or {"exit_code": 0, "output": ""},
    )

    fake_file = MagicMock()
    fake_file.is_file.return_value = True
    fake_file.relative_to.return_value = PureWindowsPath(r"nested\file.txt")
    fake_file.__str__.return_value = r"C:\workspace\nested\file.txt"

    monkeypatch.setattr(Path, "rglob", lambda self, pattern: [fake_file])

    result = ctx.upload_dir(str(tmp_path), "/sandbox/root")

    assert result == [{"exit_code": 0, "output": ""}]
    assert uploads == [
        (r"C:\workspace\nested\file.txt", "/sandbox/root/nested/file.txt"),
    ]
