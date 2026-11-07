"""Windows stdio MCP spawns no longer flash a console window.

The gateway often runs without a console (``.vbs`` launch), so each stdio MCP
spawn of a console binary (``npx.cmd``, ``node.exe``) allocated a visible
console window that flashed on the desktop. ``_hide_stdio_console_window``
defaults ``CREATE_NO_WINDOW`` on ``anyio.open_process``, which the MCP SDK's
``stdio_client()`` has no other way to pass. See #149528.
"""

import asyncio

import anyio
import pytest

from tools.mcp_tool_transport import _hide_stdio_console_window


def test_noop_off_windows(monkeypatch):
    """On non-Windows the function must not touch anyio.open_process."""
    monkeypatch.setattr("os.name", "posix")
    original = anyio.open_process
    _hide_stdio_console_window()
    assert anyio.open_process is original


def test_defaults_create_no_window_on_windows(monkeypatch):
    """On Windows anyio.open_process is wrapped to default CREATE_NO_WINDOW."""
    monkeypatch.setattr("os.name", "nt")
    import subprocess

    # CREATE_NO_WINDOW is a Windows-only constant; provide it on the test host.
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)

    seen: dict = {}

    async def fake_open_process(*args, **kwargs):
        seen["kwargs"] = kwargs
        return "proc"

    monkeypatch.setattr("anyio.open_process", fake_open_process)
    _hide_stdio_console_window()
    patched = anyio.open_process
    assert patched is not fake_open_process
    assert getattr(patched, "_hermes_no_window", False)

    asyncio.run(patched(["cmd"], stdin=subprocess.PIPE))
    assert seen["kwargs"]["creationflags"] == 0x08000000


def test_idempotent_on_windows(monkeypatch):
    """A second call must not re-wrap the already-patched callable."""
    monkeypatch.setattr("os.name", "nt")
    import subprocess

    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)

    async def fake_open_process(*args, **kwargs):
        return "proc"

    monkeypatch.setattr("anyio.open_process", fake_open_process)
    _hide_stdio_console_window()
    first = anyio.open_process
    _hide_stdio_console_window()
    assert anyio.open_process is first


def test_preserves_explicit_creationflags_on_windows(monkeypatch):
    """An explicit creationflags kwarg must not be overwritten by the default."""
    monkeypatch.setattr("os.name", "nt")
    import subprocess

    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)

    seen: dict = {}

    async def fake_open_process(*args, **kwargs):
        seen["kwargs"] = kwargs
        return "proc"

    monkeypatch.setattr("anyio.open_process", fake_open_process)
    _hide_stdio_console_window()

    asyncio.run(anyio.open_process(["cmd"], stdin=subprocess.PIPE, creationflags=0x10000))
    assert seen["kwargs"]["creationflags"] == 0x10000
