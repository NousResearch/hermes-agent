"""Windows .cmd/.bat shims must be invoked via cmd.exe /c.

Direct CreateProcess on agent-browser.CMD (npm shim) leaves stdout/stderr
pipes open because background node workers inherit them; subprocess.run(
capture_output=True) then hangs past the timeout (#107232, same class as
#96731). Wrap at _agent_browser_argv so session_cmd, attach, execute, and
lightpanda all inherit it.

``os.name`` is patched only around the argv call: leaving it as ``nt`` on
POSIX makes pathlib instantiate WindowsPath while pytest formats failures.
"""

import os
from unittest.mock import patch

from tools.browser_tool import AGENT_BROWSER_NPX_SPEC, NPX_AGENT_BROWSER_SENTINEL
from tools.browser_tool_session import _agent_browser_argv


def _argv_under_os_name(name: str, browser_cmd: str) -> list:
    with patch.object(os, "name", name):
        return _agent_browser_argv(browser_cmd)


def test_win32_cmd_shim_wrapped_with_cmd_exe():
    argv = _argv_under_os_name("nt", r"C:\Users\me\AppData\Roaming\npm\agent-browser.CMD")
    assert argv[0].lower() in ("cmd.exe", "cmd")
    assert "/c" in argv
    assert any(str(t).lower().endswith(".cmd") for t in argv)


def test_win32_bat_shim_wrapped_with_cmd_exe():
    bat = r"C:\tools\agent-browser.bat"
    argv = _argv_under_os_name("nt", bat)
    assert argv[0].lower() in ("cmd.exe", "cmd")
    assert "/c" in argv
    assert bat in argv


def test_win32_npx_cmd_sentinel_wrapped(monkeypatch):
    npx_cmd = r"C:\nodejs\npx.CMD"
    monkeypatch.setattr(
        "tools.browser_tool_session._install._resolve_npx_bin",
        lambda: npx_cmd,
    )
    argv = _argv_under_os_name("nt", NPX_AGENT_BROWSER_SENTINEL)
    assert argv[0].lower() in ("cmd.exe", "cmd")
    assert "/c" in argv
    assert npx_cmd in argv
    assert "--ignore-scripts" in argv
    assert "--prefer-offline" in argv
    assert "-y" in argv
    assert AGENT_BROWSER_NPX_SPEC in argv


def test_posix_direct_binary_unwrapped():
    assert _argv_under_os_name("posix", "/usr/bin/agent-browser") == ["/usr/bin/agent-browser"]


def test_posix_npx_sentinel_unwrapped(monkeypatch):
    monkeypatch.setattr(
        "tools.browser_tool_session._install._resolve_npx_bin",
        lambda: "/usr/bin/npx",
    )
    assert _argv_under_os_name("posix", NPX_AGENT_BROWSER_SENTINEL) == [
        "/usr/bin/npx",
        "--ignore-scripts",
        "--prefer-offline",
        "-y",
        AGENT_BROWSER_NPX_SPEC,
    ]


def test_win32_exe_unwrapped():
    exe = r"C:\Program Files\nodejs\node.exe"
    assert _argv_under_os_name("nt", exe) == [exe]


def test_win32_bare_name_unwrapped():
    assert _argv_under_os_name("nt", "agent-browser") == ["agent-browser"]
    assert _argv_under_os_name("nt", "npx") == ["npx"]


def test_win32_already_cmd_not_double_wrapped():
    assert _argv_under_os_name("nt", "cmd.exe") == ["cmd.exe"]
    assert _argv_under_os_name("nt", "cmd") == ["cmd"]
    cmd_path = r"C:\Windows\System32\cmd.exe"
    assert _argv_under_os_name("nt", cmd_path) == [cmd_path]
