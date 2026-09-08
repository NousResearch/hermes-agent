"""#87152 / #103973 regression: the browser-use CLI subprocess must pin UTF-8
on both pipes, not inherit the Windows ANSI code page.

`subprocess.run(..., text=True)` without `encoding=` decodes with
`locale.getpreferredencoding()` — cp1252 on US Windows, cp936 (GBK) on CJK
Windows. The browser-use CLI speaks UTF-8 in both directions, so any
non-ASCII byte crashed the reader thread (`UnicodeDecodeError` in
`_readerthread`, empty tool output, #87152) and GBK-encoded stdin code
crashed the CLI itself (`sys.stdin.read()`, #103973). The contract: the exec
call pins `encoding="utf-8"` (with `errors="replace"`) the same way
`install_cli` already does — on every host locale.
"""
import subprocess
import sys
from unittest.mock import patch

import pytest

import tools.browser_use_cli as browser_use_cli


def _capture_run_kwargs(cmd_prefix=None):
    """Run browser_exec with subprocess.run intercepted; return the kwargs it
    was called with and the recorded call command."""
    captured = {}

    class _FakeProc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FakeProc()

    return captured, fake_run


def test_exec_subprocess_pins_utf8_pipes(monkeypatch):
    """The exec call must pass encoding='utf-8' (and errors=) regardless of the
    host's preferred encoding — the exact kwargs pair install_cli uses."""
    captured, fake_run = _capture_run_kwargs()
    # patch subprocess.run as the module sees it
    monkeypatch.setattr(browser_use_cli.subprocess, "run", fake_run)
    # Neutralize the CLI discovery + route/backend probes that would reach the real CLI.
    monkeypatch.setattr(browser_use_cli, "_find_cli", lambda: ["browser-use", "exec"])
    monkeypatch.setattr(browser_use_cli, "_base_subprocess_env", lambda: {})
    monkeypatch.setattr(browser_use_cli, "_route_backend", lambda env, session, task_id, local: None)
    monkeypatch.setattr(browser_use_cli, "_read_browser_cfg", lambda: {})
    monkeypatch.setattr(browser_use_cli, "is_legacy_browser_use_cloud_config", lambda cfg: False)
    monkeypatch.setattr(browser_use_cli, "_workspace_dir", lambda task_id: None)

    browser_use_cli.browser_exec("print('打开一个页面')")

    kwargs = captured["kwargs"]
    assert kwargs.get("encoding") == "utf-8", (
        "browser_exec must pin encoding='utf-8' on the CLI pipes: text=True without "
        "encoding decodes with the Windows ANSI code page and crashes on non-ASCII "
        "(#87152 stdout, #103973 stdin)"
    )
    assert kwargs.get("errors") == "replace"
    assert kwargs.get("text") is True


@pytest.mark.skipif(sys.platform != "win32", reason="locale-dependent crash is Windows-specific evidence")
def test_ansi_locale_cannot_roundtrip_utf8():
    """Why the pin is load-bearing: under a non-UTF-8 preferred encoding, the
    default text pipes cannot round-trip a UTF-8 emoji — this is the exact
    byte pattern that crashed the reader thread in the field."""
    pref = subprocess.run([sys.executable, "-c", "import sys; print(sys.getpreferredencoding())"],
                          capture_output=True, text=True).stdout.strip()
    if pref.lower().replace("-", "") in ("utf8", "utf8mb4"):
        pytest.skip("host locale is already UTF-8; the crash needs an ANSI code page")
    payload = "\U0001F054"  # the burger emoji from the field report
    with pytest.raises(UnicodeError):
        subprocess.run([sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
                       input=payload, capture_output=True, text=True, timeout=10)
