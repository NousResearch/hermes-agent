"""``_is_interactive()`` must not trust ``isatty()`` alone on Windows: the CRT reports a DEVNULL /
detached stdin as a TTY, so the background gateway looked interactive and launched browser OAuth."""

import os
import subprocess
import sys

import pytest


@pytest.mark.windows_only
def test_devnull_stdin_is_not_a_console_on_windows():
    code = ("import sys, os; sys.path.insert(0, os.getcwd()); "
            "from tools.mcp_oauth import _stdin_is_console; print(_stdin_is_console(), sys.stdin.isatty())")
    proc = subprocess.run([sys.executable, "-c", code], stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, cwd=os.getcwd(), timeout=60)
    console, isatty = proc.stdout.split()
    assert isatty == "True", "premise: the Windows CRT calls DEVNULL a tty (else this guard is moot)"
    assert console == "False"


def test_non_tty_stdin_is_not_interactive(monkeypatch):
    import io
    from tools import mcp_oauth
    monkeypatch.setattr(mcp_oauth.sys, "stdin", io.StringIO())
    assert mcp_oauth._stdin_is_console() is False
    assert mcp_oauth._is_interactive() is False
