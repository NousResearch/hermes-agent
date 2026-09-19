"""The real-profile runner must return in bounded time when a survivor holds the handles.

agent-browser forks a daemon — and, on a cold session, Chromium — that inherit the parent's
output handles. Captured through PIPES, ``subprocess.run``'s post-timeout drain waits for an
EOF those survivors never give, so the real-profile launch path wedged until the OUTER tool
timeout (the #106244 bug class; 420s of dead tool call per attempt). These tests pin the
contract the fix relies on: capture into FILES and always return, even while a surviving
grandchild still holds the handles.
"""
import sys
import time

from tools import browser_tool_real_profile as rp

_SURVIVOR_SECONDS = 30
# Parent: fork a survivor that inherits our handles, then outlive the runner's timeout.
_PARENT_CODE = (
    "import subprocess, sys, time\n"
    f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep({_SURVIVOR_SECONDS})'])\n"
    "time.sleep(60)\n"
)


def test_run_agent_browser_argv_returns_bounded_with_surviving_grandchild():
    """A timeout returns None promptly — never waiting for the survivor to release the handles."""
    started = time.monotonic()
    result = rp._run_agent_browser_argv([sys.executable, "-c", _PARENT_CODE], "test-bounded", 2)
    elapsed = time.monotonic() - started

    assert result is None, "a timed-out command must report failure instead of returning output"
    # Generous bound: the survivor lives 30s, so a pipe-capturing drain cannot come back early.
    assert elapsed < 10, (
        f"returned after {elapsed:.1f}s — the capture is waiting on the surviving grandchild "
        "(pipes) instead of returning on the timeout (files)"
    )


def test_session_cmd_routes_through_bounded_runner(monkeypatch):
    """``_agent_browser_session_cmd`` must never call subprocess directly again."""
    seen = {}

    def fake_runner(argv, tag, timeout):
        seen["argv"], seen["tag"], seen["timeout"] = argv, tag, timeout
        return "ran"

    monkeypatch.setattr(rp, "_run_agent_browser_argv", fake_runner)
    monkeypatch.setattr(rp._install, "_find_agent_browser", lambda *a, **k: "agent-browser-fake")

    assert rp._agent_browser_session_cmd("sess", "get", "cdp-url", log_label="get cdp-url") == "ran"
    assert seen["timeout"] == 15
    assert seen["tag"] == "rp-get-cdp-url"
    assert seen["argv"] == ["agent-browser-fake", "--session", "sess", "get", "cdp-url"]
