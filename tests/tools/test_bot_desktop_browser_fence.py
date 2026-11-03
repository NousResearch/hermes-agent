"""The bot's browser tools obey the screen lease: while a human holds the shared browser, nothing is
dispatched, and a command whose run crossed a takeover loses its result."""

from __future__ import annotations

import json
import subprocess

import pytest

from tools.bot_desktop import lease, runtime


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    lease._reset_for_tests()
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":37"})
    yield
    lease._reset_for_tests()


def _wire_browser_exec(monkeypatch, run_cli):
    """Route browser_exec through the local session layer without starting a real browser."""
    from tools import browser_lightpanda, browser_session, browser_session_cdp, browser_session_cloud
    from tools import browser_use_cli as browser_use

    monkeypatch.setattr(browser_use, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(browser_use, "_base_subprocess_env", dict)
    monkeypatch.setattr(browser_use, "_real_profile_consented", lambda: False)
    monkeypatch.setattr(browser_lightpanda, "using_lightpanda_engine", lambda: False)
    monkeypatch.setattr(browser_session_cdp, "get_cdp_override", lambda: "")
    monkeypatch.setattr(browser_session_cdp, "get_cdp_override_raw", lambda: "")
    monkeypatch.setattr(browser_session_cloud, "get_cloud_provider", lambda: None)
    monkeypatch.setattr(browser_session, "get_session_info", lambda *_a, **_kw: {
        "session_name": "review", "cdp_url": "http://127.0.0.1:9222", "features": {"local": True}})
    monkeypatch.setattr(browser_use, "_attach_vault_supervisor", lambda *a: None)
    monkeypatch.setattr(browser_use, "_run_cli_killing_process_group", run_cli)
    return browser_use


def test_browser_exec_is_fenced_while_human_controls_shared_browser(monkeypatch):
    dispatched: list[str] = []

    def run_cli(*_args):
        dispatched.append("browser-use")
        return subprocess.CompletedProcess([], 0, "WHAT-THE-HUMAN-TYPED", "")

    browser_use = _wire_browser_exec(monkeypatch, run_cli)
    lease.acquire("human-viewer")
    raw = browser_use.browser_exec("print(page_info())", task_id="review")
    assert isinstance(raw, str)
    result = json.loads(raw)
    assert dispatched == [], "human holds the lease, yet browser_exec was dispatched"
    assert "WHAT-THE-HUMAN-TYPED" not in raw
    assert result.get("code") == "human_has_control"


def test_browser_exec_result_crossing_a_takeover_is_discarded(monkeypatch):
    def run_cli(*_args):
        lease.acquire("human-viewer")
        lease.release("human-viewer")
        return subprocess.CompletedProcess([], 0, "WHAT-THE-HUMAN-TYPED", "")

    browser_use = _wire_browser_exec(monkeypatch, run_cli)
    raw = browser_use.browser_exec("print(page_info())", task_id="review")
    assert isinstance(raw, str)
    result = json.loads(raw)
    assert "WHAT-THE-HUMAN-TYPED" not in raw
    assert result.get("code") == "human_has_control"


def test_run_fenced_refuses_a_local_session_while_human_controls_shared_browser():
    from tools.browser_session import run_fenced

    dispatched: list = []
    lease.acquire("human-viewer")
    result = run_fenced({"session_name": "review", "cdp_url": None, "features": {"local": True}},
                        lambda: dispatched.append("cmd") or {"success": True})
    assert dispatched == [], f"human holds the lease, yet a browser command was dispatched: {dispatched}"
    assert result.get("code") == "human_has_control"


def test_run_fenced_discards_a_result_crossing_a_takeover():
    from tools.browser_session import run_fenced

    def run_then_takeover():
        lease.acquire("human-viewer")
        lease.release("human-viewer")  # a full cycle, control is back — the frame is still theirs
        return {"success": True, "data": {"secret": "WHAT-THE-HUMAN-TYPED"}}

    result = run_fenced({"session_name": "review", "cdp_url": None, "features": {"local": True}},
                        run_then_takeover)
    assert "WHAT-THE-HUMAN-TYPED" not in json.dumps(result)


def test_real_profile_local_browser_is_fenced_by_provenance_even_without_a_live_display(monkeypatch):
    """A real-profile session attaches over a loopback cdp_url but is launched with the Bot Desktop
    DISPLAY, so it IS the human's browser: the fence keys on the ``local`` feature, not on the
    transport. And a stranded human lease with the screen already down must still fence (computer_use
    does), not silently unfence the browser."""
    from tools.browser_session import run_fenced

    dispatched: list = []
    monkeypatch.setattr(runtime, "published_env", dict)
    lease.acquire("human-viewer")
    result = run_fenced({"session_name": "rp_1", "cdp_url": "ws://127.0.0.1:9222/devtools/browser/x",
                         "features": {"local": True, "real_profile": True}},
                        lambda: dispatched.append("cmd") or {"success": True})
    assert dispatched == [], f"human holds the lease, yet a real-profile browser command was dispatched: {dispatched}"
    assert result.get("code") == "human_has_control"


def test_vault_page_operations_respect_the_human_lease(monkeypatch):
    """The vault tools reach the page over the supervisor socket, outside `_run_browser_command`: while a
    human holds the screen they must be refused like every other page access, and never focus, inspect or
    write to the form the human is typing into."""
    from tools import browser_session
    from tools import browser_vault_tool as vault

    monkeypatch.setitem(browser_session._active_sessions, "default",
                        {"session_name": "review", "cdp_url": None, "features": {"local": True}})
    touched = []
    for name in ("browser_vault_fill", "browser_vault_enter_code", "browser_vault_save_login"):
        monkeypatch.setattr(vault, name, lambda *a, **k: touched.append(name) or json.dumps({"success": True}))
    lease.acquire("human")
    for action in ("fill", "enter_code", "save_login"):
        res = json.loads(vault._handle_vault({"action": action, "handle": "vault_x"}, task_id="default"))
        assert res["code"] == "human_has_control", action
    assert touched == [], "no vault page access while the human holds the screen"
    lease.release("human")
    assert json.loads(vault._handle_vault({"action": "fill", "handle": "vault_x"}, task_id="default"))["success"] is True


def test_secret_write_re_admits_after_a_takeover_during_the_code_prompt(monkeypatch):
    """`enter_code` blocks on the user's code INSIDE the handler fence. A takeover during that wait must
    refuse the write itself: the outer fence only discards the result afterwards, and by then the code is
    already in the page the human is typing into."""
    from tools import browser_session
    from tools import browser_vault_tool as vault

    monkeypatch.setitem(browser_session._active_sessions, "default",
                        {"session_name": "review", "cdp_url": None, "features": {"local": True}})
    evaluated = []

    class Sup:
        def evaluate_runtime(self, expr):
            evaluated.append(expr)
            return {"ok": True, "result": "{}"}

    monkeypatch.setattr(vault, "_ensure_supervisor", lambda tid: Sup())
    assert vault._eval_js_secret("default", "fill()")["success"] is True  # agent holds: writes
    lease.acquire("human")  # takeover while the prompt was open
    res = vault._eval_js_secret("default", "fill()")
    assert res["success"] is False and res["error_type"] == "human_has_control"
    assert evaluated == ["fill()"], "the credential must not reach the page under a human lease"
