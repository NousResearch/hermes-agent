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


def _wire(monkeypatch, commands):
    from tools import browser_tool as browser
    from tools import browser_tool_session as session

    monkeypatch.delenv("AGENT_BROWSER_PROFILE", raising=False)
    monkeypatch.setattr(browser, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser, "_blocked_private_page_action", lambda *a: None)
    monkeypatch.setattr(session, "_browser_command_preflight", lambda: {"browser_cmd": "agent-browser"})
    monkeypatch.setattr(session, "_get_session_info", lambda *a: {"session_name": "review", "cdp_url": None, "features": {"local": True}})
    monkeypatch.setattr(session._cloud, "_get_browser_engine", lambda: "chrome")
    monkeypatch.setattr(session._cloud, "_is_headed_mode", lambda: True)

    def spawn(*args):
        commands.append(args[2])
        return {"success": True, "data": {"secret": "WHAT-THE-HUMAN-TYPED"}}

    monkeypatch.setattr(session, "_spawn_and_collect", spawn)
    return browser, session


def _wire_browser_exec(monkeypatch, run_cli):
    """Route browser_exec through local Chromium without starting a real browser."""
    from tools import browser_tool_cloud as cloud
    from tools import browser_tool_session as session
    from tools import browser_use_cli as browser_use

    monkeypatch.setattr(browser_use, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(browser_use, "_base_subprocess_env", lambda: {})
    monkeypatch.setattr(browser_use, "_real_profile_consented", lambda: False)
    monkeypatch.setattr(browser_use, "_resolve_lightpanda_cdp", lambda *a: None)
    monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: "")
    monkeypatch.setattr("tools.browser_tool._get_open_command_timeout", lambda **_kw: 5)
    monkeypatch.setattr(cloud, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr(session, "_run_browser_command", lambda *_a, **_kw: {
        "success": True, "data": {"cdpUrl": "http://127.0.0.1:9222"}})
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


def test_browser_click_is_fenced_while_human_controls_shared_browser(monkeypatch):
    commands: list = []
    browser, _ = _wire(monkeypatch, commands)
    lease.acquire("human-viewer")
    result = json.loads(browser.browser_click("e1", task_id="review"))
    assert commands == [], f"human holds the lease, yet a browser command was dispatched: {commands}"
    assert result.get("code") == "human_has_control"


def test_browser_result_crossing_a_takeover_is_discarded(monkeypatch):
    commands: list = []
    browser, session = _wire(monkeypatch, commands)

    def spawn_then_takeover(*args):
        lease.acquire("human-viewer")
        lease.release("human-viewer")  # a full cycle, control is back — the frame is still theirs
        return {"success": True, "data": {"secret": "WHAT-THE-HUMAN-TYPED"}}

    monkeypatch.setattr(session, "_spawn_and_collect", spawn_then_takeover)
    result = browser.browser_click("e1", task_id="review")
    assert "WHAT-THE-HUMAN-TYPED" not in result


def test_real_profile_local_browser_is_fenced_by_provenance_even_without_a_live_display(monkeypatch):
    """A real-profile session attaches over a loopback cdp_url but is launched with the Bot Desktop
    DISPLAY, so it IS the human's browser: the fence keys on the ``local`` feature, not on the
    transport. And a stranded human lease with the screen already down must still fence (computer_use
    does), not silently unfence the browser."""
    commands: list = []
    browser, session = _wire(monkeypatch, commands)
    monkeypatch.setattr(session, "_get_session_info", lambda *a: {
        "session_name": "rp_1", "cdp_url": "ws://127.0.0.1:9222/devtools/browser/x",
        "features": {"local": True, "real_profile": True}})
    monkeypatch.setattr(runtime, "published_env", lambda: {})
    lease.acquire("human-viewer")
    result = json.loads(browser.browser_click("e1", task_id="review"))
    assert commands == [], f"human holds the lease, yet a real-profile browser command was dispatched: {commands}"
    assert result.get("code") == "human_has_control"


def test_browser_console_supervisor_fast_path_is_fenced_while_human_controls_shared_browser(monkeypatch):
    """`browser_console(expression=...)` answers over the CDP supervisor's WebSocket without ever reaching
    `_run_browser_command`, so the fence must sit in front of that fast path too — otherwise the one
    command that reads arbitrary page state is the one command the human's takeover does not stop."""
    import tools.browser_supervisor as supervisor_mod

    commands: list = []
    browser, _ = _wire(monkeypatch, commands)
    browser._active_sessions["review"] = {"session_name": "review", "cdp_url": "ws://127.0.0.1:9222/devtools/browser/x",
                                          "features": {"local": True}}
    evaluated: list = []

    class FakeSupervisor:
        def evaluate_runtime(self, expression, **_kw):
            evaluated.append(expression)
            return {"ok": True, "result": "WHAT-THE-HUMAN-TYPED", "result_type": "string"}

    class FakeRegistry:
        def get(self, task_id):
            return FakeSupervisor()

    monkeypatch.setattr(supervisor_mod, "SUPERVISOR_REGISTRY", FakeRegistry())
    try:
        lease.acquire("human-viewer")
        raw = browser.browser_console(expression="document.title", task_id="review")
    finally:
        browser._active_sessions.pop("review", None)
    result = json.loads(raw)
    assert evaluated == [] and commands == [], "human holds the lease, yet the page was evaluated"
    assert "WHAT-THE-HUMAN-TYPED" not in raw
    assert result.get("code") == "human_has_control"


def test_vault_page_operations_respect_the_human_lease(monkeypatch):
    """The vault tools reach the page over the supervisor socket, outside `_run_browser_command`: while a
    human holds the screen they must be refused like every other page access, and never focus, inspect or
    write to the form the human is typing into."""
    from tools import browser_tool as browser
    from tools import browser_vault_tool as vault

    browser._active_sessions["default"] = {"session_name": "review", "cdp_url": None, "features": {"local": True}}
    touched = []
    for name in ("browser_vault_fill", "browser_vault_enter_code", "browser_vault_save_login"):
        monkeypatch.setattr(vault, name, lambda *a, **k: touched.append(name) or json.dumps({"success": True}))
    lease.acquire("human")
    for handler in (vault._handle_vault_fill, vault._handle_vault_enter_code, vault._handle_vault_save_login):
        res = json.loads(handler({"handle": "vault_x"}, task_id="default"))
        assert res["code"] == "human_has_control", handler.__name__
    assert touched == [], "no vault page access while the human holds the screen"
    lease.release("human")
    assert json.loads(vault._handle_vault_fill({"handle": "vault_x"}, task_id="default"))["success"] is True


def test_secret_write_re_admits_after_a_takeover_during_the_code_prompt(monkeypatch):
    """`enter_code` blocks on the user's code INSIDE the handler fence. A takeover during that wait must
    refuse the write itself: the outer fence only discards the result afterwards, and by then the code is
    already in the page the human is typing into."""
    from tools import browser_tool as browser
    from tools import browser_vault_tool as vault

    browser._active_sessions["default"] = {"session_name": "review", "cdp_url": None, "features": {"local": True}}
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
