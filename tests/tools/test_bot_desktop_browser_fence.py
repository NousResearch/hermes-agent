"""The bot's browser tools obey the screen lease: while a human holds the shared browser, nothing is
dispatched, and a command whose run crossed a takeover loses its result."""

from __future__ import annotations

import json

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


class _RecordingSupervisor:
    """Records every page access made over the CDP supervisor WebSocket."""

    def __init__(self):
        self.calls: list = []

    def focus_page(self, origin, *, accept=None, timeout=10.0):
        self.calls.append(("focus_page", origin))
        return {"ok": True, "url": "https://example.com/login"}

    def evaluate_runtime(self, expression, **_kw):
        self.calls.append(("evaluate_runtime", expression))
        return {"ok": True, "result": "https://example.com/login", "result_type": "string"}


def _wire_vault(monkeypatch, sup):
    """Wire a LOCAL browser session with a recording supervisor, as the vault tools see it."""
    import tools.browser_supervisor as supervisor_mod
    from tools import browser_tool as browser

    class FakeRegistry:
        def get(self, task_id):
            return sup

        def get_or_start(self, task_id, **kw):
            return sup

    monkeypatch.setattr(supervisor_mod, "SUPERVISOR_REGISTRY", FakeRegistry())
    local = {"session_name": "review", "cdp_url": None, "features": {"local": True}}
    browser._active_sessions["review"] = local
    return browser, local


def test_browser_vault_fill_is_fenced_while_human_controls_shared_browser(monkeypatch):
    """`browser_vault_fill` writes a password into the page over the CDP supervisor, bypassing
    `_run_browser_command`, so it must consult the same lease fence: the browser lives on the
    Bot Desktop screen a human who took over is typing into."""
    from tools import browser_vault_tool as vault

    sup = _RecordingSupervisor()
    browser, _ = _wire_vault(monkeypatch, sup)
    lease.acquire("human-viewer")
    try:
        raw = vault.browser_vault_fill("vault_anything", task_id="review")
    finally:
        browser._active_sessions.pop("review", None)
    result = json.loads(raw)
    assert sup.calls == [], f"human holds the lease, yet a vault fill reached the page: {sup.calls}"
    assert result.get("code") == "human_has_control", result


def test_browser_vault_enter_code_is_fenced_while_human_controls_shared_browser(monkeypatch):
    """`browser_vault_enter_code` writes an OTP into the page over the supervisor, so it too must
    refuse while the human holds — the person may be typing the same code the bot is about to write."""
    from tools import browser_vault_tool as vault

    sup = _RecordingSupervisor()
    browser, _ = _wire_vault(monkeypatch, sup)
    lease.acquire("human-viewer")
    try:
        raw = vault.browser_vault_enter_code(handle="", task_id="review")
    finally:
        browser._active_sessions.pop("review", None)
    result = json.loads(raw)
    assert sup.calls == [], f"human holds the lease, yet a vault OTP fill reached the page: {sup.calls}"
    assert result.get("code") == "human_has_control", result


def test_browser_vault_save_login_is_fenced_while_human_controls_shared_browser(monkeypatch):
    """`browser_vault_save_login` focuses a tab and reads its origin before prompting, so the two
    unfenced supervisor reads must be refused while the human holds the screen."""
    from tools import browser_vault_tool as vault

    sup = _RecordingSupervisor()
    browser, _ = _wire_vault(monkeypatch, sup)
    lease.acquire("human-viewer")
    try:
        raw = vault.browser_vault_save_login(task_id="review")
    finally:
        browser._active_sessions.pop("review", None)
    result = json.loads(raw)
    assert sup.calls == [], f"human holds the lease, yet a vault save-login read reached the page: {sup.calls}"
    assert result.get("code") == "human_has_control", result


def test_vault_eval_result_crossing_a_takeover_is_discarded(monkeypatch):
    """A vault page read admitted while the agent held must lose its result if a human takes over
    mid-call (the same epoch fence that discards a browser command's result)."""
    from tools import browser_vault_tool as vault

    class TakeoverSupervisor:
        def evaluate_runtime(self, expression, **_kw):
            lease.acquire("human-viewer")  # a full takeover mid-read
            return {"ok": True, "result": "https://example.com/login", "result_type": "string"}

    browser, _ = _wire_vault(monkeypatch, TakeoverSupervisor())
    try:
        result = vault._eval_js("review", "window.location.href")
    finally:
        browser._active_sessions.pop("review", None)
    assert result.get("code") == "human_has_control", result
    assert result.get("success") is False
