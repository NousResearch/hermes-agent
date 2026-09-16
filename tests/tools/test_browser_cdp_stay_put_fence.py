"""Opt-in stay-put CDP honours the Bot Screen lease when present.

Unmarked user/cloud CDP stays unfenced (#108914). ``tools.bot_desktop`` is not
on main; the fence soft-imports the lease and is a no-op when the module is missing.
"""

from __future__ import annotations

import json
import sys
import types

from tools import browser_tool_session as session
from tools.browser_tool_session import _create_cdp_session, run_fenced

CAP = "http://127.0.0.1:9222"
LAB2 = "http://127.0.0.1:9223"


class HumanHasControl(RuntimeError):
    pass


def _install_fake_lease(monkeypatch, *, human: bool, epoch: int = 1):
    """Inject a minimal ``tools.bot_desktop.lease`` (Bot Screen is not merged)."""
    state = {"human": human, "epoch": epoch}

    class Lease:
        def __init__(self):
            self.epoch = state["epoch"]

    def assert_agent_may_act():
        if state["human"]:
            raise HumanHasControl(
                "A human has taken over this desktop (they may be entering a credential). "
                "Screen actions and captures are refused until they hand control back."
            )
        return Lease()

    lease_mod = types.ModuleType("tools.bot_desktop.lease")
    lease_mod.HumanHasControl = HumanHasControl
    lease_mod.assert_agent_may_act = assert_agent_may_act
    lease_mod.get = lambda: Lease()
    lease_mod.human_holds = lambda: state["human"]

    runtime_mod = types.ModuleType("tools.bot_desktop.runtime")
    runtime_mod.published_env = lambda: {}

    pkg = types.ModuleType("tools.bot_desktop")
    pkg.lease = lease_mod
    pkg.runtime = runtime_mod

    monkeypatch.setitem(sys.modules, "tools.bot_desktop", pkg)
    monkeypatch.setitem(sys.modules, "tools.bot_desktop.lease", lease_mod)
    monkeypatch.setitem(sys.modules, "tools.bot_desktop.runtime", runtime_mod)
    return state


def _wire_command(monkeypatch, session_info):
    commands: list = []
    monkeypatch.setattr(session, "_browser_command_preflight", lambda: {"browser_cmd": "agent-browser"})
    monkeypatch.setattr(session, "_get_session_info", lambda *a, **k: session_info)
    monkeypatch.setattr(session._cdp, "_ensure_cdp_supervisor", lambda *a, **k: None)
    monkeypatch.setattr(session._cloud, "_get_browser_engine", lambda: "chrome")
    monkeypatch.setattr(
        session, "_spawn_and_collect",
        lambda *a, **k: commands.append(a) or {"success": True, "data": {"secret": "HUMAN-TYPED"}},
    )
    return commands


def test_stay_put_refused_when_human_holds_lease(monkeypatch):
    _install_fake_lease(monkeypatch, human=True)
    commands = _wire_command(monkeypatch, {
        "session_name": "cdp_1", "cdp_url": CAP,
        "features": {"cdp_override": True, "stay_put": True},
    })
    result = session._run_browser_command("t1", "snapshot")
    assert commands == []
    assert result.get("code") == "human_has_control"
    assert result.get("success") is False


def test_unmarked_cdp_stays_unfenced_while_human_holds(monkeypatch):
    _install_fake_lease(monkeypatch, human=True)
    commands = _wire_command(monkeypatch, {
        "session_name": "cdp_2", "cdp_url": LAB2,
        "features": {"cdp_override": True},
    })
    result = session._run_browser_command("t1", "snapshot")
    assert commands, "unmarked user CDP must still dispatch while a human holds Bot Screen"
    assert result.get("code") != "human_has_control"
    assert result.get("success") is True


def test_stay_put_is_noop_without_bot_desktop_module(monkeypatch):
    """On main today ``tools.bot_desktop`` is absent — stay-put must not crash or refuse."""
    sys.modules.pop("tools.bot_desktop", None)
    sys.modules.pop("tools.bot_desktop.lease", None)
    sys.modules.pop("tools.bot_desktop.runtime", None)
    commands = _wire_command(monkeypatch, {
        "session_name": "cdp_1", "cdp_url": CAP,
        "features": {"cdp_override": True, "stay_put": True},
    })
    result = session._run_browser_command("t1", "click")
    assert commands, "missing bot_desktop must not fence stay-put CDP"
    assert result.get("success") is True
    assert result.get("code") != "human_has_control"


def test_create_cdp_session_records_stay_put_feature():
    marked = _create_cdp_session("t", CAP, stay_put=True)
    assert marked["features"]["stay_put"] is True
    assert marked["features"]["cdp_override"] is True
    unmarked = _create_cdp_session("t", LAB2)
    assert "stay_put" not in unmarked["features"]


def test_takeover_mid_command_discards_stay_put_result(monkeypatch):
    state = _install_fake_lease(monkeypatch, human=False, epoch=3)
    session_info = {
        "session_name": "cdp_1", "cdp_url": CAP,
        "features": {"cdp_override": True, "stay_put": True},
    }
    commands = _wire_command(monkeypatch, session_info)

    def spawn_then_takeover(*args, **kwargs):
        state["epoch"] += 1
        return {"success": True, "data": {"secret": "HUMAN-TYPED"}}

    monkeypatch.setattr(session, "_spawn_and_collect", spawn_then_takeover)
    result = session._run_browser_command("t1", "snapshot")
    assert "HUMAN-TYPED" not in json.dumps(result)
    assert result.get("code") == "human_has_control"
    assert commands == []  # spawn_then_takeover replaced the list-appending stub


def test_run_fenced_direct_contract(monkeypatch):
    _install_fake_lease(monkeypatch, human=True)
    ran = []
    refused = run_fenced(
        {"features": {"stay_put": True}},
        lambda: ran.append("acted") or {"success": True},
    )
    assert ran == []
    assert refused["code"] == "human_has_control"

    allowed = run_fenced(
        {"features": {"cdp_override": True}},
        lambda: ran.append("acted") or {"success": True},
    )
    assert ran == ["acted"]
    assert allowed == {"success": True}
