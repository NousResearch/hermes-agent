"""A backend teardown must stop its raw-task vault supervisors, not other browsers."""
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from tools import browser_supervisor as bs
from tools import browser_tool as bt
from tools import browser_tool_cdp as cdp
from tools import browser_tool_cloud as cloud
from tools import browser_tool_lifecycle as lifecycle
from tools import browser_tool_session as session
from tools import browser_use_cli as cli


@pytest.fixture
def browsers(monkeypatch):
    registry = bs._SupervisorRegistry()
    stopped = []

    def start(supervisor, timeout=15):
        supervisor._thread = SimpleNamespace(is_alive=lambda: True)
        supervisor._loop = SimpleNamespace(is_running=lambda: True)

    monkeypatch.setattr(bs.CDPSupervisor, "start", start)
    monkeypatch.setattr(bs.CDPSupervisor, "stop", lambda s: stopped.append(s.task_id))
    monkeypatch.setattr(bs, "SUPERVISOR_REGISTRY", registry)
    monkeypatch.setattr(cdp, "_get_cdp_override", lambda: "")
    monkeypatch.setattr(cdp, "_resolve_cdp_override", lambda url: url)
    monkeypatch.setattr(cdp, "_get_dialog_policy_config", lambda: (bs.DEFAULT_DIALOG_POLICY, 30))
    monkeypatch.setattr(cloud, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr(cli, "_resolve_lightpanda_cdp", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_reach_sandbox_cdp", lambda url: url)
    monkeypatch.setattr(bt, "_get_open_command_timeout", lambda **kw: 10)
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(bt, "_maybe_stop_recording", lambda key: None)
    monkeypatch.setattr(bt, "_active_sessions", {})

    def command(key, command, args=None, timeout=None):
        if command == "close":
            assert all(s.browser_session_key != key for s in registry._by_task.values())
        else:
            bt._active_sessions[key] = {"session_name": key, "bb_session_id": None}
        return {"success": True, "data": {"cdpUrl": f"ws://localhost/{key}"}}

    monkeypatch.setattr(session, "_run_browser_command", command)

    def attach(task, name, profile="p1", override=None):
        monkeypatch.setattr(cli, "_served_profile_tag", lambda: profile)
        env = {"BU_CDP_WS": override} if override else {}
        assert cli._resolve_backend_cdp(env, task, name) is None
        cli._attach_vault_supervisor(env, task)
        supervisor = registry.get(task)
        assert supervisor is not None
        return supervisor

    yield registry, stopped, attach
    registry.stop_all()


@pytest.mark.parametrize("force", [False, True])
def test_backend_teardown_stops_raw_task_supervisors_before_browser(browsers, monkeypatch, force):
    registry, stopped, attach = browsers
    attach("cron-run-1", "apple")
    attach("cron-run-2", "apple")
    other = attach("other-session", "idigital")
    profile = attach("other-profile", "apple", profile="p2")
    key = "bu-named-apple@p1"

    def release(owner, info):
        assert registry.get("cron-run-1") is None
        assert registry.get("cron-run-2") is None
        assert registry.get("other-session") is other
        assert registry.get("other-profile") is profile
        bt._active_sessions.pop(owner)

    monkeypatch.setattr(lifecycle, "_release_session_resources", release)
    cleanup = lifecycle._force_reap_browser_session if force else lifecycle._cleanup_single_browser_session
    cleanup(key)
    assert set(stopped) == {"cron-run-1", "cron-run-2"}


def test_rebound_and_operator_supervisors_survive_old_backend_cleanup(browsers, monkeypatch):
    registry, stopped, attach = browsers
    attach("switched", "apple")
    rebound = attach("switched", "idigital")
    attach("operator", "apple")
    operator = attach("operator", "apple", override="ws://operator/browser")
    # Reusing the same endpoint as an operator override must also clear ownership.
    attach("same-endpoint", "apple")
    shared = attach("same-endpoint", "apple", override="ws://localhost/bu-named-apple@p1")
    stopped.clear()
    monkeypatch.setattr(lifecycle, "_release_session_resources", lambda key, info: bt._active_sessions.pop(key))
    lifecycle._cleanup_single_browser_session("bu-named-apple@p1")
    assert registry.get("switched") is rebound
    assert registry.get("operator") is operator
    assert registry.get("same-endpoint") is shared
    assert stopped == []


def test_unnamed_task_rebound_to_operator_survives_backend_cleanup(browsers, monkeypatch):
    registry, stopped, attach = browsers
    attach("same-task", "", profile="")
    operator = attach("same-task", "", profile="", override="ws://operator/browser")
    stopped.clear()
    monkeypatch.setattr(lifecycle, "_release_session_resources", lambda key, info: bt._active_sessions.pop(key))
    lifecycle._cleanup_single_browser_session("same-task")
    assert registry.get("same-task") is operator
    assert stopped == []


def test_cleanup_cancels_inflight_supervisor_attach(browsers, monkeypatch):
    registry, stopped, _ = browsers
    entered, release = threading.Event(), threading.Event()

    def start(supervisor, timeout=15):
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(bs.CDPSupervisor, "start", start)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(registry.get_or_start, "inflight", "ws://browser",
                                  browser_session_key="bu-named-apple@p1")
        try:
            assert entered.wait(5)
            registry.stop_session("bu-named-apple@p1")
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="closed during"):
            pending.result(timeout=5)
    assert registry.get("inflight") is None
    assert stopped == ["inflight"]
