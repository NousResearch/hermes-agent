"""Identical caller session IDs must not share a browser between profile homes.

Session caches, activity tracking, cleanup and supervisor ownership are real.
Only external browser creation/commands and the supervisor's network thread are
substituted; the matching live-browser canary exercises those external edges.
"""
from types import SimpleNamespace

import pytest

from agent import secret_scope
from gateway.run import _profile_runtime_scope


@pytest.fixture
def homes(tmp_path):
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    result = []
    for name in ("a", "b"):
        home = tmp_path / name
        home.mkdir()
        (home / "config.yaml").write_text(
            "browser:\n  backend: off\n  cloud_provider: local\n", encoding="utf-8")
        result.append(home)
    try:
        yield result
    finally:
        secret_scope.set_multiplex_active(previous)


def test_same_caller_session_uses_separate_cache_and_cleanup(homes, monkeypatch):
    from tools import browser_tool as bt
    from tools import browser_tool_cdp as cdp
    from tools import browser_tool_lifecycle as lifecycle
    from tools import browser_tool_session as sessions

    for name in ("_active_sessions", "_last_active_session_key", "_session_last_activity",
                 "_session_owner_homes", "_cleanup_failures", "_suspect_browser_sessions"):
        monkeypatch.setattr(bt, name, {})
    monkeypatch.setattr(bt, "_recording_sessions", set())
    monkeypatch.setattr(lifecycle, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(cdp, "_ensure_cdp_supervisor", lambda *_args: None)
    monkeypatch.setattr(cdp, "_stop_cdp_supervisor", lambda *_args: None)
    created = []

    def create(_key, _force_local):
        value = {"session_name": "", "bb_session_id": None,
                 "cdp_url": f"ws://browser-fixture/{len(created)}", "features": {"cdp_override": True}}
        created.append(value)
        return value

    monkeypatch.setattr(sessions, "_create_session_for_key", create)
    monkeypatch.setattr(sessions, "_run_browser_command", lambda *_args, **_kwargs: {"success": True})
    a, b = homes
    with _profile_runtime_scope(a):
        first = sessions._get_session_info("shared-session")
        first["synthetic_marker"] = "only-a"
    with _profile_runtime_scope(b):
        second = sessions._get_session_info("shared-session")
        assert second is not first
        assert "synthetic_marker" not in second
        assert second["session_key"] != first["session_key"]
        lifecycle.cleanup_browser("shared-session")
        assert second["session_key"] not in bt._active_sessions
    with _profile_runtime_scope(a):
        assert sessions._get_session_info("shared-session") is first
        assert first["synthetic_marker"] == "only-a"
        lifecycle.cleanup_browser("shared-session")
    assert len(created) == 2
    assert not bt._active_sessions


@pytest.fixture
def supervisors(monkeypatch):
    from tools import browser_supervisor as module

    class NetworkEndpoint:
        def __init__(self, *, task_id, cdp_url, **_kwargs):
            self.task_id, self.cdp_url = task_id, cdp_url
            self.stopped = False
            self._thread = SimpleNamespace(is_alive=lambda: not self.stopped)
            self._loop = SimpleNamespace(is_running=lambda: not self.stopped)

        def start(self, **_kwargs):
            pass

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(module, "CDPSupervisor", NetworkEndpoint)
    registry = module._SupervisorRegistry()
    yield registry
    registry.stop_all()


def test_same_caller_supervisors_and_background_cleanup_are_isolated(homes, supervisors):
    a, b = homes
    with _profile_runtime_scope(a):
        first = supervisors.get_or_start("shared-session", "ws://browser-fixture/a")
    with _profile_runtime_scope(b):
        assert supervisors.get("shared-session") is None
        second = supervisors.get_or_start("shared-session", "ws://browser-fixture/b")
        assert second is not first and not first.stopped
    # Background teardown retains the resolved internal identity without an
    # active profile override; it must stop B and keep A's owner intact.
    supervisors.stop(second.task_id)
    assert second.stopped and not first.stopped
    with _profile_runtime_scope(a):
        assert supervisors.get("shared-session") is first
    with _profile_runtime_scope(b):
        assert supervisors.get("shared-session") is None


def test_raw_caller_id_cannot_impersonate_another_internal_browser_key(homes, supervisors):
    a, b = homes
    with _profile_runtime_scope(a):
        first = supervisors.get_or_start("shared-session", "ws://browser-fixture/a")
    with _profile_runtime_scope(b):
        assert supervisors.get(str(first.task_id)) is None
        supervisors.stop(str(first.task_id))
    assert not first.stopped
