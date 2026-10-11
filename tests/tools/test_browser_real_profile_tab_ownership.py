"""Tests for per-task tab ownership in the shared real-profile Chromium lane.

The real-profile lane is ONE shared Chromium: every task's agent-browser
daemon attaches to it via ``--cdp``, so a daemon's active tab follows the
browser's currently-active target and a rival task's tab steals another
task's commands (2026-10-11 live diagnosis). The fix: each task binds its
own tab (labeled with its unique session name) and every real-profile
command re-pins the daemon's active tab to that bound tab first; the
janitor closes the bound tab on teardown, including when the daemon is
already dead (HTTP CDP close of the targetId recorded at bind time).

Invariants under test:
- two tasks each bind their OWN tab; a later command re-pins to the
  owner's tab, never the browser's current one
- a gone bound tab is rebound (and the stale Chrome target closed)
- the janitor closes an orphaned bound tab even with a dead daemon
- the dispatch hook pins real-profile commands and skips others
"""
import json
from unittest.mock import Mock

import pytest
from tools import browser_tool_real_profile as bt_real_profile
from tools import browser_tool_session as bt_session


@pytest.fixture(autouse=True)
def _isolated_socket_tmp(tmp_path, monkeypatch):
    """Keep every socket-dir touch out of the real hermes home (home_io_guard)."""
    import tools.browser_tool as bt
    monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))


def _rp_info(session_name: str, cdp_url: str = "ws://127.0.0.1:41000/devtools/browser/x") -> dict:
    return {"session_name": session_name, "cdp_url": cdp_url,
            "features": {"local": True, "real_profile": True}}


def _ok(stdout_body: dict | None = None) -> Mock:
    body = {"success": True, "data": {}}
    if stdout_body is not None:
        body = stdout_body
    return Mock(returncode=0, stdout=json.dumps(body), stderr="")


def _fail(msg: str) -> Mock:
    return Mock(returncode=0, stdout=json.dumps({"success": False, "error": msg}), stderr="")


class FakeTabRecorder:
    """Records ``(session_name, cmd)`` per daemon and answers per script."""

    def __init__(self, script=None):
        self.calls: list[tuple[str, tuple]] = []
        self.script = script or (lambda session, cmd: _ok())

    def __call__(self, session_name, socket_dir, *cmd, timeout=15.0):
        self.calls.append((session_name, cmd))
        return self.script(session_name, cmd)


class TestTwoTasksBindOwnTabs:
    def test_each_task_binds_its_own_tab(self, monkeypatch):
        rec = FakeTabRecorder()
        monkeypatch.setattr(bt_real_profile, "_task_session_cmd", rec)
        a, b = _rp_info("rp_aaaa111111"), _rp_info("rp_bbbb222222")

        assert bt_real_profile._pin_task_tab(a) is None
        assert bt_real_profile._pin_task_tab(b) is None

        a_cmds = [c for s, c in rec.calls if s == "rp_aaaa111111"]
        b_cmds = [c for s, c in rec.calls if s == "rp_bbbb222222"]
        assert ("tab", "new", "--label", "rp_aaaa111111",
                "about:blank#hermes-rp-tab=rp_aaaa111111") in a_cmds
        assert ("tab", "new", "--label", "rp_bbbb222222",
                "about:blank#hermes-rp-tab=rp_bbbb222222") in b_cmds
        assert a["rp_tab_bound"] and b["rp_tab_bound"]

    def test_later_command_repins_own_tab_not_the_current_one(self, monkeypatch):
        """After task B's tab became the browser's active one, task A's next
        command must still be routed to A's bound tab (the re-pin), never to
        whatever tab is currently active in the shared browser."""
        rec = FakeTabRecorder()
        monkeypatch.setattr(bt_real_profile, "_task_session_cmd", rec)
        a, b = _rp_info("rp_aaaa111111"), _rp_info("rp_bbbb222222")
        bt_real_profile._pin_task_tab(a)
        bt_real_profile._pin_task_tab(b)  # B's tab is now the browser's active tab

        assert bt_real_profile._pin_task_tab(a) is None
        a_after = [c for s, c in rec.calls if s == "rp_aaaa111111"][1:]
        # Re-pin to A's OWN tab; no new tab opened, no touch of B's tab.
        assert ("tab", "rp_aaaa111111") in a_after
        assert not any(c[:2] == ("tab", "new") for c in a_after)
        assert not any("rp_bbbb222222" in str(c) for c in a_after)

    def test_bind_failure_fails_the_command(self, monkeypatch):
        rec = FakeTabRecorder(script=lambda s, c: _fail("browser gone"))
        monkeypatch.setattr(bt_real_profile, "_task_session_cmd", rec)
        err = bt_real_profile._pin_task_tab(_rp_info("rp_aaaa111111"))
        assert err and "could not open this task's own tab" in err


class TestRebindOnGoneTab:
    def test_gone_tab_is_rebound_and_stale_target_closed(self, monkeypatch):
        script = FakeTabRecorder()
        # Re-pin fails with "not found" → rebind path.
        script.script = lambda s, c: (_fail("no such tab: rp_aaaa111111")
                                      if c[:1] == ("tab",) and len(c) == 2 else _ok())
        monkeypatch.setattr(bt_real_profile, "_task_session_cmd", script)
        monkeypatch.setattr(bt_real_profile, "_cdp_target_for_marker",
                            lambda http, url: "TID-OLD")
        closed = []
        monkeypatch.setattr(bt_real_profile, "_cdp_close_target",
                            lambda http, tid: closed.append((http, tid)) or True)
        info = _rp_info("rp_aaaa111111")
        assert bt_real_profile._pin_task_tab(info) is None
        assert info["rp_tab_target"] == "TID-OLD"

        script.script = lambda s, c: (_fail("no such tab: rp_aaaa111111")
                                      if c[:2] == ("tab", "rp_aaaa111111") else _ok())
        monkeypatch.setattr(bt_real_profile, "_cdp_target_for_marker",
                            lambda http, url: "TID-NEW")
        assert bt_real_profile._pin_task_tab(info) is None

        assert closed == [("http://127.0.0.1:41000", "TID-OLD")]
        assert info["rp_tab_target"] == "TID-NEW"
        rebinds = [c for s, c in script.calls if c[:2] == ("tab", "new")]
        assert len(rebinds) == 2  # original bind + the fallback rebind

    def test_transient_pin_failure_does_not_rebind(self, monkeypatch):
        """A non-'not found' failure must fail the command, not silently leak
        a second bound tab next to the live one."""
        script = FakeTabRecorder()
        monkeypatch.setattr(bt_real_profile, "_task_session_cmd", script)
        info = _rp_info("rp_aaaa111111")
        assert bt_real_profile._pin_task_tab(info) is None  # bind ok
        script.script = lambda s, c: _fail("page is navigating")
        err = bt_real_profile._pin_task_tab(info)
        assert err and "navigating" in err
        assert info.get("rp_tab_bound") is True  # binding kept


class TestJanitorClosesOrphanedTab:
    def test_close_uses_daemon_tab_close(self, monkeypatch):
        rec = FakeTabRecorder()
        monkeypatch.setattr(bt_real_profile, "_task_session_cmd", rec)
        fallback = []
        monkeypatch.setattr(bt_real_profile, "_cdp_close_target",
                            lambda http, tid: fallback.append((http, tid)) or True)
        bt_real_profile.close_task_tab(_rp_info("rp_aaaa111111"))
        assert ("rp_aaaa111111", ("tab", "close", "rp_aaaa111111")) in rec.calls
        assert fallback == []  # daemon alive: no HTTP fallback

    def test_close_falls_back_to_http_cdp_when_daemon_dead(self, monkeypatch):
        """Crashed task: the daemon is gone but its bound tab lives on in the
        shared Chrome — the janitor reaps it via the recorded targetId."""
        rec = FakeTabRecorder(script=lambda s, c: None)  # daemon unreachable
        monkeypatch.setattr(bt_real_profile, "_task_session_cmd", rec)
        fallback = []
        monkeypatch.setattr(bt_real_profile, "_cdp_close_target",
                            lambda http, tid: fallback.append((http, tid)) or True)
        info = _rp_info("rp_aaaa111111")
        info["rp_tab_target"] = "TID-ORPHAN"
        bt_real_profile.close_task_tab(info)
        assert fallback == [("http://127.0.0.1:41000", "TID-ORPHAN")]

    def test_release_session_resources_closes_bound_tab_before_daemon_kill(self, monkeypatch):
        from tools import browser_tool_lifecycle as bt_lifecycle
        import tools.browser_tool as bt

        closed = []
        monkeypatch.setattr(bt_real_profile, "close_task_tab",
                            lambda info: closed.append(info["session_name"]))
        monkeypatch.setattr(bt_lifecycle, "_kill_verified_daemon", lambda *a, **k: True)
        monkeypatch.setattr(bt_lifecycle, "_forget_session_tracking", lambda *a, **k: None)
        monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: "/nonexistent-socket-dir")
        info = _rp_info("rp_aaaa111111")
        bt_lifecycle._release_session_resources("task-1", info)
        assert closed == ["rp_aaaa111111"]

    def test_release_session_resources_skips_non_real_profile(self, monkeypatch):
        from tools import browser_tool_lifecycle as bt_lifecycle
        import tools.browser_tool as bt

        closed = []
        monkeypatch.setattr(bt_real_profile, "close_task_tab",
                            lambda info: closed.append(info["session_name"]))
        monkeypatch.setattr(bt_lifecycle, "_kill_verified_daemon", lambda *a, **k: True)
        monkeypatch.setattr(bt_lifecycle, "_forget_session_tracking", lambda *a, **k: None)
        monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: "/nonexistent-socket-dir")
        bt_lifecycle._release_session_resources(
            "task-1", {"session_name": "h_abcdef1234", "features": {"local": True}})
        assert closed == []


class TestDispatchHook:
    def _run_dispatch(self, info, command, monkeypatch, pin_result=None, tmp_path=None):
        import tools.browser_tool as bt
        import tools.browser_tool_cloud as bt_cloud

        spawned = []
        pins = []
        if tmp_path is not None:
            monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
        monkeypatch.setattr(bt_session, "_browser_command_preflight",
                            lambda: {"browser_cmd": "agent-browser"})
        monkeypatch.setattr(bt_session, "_spawn_and_collect",
                            lambda *a, **k: spawned.append(a) or {"success": True, "data": {}})
        monkeypatch.setattr(bt_session, "_unwrap_batch_result", lambda r, c: r)
        monkeypatch.setattr(bt_cloud, "_get_browser_engine", lambda: "auto")
        import tools.browser_tool_cdp as bt_cdp
        monkeypatch.setattr(bt_cdp, "_ensure_cdp_supervisor", lambda *a, **k: None)
        real_pin = bt_real_profile._pin_task_tab

        def spy(info_):
            pins.append(info_["session_name"])
            if pin_result is not None:
                return pin_result
            # Exercise the real pin logic but with the agent-browser CLI faked
            # (the real CLI would touch the real hermes home via its manifest).
            monkeypatch.setattr(bt_real_profile, "_task_session_cmd", FakeTabRecorder())
            monkeypatch.setattr(bt_real_profile, "_cdp_target_for_marker", lambda http, url: None)
            return real_pin(info_)

        monkeypatch.setattr(bt_real_profile, "_pin_task_tab", spy)
        engine, result = bt_session._dispatch_browser_command(
            "task-1", info, "agent-browser", command, [], 5, None)
        return engine, result, spawned, pins

    def test_real_profile_command_is_pinned(self, monkeypatch, tmp_path):
        _, result, spawned, pins = self._run_dispatch(_rp_info("rp_aaaa111111"), "snapshot", monkeypatch, tmp_path=tmp_path)
        assert pins == ["rp_aaaa111111"]
        assert result.get("success") and len(spawned) == 1

    def test_pin_failure_fails_the_command_without_spawning(self, monkeypatch):
        _, result, spawned, pins = self._run_dispatch(
            _rp_info("rp_aaaa111111"), "snapshot", monkeypatch, pin_result="no tab")
        assert pins == ["rp_aaaa111111"]
        assert result.get("success") is False and "no tab" in result.get("error", "")
        assert spawned == []

    def test_close_is_not_pinned(self, monkeypatch):
        _, _result, spawned, pins = self._run_dispatch(_rp_info("rp_aaaa111111"), "close", monkeypatch)
        assert pins == [] and len(spawned) == 1

    def test_plain_local_session_is_not_pinned(self, monkeypatch):
        _, _result, spawned, pins = self._run_dispatch(
            {"session_name": "h_abcdef1234", "features": {"local": True}}, "snapshot", monkeypatch)
        assert pins == [] and len(spawned) == 1


class TestTargetDiscovery:
    def test_marker_target_found_in_json_list(self, monkeypatch):
        import tools.browser_tool_real_profile as rp

        resp = Mock(status_code=200)
        resp.json.return_value = [
            {"type": "page", "url": "about:blank", "id": "TID-OTHER"},
            {"type": "page", "url": rp._RP_TAB_MARKER + "rp_aaaa111111", "id": "TID-MINE"},
        ]
        # Marker URL: about:blank#<marker><name>
        url = f"about:blank#{rp._RP_TAB_MARKER}rp_aaaa111111"
        resp.json.return_value[1]["url"] = url
        monkeypatch.setattr("requests.get", lambda *a, **k: resp)
        monkeypatch.setattr(rp.loopback_request_kwargs, "__call__", lambda self, u: {})
        assert rp._cdp_target_for_marker("http://127.0.0.1:41000", url) == "TID-MINE"

    def test_http_root_derived_from_ws_cdp_url(self):
        assert bt_real_profile._rp_http_cdp_root(
            _rp_info("rp_aaaa111111")) == "http://127.0.0.1:41000"
        assert bt_real_profile._rp_http_cdp_root(_rp_info("rp_aaaa111111", cdp_url="")) is None


@pytest.mark.parametrize("proc", [None, Mock(returncode=1, stdout="{}", stderr="")])
def test_tab_cmd_ok_false_on_dead_cli(proc):
    assert bt_real_profile._tab_cmd_ok(proc) is False
