"""Regression tests: the /loop and /heartbeat idle-poll tick drivers must bind the session's OWN
profile home before touching ``LoopManager``/``HeartbeatManager`` state.

``hermes_cli.goals._get_session_db()`` (which both ``hermes_cli.loops`` and ``hermes_cli.heartbeat``
delegate to) caches its ``SessionDB`` handle keyed on ``str(get_hermes_home())``. Under multiplex, a
Desktop/TUI session bound to a secondary profile (``session["profile_home"]``) must read/write THAT
profile's ``state.db`` for its loop/heartbeat state, not whatever profile happens to be the process's
current ``HERMES_HOME`` at poll time.

``tui_gateway/methods_tools.py::_cmd_goal`` already binds ``_session_profile_runtime_scope(session)``
before touching ``GoalManager``; the sibling ``/loop`` command handler (``_cmd_loop``) and the two
per-session poller-thread tick drivers in ``tui_gateway/session_notifications.py``
(``_maybe_fire_tui_heartbeat_tick`` / ``_maybe_fire_tui_loop_tick``) previously constructed
``LoopManager``/``HeartbeatManager`` directly, without binding the session's profile home first — a
secondary-profile session's due loop/heartbeat tick would silently look at (or clobber) the launch
profile's DB instead of its own, wedging automation for any session not on the "currently
home-scoped" profile.
"""

from __future__ import annotations

import importlib
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # tests/conftest.py's autouse fixture re-points hermes_state.DEFAULT_DB_PATH to ITS OWN
    # per-test fake home (a module-level constant pinned once collection has imported
    # hermes_state), so a bare SessionDB() would otherwise ignore both HERMES_HOME and any
    # context-local hermes_home override and always resolve to that pinned path. Undo the pin so
    # SessionDB() resolves at CALL time via get_hermes_home() again — the same restore
    # tests/tui_gateway/test_goal_command.py's identical fixture performs, for the identical
    # reason (goals.py / loops.py / heartbeat.py all bottom out in a bare SessionDB()).
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)

    # Bust the goal-module DB cache (shared by loops.py/heartbeat.py) so it re-resolves HERMES_HOME.
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.fixture()
def server(hermes_home, monkeypatch):
    # Mocks are scoped to the initial import only (see tests/tui_gateway/test_protocol.py).
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")

    monkeypatch.setattr(mod, "_hermes_home", hermes_home)
    monkeypatch.setattr(mod, "_cfg_cache", None)
    monkeypatch.setattr(mod, "_cfg_mtime", None)
    monkeypatch.setattr(mod, "_cfg_path", None)
    yield mod
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()


@pytest.fixture()
def secondary_profile(tmp_path):
    secondary = tmp_path / "profiles" / "secondary"
    secondary.mkdir(parents=True)
    (secondary / "config.yaml").write_text("", encoding="utf-8")
    return secondary


def _poller_session(session_key: str, profile_home) -> dict:
    return {
        "session_key": session_key,
        "profile_home": str(profile_home),
        "history_lock": threading.Lock(),
        "running": False,
    }


def _wire_stub_dispatch(server, monkeypatch):
    """Capture status.update emits and turn submissions without running a real turn."""
    emits: list = []
    submits: list = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emits.append((event, sid, payload)))

    def fake_submit(rid, sid, session, text, **kwargs):
        submits.append(text)
        with session["history_lock"]:
            session["running"] = False
        return True

    monkeypatch.setattr(server, "_run_prompt_submit", fake_submit)
    return emits, submits


class TestHeartbeatTickProfileScope:
    def test_heartbeat_tick_fires_from_the_session_own_profile(self, server, monkeypatch, secondary_profile):
        from hermes_cli.heartbeat import HeartbeatManager

        session_key = "hb-secondary-session"
        # Seed a DUE heartbeat directly in the secondary profile's DB.
        with server._session_profile_runtime_scope({"profile_home": str(secondary_profile)}):
            mgr = HeartbeatManager(session_id=session_key)
            mgr.set("check the deploy", 60)
            mgr.state.created_at -= 120  # backdate so it's immediately due
            from hermes_cli.heartbeat import save_heartbeat

            save_heartbeat(session_key, mgr.state)

        # The launch (default) profile has no heartbeat for this key.
        assert HeartbeatManager(session_id=session_key).state is None

        session = _poller_session(session_key, secondary_profile)
        emits, submits = _wire_stub_dispatch(server, monkeypatch)

        server._maybe_fire_tui_heartbeat_tick("sid-hb", session)

        assert submits, "heartbeat tick never dispatched — it did not see the secondary profile's due state"
        assert "check the deploy" in submits[0]
        assert any(e == "status.update" for e, _, _ in emits)

        # And the fire was persisted into the SECONDARY profile's DB, not the launch one.
        with server._session_profile_runtime_scope({"profile_home": str(secondary_profile)}):
            assert HeartbeatManager(session_id=session_key).state.fire_count == 1
        assert HeartbeatManager(session_id=session_key).state is None  # launch profile still untouched

    def test_heartbeat_tick_is_a_noop_without_a_profile_home(self, server, monkeypatch):
        """No profile_home (single-profile deployment): falls back to the process HERMES_HOME, unchanged."""
        from hermes_cli.heartbeat import HeartbeatManager, save_heartbeat

        session_key = "hb-launch-session"
        mgr = HeartbeatManager(session_id=session_key)
        mgr.set("launch profile heartbeat", 60)
        mgr.state.created_at -= 120
        save_heartbeat(session_key, mgr.state)

        session = {
            "session_key": session_key,
            "history_lock": threading.Lock(),
            "running": False,
        }
        emits, submits = _wire_stub_dispatch(server, monkeypatch)

        server._maybe_fire_tui_heartbeat_tick("sid-hb2", session)

        assert submits, "heartbeat tick must still fire for a session with no profile_home"
        assert "launch profile heartbeat" in submits[0]


class TestLoopTickProfileScope:
    def test_loop_tick_fires_from_the_session_own_profile(self, server, monkeypatch, secondary_profile):
        from hermes_cli.loops import LoopManager

        session_key = "loop-secondary-session"
        with server._session_profile_runtime_scope({"profile_home": str(secondary_profile)}):
            mgr = LoopManager(session_id=session_key)
            mgr.set("ping the service")  # self-paced: next_due_at = now, immediately due

        assert LoopManager(session_id=session_key).state is None  # launch profile: nothing there

        session = _poller_session(session_key, secondary_profile)
        emits, submits = _wire_stub_dispatch(server, monkeypatch)

        server._maybe_fire_tui_loop_tick("sid-loop", session)

        assert submits, "loop tick never dispatched — it did not see the secondary profile's due loop"
        assert "ping the service" in submits[0]

        with server._session_profile_runtime_scope({"profile_home": str(secondary_profile)}):
            assert LoopManager(session_id=session_key).state.ticks_fired == 1
        assert LoopManager(session_id=session_key).state is None  # launch profile untouched


class TestCmdLoopProfileScope:
    def test_cmd_loop_persists_into_the_session_own_profile(self, server, secondary_profile):
        from hermes_cli.loops import LoopManager

        sid = "sid-cmdloop"
        session_key = "loop-cmd-secondary-session"
        session = {
            "session_key": session_key,
            "profile_home": str(secondary_profile),
            "history": [],
            "history_lock": threading.Lock(),
            "history_version": 0,
            "running": False,
            "attached_images": [],
            "cols": 120,
        }
        server._sessions[sid] = session

        handler = server._methods["command.dispatch"]
        result = handler(1, {"session_id": sid, "name": "loop", "arg": "keep checking status"})

        assert "error" not in result, result

        # The loop landed in the SECONDARY profile's DB.
        with server._session_profile_runtime_scope(session):
            state = LoopManager(session_id=session_key).state
            assert state is not None
            assert "keep checking status" in state.prompt

        # ...and did NOT leak into the launch profile's DB.
        assert LoopManager(session_id=session_key).state is None
