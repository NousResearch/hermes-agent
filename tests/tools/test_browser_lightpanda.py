"""Tests for the Lightpanda engine (``browser.engine: lightpanda``) in the browser session layer."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from tools.bot_desktop import placement
from tools import browser_launcher as launcher
from tools import browser_lightpanda as lp
from tools import browser_session as bs
from tools import browser_session_cdp as bs_cdp
from tools import browser_session_cloud as bs_cloud
from tools import browser_session_lifecycle as lifecycle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_engine_cache():
    """Reset the per-profile config memo (engine included) before and after each test."""
    bs_cloud.reset_caches()
    yield
    bs_cloud.reset_caches()


# ---------------------------------------------------------------------------
# get_browser_engine
# ---------------------------------------------------------------------------

class TestGetBrowserEngine:
    """Test engine resolution from config."""


    def test_config_lightpanda(self):
        """Config browser.engine = 'lightpanda' is respected."""
        cfg = {"browser": {"engine": "lightpanda"}}
        with patch("hermes_cli.config_effective.load_user_config_effective", return_value=cfg):
            assert bs_cloud.get_browser_engine() == "lightpanda"


# ---------------------------------------------------------------------------
# Lightpanda never needs local Chromium
# ---------------------------------------------------------------------------

class _FakeServer:
    def __init__(self, port=4321, alive=True):
        self.port = port
        self.cdp_url = f"http://127.0.0.1:{port}"
        self._alive = alive

    def is_alive(self):
        return self._alive


class TestLightpandaRequirements:
    """Lightpanda should serve browser sessions without local Chromium."""

    def _local(self, monkeypatch, engine_lightpanda):
        monkeypatch.setattr("tools.browser_session_real_profile.real_profile_cdp", lambda: (None, None))
        monkeypatch.setattr(lp, "using_lightpanda_engine", lambda: engine_lightpanda)
        monkeypatch.setattr(bs_cloud, "is_local_backend", lambda: True)
        monkeypatch.setattr("tools.bot_desktop.runtime.tool_placement", lambda: placement.GATEWAY)  # host browser
        monkeypatch.setattr(launcher, "chromium_ready", lambda: (False, "Chromium browser is missing."))
        monkeypatch.setattr(lp, "launch_lightpanda", lambda name, **kw: (_FakeServer(), None))

    def test_lightpanda_local_mode_does_not_require_chromium(self, monkeypatch):
        self._local(monkeypatch, engine_lightpanda=True)
        monkeypatch.setattr(launcher, "launch_chromium",
                            lambda *a, **kw: pytest.fail("Lightpanda must not launch Chromium"))
        info = bs._create_local_session("task-1")
        assert info["features"] == {"local": True, "lightpanda": True}

    def test_chrome_local_mode_still_requires_chromium(self, monkeypatch):
        self._local(monkeypatch, engine_lightpanda=False)
        with pytest.raises(RuntimeError, match="Chromium browser is missing"):
            bs._create_local_session("task-1")


# ---------------------------------------------------------------------------
# Hybrid ::local sidecar honours the engine
# ---------------------------------------------------------------------------

class TestEngineOverride:

    def test_hybrid_local_sidecar_uses_lightpanda_even_with_cloud_provider(self, monkeypatch):
        """A task::local sidecar is local even when global cloud config exists, so the engine applies."""
        mock_provider = MagicMock()
        launched = []
        monkeypatch.setattr(bs, "_active_sessions", {})
        monkeypatch.setattr(lifecycle, "start_cleanup_thread", lambda: None)
        monkeypatch.setattr(lifecycle, "update_session_activity", lambda key: None)
        monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: mock_provider)
        monkeypatch.setattr(bs_cdp, "get_cdp_override", lambda: "")
        monkeypatch.setattr(lp, "using_lightpanda_engine", lambda: True)
        monkeypatch.setattr(bs_cloud, "is_local_backend", lambda: False)
        monkeypatch.setattr(lp, "launch_lightpanda",
                            lambda name, **kw: launched.append(name) or (_FakeServer(), None))

        info = bs.get_session_info("task::local")

        assert len(launched) == 1
        assert info["features"]["lightpanda"] is True
        mock_provider.create_session.assert_not_called()


# ---------------------------------------------------------------------------
# lightpanda_engine_status — is the engine in effect, or shadowed?
# ---------------------------------------------------------------------------

class TestLightpandaEngineStatus:
    def _gates(self, monkeypatch, **overrides):
        gates = dict(
            using_lightpanda_engine=lambda: True,
            get_cdp_override_raw=lambda: "",
            get_cloud_provider=lambda: None,
            use_real_profile=lambda: False,
        )
        gates.update(overrides)
        homes = {
            "using_lightpanda_engine": lp, "get_cdp_override_raw": bs_cdp,
            "get_cloud_provider": bs_cloud, "use_real_profile": bs_cloud,
        }
        for name, fn in gates.items():
            monkeypatch.setattr(homes[name], name, fn)
        monkeypatch.setattr(
            "tools.browser_use_cli.is_legacy_browser_use_cloud_config", lambda cfg: False
        )

    def test_not_lightpanda(self, monkeypatch):
        self._gates(monkeypatch, using_lightpanda_engine=lambda: False)
        assert lp.lightpanda_engine_status() == (False, "")

    def test_used_when_nothing_shadows_it(self, monkeypatch):
        self._gates(monkeypatch)
        used, reason = lp.lightpanda_engine_status()
        assert used is True
        assert "lightpanda serve" in reason

    def test_shadowed_by_cdp_override(self, monkeypatch):
        self._gates(monkeypatch, get_cdp_override_raw=lambda: "ws://x")
        used, reason = lp.lightpanda_engine_status()
        assert used is False and "CDP override" in reason

    def test_shadowed_by_cloud_provider(self, monkeypatch):
        provider = MagicMock()
        provider.display_name = "Browserbase"
        self._gates(monkeypatch, get_cloud_provider=lambda: provider)
        used, reason = lp.lightpanda_engine_status()
        assert used is False and "Browserbase" in reason

    def test_shadowed_by_legacy_browser_use_cloud(self, monkeypatch):
        self._gates(monkeypatch)
        monkeypatch.setattr(
            "tools.browser_use_cli.is_legacy_browser_use_cloud_config", lambda cfg: True
        )
        used, reason = lp.lightpanda_engine_status()
        assert used is False and "Browser Use cloud" in reason

    def test_shadowed_by_real_profile(self, monkeypatch):
        self._gates(monkeypatch, use_real_profile=lambda: True)
        used, reason = lp.lightpanda_engine_status()
        assert used is False and "use_real_profile" in reason

    def test_real_profile_wins_over_cloud_provider(self, monkeypatch):
        """browser_exec resolves real-profile before the backend, so with
        both set the real-profile toggle is the actual shadow."""
        provider = MagicMock()
        provider.display_name = "Browserbase"
        self._gates(
            monkeypatch,
            use_real_profile=lambda: True,
            get_cloud_provider=lambda: provider,
        )
        used, reason = lp.lightpanda_engine_status()
        assert used is False and "use_real_profile" in reason


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

class TestLightpandaSessionCreation:
    def _common(self, monkeypatch, *, local_backend=True, launch=None):
        calls = []

        def fake_launch(session_name, *, block_private_networks=False):
            calls.append((session_name, block_private_networks))
            if launch is not None:
                return launch
            return _FakeServer(), None

        monkeypatch.setattr("tools.browser_session_real_profile.real_profile_cdp", lambda: (None, None))
        monkeypatch.setattr(lp, "using_lightpanda_engine", lambda: True)
        monkeypatch.setattr(bs_cloud, "is_local_backend", lambda: local_backend)
        monkeypatch.setattr(lp, "launch_lightpanda", fake_launch)
        return calls

    def test_spawns_lightpanda(self, monkeypatch):
        calls = self._common(monkeypatch)
        info = bs._create_local_session("task-1")
        assert info["session_name"].startswith("lp_")
        assert info["cdp_url"] == "http://127.0.0.1:4321"
        assert info["features"] == {"local": True, "lightpanda": True}
        assert info["bb_session_id"] is None
        assert calls == [(info["session_name"], False)]

    def test_blocks_private_networks_for_containerised_terminal(self, monkeypatch):
        calls = self._common(monkeypatch, local_backend=False)
        bs._create_local_session("task-1")
        assert calls[0][1] is True

    def test_consented_real_profile_with_lightpanda_reports_the_conflict(self, monkeypatch):
        """Real profile precedes Lightpanda, but Lightpanda cannot load a Chromium profile: the session is
        Lightpanda and carries the fallback note naming the engine conflict."""
        calls = []
        monkeypatch.setattr(lp, "using_lightpanda_engine", lambda: True)
        monkeypatch.setattr(bs_cloud, "is_local_backend", lambda: True)
        monkeypatch.setattr(bs_cloud, "use_real_profile", lambda: True)
        monkeypatch.setattr(lp, "launch_lightpanda",
                            lambda name, **kw: calls.append((name, False)) or (_FakeServer(), None))
        info = bs._create_local_session("task-1")
        assert info["features"] == {"local": True, "lightpanda": True}
        assert "lightpanda" in info["real_profile_unavailable"]
        assert len(calls) == 1

    def test_launch_failure_raises(self, monkeypatch):
        self._common(monkeypatch, launch=(None, "no lightpanda binary was found"))
        with pytest.raises(RuntimeError, match="no lightpanda binary"):
            bs._create_local_session("task-1")


class TestLightpandaSessionLifecycle:
    def setup_method(self):
        self.orig_sessions = bs._active_sessions.copy()
        self.orig_activity = bs._session_last_activity.copy()
        self.orig_cleanup_done = bs._cleanup_done
        bs._active_sessions.clear()
        bs._session_last_activity.clear()

    def teardown_method(self):
        bs._active_sessions.clear()
        bs._active_sessions.update(self.orig_sessions)
        bs._session_last_activity.clear()
        bs._session_last_activity.update(self.orig_activity)
        bs._cleanup_done = self.orig_cleanup_done

    def _seed(self, key="task-1", name="lp_dead"):
        info = {
            "session_name": name,
            "bb_session_id": None,
            "cdp_url": "http://127.0.0.1:1",
            "features": {"local": True, "lightpanda": True},
        }
        bs._active_sessions[key] = info
        bs._session_last_activity[key] = 1.0
        return info

    def test_dead_process_is_detected(self, monkeypatch):
        info = self._seed()
        monkeypatch.setattr(lp, "get_server", lambda name: None)
        assert bs._backend_dead(info) is True
        monkeypatch.setattr(lp, "get_server", lambda name: _FakeServer(alive=False))
        assert bs._backend_dead(info) is True
        monkeypatch.setattr(lp, "get_server", lambda name: _FakeServer())
        assert bs._backend_dead(info) is False
        assert bs._backend_dead({"features": {"local": True}}) is False

    def test_get_session_info_respawns_dead_lightpanda(self, monkeypatch):
        stale = self._seed()
        fresh = {
            "session_name": "lp_fresh",
            "bb_session_id": None,
            "cdp_url": "http://127.0.0.1:2",
            "features": {"local": True, "lightpanda": True},
        }
        cleaned = []

        def fake_cleanup(key):
            cleaned.append(key)
            bs._active_sessions.pop(key, None)

        monkeypatch.setattr(lifecycle, "start_cleanup_thread", lambda: None)
        monkeypatch.setattr(lp, "get_server", lambda name: None)
        monkeypatch.setattr(lifecycle, "cleanup_single_session", fake_cleanup)
        monkeypatch.setattr(bs_cdp, "get_cdp_override", lambda: "")
        monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: None)
        monkeypatch.setattr(bs, "_create_local_session", lambda *a, **k: fresh)
        supervised = []
        monkeypatch.setattr(bs_cdp, "ensure_cdp_supervisor", supervised.append)

        info = bs.get_session_info("task-1")
        assert cleaned == ["task-1"]
        assert info["session_name"] == "lp_fresh"
        assert bs._active_sessions["task-1"]["session_name"] == "lp_fresh"
        assert info["session_name"] != stale["session_name"]
        # A local (Lightpanda) session never gets the session-layer supervisor;
        # browser_exec attaches one per call.
        assert supervised == []

    def test_cleanup_stops_lightpanda_without_a_chromium_stop(self, monkeypatch):
        self._seed()
        stopped = []
        monkeypatch.setattr(lp, "stop_lightpanda", stopped.append)
        with patch("tools.browser_launcher.stop_browser") as chromium_stop:
            lifecycle.cleanup_browser("task-1")
        chromium_stop.assert_not_called()
        assert stopped == ["lp_dead"]
        assert "task-1" not in bs._active_sessions
        assert "task-1" not in bs._session_last_activity

    def test_emergency_cleanup_stops_all_lightpanda(self, monkeypatch):
        bs._cleanup_done = False
        with patch("tools.browser_lightpanda.stop_all_lightpanda") as stop_all, \
             patch("tools.browser_launcher.stop_all_browsers"), \
             patch("tools.browser_session_lifecycle.reap_orphaned_browser_sessions"):
            lifecycle.emergency_cleanup_all_sessions()
        stop_all.assert_called_once()

    def test_orphan_reaper_sweeps_lightpanda_records(self):
        with patch("tools.browser_lightpanda.reap_orphaned_lightpanda", return_value=0) as reap, \
             patch("tools.browser_launcher.reap_orphaned_browsers", return_value=0):
            lifecycle.reap_orphaned_browser_sessions()
        reap.assert_called_once()

    def test_orphan_reaper_survives_non_dict_record(self, tmp_path, monkeypatch):
        """A parseable-but-non-object state file is swept like an unreadable one —
        never an AttributeError that wedges the reaper."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        state_dir = lp._state_dir()
        bad = state_dir / "broken.json"
        bad.write_text('"not a record"', encoding="utf-8")
        dead_owner = state_dir / "dead.json"
        dead_owner.write_text(json.dumps({"owner_pid": 2**22 + 7}), encoding="utf-8")
        assert lp.reap_orphaned_lightpanda() == 0
        assert not bad.exists() and not dead_owner.exists()

    def test_orphan_sweep_mid_record_write_keeps_the_live_record(self, tmp_path, monkeypatch):
        """A sweep running while the owner writes its record must not see a torn (empty) file:
        it would unlink it as corrupt and the live server would never be reaped after a crash."""
        from tests.file_write_observer import observe_file_writes

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        server = lp.LightpandaServer("live", 9222, MagicMock(pid=os.getpid()), "x.log")
        monkeypatch.setitem(lp._servers, "live", server)
        with observe_file_writes(lp.reap_orphaned_lightpanda):
            lp._write_record(server)
        record = lp._state_dir() / "live.json"
        assert record.exists()
        assert json.loads(record.read_text(encoding="utf-8"))["port"] == 9222
