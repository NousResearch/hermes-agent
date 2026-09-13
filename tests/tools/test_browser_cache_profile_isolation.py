"""Regression tests for #110032: the browser-session and computer_use backend caches are
keyed by session/task id alone, so in a multiplexed process (or any process serving more
than one profile) one profile can receive another profile's live browser session, CDP
supervisor, or cua backend, and teardown bookkeeping stays pinned to the first profile's
secret scope.

Expected behavior (issue contract): the identity of both caches includes the owning home
as ``(hermes_home_key, task_or_session_id)``; entries for both homes coexist under
different keys (eviction-on-mismatch is explicitly rejected because concurrent requests
would evict each other's live sessions).

No browser, display, driver or network is started — caches are seeded and read through
the production entry points with the process boundaries mocked.
"""

import contextlib
import time
from unittest.mock import Mock, patch

import pytest

from hermes_constants import hermes_home_key, reset_hermes_home_key_cache

import tools.browser_tool as bt
from tools import browser_tool_cdp as bt_cdp
from tools import browser_tool_lifecycle as bt_lifecycle
from tools import browser_tool_session as bt_session
from tools.computer_use import tool as cu


@pytest.fixture(autouse=True)
def _reset_caches():
    bt._active_sessions.clear()
    bt._session_last_activity.clear()
    bt._last_active_session_key.clear()
    bt._suspect_browser_sessions.clear()
    bt._recording_sessions.clear()
    bt._cleanup_failures.clear()
    cu._backends.clear()
    cu._backend_call_locks.clear()
    cu._backend_permission_modes.clear()
    yield
    bt._active_sessions.clear()
    bt._session_last_activity.clear()
    bt._last_active_session_key.clear()
    bt._suspect_browser_sessions.clear()
    bt._recording_sessions.clear()
    bt._cleanup_failures.clear()
    cu._backends.clear()
    cu._backend_call_locks.clear()
    cu._backend_permission_modes.clear()


def _two_homes(tmp_path):
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    return home_a, home_b


def _switch_home(monkeypatch, home):
    monkeypatch.setenv("HERMES_HOME", str(home))
    reset_hermes_home_key_cache()


def _session(name):
    return {"session_name": name, "bb_session_id": None, "features": {}}


def _session_creator(name):
    def _create(task_id, force_local):
        return _session(name)

    return _create


def _stub_browser_creation(monkeypatch, name):
    monkeypatch.setattr(bt_session, "_create_session_for_key", _session_creator(name))
    monkeypatch.setattr(bt_lifecycle, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(bt_cdp, "_ensure_cdp_supervisor", lambda _tid: None)


class TestBrowserSessionCacheProfileIsolation:
    def test_session_not_handed_to_another_home(self, tmp_path, monkeypatch):
        home_a, home_b = _two_homes(tmp_path)
        _switch_home(monkeypatch, home_a)
        _stub_browser_creation(monkeypatch, "SESSION_OWNED_BY_A")
        assert bt_session._get_session_info("default")["session_name"] == "SESSION_OWNED_BY_A"

        _switch_home(monkeypatch, home_b)
        _stub_browser_creation(monkeypatch, "SESSION_OWNED_BY_B")
        got = bt_session._get_session_info("default")
        assert got["session_name"] == "SESSION_OWNED_BY_B"

        # Entries for both homes coexist: switching back to A still serves A's
        # cached session without creating a replacement.
        _switch_home(monkeypatch, home_a)
        monkeypatch.setattr(
            bt_session, "_create_session_for_key",
            lambda task_id, force_local: pytest.fail("profile A's cached session was not reused"))
        assert bt_session._get_session_info("default")["session_name"] == "SESSION_OWNED_BY_A"

    def test_activity_touch_is_home_scoped(self, tmp_path, monkeypatch):
        home_a, home_b = _two_homes(tmp_path)
        _switch_home(monkeypatch, home_a)
        bt_lifecycle._update_session_activity("default")
        _switch_home(monkeypatch, home_b)
        bt_lifecycle._update_session_activity("default")

        key_a = (hermes_home_key(str(home_a)), "default")
        key_b = (hermes_home_key(str(home_b)), "default")
        assert key_a != key_b
        assert key_a in bt._session_last_activity
        assert key_b in bt._session_last_activity

    def test_janitor_selects_sessions_from_every_home(self, tmp_path, monkeypatch):
        home_a, home_b = _two_homes(tmp_path)
        _switch_home(monkeypatch, home_a)
        bt_lifecycle._update_session_activity("default")
        _switch_home(monkeypatch, home_b)
        bt_lifecycle._update_session_activity("default")
        for key in list(bt._session_last_activity):
            bt._session_last_activity[key] = time.time() - 100000.0

        cleaned = []
        monkeypatch.setattr(bt_lifecycle, "_session_owner_scope", contextlib.nullcontext)
        monkeypatch.setattr(bt_lifecycle, "cleanup_browser", lambda tid: cleaned.append(tid))
        bt_lifecycle._cleanup_inactive_browser_sessions()

        assert cleaned == ["default", "default"]

    def test_owner_scope_enters_the_entrys_home(self, tmp_path, monkeypatch):
        from hermes_constants import get_hermes_home_override

        home_a, home_b = _two_homes(tmp_path)
        key_a = (hermes_home_key(str(home_a)), "default")
        monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", lambda _p: None)
        monkeypatch.setattr("agent.secret_scope.build_profile_secret_scope", lambda _p: {})

        with bt_lifecycle._session_owner_scope(key_a):
            assert get_hermes_home_override() == hermes_home_key(str(home_a))

    def test_suspect_flag_is_home_scoped(self, tmp_path, monkeypatch):
        home_a, home_b = _two_homes(tmp_path)
        _switch_home(monkeypatch, home_a)
        bt._browser_session_backend("default").mark_suspect("browser command timed out")

        _switch_home(monkeypatch, home_b)
        assert bt._browser_session_backend("default").ensure_healthy() is True

        _switch_home(monkeypatch, home_a)
        monkeypatch.setattr(bt_lifecycle, "_cleanup_single_browser_session", lambda _tid: None)
        assert bt._browser_session_backend("default").ensure_healthy() is False

    def test_sidecar_cleanup_drops_last_active_binding(self, tmp_path, monkeypatch):
        """Cleaning a ``::local`` sidecar must drop the binding stored under the bare
        task's home-scoped key (the map is keyed by bare task id, values are session keys)."""
        home_a, _ = _two_homes(tmp_path)
        _switch_home(monkeypatch, home_a)
        bare_key = (hermes_home_key(str(home_a)), "default")
        bt._last_active_session_key[bare_key] = "default::local"

        monkeypatch.setattr(bt_lifecycle, "_cleanup_single_browser_session", lambda _tid: None)
        bt_lifecycle.cleanup_browser("default::local")

        assert bare_key not in bt._last_active_session_key

    def test_cdp_supervisor_registry_key_is_home_scoped(self, tmp_path, monkeypatch):
        home_a, home_b = _two_homes(tmp_path)
        _switch_home(monkeypatch, home_a)

        with patch.object(bt_cdp, "_get_cdp_override", return_value="ws://test"), \
                patch("tools.browser_supervisor.SUPERVISOR_REGISTRY") as registry:
            bt_cdp._ensure_cdp_supervisor("default")

        key_a = (hermes_home_key(str(home_a)), "default")
        assert registry.get_or_start.call_args.kwargs["task_id"] == key_a


class TestComputerUseBackendCacheProfileIsolation:
    def test_backend_not_handed_to_another_home(self, tmp_path, monkeypatch):
        home_a, home_b = _two_homes(tmp_path)
        _switch_home(monkeypatch, home_a)
        monkeypatch.setattr(cu, "_cua_permission_mode", lambda _sid: "standard")
        sentinel_a = Mock()
        monkeypatch.setattr(cu, "_new_backend", lambda mode: sentinel_a)
        assert cu._get_backend("shared") is sentinel_a

        _switch_home(monkeypatch, home_b)
        sentinel_b = Mock()
        monkeypatch.setattr(cu, "_new_backend", lambda mode: sentinel_b)
        got = cu._get_backend("shared")
        assert got is sentinel_b

        # Entries for both homes coexist: switching back to A still serves A's
        # cached backend without creating a replacement.
        _switch_home(monkeypatch, home_a)
        monkeypatch.setattr(
            cu, "_new_backend",
            lambda mode: pytest.fail("profile A's cached backend was not reused"))
        assert cu._get_backend("shared") is sentinel_a

    def test_release_session_only_releases_the_current_home(self, tmp_path, monkeypatch):
        home_a, home_b = _two_homes(tmp_path)
        _switch_home(monkeypatch, home_a)
        monkeypatch.setattr(cu, "_cua_permission_mode", lambda _sid: "standard")
        sentinel_a = Mock()
        monkeypatch.setattr(cu, "_new_backend", lambda mode: sentinel_a)
        assert cu._get_backend("shared") is sentinel_a

        _switch_home(monkeypatch, home_b)
        monkeypatch.setattr(cu, "_new_backend", lambda mode: Mock())
        assert cu._get_backend("shared") is not sentinel_a
        assert cu.release_computer_use_session("shared") is True

        key_b = (hermes_home_key(str(home_b)), "shared")
        assert key_b not in cu._backends
        assert key_b not in cu._backend_call_locks

        # Profile A's cached backend survived profile B's release.
        _switch_home(monkeypatch, home_a)
        monkeypatch.setattr(
            cu, "_new_backend",
            lambda mode: pytest.fail("profile A's cached backend was not reused"))
        assert cu._get_backend("shared") is sentinel_a
