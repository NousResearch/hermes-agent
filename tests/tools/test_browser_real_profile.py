"""Tests for real-profile browsing: resolvers, snapshot, launch routing, consent.

The consent path never drives the live default profile: it snapshots into
``~/.hermes/browser-profile/<browser>/`` and launches the user's real binary
on the copy with a devtools port (see hermes_cli.browser_connect). These tests
exercise the real functions with real file I/O wherever possible — the mocks
are limited to OS detection and process launch.
"""
import json
import os
import ntpath
from unittest.mock import Mock, patch

import pytest


class TestRealProfileResolvers:
    def test_data_dir_windows(self):
        import hermes_cli.browser_connect as bc
        with patch.dict(os.environ, {"LOCALAPPDATA": r"C:\Users\T\AppData\Local"}, clear=False):
            got = bc.real_profile_data_dir("chrome", "Windows")
        # Use ntpath basename checks so this passes on Linux CI too.
        assert got.endswith(ntpath.join("Google", "Chrome", "User Data")) or got.endswith(
            "Google\\Chrome\\User Data"
        )

    def test_data_dir_linux_edge(self):
        import hermes_cli.browser_connect as bc
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/home/t/.config"}, clear=False):
            got = bc.real_profile_data_dir("edge", "Linux")
        assert got == "/home/t/.config/microsoft-edge"

    def test_data_dir_unknown_browser_is_none(self):
        import hermes_cli.browser_connect as bc
        assert bc.real_profile_data_dir("firefox", "Windows") is None

    def test_detect_default_windows_progid_maps(self):
        import hermes_cli.browser_connect as bc
        # Non-Windows host: _detect_default_windows short-circuits via winreg
        # ImportError → None. Assert the ProgId map itself is correct instead.
        m = dict(bc._WINDOWS_PROGID_MAP)
        assert m["chromehtml"] == "chrome"
        assert m["msedgehtm"] == "edge"
        assert m["bravehtml"] == "brave"

    def test_detect_default_non_chromium_is_none(self):
        import hermes_cli.browser_connect as bc
        with patch.object(bc, "_detect_default_linux", return_value=None):
            assert bc.detect_default_chromium("Linux") is None


class TestSnapshotRealProfile:
    """Real file I/O: the snapshot copier against a synthetic profile tree."""

    def _make_profile(self, root):
        """Build a minimal real-looking Chromium user-data-dir."""
        (root / "Default" / "Network").mkdir(parents=True)
        (root / "Default" / "Cache" / "Cache_Data").mkdir(parents=True)
        (root / "Code Cache" / "js").mkdir(parents=True)
        (root / "Crashpad").mkdir()
        (root / "Local State").write_text('{"os_crypt": {}}')
        (root / "Default" / "Cookies").write_text("sqlite-cookies")
        (root / "Default" / "Network" / "Cookies").write_text("sqlite-net-cookies")
        (root / "Default" / "Login Data").write_text("sqlite-logins")
        (root / "Default" / "Preferences").write_text("{}")
        (root / "Default" / "Cache" / "Cache_Data" / "big").write_text("x" * 1000)
        (root / "Code Cache" / "js" / "blob").write_text("y" * 1000)
        (root / "Crashpad" / "dump").write_text("z")
        # Live-instance leftovers that must never reach the copy
        os.symlink("dead-target-1", root / "SingletonLock")
        return root

    def test_fresh_snapshot_copies_auth_and_skips_caches(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc
        src = self._make_profile(tmp_path / "real")
        home = tmp_path / "hermes-home"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)

        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert err is None
        assert dst == str(home / "browser-profile" / "chrome")
        # Auth files present
        assert (home / "browser-profile" / "chrome" / "Default" / "Cookies").read_text() == "sqlite-cookies"
        assert (home / "browser-profile" / "chrome" / "Default" / "Network" / "Cookies").exists()
        assert (home / "browser-profile" / "chrome" / "Default" / "Login Data").exists()
        assert (home / "browser-profile" / "chrome" / "Local State").exists()
        # Caches, crash dirs, singleton leftovers excluded
        assert not (home / "browser-profile" / "chrome" / "Default" / "Cache").exists()
        assert not (home / "browser-profile" / "chrome" / "Code Cache").exists()
        assert not (home / "browser-profile" / "chrome" / "Crashpad").exists()
        assert not (home / "browser-profile" / "chrome" / "SingletonLock").exists()

    def test_existing_snapshot_refreshes_auth_files_only(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc
        src = self._make_profile(tmp_path / "real")
        home = tmp_path / "hermes-home"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)

        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert err is None
        # Simulate: user logs into a new site in their own browser, and the
        # copy has drifted state that must survive (History not in refresh set).
        (src / "Default" / "Cookies").write_text("sqlite-cookies-v2")
        copy_history = home / "browser-profile" / "chrome" / "Default" / "History"
        copy_history.write_text("agent-session-history")

        dst2, err2 = bc.snapshot_real_profile("chrome", src=str(src))
        assert err2 is None and dst2 == dst
        assert (home / "browser-profile" / "chrome" / "Default" / "Cookies").read_text() == "sqlite-cookies-v2"
        assert copy_history.read_text() == "agent-session-history"

    def test_missing_source_fails_closed(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc
        monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path / "hh")
        dst, err = bc.snapshot_real_profile("chrome", src=str(tmp_path / "nope"))
        assert dst is None
        assert err and "was not found" in err


class TestRealProfileCdpLaunch:
    """The agent-browser-based launcher in browser_tool._real_profile_cdp."""

    def _reset(self):
        import tools.browser_tool as bt
        bt._real_profile_cdp_cache.clear()

    def test_consent_off_is_noop(self):
        import tools.browser_tool as bt
        self._reset()
        with patch.object(bt, "_use_real_profile", return_value=False):
            cdp, err = bt._real_profile_cdp()
        assert cdp is None and err is None

    def test_non_chromium_default_fails_closed(self):
        import tools.browser_tool as bt
        self._reset()
        with patch.object(bt, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value=None):
            cdp, err = bt._real_profile_cdp()
        assert cdp is None
        assert err and "not a supported Chromium" in err

    def test_snapshot_failure_fails_closed(self):
        import tools.browser_tool as bt
        self._reset()
        with patch.object(bt, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
             patch("hermes_cli.browser_connect.snapshot_real_profile", return_value=(None, "boom")):
            cdp, err = bt._real_profile_cdp()
        assert cdp is None
        assert err and "boom" in err

    def test_launch_returns_http_cdp(self, tmp_path):
        import tools.browser_tool as bt
        self._reset()
        proc = Mock(returncode=0, stdout="", stderr="")
        with patch.object(bt, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
             patch("hermes_cli.browser_connect.snapshot_real_profile", return_value=(str(tmp_path), None)), \
             patch.object(bt, "_agent_browser_get_cdp",
                          side_effect=[None, "http://127.0.0.1:41000"]), \
             patch.object(bt, "_find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch.object(bt.subprocess, "run", return_value=proc), \
             patch.object(bt, "_is_headed_mode", return_value=False):
            cdp, err = bt._real_profile_cdp()
        assert err is None
        assert cdp == "http://127.0.0.1:41000"
        self._reset()

    def test_launch_never_passes_headless(self, tmp_path):
        """--headless would use a separate cookie store → 0 real cookies."""
        import tools.browser_tool as bt
        self._reset()
        proc = Mock(returncode=0, stdout="", stderr="")
        captured = {}

        def fake_run(argv, **kw):
            captured["argv"] = argv
            return proc

        with patch.object(bt, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
             patch("hermes_cli.browser_connect.snapshot_real_profile", return_value=(str(tmp_path), None)), \
             patch.object(bt, "_agent_browser_get_cdp",
                          side_effect=[None, "http://127.0.0.1:41000"]), \
             patch.object(bt, "_find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch.object(bt.subprocess, "run", side_effect=fake_run), \
             patch.object(bt, "_is_headed_mode", return_value=False):
            bt._real_profile_cdp()
        assert "--headless" not in captured["argv"]
        assert "--profile" in captured["argv"]
        assert str(tmp_path) in captured["argv"]
        self._reset()

    def test_reuses_only_session_on_our_copy_dir(self, tmp_path):
        """A live session on a DIFFERENT dir (stale/throwaway) is closed, not reused."""
        import tools.browser_tool as bt
        self._reset()
        proc = Mock(returncode=0, stdout="", stderr="")
        closed = {"n": 0}
        with patch.object(bt, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
             patch("hermes_cli.browser_connect.snapshot_real_profile", return_value=(str(tmp_path), None)), \
             patch.object(bt, "_agent_browser_get_cdp",
                          side_effect=["http://127.0.0.1:5000", "http://127.0.0.1:41000"]), \
             patch.object(bt, "_cdp_http_ready", return_value=True), \
             patch.object(bt, "_cdp_on_data_dir", return_value=False), \
             patch.object(bt, "_agent_browser_close_session",
                          side_effect=lambda s: closed.__setitem__("n", closed["n"] + 1)), \
             patch.object(bt, "_find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch.object(bt.subprocess, "run", return_value=proc), \
             patch.object(bt, "_is_headed_mode", return_value=False):
            cdp, err = bt._real_profile_cdp()
        assert closed["n"] == 1  # stale wrong-dir session was closed
        assert cdp == "http://127.0.0.1:41000"
        self._reset()

    def test_cdp_on_data_dir_matches_devtoolsactiveport(self, tmp_path):
        import tools.browser_tool as bt
        (tmp_path / "DevToolsActivePort").write_text("41000\n/devtools/browser/x\n")
        assert bt._cdp_on_data_dir("http://127.0.0.1:41000", str(tmp_path))
        assert not bt._cdp_on_data_dir("http://127.0.0.1:9999", str(tmp_path))


class TestConsentConfigRead:
    """Unmocked config read: _use_real_profile against a real config.yaml."""

    def test_consent_read_from_config(self, tmp_path, monkeypatch):
        import tools.browser_tool as bt
        cfg = tmp_path / "config.yaml"
        cfg.write_text("browser:\n  use_real_profile: true\n")
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"browser": {"use_real_profile": True}}):
            assert bt._use_real_profile() is True

    def test_consent_default_off(self):
        import tools.browser_tool as bt
        with patch("hermes_cli.config.read_raw_config", return_value={}):
            assert bt._use_real_profile() is False

    def test_consent_revocation_takes_effect_immediately(self):
        """No process-lifetime caching: consent is a per-use read."""
        import tools.browser_tool as bt
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"browser": {"use_real_profile": True}}):
            assert bt._use_real_profile() is True
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"browser": {"use_real_profile": False}}):
            assert bt._use_real_profile() is False


class TestLocalSessionRealProfile:
    def test_local_session_attaches_to_real_profile_cdp(self):
        import tools.browser_tool as bt
        with patch.object(bt, "_real_profile_cdp",
                          return_value=("http://127.0.0.1:9251", None)), \
             patch.object(bt, "_resolve_cdp_override", side_effect=lambda u: u):
            info = bt._create_local_session("t1")
        assert info["cdp_url"] == "http://127.0.0.1:9251"
        assert info["features"]["real_profile"] is True
        assert info["session_name"].startswith("rp_")

    def test_local_session_fails_closed_on_error(self):
        import tools.browser_tool as bt
        with patch.object(bt, "_real_profile_cdp", return_value=(None, "no chromium")):
            with pytest.raises(RuntimeError, match="no chromium"):
                bt._create_local_session("t1")

    def test_local_session_without_consent_is_throwaway(self):
        import tools.browser_tool as bt
        with patch.object(bt, "_real_profile_cdp", return_value=(None, None)):
            info = bt._create_local_session("t1")
        assert info["cdp_url"] is None
        assert "real_profile" not in info["features"]
        assert info["session_name"].startswith("h_")


class TestBrowserExecLocalArg:
    def _env(self):
        return {}

    def test_local_forces_real_profile_under_cloud_backend(self):
        import tools.browser_use_cli as bu
        env = self._env()
        with patch.object(bu, "_real_profile_consented", return_value=True), \
             patch("tools.browser_tool._get_cdp_override_raw", return_value=""), \
             patch("tools.browser_tool._get_cloud_provider", return_value=Mock()), \
             patch("tools.browser_tool._real_profile_cdp",
                   return_value=("http://127.0.0.1:9251", None)):
            err = bu._resolve_real_profile_cdp(env, force_local=True)
        assert err is None
        assert env.get("BU_CDP_URL") == "http://127.0.0.1:9251"

    def test_no_force_keeps_cloud_backend(self):
        import tools.browser_use_cli as bu
        env = self._env()
        with patch.object(bu, "_real_profile_consented", return_value=True), \
             patch("tools.browser_tool._get_cdp_override_raw", return_value=""), \
             patch("tools.browser_tool._get_cloud_provider", return_value=Mock()):
            err = bu._resolve_real_profile_cdp(env, force_local=False)
        assert err is None
        assert "BU_CDP_URL" not in env and "BU_CDP_WS" not in env

    def test_local_backend_upgrades_without_force(self):
        import tools.browser_use_cli as bu
        env = self._env()
        with patch.object(bu, "_real_profile_consented", return_value=True), \
             patch.object(bu, "_read_browser_cfg", return_value={}), \
             patch("tools.browser_tool._get_cdp_override_raw", return_value=""), \
             patch("tools.browser_tool._get_cloud_provider", return_value=None), \
             patch("tools.browser_tool._real_profile_cdp",
                   return_value=("http://127.0.0.1:9251", None)):
            err = bu._resolve_real_profile_cdp(env, force_local=False)
        assert err is None
        assert env.get("BU_CDP_URL") == "http://127.0.0.1:9251"

    def test_consent_off_is_inert(self):
        import tools.browser_use_cli as bu
        env = self._env()
        with patch.object(bu, "_real_profile_consented", return_value=False):
            err = bu._resolve_real_profile_cdp(env, force_local=True)
        assert err is None and env == {}

    def test_launch_failure_fails_closed(self):
        import tools.browser_use_cli as bu
        env = self._env()
        with patch.object(bu, "_real_profile_consented", return_value=True), \
             patch("tools.browser_tool._get_cdp_override_raw", return_value=""), \
             patch("tools.browser_tool._real_profile_cdp",
                   return_value=(None, "chrome exited")):
            err = bu._resolve_real_profile_cdp(env, force_local=True)
        assert err == "chrome exited"
        assert "BU_CDP_URL" not in env

    def test_explicit_bu_env_override_wins(self):
        import tools.browser_use_cli as bu
        env = {"BU_CDP_WS": "ws://operator-override"}
        with patch.object(bu, "_real_profile_consented", return_value=True):
            err = bu._resolve_real_profile_cdp(env, force_local=True)
        assert err is None
        assert env["BU_CDP_WS"] == "ws://operator-override"
        assert "BU_CDP_URL" not in env

    def test_operator_cdp_override_wins(self):
        import tools.browser_use_cli as bu
        env = self._env()
        with patch.object(bu, "_real_profile_consented", return_value=True), \
             patch("tools.browser_tool._get_cdp_override_raw", return_value="ws://connect"):
            err = bu._resolve_real_profile_cdp(env, force_local=True)
        assert err is None and env == {}


class TestBrowserExecSchemaGating:
    def test_local_arg_absent_without_consent(self):
        import tools.browser_use_cli as bu
        with patch.object(bu, "_real_profile_consented", return_value=False):
            overrides = bu._dynamic_schema_overrides()
        assert "parameters" not in overrides
        assert "local" not in bu.BROWSER_EXEC_SCHEMA["parameters"]["properties"]

    def test_local_arg_present_with_consent(self):
        import tools.browser_use_cli as bu
        with patch.object(bu, "_real_profile_consented", return_value=True):
            overrides = bu._dynamic_schema_overrides()
        props = overrides["parameters"]["properties"]
        assert "local" in props
        assert props["local"]["type"] == "boolean"
        # Static schema must stay untouched (override is a copy).
        assert "local" not in bu.BROWSER_EXEC_SCHEMA["parameters"]["properties"]
        # 'local' must not be required — pure opt-in.
        assert "local" not in overrides["parameters"].get("required", [])


class TestNavigationRouting:
    def test_private_url_routing_unchanged(self):
        import tools.browser_tool as bt
        with patch.object(bt, "_get_cdp_override_raw", return_value=""), \
             patch.object(bt, "_is_camofox_mode", return_value=False), \
             patch.object(bt, "_get_cloud_provider", return_value=Mock()), \
             patch.object(bt, "_auto_local_for_private_urls", return_value=True), \
             patch.object(bt, "_url_is_private", return_value=True):
            key = bt._navigation_session_key("t1", "http://192.168.1.1/x")
        assert key == "t1::local"

    def test_public_url_stays_on_cloud(self):
        import tools.browser_tool as bt
        with patch.object(bt, "_get_cdp_override_raw", return_value=""), \
             patch.object(bt, "_is_camofox_mode", return_value=False), \
             patch.object(bt, "_get_cloud_provider", return_value=Mock()), \
             patch.object(bt, "_url_is_private", return_value=False):
            key = bt._navigation_session_key("t1", "https://example.com")
        assert key == "t1"


class TestChannelIdentity:
    """#95549 invariant: pre-release channels must NOT normalize to stable.

    Swallowing Beta/Dev/Canary into the stable family drives a different
    profile/account — a wrong-principal bug. Detection must flag the channel
    (UNSUPPORTED_CHANNEL) so the caller fails closed, never returning 'chrome'
    for a Beta default.
    """

    def test_linux_beta_not_normalized_to_stable(self):
        import hermes_cli.browser_connect as bc
        with patch.object(bc.subprocess, "run",
                          return_value=Mock(stdout="google-chrome-beta.desktop\n")):
            assert bc._detect_default_linux() == bc.UNSUPPORTED_CHANNEL

    def test_linux_stable_still_resolves(self):
        import hermes_cli.browser_connect as bc
        with patch.object(bc.subprocess, "run",
                          return_value=Mock(stdout="google-chrome.desktop\n")):
            assert bc._detect_default_linux() == "chrome"

    def test_linux_flatpak_beta_not_stable(self):
        import hermes_cli.browser_connect as bc
        with patch.object(bc.subprocess, "run",
                          return_value=Mock(stdout="com.google.chrome.beta.desktop\n")):
            assert bc._detect_default_linux() == bc.UNSUPPORTED_CHANNEL

    def test_darwin_canary_not_normalized(self):
        import hermes_cli.browser_connect as bc
        with patch.object(bc, "_launchservices_https_handler",
                          return_value="com.google.chrome.canary"):
            with patch.object(bc.subprocess, "run", return_value=Mock(stdout="")):
                assert bc._detect_default_darwin() == bc.UNSUPPORTED_CHANNEL

    def test_darwin_stable_exact_match(self):
        import hermes_cli.browser_connect as bc
        with patch.object(bc, "_launchservices_https_handler",
                          return_value="com.google.chrome"):
            with patch.object(bc.subprocess, "run", return_value=Mock(stdout="")):
                assert bc._detect_default_darwin() == "chrome"

    def test_windows_progid_maps(self):
        import hermes_cli.browser_connect as bc
        # Stable ProgIds → family; channel ProgIds are in the channel set.
        assert dict(bc._WINDOWS_PROGID_MAP)["chromehtml"] == "chrome"
        assert "chromebhtml" in bc._WINDOWS_CHANNEL_PROGIDS   # Beta
        assert "msedgebhtml" in bc._WINDOWS_CHANNEL_PROGIDS   # Edge Beta
        # A channel ProgId must not be a prefix hit for any stable entry.
        for chan in bc._WINDOWS_CHANNEL_PROGIDS:
            assert not any(chan.startswith(p) for p, _ in bc._WINDOWS_PROGID_MAP)

    def test_channel_sentinel_fails_closed_in_cdp(self):
        """A channel default → _real_profile_cdp fails closed, never launches."""
        import tools.browser_tool as bt
        import hermes_cli.browser_connect as bc
        bt._real_profile_cdp_cache.clear()
        with patch.object(bt, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium",
                   return_value=bc.UNSUPPORTED_CHANNEL), \
             patch("hermes_cli.browser_connect.snapshot_real_profile") as snap:
            cdp, err = bt._real_profile_cdp()
        assert cdp is None
        assert err and "pre-release" in err.lower()
        snap.assert_not_called()  # never even snapshotted a stable profile
        bt._real_profile_cdp_cache.clear()

    def test_data_dir_rejects_sentinel(self):
        import hermes_cli.browser_connect as bc
        assert bc.real_profile_data_dir(bc.UNSUPPORTED_CHANNEL, "Linux") is None
        assert bc.chromium_executable(bc.UNSUPPORTED_CHANNEL, "Linux") is None


class TestSnapshotIsCredentialStore:
    """The copied Cookies/Login Data must live inside Hermes' secret lifecycle."""

    def test_excluded_from_backup(self):
        import hermes_cli.backup as bk
        # Exact-component match (both singular and plural browser dirs).
        assert "browser-profile" in bk._EXCLUDED_DIRS
        assert bk._should_exclude(
            __import__("pathlib").Path("browser-profile/chrome/Default/Cookies")
        )

    def test_read_guard_blocks_snapshot(self, tmp_path, monkeypatch):
        import agent.file_safety as fs
        home = tmp_path / ".hermes"
        (home / "browser-profile" / "chrome" / "Default").mkdir(parents=True)
        cookies = home / "browser-profile" / "chrome" / "Default" / "Cookies"
        cookies.write_text("secret-cookie-db")
        monkeypatch.setenv("HERMES_HOME", str(home))
        err = fs.get_read_block_error(str(cookies))
        assert err and "snapshot" in err.lower()

    def test_read_guard_allows_normal_file(self, tmp_path, monkeypatch):
        import agent.file_safety as fs
        home = tmp_path / ".hermes"
        home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        normal = tmp_path / "notes.txt"
        normal.write_text("hello")
        assert fs.get_read_block_error(str(normal)) is None

    def test_snapshot_dir_secured(self, tmp_path, monkeypatch):
        """snapshot_real_profile locks the dir via the canonical _secure_dir."""
        import hermes_cli.browser_connect as bc
        src = tmp_path / "real" / "Default"
        src.mkdir(parents=True)
        (tmp_path / "real" / "Local State").write_text("{}")
        (src / "Cookies").write_text("db")
        monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path / "hh")
        called = {}
        with patch("hermes_cli.config._secure_dir",
                   side_effect=lambda p: called.__setitem__("p", p)):
            dst, err = bc.snapshot_real_profile("chrome", src=str(tmp_path / "real"))
        assert err is None
        assert called.get("p") == dst  # secured through the canonical owner
