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
from tools import browser_tool_cdp as bt_cdp
from tools import browser_tool_cloud as bt_cloud
from tools import browser_tool_lightpanda_fallback as bt_lightpanda_fallback
from tools import browser_tool_real_profile as bt_real_profile
from tools import browser_tool_session as bt_session
from tools import browser_tool_install as bt_install


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
        assert m["braveohtml"] == "brave-origin"

    def test_brave_origin_data_dirs(self):
        import hermes_cli.browser_connect as bc
        with patch.dict(os.environ, {"LOCALAPPDATA": r"C:\Users\T\AppData\Local"}, clear=False):
            win = bc.real_profile_data_dir("brave-origin", "Windows")
        assert win and win.endswith(ntpath.join("BraveSoftware", "Brave-Origin", "User Data"))
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/home/t/.config"}, clear=False):
            assert (
                bc.real_profile_data_dir("brave-origin", "Linux")
                == "/home/t/.config/BraveSoftware/Brave-Origin"
            )
        mac = bc.real_profile_data_dir("brave-origin", "Darwin")
        assert mac and mac.endswith("Library/Application Support/BraveSoftware/Brave-Origin")

    def test_brave_origin_channel_progids_fail_closed(self):
        import hermes_cli.browser_connect as bc
        # Beta=BraveOBHTML, Dev=BraveODHTML, Nightly=BraveOSHTM must be caught
        # by the channel list, and must be checked BEFORE the stable map — note
        # none of them share the braveohtml stable prefix, but ordering is the
        # invariant the detector relies on for the other families.
        for chan in ("braveobhtml", "braveodhtml", "braveoshtm"):
            assert chan in bc._WINDOWS_CHANNEL_PROGIDS

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

    def test_snapshot_files_are_owner_only(self, tmp_path, monkeypatch):
        """Every copied file must be 0600 and every dir 0700 (#96729).

        copy2 preserves Chrome's 0644 source modes and sqlite-backup files
        land umask-wide, so without explicit reconciliation the user's
        session-cookie copies are group/world-readable.
        """
        import stat

        import hermes_cli.browser_connect as bc
        src = self._make_profile(tmp_path / "real")
        home = tmp_path / "hermes-home"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        old_umask = os.umask(0o022)  # the common default that produced 0644
        try:
            dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        finally:
            os.umask(old_umask)
        assert err is None and dst
        offenders = []
        for root, dirs, files in os.walk(dst):
            for d in dirs:
                mode = stat.S_IMODE(os.stat(os.path.join(root, d)).st_mode)
                if mode & 0o077:
                    offenders.append((os.path.join(root, d), oct(mode)))
            for f in files:
                mode = stat.S_IMODE(os.stat(os.path.join(root, f)).st_mode)
                if mode & 0o077:
                    offenders.append((os.path.join(root, f), oct(mode)))
        assert not offenders, f"group/world-accessible snapshot entries: {offenders}"

    def test_existing_lax_snapshot_heals_on_refresh(self, tmp_path, monkeypatch):
        """A snapshot left 0644 by an older build tightens on the next pass."""
        import stat

        import hermes_cli.browser_connect as bc
        src = self._make_profile(tmp_path / "real")
        home = tmp_path / "hermes-home"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert err is None and dst
        cookies = os.path.join(dst, "Default", "Cookies")
        os.chmod(cookies, 0o644)  # simulate the pre-fix on-disk state
        dst2, err2 = bc.snapshot_real_profile("chrome", src=str(src))
        assert err2 is None and dst2 == dst
        assert stat.S_IMODE(os.stat(cookies).st_mode) == 0o600


class TestRealProfileCdpLaunch:
    """The agent-browser-based launcher in browser_tool_real_profile._real_profile_cdp."""

    def _reset(self):
        import tools.browser_tool as bt
        bt._real_profile_cdp_cache.clear()

    def test_consent_off_is_noop(self):
        self._reset()
        with patch.object(bt_cloud, "_use_real_profile", return_value=False):
            cdp, err = bt_real_profile._real_profile_cdp()
        assert cdp is None and err is None

    def test_non_chromium_default_fails_closed(self):
        self._reset()
        with patch.object(bt_cloud, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value=None):
            cdp, err = bt_real_profile._real_profile_cdp()
        assert cdp is None
        assert err and "not a supported Chromium" in err

    def test_snapshot_failure_fails_closed(self):
        self._reset()
        with patch.object(bt_cloud, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
             patch("hermes_cli.browser_connect.snapshot_real_profile", return_value=(None, "boom")):
            cdp, err = bt_real_profile._real_profile_cdp()
        assert cdp is None
        assert err and "boom" in err

    def test_launch_returns_http_cdp(self, tmp_path):
        import tools.browser_tool as bt
        self._reset()
        proc = Mock(return_value=None, returncode=0, stdout="", stderr="")

        class FakeChrome:
            def poll(self):
                return None

        def fake_popen(argv, **kw):
            (tmp_path / "DevToolsActivePort").write_text("41000\n/devtools/browser/x\n")
            return FakeChrome()

        with patch.object(bt_cloud, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
             patch("hermes_cli.browser_connect.snapshot_real_profile", return_value=(str(tmp_path), None)), \
             patch("hermes_cli.browser_connect.chromium_executable", return_value="/usr/bin/chrome"), \
             patch.object(bt.subprocess, "Popen", side_effect=fake_popen), \
             patch.object(bt_real_profile, "_agent_browser_get_cdp",
                          side_effect=[None, "http://127.0.0.1:41000"]), \
             patch.object(bt_install, "_find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch.object(bt.subprocess, "run", return_value=proc), \
             patch.object(bt_cloud, "_is_headed_mode", return_value=False):
            cdp, err = bt_real_profile._real_profile_cdp()
        assert err is None
        assert cdp == "http://127.0.0.1:41000"
        self._reset()

    def test_launch_is_headless_and_agent_browser_attaches(self, tmp_path):
        """Real-profile browsing runs headless (no focus-stealing window).

        Two argv paths are checked:

        1. The REAL Chrome binary we launch ourselves (via Popen) MUST pass
           ``--headless=new``. Real-profile browsing is a background
           capability — a visible window that grabs focus every turn defeats
           the point. NEW headless shares the profile's normal cookie store
           (unlike legacy ``--headless``), and cookie decryption is unaffected
           by headless — the drop we avoid comes from ``--use-mock-keychain``,
           not from headless. We launch without mock-keychain switches, so the
           copied auth/login state still loads.
        2. agent-browser ATTACHES to that running Chrome via ``--cdp``, so its
           argv must contain ``--cdp`` and must NOT contain launch-mode
           switches (``--headless`` / ``--profile``).
        """
        import tools.browser_tool as bt
        self._reset()
        proc = Mock(return_value=None, returncode=0, stdout="", stderr="")
        captured = {}

        def fake_run(argv, **kw):
            captured["argv"] = argv
            return proc

        class FakeChrome:
            def poll(self):
                return None

        def fake_popen(argv, **kw):
            captured["chrome_argv"] = argv
            (tmp_path / "DevToolsActivePort").write_text("41000\n/devtools/browser/x\n")
            return FakeChrome()

        with patch.object(bt_cloud, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
             patch("hermes_cli.browser_connect.snapshot_real_profile", return_value=(str(tmp_path), None)), \
             patch("hermes_cli.browser_connect.chromium_executable", return_value="/usr/bin/chrome"), \
             patch.object(bt.subprocess, "Popen", side_effect=fake_popen), \
             patch.object(bt_real_profile, "_agent_browser_get_cdp",
                          side_effect=[None, "http://127.0.0.1:41000"]), \
             patch.object(bt_install, "_find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch.object(bt.subprocess, "run", side_effect=fake_run), \
             patch.object(bt_cloud, "_is_headed_mode", return_value=False):
            bt_real_profile._real_profile_cdp()
        # The chrome launch itself is headless (no window, no focus steal).
        assert "--headless=new" in captured["chrome_argv"]
        # agent-browser attaches, it does not launch.
        assert "--headless" not in captured["argv"]
        assert "--profile" not in captured["argv"]
        assert "--cdp" in captured["argv"]
        self._reset()

    def test_reuses_only_session_on_our_copy_dir(self, tmp_path):
        """A live session on a DIFFERENT dir (stale/throwaway) is closed, not reused."""
        import tools.browser_tool as bt
        self._reset()
        proc = Mock(return_value=None, returncode=0, stdout="", stderr="")
        closed = {"n": 0}

        class FakeChrome:
            def poll(self):
                return None

        def fake_popen(argv, **kw):
            (tmp_path / "DevToolsActivePort").write_text("41000\n/devtools/browser/x\n")
            return FakeChrome()

        with patch.object(bt_cloud, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
             patch("hermes_cli.browser_connect.snapshot_real_profile", return_value=(str(tmp_path), None)), \
             patch("hermes_cli.browser_connect.chromium_executable", return_value="/usr/bin/chrome"), \
             patch.object(bt.subprocess, "Popen", side_effect=fake_popen), \
             patch.object(bt_real_profile, "_agent_browser_get_cdp",
                          side_effect=["http://127.0.0.1:5000", "http://127.0.0.1:41000"]), \
             patch.object(bt_real_profile, "_cdp_http_ready", return_value=True), \
             patch.object(bt_real_profile, "_cdp_on_data_dir", return_value=False), \
             patch.object(bt_real_profile, "_agent_browser_close_session",
                          side_effect=lambda s: closed.__setitem__("n", closed["n"] + 1)), \
             patch.object(bt_install, "_find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch.object(bt.subprocess, "run", return_value=proc), \
             patch.object(bt_cloud, "_is_headed_mode", return_value=False):
            cdp, err = bt_real_profile._real_profile_cdp()
        assert closed["n"] == 1  # stale wrong-dir session was closed
        assert cdp == "http://127.0.0.1:41000"
        self._reset()

    def test_cdp_on_data_dir_matches_devtoolsactiveport(self, tmp_path):
        (tmp_path / "DevToolsActivePort").write_text("41000\n/devtools/browser/x\n")
        assert bt_real_profile._cdp_on_data_dir("http://127.0.0.1:41000", str(tmp_path))
        assert not bt_real_profile._cdp_on_data_dir("http://127.0.0.1:9999", str(tmp_path))


class TestConsentConfigRead:
    """Unmocked config read: _use_real_profile against a real config.yaml."""

    def test_consent_read_from_config(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("browser:\n  use_real_profile: true\n")
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"browser": {"use_real_profile": True}}):
            assert bt_cloud._use_real_profile() is True

    def test_consent_default_off(self):
        with patch("hermes_cli.config.read_raw_config", return_value={}):
            assert bt_cloud._use_real_profile() is False

    def test_consent_revocation_takes_effect_immediately(self):
        """No process-lifetime caching: consent is a per-use read."""
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"browser": {"use_real_profile": True}}):
            assert bt_cloud._use_real_profile() is True
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"browser": {"use_real_profile": False}}):
            assert bt_cloud._use_real_profile() is False


class TestLocalSessionRealProfile:
    def test_local_session_attaches_to_real_profile_cdp(self):
        with patch.object(bt_real_profile, "_real_profile_cdp",
                          return_value=("http://127.0.0.1:9251", None)), \
             patch.object(bt_cdp, "_resolve_cdp_override", side_effect=lambda u: u):
            info = bt_session._create_local_session("t1")
        assert info["cdp_url"] == "http://127.0.0.1:9251"
        assert info["features"]["real_profile"] is True
        assert info["session_name"].startswith("rp_")

    def test_local_session_fails_closed_on_error(self):
        with patch.object(bt_real_profile, "_real_profile_cdp", return_value=(None, "no chromium")):
            with pytest.raises(RuntimeError, match="no chromium"):
                bt_session._create_local_session("t1")

    def test_local_session_without_consent_is_throwaway(self):
        with patch.object(bt_real_profile, "_real_profile_cdp", return_value=(None, None)):
            info = bt_session._create_local_session("t1")
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
             patch("tools.browser_tool_cdp._get_cdp_override_raw", return_value=""), \
             patch("tools.browser_tool_cloud._get_cloud_provider", return_value=Mock()), \
             patch("tools.browser_tool_real_profile._real_profile_cdp",
                   return_value=("http://127.0.0.1:9251", None)):
            err = bu._resolve_real_profile_cdp(env, force_local=True)
        assert err is None
        assert env.get("BU_CDP_URL") == "http://127.0.0.1:9251"

    def test_no_force_keeps_cloud_backend(self):
        import tools.browser_use_cli as bu
        env = self._env()
        with patch.object(bu, "_real_profile_consented", return_value=True), \
             patch("tools.browser_tool_cdp._get_cdp_override_raw", return_value=""), \
             patch("tools.browser_tool_cloud._get_cloud_provider", return_value=Mock()):
            err = bu._resolve_real_profile_cdp(env, force_local=False)
        assert err is None
        assert "BU_CDP_URL" not in env and "BU_CDP_WS" not in env

    def test_local_backend_upgrades_without_force(self):
        import tools.browser_use_cli as bu
        env = self._env()
        with patch.object(bu, "_real_profile_consented", return_value=True), \
             patch.object(bu, "_read_browser_cfg", return_value={}), \
             patch("tools.browser_tool_cdp._get_cdp_override_raw", return_value=""), \
             patch("tools.browser_tool_cloud._get_cloud_provider", return_value=None), \
             patch("tools.browser_tool_real_profile._real_profile_cdp",
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
             patch("tools.browser_tool_cdp._get_cdp_override_raw", return_value=""), \
             patch("tools.browser_tool_real_profile._real_profile_cdp",
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
             patch("tools.browser_tool_cdp._get_cdp_override_raw", return_value="ws://connect"):
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
        with patch.object(bt_cdp, "_get_cdp_override_raw", return_value=""), \
             patch.object(bt, "_is_camofox_mode", return_value=False), \
             patch.object(bt_cloud, "_get_cloud_provider", return_value=Mock()), \
             patch.object(bt_cloud, "_auto_local_for_private_urls", return_value=True), \
             patch.object(bt, "_url_is_private", return_value=True):
            key = bt._navigation_session_key("t1", "http://192.168.1.1/x")
        assert key == "t1::local"

    def test_public_url_stays_on_cloud(self):
        import tools.browser_tool as bt
        with patch.object(bt_cdp, "_get_cdp_override_raw", return_value=""), \
             patch.object(bt, "_is_camofox_mode", return_value=False), \
             patch.object(bt_cloud, "_get_cloud_provider", return_value=Mock()), \
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
        with patch.object(bt_cloud, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium",
                   return_value=bc.UNSUPPORTED_CHANNEL), \
             patch("hermes_cli.browser_connect.snapshot_real_profile") as snap:
            cdp, err = bt_real_profile._real_profile_cdp()
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
        called = {"paths": []}
        with patch("hermes_cli.config._secure_dir",
                   side_effect=lambda p: called["paths"].append(p)):
            dst, err = bc.snapshot_real_profile("chrome", src=str(tmp_path / "real"))
        assert err is None
        # Secured through the canonical owner; since #96729 the walk also
        # secures every nested dir, so dst is IN the set rather than last.
        assert dst in called["paths"]


class TestReviewBugFixes:
    """Regressions for the five PR #95620 review findings."""

    # ── Bug 2: launch the profile the user actually browses (last_used) ──
    def _multi_profile(self, root):
        """Build a data-dir where the SIGNED-IN session lives in 'Profile 6'."""
        for prof in ("Default", "Profile 6"):
            (root / prof / "Network").mkdir(parents=True)
        (root / "Local State").write_text(
            '{"profile": {"last_used": "Profile 6"}}'
        )
        # Default is signed OUT (tracking cookies only); Profile 6 has the session.
        (root / "Default" / "Cookies").write_text("default-tracking-only")
        (root / "Profile 6" / "Cookies").write_text("PROFILE6-SESSION-AUTH")
        (root / "Profile 6" / "Login Data").write_text("profile6-logins")
        (root / "Profile 6" / "Preferences").write_text("{}")
        return root

    def test_last_used_profile_lands_in_copy_default(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc
        src = self._multi_profile(tmp_path / "real")
        home = tmp_path / "hh"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert err is None
        # The copy's Default must carry PROFILE 6's session, not Default's.
        got = (home / "browser-profile" / "chrome" / "Default" / "Cookies").read_text()
        assert got == "PROFILE6-SESSION-AUTH"
        assert (home / "browser-profile" / "chrome" / "Default" / "Login Data").read_text() == "profile6-logins"

    def test_last_used_falls_back_to_default(self, tmp_path):
        import hermes_cli.browser_connect as bc
        root = tmp_path / "d"
        (root / "Default").mkdir(parents=True)
        (root / "Local State").write_text('{"profile": {"last_used": "Profile 9"}}')  # not present
        assert bc._last_used_profile(str(root)) == "Default"

    def test_last_used_reads_local_state(self, tmp_path):
        import hermes_cli.browser_connect as bc
        root = tmp_path / "d"
        (root / "Profile 6").mkdir(parents=True)
        (root / "Local State").write_text('{"profile": {"last_used": "Profile 6"}}')
        assert bc._last_used_profile(str(root)) == "Profile 6"

    def test_refresh_remirrors_last_used(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc
        src = self._multi_profile(tmp_path / "real")
        home = tmp_path / "hh"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        bc.snapshot_real_profile("chrome", src=str(src))          # fresh
        (src / "Profile 6" / "Cookies").write_text("PROFILE6-REFRESHED")
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))  # refresh
        assert err is None
        assert (home / "browser-profile" / "chrome" / "Default" / "Cookies").read_text() == "PROFILE6-REFRESHED"

    # ── Bug 3: private-URL sidecar must NOT carry the real profile ──
    def test_sidecar_never_uses_real_profile(self):
        # Even with consent resolving a real-profile CDP, the sidecar path
        # (allow_real_profile=False) must return a throwaway session.
        with patch.object(bt_real_profile, "_real_profile_cdp",
                          return_value=("http://127.0.0.1:9251", None)):
            info = bt_session._create_local_session("t::local", allow_real_profile=False)
        assert info["cdp_url"] is None
        assert "real_profile" not in info["features"]
        assert info["session_name"].startswith("h_")

    def test_sidecar_ignores_real_profile_error(self):
        """A real-profile resolve failure must not break private-URL routing."""
        with patch.object(bt_real_profile, "_real_profile_cdp",
                          return_value=(None, "non-chromium default")):
            info = bt_session._create_local_session("t::local", allow_real_profile=False)
        assert info["cdp_url"] is None  # no raise, throwaway session

    def test_bare_local_still_uses_real_profile(self):
        with patch.object(bt_real_profile, "_real_profile_cdp",
                          return_value=("http://127.0.0.1:9251", None)), \
             patch.object(bt_cdp, "_resolve_cdp_override", side_effect=lambda u: u):
            info = bt_session._create_local_session("t1")  # allow_real_profile defaults True
        assert info["features"].get("real_profile") is True

    # ── Bug 1: macOS 26 LSHandlers parser ──
    def test_macos26_parser_returns_bundle_not_version(self):
        import hermes_cli.browser_connect as bc
        dump = (
            "( { LSHandlerPreferredVersions = { LSHandlerRoleAll = \"7559.97\"; }; "
            "LSHandlerRoleAll = \"com.google.chrome\"; LSHandlerURLScheme = https; } )"
        )
        assert bc._launchservices_https_handler(dump) == "com.google.chrome"

    def test_macos26_detect_returns_chrome(self):
        import hermes_cli.browser_connect as bc
        dump = (
            "( { LSHandlerPreferredVersions = { LSHandlerRoleAll = \"7559.97\"; }; "
            "LSHandlerRoleAll = \"com.google.chrome\"; LSHandlerURLScheme = https; } )"
        )
        with patch.object(bc.subprocess, "run", return_value=Mock(stdout=dump)):
            assert bc._detect_default_darwin() == "chrome"

    # ── Bug 4: permissions applied on refresh, not only fresh ──
    def test_permissions_secured_on_refresh(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc
        src = self._multi_profile(tmp_path / "real")
        home = tmp_path / "hh"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        bc.snapshot_real_profile("chrome", src=str(src))  # fresh
        secured = []
        with patch("hermes_cli.config._secure_dir", side_effect=secured.append):
            bc.snapshot_real_profile("chrome", src=str(src))  # refresh
        # Refresh still secures BOTH the snapshot dir and its browser-profile parent.
        assert str(home / "browser-profile" / "chrome") in secured
        assert str(home / "browser-profile") in secured

    # ── Bug 5: lightpanda engine + consent fails with an actionable message ──
    def test_lightpanda_engine_fails_actionably(self):
        import tools.browser_tool as bt
        bt._real_profile_cdp_cache.clear()
        with patch.object(bt_cloud, "_use_real_profile", return_value=True), \
             patch.object(bt_lightpanda_fallback, "_using_lightpanda_engine", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium") as det:
            cdp, err = bt_real_profile._real_profile_cdp()
        assert cdp is None
        assert err and "lightpanda" in err.lower() and "browser.engine" in err.lower()
        det.assert_not_called()  # guard fires before detection
        bt._real_profile_cdp_cache.clear()


class TestReviewRound3:
    """Regressions for the round-3 review findings (Adolanium + kshitij)."""

    def _multi(self, root):
        for prof in ("Default", "Profile 6"):
            (root / prof / "Network").mkdir(parents=True)
        (root / "Local State").write_text('{"profile": {"last_used": "Profile 6"}}')
        (root / "Default" / "Cookies").write_text("default-signed-out")
        (root / "Profile 6" / "Cookies").write_text("PROFILE6-SESSION")
        (root / "Profile 6" / "Preferences").write_text("{}")
        return root

    # ── ② torn first copy must not poison freshness ──
    def test_done_marker_gates_fresh(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc
        src = self._multi(tmp_path / "real")
        home = tmp_path / "hh"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert err is None
        assert os.path.isfile(os.path.join(dst, bc._SNAPSHOT_DONE_MARKER))

    def test_torn_copy_is_redone_not_overlaid(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc
        src = self._multi(tmp_path / "real")
        home = tmp_path / "hh"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        dst = bc.real_profile_copy_dir("chrome")
        # Simulate a torn first copy: Default exists but NO done marker.
        os.makedirs(os.path.join(dst, "Default"))
        open(os.path.join(dst, "Default", "Cookies"), "w").write("HALF-COPY-GARBAGE")
        d, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert err is None
        # Rebuilt from the active profile, not treated as populated.
        assert (home / "browser-profile" / "chrome" / "Default" / "Cookies").read_text() == "PROFILE6-SESSION"
        assert os.path.isfile(os.path.join(dst, bc._SNAPSHOT_DONE_MARKER))

    # ── ④ only the active profile is copied, never the others ──
    def test_only_active_profile_copied(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc
        src = self._multi(tmp_path / "real")
        # Add a non-active profile with its own cookies — must NOT be copied.
        (src / "Profile 3").mkdir()
        (src / "Profile 3" / "Cookies").write_text("PROFILE3-SHOULD-NOT-COPY")
        home = tmp_path / "hh"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert err is None
        copy = home / "browser-profile" / "chrome"
        # Active profile (Profile 6) landed in Default; other profiles absent.
        assert (copy / "Default" / "Cookies").read_text() == "PROFILE6-SESSION"
        assert not (copy / "Profile 3").exists()
        assert not (copy / "Profile 6").exists()

    # ── ③ consent-off deletes the snapshot store ──
    def test_cleanup_removes_store(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc
        home = tmp_path / "hh"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        store = home / "browser-profile" / "chrome" / "Default"
        store.mkdir(parents=True)
        (store / "Cookies").write_text("secret")
        bc.cleanup_real_profile_snapshots()
        assert not (home / "browser-profile").exists()

    def test_cleanup_idempotent_when_absent(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc
        monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path / "hh")
        bc.cleanup_real_profile_snapshots()  # no raise

    # ── Windows lock probe (unit; the live share-lock is proven in the
    #    windows-latest E2E — here we cover the probe's contract portably) ──
    def test_lock_probe_false_when_readable(self, tmp_path):
        import hermes_cli.browser_connect as bc
        (tmp_path / "Default" / "Network").mkdir(parents=True)
        (tmp_path / "Default" / "Network" / "Cookies").write_bytes(b"db")
        assert bc._profile_is_locked(str(tmp_path), "Default") is False

    def test_lock_probe_false_when_no_cookie_db(self, tmp_path):
        import hermes_cli.browser_connect as bc
        (tmp_path / "Default").mkdir(parents=True)
        assert bc._profile_is_locked(str(tmp_path), "Default") is False

    def test_lock_probe_true_on_permissionerror(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc
        (tmp_path / "Default").mkdir(parents=True)
        (tmp_path / "Default" / "Cookies").write_bytes(b"db")
        import builtins
        real_open = builtins.open

        def deny(path, *a, **k):
            if str(path).endswith("Cookies"):
                raise PermissionError("locked")
            return real_open(path, *a, **k)

        monkeypatch.setattr(builtins, "open", deny)
        assert bc._profile_is_locked(str(tmp_path), "Default") is True

    def test_snapshot_fails_fast_when_locked(self, tmp_path, monkeypatch):
        """snapshot_real_profile always BLOCKS when locked — never kills, never
        proceeds to a heavy copy. autoclose off → plain quit guidance."""
        import hermes_cli.browser_connect as bc
        src = self._multi(tmp_path / "real")
        home = tmp_path / "hh"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        monkeypatch.setattr(bc, "_profile_is_locked", lambda s, p: True)
        monkeypatch.setattr(bc, "_real_profile_autoclose", lambda: False)
        called = {"copytree": 0}
        import shutil as _sh
        orig_ct = _sh.copytree
        monkeypatch.setattr(_sh, "copytree",
                            lambda *a, **k: (called.__setitem__("copytree", called["copytree"] + 1), orig_ct(*a, **k))[1])
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None
        assert err and err.startswith(bc._PROFILE_LOCKED_PREFIX)
        assert "quit" in err.lower()
        assert called["copytree"] == 0  # bailed before any copy

    def test_snapshot_blocks_when_locked_even_with_autoclose(self, tmp_path, monkeypatch):
        """Even with autoclose armed, snapshot_real_profile does NOT kill — it
        blocks and defers the close to the explicit, user-approved step. The
        message offers the close (mentions Hermes can close it)."""
        import hermes_cli.browser_connect as bc
        src = self._multi(tmp_path / "real")
        home = tmp_path / "hh"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        monkeypatch.setattr(bc, "_profile_is_locked", lambda s, p: True)
        monkeypatch.setattr(bc, "_real_profile_autoclose", lambda: True)
        killed = {"n": 0}
        monkeypatch.setattr(bc, "close_browser_holding_profile",
                            lambda *a, **k: (killed.__setitem__("n", killed["n"] + 1), (True, "x"))[1])
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None
        assert err and err.startswith(bc._PROFILE_LOCKED_PREFIX)
        assert "close it for you" in err.lower() or "can close it" in err.lower()
        assert killed["n"] == 0  # snapshot must NOT invoke the killer itself

    def test_processes_holding_profile_identity_binding(self, tmp_path, monkeypatch):
        """The process matcher requires BOTH a browser binary AND this exact
        user-data-dir in the cmdline — never a same-name process on another dir."""
        import hermes_cli.browser_connect as bc

        class FakeProc:
            def __init__(self, name, cmdline):
                self.info = {"name": name, "cmdline": cmdline}

        ud = str(tmp_path / "ud")
        procs = [
            FakeProc("chrome.exe", ["chrome.exe", f"--user-data-dir={ud}"]),      # match
            FakeProc("chrome.exe", ["chrome.exe", "--user-data-dir=C:\\Other"]),  # wrong dir
            FakeProc("python.exe", ["python.exe", f"--user-data-dir={ud}"]),      # not a browser
        ]

        class FakePsutil:
            NoSuchProcess = psutil_exc = type("E", (Exception,), {})
            AccessDenied = type("E2", (Exception,), {})

            def process_iter(self, attrs=None):
                return iter(procs)

        import sys as _sys
        monkeypatch.setitem(_sys.modules, "psutil", FakePsutil())
        matched = list(bc._processes_holding_profile(ud))
        assert len(matched) == 1
        assert matched[0].info["name"] == "chrome.exe"
        assert f"--user-data-dir={ud}" in " ".join(matched[0].info["cmdline"])

    def test_consent_off_triggers_cleanup(self, tmp_path, monkeypatch):
        called = {"n": 0}
        with patch.object(bt_cloud, "_use_real_profile", return_value=False), \
             patch("hermes_cli.browser_connect.cleanup_real_profile_snapshots",
                   side_effect=lambda: called.__setitem__("n", called["n"] + 1)):
            cdp, err = bt_real_profile._real_profile_cdp()
        assert cdp is None and err is None
        assert called["n"] == 1

    # ── ① overlay must not run before the reuse check (live-browser safety) ──
    def test_reuse_skips_snapshot_overlay(self, tmp_path):
        """When a live session on our copy dir is reused, snapshot_real_profile
        must NOT be called — otherwise it rewrites cookie DBs under a live
        browser."""
        import tools.browser_tool as bt
        bt._real_profile_cdp_cache.clear()
        with patch.object(bt_cloud, "_use_real_profile", return_value=True), \
             patch.object(bt_lightpanda_fallback, "_using_lightpanda_engine", return_value=False), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
             patch("hermes_cli.browser_connect.real_profile_copy_dir", return_value=str(tmp_path)), \
             patch.object(bt_real_profile, "_agent_browser_get_cdp", return_value="http://127.0.0.1:9251"), \
             patch.object(bt_real_profile, "_cdp_http_ready", return_value=True), \
             patch.object(bt_real_profile, "_cdp_on_data_dir", return_value=True), \
             patch("hermes_cli.browser_connect.snapshot_real_profile") as snap:
            cdp, err = bt_real_profile._real_profile_cdp()
        assert cdp == "http://127.0.0.1:9251" and err is None
        snap.assert_not_called()  # ← the fix: no overlay while a live browser owns the dir
        bt._real_profile_cdp_cache.clear()

    def test_relaunch_path_does_snapshot(self, tmp_path):
        """When there's no reusable session, the overlay DOES run (relaunch)."""
        import tools.browser_tool as bt
        bt._real_profile_cdp_cache.clear()
        proc = Mock(returncode=0, stdout="", stderr="")
        with patch.object(bt_cloud, "_use_real_profile", return_value=True), \
             patch.object(bt_lightpanda_fallback, "_using_lightpanda_engine", return_value=False), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
             patch("hermes_cli.browser_connect.real_profile_copy_dir", return_value=str(tmp_path)), \
             patch("hermes_cli.browser_connect.snapshot_real_profile",
                   return_value=(str(tmp_path), None)) as snap, \
             patch.object(bt_real_profile, "_agent_browser_get_cdp",
                          side_effect=[None, "http://127.0.0.1:9251"]), \
             patch.object(bt_install, "_find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch.object(bt.subprocess, "run", return_value=proc), \
             patch.object(bt_cloud, "_is_headed_mode", return_value=False):
            cdp, err = bt_real_profile._real_profile_cdp()
        assert err is None
        snap.assert_called_once()
        bt._real_profile_cdp_cache.clear()


class TestWindowsLockedProfileCopy:
    """Windows: a running Chrome holds Cookies/Login Data with an exclusive
    lock. The auth DBs must be copied via SQLite online-backup (works under the
    lock), not a raw copy that fails and leaves a signed-out snapshot."""

    def _locked_src(self, root):
        import sqlite3, json
        (root / "Default" / "Network").mkdir(parents=True)
        (root / "Local State").write_text(json.dumps({"profile": {"last_used": "Default"}}))
        (root / "Default" / "Preferences").write_text("{}")
        ck = str(root / "Default" / "Cookies")
        con = sqlite3.connect(ck)
        con.execute("create table cookies(host_key, name)")
        con.executemany("insert into cookies values(?,?)",
                        [("nous.ai", f"c{i}") for i in range(42)])
        con.commit()
        return root, con  # caller keeps con open to simulate the live lock

    def test_locked_cookie_db_copied_via_backup(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc
        import sqlite3, shutil
        src, con = self._locked_src(tmp_path / "real")
        con.execute("BEGIN"); con.execute("insert into cookies values('u','uncommitted')")
        home = tmp_path / "hh"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        try:
            dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        finally:
            con.rollback(); con.close()
        assert err is None
        copy_ck = str(home / "browser-profile" / "chrome" / "Default" / "Cookies")
        t = str(tmp_path / "probe"); shutil.copy2(copy_ck, t)
        n = sqlite3.connect(t).execute("select count(*) from cookies").fetchone()[0]
        assert n == 42  # committed rows copied under the lock; uncommitted excluded
        # No stale journal/wal sidecar left next to the backed-up DB.
        assert not (home / "browser-profile" / "chrome" / "Default" / "Cookies-journal").exists()

    def test_copy_auth_file_backs_up_db(self, tmp_path):
        import hermes_cli.browser_connect as bc
        import sqlite3
        src = str(tmp_path / "Cookies")
        con = sqlite3.connect(src); con.execute("create table cookies(x)"); con.execute("insert into cookies values(1)"); con.commit(); con.close()
        dst = str(tmp_path / "out" / "Cookies")
        assert bc._copy_auth_file(src, dst) is True
        assert sqlite3.connect(dst).execute("select count(*) from cookies").fetchone()[0] == 1

    def test_copy_auth_file_plain_for_non_db(self, tmp_path):
        import hermes_cli.browser_connect as bc
        src = str(tmp_path / "Preferences"); open(src, "w").write('{"k":1}')
        dst = str(tmp_path / "out" / "Preferences")
        assert bc._copy_auth_file(src, dst) is True
        assert open(dst).read() == '{"k":1}'

    def test_fail_closed_when_db_unreadable(self, tmp_path, monkeypatch):
        """If even the online-backup can't read the DB, snapshot fails closed
        rather than launching a silently signed-out session."""
        import hermes_cli.browser_connect as bc
        import json
        root = tmp_path / "real"
        (root / "Default").mkdir(parents=True)
        (root / "Local State").write_text(json.dumps({"profile": {"last_used": "Default"}}))
        (root / "Default" / "Cookies").write_text("not-a-db")
        (root / "Default" / "Preferences").write_text("{}")
        home = tmp_path / "hh"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        # Force both sqlite-backup and raw copy to fail for the DB.
        monkeypatch.setattr(bc, "_copy_auth_file",
                            lambda s, d: False if os.path.basename(s) in bc._SQLITE_AUTH_DBS else True)
        dst, err = bc.snapshot_real_profile("chrome", src=str(root))
        assert dst is None
        assert err and "login data" in err.lower() and "close" in err.lower()
