"""Tests for real-profile browsing: resolvers, snapshot, launch routing, consent.

The consent path never drives the live default profile: it snapshots into
``~/.hermes/browser-profile/<browser>/`` and launches the user's real binary
on the copy with a devtools port (see hermes_cli.browser_connect). These tests
exercise the real functions with real file I/O wherever possible — the mocks
are limited to OS detection and process launch.
"""
import json
import os
from unittest.mock import Mock, patch

import pytest
from tools import browser_tool_cdp as bt_cdp
from tools import browser_tool_cloud as bt_cloud
from tools import browser_tool_lightpanda_fallback as bt_lightpanda_fallback
from tools import browser_tool_real_profile as bt_real_profile
from tools import browser_tool_session as bt_session
from tools import browser_tool_install as bt_install


def _auth_db(path, value=None):
    """Store/read a marker in a real auth DB so snapshot fixtures exercise SQLite."""
    import sqlite3
    from contextlib import closing

    with closing(sqlite3.connect(path)) as conn, conn:
        if value is not None:
            conn.execute("create table if not exists marker(value)")
            conn.execute("delete from marker")
            conn.execute("insert into marker values(?)", (value,))
        return conn.execute("select value from marker").fetchone()[0]


class TestRealProfileResolvers:


    def test_data_dir_unknown_browser_is_none(self):
        import hermes_cli.browser_connect as bc
        assert bc.real_profile_data_dir("firefox", "Windows") is None






class TestSnapshotRealProfile:
    """Real file I/O: the snapshot copier against a synthetic profile tree."""

    def _make_profile(self, root):
        """Build a minimal real-looking Chromium user-data-dir."""
        (root / "Default" / "Network").mkdir(parents=True)
        (root / "Default" / "Cache" / "Cache_Data").mkdir(parents=True)
        (root / "Code Cache" / "js").mkdir(parents=True)
        (root / "Crashpad").mkdir()
        (root / "Local State").write_text('{"os_crypt": {}}')
        _auth_db((root / "Default" / "Cookies"), "sqlite-cookies")
        _auth_db((root / "Default" / "Network" / "Cookies"), "sqlite-net-cookies")
        _auth_db((root / "Default" / "Login Data"), "sqlite-logins")
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
        assert _auth_db((home / "browser-profile" / "chrome" / "Default" / "Cookies")) == "sqlite-cookies"
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
        _auth_db((src / "Default" / "Cookies"), "sqlite-cookies-v2")
        copy_history = home / "browser-profile" / "chrome" / "Default" / "History"
        copy_history.write_text("agent-session-history")

        dst2, err2 = bc.snapshot_real_profile("chrome", src=str(src))
        assert err2 is None and dst2 == dst
        assert _auth_db((home / "browser-profile" / "chrome" / "Default" / "Cookies")) == "sqlite-cookies-v2"
        assert copy_history.read_text() == "agent-session-history"

    def test_missing_source_fails_closed(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc
        monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path / "hh")
        dst, err = bc.snapshot_real_profile("chrome", src=str(tmp_path / "nope"))
        assert dst is None
        assert err

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
        assert err

    def test_snapshot_failure_fails_closed(self):
        self._reset()
        with patch.object(bt_cloud, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
             patch("hermes_cli.browser_connect.snapshot_real_profile", return_value=(None, "boom")):
            cdp, err = bt_real_profile._real_profile_cdp()
        assert cdp is None
        assert err and "boom" in err


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
            captured["env"] = kw["env"]
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
             patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmp_path)), \
             patch.object(bt_cloud, "_is_headed_mode", return_value=False):
            bt_real_profile._real_profile_cdp()
        # The chrome launch itself is headless (no window, no focus steal).
        assert "--headless=new" in captured["chrome_argv"]
        # agent-browser attaches, it does not launch.
        assert "--headless" not in captured["argv"]
        assert "--profile" not in captured["argv"]
        assert "--cdp" in captured["argv"]
        # #100855: the attach daemon lives in a reaper-visible socket dir claimed by this
        # process, and never self-terminates (Chrome is ours, not the daemon's).
        socket_dir = captured["env"]["AGENT_BROWSER_SOCKET_DIR"]
        assert socket_dir == str(tmp_path / f"agent-browser-{bt._REAL_PROFILE_SESSION}")
        assert (tmp_path / f"agent-browser-{bt._REAL_PROFILE_SESSION}" / f"{bt._REAL_PROFILE_SESSION}.owner_pid").read_text() == str(os.getpid())
        assert "AGENT_BROWSER_IDLE_TIMEOUT_MS" not in captured["env"]
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

    @pytest.mark.parametrize("live_browser_id", ["/devtools/browser/x", "/devtools/browser/other"])
    def test_reattaches_to_surviving_chrome_instead_of_overlaying_its_profile(self, tmp_path, live_browser_id):
        """The attach daemon of a crashed owner gets reaped, but its Chrome (Hermes-launched,
        own session) survives holding the copy dir: re-attach, never re-run the snapshot.
        A DevToolsActivePort left by a crash whose port was recycled by ANOTHER CDP server
        (browser id mismatch) must not be attached to; the normal launch path runs."""
        self._reset()
        (tmp_path / "DevToolsActivePort").write_text("41000\n/devtools/browser/x\n")
        version = Mock()
        version.json.return_value = {"webSocketDebuggerUrl": f"ws://127.0.0.1:41000{live_browser_id}"}
        with patch.object(bt_cloud, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
             patch("hermes_cli.browser_connect.real_profile_copy_dir", return_value=str(tmp_path)), \
             patch("hermes_cli.browser_connect.snapshot_real_profile", return_value=(None, "boom")) as snapshot, \
             patch("requests.get", return_value=version), \
             patch.object(bt_real_profile, "_agent_browser_get_cdp", return_value=None), \
             patch.object(bt_real_profile, "_attach_agent_browser_to_real_profile",
                          return_value=("http://127.0.0.1:41000", None)) as attach:
            cdp, err = bt_real_profile._real_profile_cdp()
        if live_browser_id == "/devtools/browser/x":
            assert (cdp, err) == ("http://127.0.0.1:41000", None)
            attach.assert_called_once_with(41000, str(tmp_path))
            snapshot.assert_not_called()
        else:
            attach.assert_not_called()
            snapshot.assert_called_once()
        self._reset()

    def test_cdp_on_data_dir_matches_devtoolsactiveport(self, tmp_path):
        (tmp_path / "DevToolsActivePort").write_text("41000\n/devtools/browser/x\n")
        assert bt_real_profile._cdp_on_data_dir("http://127.0.0.1:41000", str(tmp_path))
        assert not bt_real_profile._cdp_on_data_dir("http://127.0.0.1:9999", str(tmp_path))


class TestConsentConfigRead:
    """Unmocked config read: _use_real_profile against a real config.yaml."""


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
        assert err
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
        # Hyphen snapshot dirs are any-depth; Browser Use CLI underscore dir is root-scoped.
        assert bk._should_exclude(
            __import__("pathlib").Path("browser-profile/chrome/Default/Cookies")
        )
        assert bk._should_exclude(
            __import__("pathlib").Path(
                "browser_profiles/browser-use-default/Default/Login Data"
            )
        )

    def test_read_guard_blocks_snapshot(self, tmp_path, monkeypatch):
        import agent.file_safety as fs
        home = tmp_path / ".hermes"
        (home / "browser-profile" / "chrome" / "Default").mkdir(parents=True)
        cookies = home / "browser-profile" / "chrome" / "Default" / "Cookies"
        cookies.write_text("secret-cookie-db")
        monkeypatch.setenv("HERMES_HOME", str(home))
        err = fs.get_read_block_error(str(cookies))
        assert err

    def test_read_guard_allows_normal_file(self, tmp_path, monkeypatch):
        import agent.file_safety as fs
        home = tmp_path / ".hermes"
        home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        normal = tmp_path / "notes.txt"
        normal.write_text("hello")
        assert fs.get_read_block_error(str(normal)) is None



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
        _auth_db((root / "Default" / "Cookies"), "default-tracking-only")
        _auth_db((root / "Profile 6" / "Cookies"), "PROFILE6-SESSION-AUTH")
        _auth_db((root / "Profile 6" / "Login Data"), "profile6-logins")
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
        got = _auth_db((home / "browser-profile" / "chrome" / "Default" / "Cookies"))
        assert got == "PROFILE6-SESSION-AUTH"
        assert _auth_db((home / "browser-profile" / "chrome" / "Default" / "Login Data")) == "profile6-logins"

    def test_last_used_falls_back_to_default(self, tmp_path):
        import hermes_cli.browser_connect as bc
        root = tmp_path / "d"
        (root / "Default").mkdir(parents=True)
        (root / "Local State").write_text('{"profile": {"last_used": "Profile 9"}}')  # not present
        assert bc._last_used_profile(str(root)) == "Default"


    def test_refresh_remirrors_last_used(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc
        src = self._multi_profile(tmp_path / "real")
        home = tmp_path / "hh"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        bc.snapshot_real_profile("chrome", src=str(src))          # fresh
        _auth_db((src / "Profile 6" / "Cookies"), "PROFILE6-REFRESHED")
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))  # refresh
        assert err is None
        assert _auth_db((home / "browser-profile" / "chrome" / "Default" / "Cookies")) == "PROFILE6-REFRESHED"

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



    # ── Bug 1: macOS 26 LSHandlers parser ──

    def test_macos26_detect_returns_chrome(self):
        import hermes_cli.browser_connect as bc
        dump = (
            "( { LSHandlerPreferredVersions = { LSHandlerRoleAll = \"7559.97\"; }; "
            "LSHandlerRoleAll = \"com.google.chrome\"; LSHandlerURLScheme = https; } )"
        )
        with patch.object(bc.subprocess, "run", return_value=Mock(stdout=dump)):
            assert bc._detect_default_darwin() == "chrome"

    # ── Bug 4: permissions applied on refresh, not only fresh ──

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
        _auth_db((root / "Default" / "Cookies"), "default-signed-out")
        _auth_db((root / "Profile 6" / "Cookies"), "PROFILE6-SESSION")
        (root / "Profile 6" / "Preferences").write_text("{}")
        return root

    # ── ② torn first copy must not poison freshness ──

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
        assert _auth_db((home / "browser-profile" / "chrome" / "Default" / "Cookies")) == "PROFILE6-SESSION"
        assert os.path.isfile(os.path.join(dst, bc._SNAPSHOT_DONE_MARKER))

    # ── ④ only the active profile is copied, never the others ──
    def test_only_active_profile_copied(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc
        src = self._multi(tmp_path / "real")
        # Add a non-active profile with its own cookies — must NOT be copied.
        (src / "Profile 3").mkdir()
        _auth_db((src / "Profile 3" / "Cookies"), "PROFILE3-SHOULD-NOT-COPY")
        home = tmp_path / "hh"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert err is None
        copy = home / "browser-profile" / "chrome"
        # Active profile (Profile 6) landed in Default; other profiles absent.
        assert _auth_db((copy / "Default" / "Cookies")) == "PROFILE6-SESSION"
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

    def test_processes_holding_profile_default_profile_main_process(self, tmp_path, monkeypatch):
        """A browser launched normally carries NO --user-data-dir; its main process must still
        bind when the profile is that install's default dir — but not a helper, and not a
        different install that happens to share the binary name."""
        import hermes_cli.browser_connect as bc

        class FakeProc:
            def __init__(self, pid, cmdline):
                self.pid = pid
                self.info = {"name": "chrome.exe", "cmdline": cmdline}

            def cmdline(self):
                return self.info["cmdline"]

            def parent(self):
                return None

        local = tmp_path / "Local"
        monkeypatch.setenv("LOCALAPPDATA", str(local))
        monkeypatch.setattr(bc.platform, "system", lambda: "Windows")
        ud = bc.real_profile_data_dir("chrome", "Windows")
        chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        procs = [
            FakeProc(1, [chrome, "about:blank"]),                                   # match
            FakeProc(2, [chrome, "--type=renderer"]),                               # helper
            FakeProc(3, [r"C:\Apps\Chromium\Application\chrome.exe"]),              # other install
            FakeProc(4, [chrome, "--user-data-dir=C:\\Other"]),                     # other dir
        ]

        class FakePsutil:
            NoSuchProcess = type("E", (Exception,), {})
            AccessDenied = type("E2", (Exception,), {})

            def process_iter(self, attrs=None):
                return iter(procs)

        import sys as _sys
        monkeypatch.setitem(_sys.modules, "psutil", FakePsutil())
        assert [p.pid for p in bc._processes_holding_profile(ud)] == [1]

    def test_processes_holding_profile_climbs_helper_to_browser_root(self, tmp_path, monkeypatch):
        """Only a helper (crashpad) names the dir; the matcher must yield the main browser
        process above it, once, so the close takes the whole browser down."""
        import hermes_cli.browser_connect as bc

        ud = str(tmp_path / "ud")

        class FakeProc:
            def __init__(self, pid, name, cmdline, parent=None):
                self.pid, self._name, self._parent = pid, name, parent
                self.info = {"name": name, "cmdline": cmdline}

            def name(self):
                return self._name

            def cmdline(self):
                return self.info["cmdline"]

            def parent(self):
                return self._parent

        shell = FakeProc(10, "explorer.exe", ["explorer.exe"])
        main = FakeProc(20, "chrome.exe", ["chrome.exe"], parent=shell)
        crashpad = FakeProc(21, "chrome.exe", ["chrome.exe", "--type=crashpad-handler",
                                               f"--database={ud}\\Crashpad"], parent=main)
        crashpad2 = FakeProc(22, "chrome.exe", ["chrome.exe", "--type=crashpad-handler",
                                                f"--database={ud}\\Crashpad"], parent=crashpad)
        procs = [shell, main, crashpad, crashpad2]

        class FakePsutil:
            NoSuchProcess = type("E", (Exception,), {})
            AccessDenied = type("E2", (Exception,), {})

            def process_iter(self, attrs=None):
                return iter(procs)

        import sys as _sys
        monkeypatch.setitem(_sys.modules, "psutil", FakePsutil())
        assert [p.pid for p in bc._processes_holding_profile(ud)] == [20]

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



class TestWindowsLockedProfileCopy:
    """Windows: a running Chrome holds Cookies/Login Data with an exclusive
    lock. The auth DBs must be copied via SQLite online-backup (works under the
    lock), not a raw copy that fails and leaves a signed-out snapshot."""

    def _locked_src(self, root):
        import sqlite3
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
        assert bc._copy_auth_file(src, dst) is None
        assert sqlite3.connect(dst).execute("select count(*) from cookies").fetchone()[0] == 1

    @pytest.mark.parametrize("locked", ["source", "destination"])
    def test_copy_auth_file_bounds_locks_without_overwriting(self, tmp_path, locked):
        import sqlite3
        import subprocess
        import sys
        import hermes_cli.browser_connect as bc

        src, dst = tmp_path / "Cookies", tmp_path / "out" / "Cookies"
        dst.parent.mkdir()
        for path, value in ((src, 7), (dst, 99)):
            with sqlite3.connect(path) as conn:
                conn.execute("create table cookies(x)")
                conn.execute("insert into cookies values(?)", (value,))
            conn.close()
        holder = sqlite3.connect(src if locked == "source" else dst)
        holder.execute("begin exclusive")
        try:
            result = subprocess.run(
                [sys.executable, "-c",
                 "from hermes_cli.browser_connect import _copy_auth_file; "
                 "import sys; print(_copy_auth_file(sys.argv[1], sys.argv[2]))",
                 str(src), str(dst)],
                capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL)
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip() == bc._AUTH_DB_LOCKED
        finally:
            holder.rollback()
            holder.close()
        with sqlite3.connect(dst) as conn:
            assert conn.execute("select x from cookies").fetchall() == [(99,)]
        conn.close()
        assert bc._copy_auth_file(str(src), str(dst)) is None
        with sqlite3.connect(dst) as conn:
            assert conn.execute("select x from cookies").fetchall() == [(7,)]
        conn.close()

    def test_copy_auth_file_preserves_source_wal_not_abandoned_destination_wal(self, tmp_path):
        import sqlite3
        import subprocess
        import sys
        import hermes_cli.browser_connect as bc

        src, dst = tmp_path / "Cookies", tmp_path / "out" / "Cookies"
        dst.parent.mkdir()
        source = sqlite3.connect(src)
        source.execute("create table cookies(x)")
        source.execute("insert into cookies values(7)")
        source.commit()
        source.execute("pragma journal_mode=wal")
        source.execute("update cookies set x=8")
        source.commit()
        subprocess.run(
            [sys.executable, "-c",
             "import sqlite3, os, sys; c=sqlite3.connect(sys.argv[1]); "
             "c.execute('create table cookies(x)'); c.commit(); "
             "c.execute('pragma journal_mode=wal'); "
             "c.execute('insert into cookies values(99)'); c.commit(); os._exit(0)",
             str(dst)], check=True, timeout=15, stdin=subprocess.DEVNULL)
        assert os.path.exists(str(dst) + "-wal")
        try:
            assert bc._copy_auth_file(str(src), str(dst)) is None
            with sqlite3.connect(dst) as conn:
                assert conn.execute("select x from cookies").fetchall() == [(8,)]
            conn.close()
        finally:
            source.close()

    def test_copy_auth_file_slow_but_progressing_backup_is_not_called_locked(self, tmp_path, monkeypatch):
        """A large DB on a slow disk that is still copying pages past the deadline must not
        get the lock wording (whose all-locked message tells the user to quit the browser)."""
        import sqlite3
        import hermes_cli.browser_connect as bc
        src = str(tmp_path / "Web Data")
        con = sqlite3.connect(src)
        con.execute("create table t(x)")
        con.executemany("insert into t values(?)", ((b"x" * 4000,) for _ in range(600)))  # > 256 pages
        con.commit(); con.close()
        monkeypatch.setattr(bc, "_AUTH_BACKUP_DEADLINE_S", -1.0)  # first callback is already past due
        reason = bc._copy_auth_file(src, str(tmp_path / "out" / "Web Data"))
        assert reason and reason != bc._AUTH_DB_LOCKED
        assert "write lock" not in reason and "exceeded" in reason

    def test_copy_auth_file_plain_for_non_db(self, tmp_path):
        import hermes_cli.browser_connect as bc
        src = str(tmp_path / "Preferences"); open(src, "w").write('{"k":1}')
        dst = str(tmp_path / "out" / "Preferences")
        assert bc._copy_auth_file(src, dst) is None
        assert open(dst).read() == '{"k":1}'

    def test_fail_closed_when_db_unreadable(self, tmp_path, monkeypatch):
        """If even the online-backup can't read the DB, snapshot fails closed
        rather than launching a silently signed-out session."""
        import hermes_cli.browser_connect as bc
        root = tmp_path / "real"
        (root / "Default").mkdir(parents=True)
        (root / "Local State").write_text(json.dumps({"profile": {"last_used": "Default"}}))
        (root / "Default" / "Cookies").write_text("not-a-db")
        (root / "Default" / "Preferences").write_text("{}")
        home = tmp_path / "hh"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        # Force both sqlite-backup and raw copy to fail for the DB.
        monkeypatch.setattr(bc, "_copy_auth_file",
                            lambda s, d: "file is not a database" if os.path.basename(s) in bc._SQLITE_AUTH_DBS else None)
        dst, err = bc.snapshot_real_profile("chrome", src=str(root))
        assert dst is None
        assert err
