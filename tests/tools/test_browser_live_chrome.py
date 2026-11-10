"""``browser.live_chrome``: browser_exec drives the user's own running Chrome (chrome://inspect route).

Contracts: off = routing unchanged; on = no BU_CDP_* exported (the harness finds Chrome), operator overrides
win, own-tab preamble kept, Hermes never launches/closes/approves anything, a macOS privacy block is reported.
No real Chrome is touched: profile dirs are tmp dirs, the CLI is a shell stub.
"""
import json
import os
import stat
import subprocess
import sys

import pytest

import tools.browser_use_cli as bu_cli
import tools.browser_use_live_chrome as live
from tools import browser_launcher
from tools import browser_session as bs
from tools import browser_session_cloud as bs_cloud
from tools import browser_session_lifecycle as lifecycle

_ENV_PROBE = ('code=$(cat)\n'
              'echo "ws:${BU_CDP_WS:-unset} url:${BU_CDP_URL:-unset} marker:${BH_TAB_MARKER:-unset} '
              'autospawn:${BU_AUTOSPAWN:-unset} live:${_HERMES_BU_LIVE_CHROME:-unset}"\n'
              'echo "$code"\n')


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    for var in ("BU_NAME", "BU_AUTOSPAWN", "BU_CDP_WS", "BU_CDP_URL", "BROWSER_USE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override", lambda: "")
    monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override_raw", lambda: "")
    monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: None)
    monkeypatch.setattr(bu_cli, "_real_profile_consented", lambda: False)
    monkeypatch.setattr(bs, "get_session_info", lambda key: {
        "cdp_url": f"ws://127.0.0.1:47000/devtools/browser/{key}", "features": {"local": True}})
    bs_cloud.reset_caches()
    yield
    bs_cloud.reset_caches()


@pytest.fixture
def config(monkeypatch):
    cfg: dict = {}
    monkeypatch.setattr(bs_cloud, "browser_section", lambda: cfg)
    return cfg


@pytest.fixture
def chrome_running(monkeypatch):
    """preflight answers 'Chrome is running'; records calls."""
    calls = []
    monkeypatch.setattr(live, "preflight", lambda system=None: calls.append(1))
    return calls


@pytest.fixture
def supervisor_attaches(monkeypatch):
    from tools import browser_supervisor
    attached = []

    class _Registry:
        def get_or_start(self, task_id, cdp_url, **kw):
            attached.append((task_id, cdp_url))

        def get(self, task_id):
            return None

        def stop(self, task_id):
            pass

        def stop_all(self):
            pass

    monkeypatch.setattr(browser_supervisor, "SUPERVISOR_REGISTRY", _Registry())
    return attached


def _cli(tmp_path, monkeypatch, body=_ENV_PROBE):
    script = tmp_path / "browser-use"
    script.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: [str(script)])
    return script


def _exec(code="print(1)", **kw):
    return json.loads(bu_cli.browser_exec(code, **kw))


class TestOff:
    """live_chrome off (the default): routing is what it was."""

    def test_local_routes_to_packaged_chromium(self, config, monkeypatch):
        monkeypatch.setattr(live, "preflight", lambda system=None: pytest.fail("live preflight ran"))
        env = {}
        assert bu_cli._route_backend(env, "", "t1", False) is None
        assert env["BU_CDP_WS"] == "ws://127.0.0.1:47000/devtools/browser/t1"
        assert env[bu_cli._PRIVATE_BROWSER_SENTINEL] == "1"
        assert live.LIVE_CHROME_SENTINEL not in env

    def test_real_profile_routes_to_snapshot(self, config, monkeypatch):
        config["use_real_profile"] = False
        monkeypatch.setattr(bu_cli, "_real_profile_consented", lambda: True)
        monkeypatch.setattr("tools.browser_session_real_profile.real_profile_cdp",
                            lambda: ("ws://127.0.0.1:47100/devtools/browser/rp", None))
        monkeypatch.setattr(live, "preflight", lambda system=None: pytest.fail("live preflight ran"))
        env = {}
        assert bu_cli._route_backend(env, "", "t1", False) is None
        assert env["BU_CDP_WS"] == "ws://127.0.0.1:47100/devtools/browser/rp"
        assert live.LIVE_CHROME_SENTINEL not in env

    def test_cloud_provider_session_exported(self, config, monkeypatch):
        monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: object())
        monkeypatch.setattr(bs, "get_session_info", lambda key: {"cdp_url": "wss://browser.example/cdp/abc"})
        monkeypatch.setattr(live, "preflight", lambda system=None: pytest.fail("live preflight ran"))
        env = {}
        assert bu_cli._route_backend(env, "", "t1", False) is None
        assert env["BU_CDP_WS"] == "wss://browser.example/cdp/abc"

    def test_description_is_unchanged(self, config):
        assert "LIVE CHROME" not in bu_cli._dynamic_schema_overrides()["description"]

    def test_default_config_is_off(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["browser"]["live_chrome"] is False


class TestOverridesWin:
    @pytest.fixture(autouse=True)
    def _on(self, config, monkeypatch):
        config["live_chrome"] = True
        monkeypatch.setattr(live, "preflight", lambda system=None: pytest.fail("live preflight ran"))

    def test_bu_cdp_env(self):
        env = {"BU_CDP_WS": "ws://operator:9222"}
        assert bu_cli._route_backend(env, "", "t1", False) is None
        assert env["BU_CDP_WS"] == "ws://operator:9222"
        assert live.LIVE_CHROME_SENTINEL not in env

    @pytest.mark.parametrize("override", ["http://127.0.0.1:9222", "wss://connect.example/x"])
    def test_cdp_override(self, monkeypatch, override):
        """/browser connect, BROWSER_CDP_URL and browser.cdp_url all arrive through get_cdp_override."""
        monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override", lambda: override)
        monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override_raw", lambda: override)
        env = {}
        assert bu_cli._route_backend(env, "", "t1", False) is None
        assert override in env.values()
        assert live.LIVE_CHROME_SENTINEL not in env


class TestOn:
    @pytest.fixture(autouse=True)
    def _on(self, config):
        config["live_chrome"] = True

    def test_exports_no_cdp_and_marks_local(self, chrome_running):
        env = {}
        assert bu_cli._route_backend(env, "", "t1", False) is None
        assert not bu_cli._has_cdp_env(env)
        assert env[bu_cli._BOT_DESKTOP_BROWSER_SENTINEL] == "local"
        assert env[live.LIVE_CHROME_SENTINEL] == "1"
        assert bu_cli._PRIVATE_BROWSER_SENTINEL not in env
        assert chrome_running

    def test_beats_cloud_provider(self, chrome_running, monkeypatch):
        monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: object())
        monkeypatch.setattr(bs, "get_session_info", lambda key: pytest.fail("cloud session requested"))
        env = {}
        assert bu_cli._route_backend(env, "", "t1", False) is None
        assert not bu_cli._has_cdp_env(env)

    def test_private_url_still_uses_sidecar(self, chrome_running, monkeypatch):
        monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: object())
        env = {}
        assert bu_cli._route_backend(env, "", "t1", False, force_local_sidecar=True) is None
        assert env["BU_CDP_WS"].endswith("t1::local")
        assert live.LIVE_CHROME_SENTINEL not in env
        assert not chrome_running

    def test_beats_real_profile_and_serves_local_arg(self, chrome_running, monkeypatch):
        monkeypatch.setattr(bu_cli, "_real_profile_consented", lambda: True)
        monkeypatch.setattr("tools.browser_session_real_profile.real_profile_cdp",
                            lambda: pytest.fail("snapshot browser started"))
        env = {}
        assert bu_cli._route_backend(env, "", "t1", True) is None
        assert not bu_cli._has_cdp_env(env)
        assert env[live.LIVE_CHROME_SENTINEL] == "1"

    def test_exec_subprocess_env_and_own_tab(self, tmp_path, monkeypatch, chrome_running, supervisor_attaches):
        _cli(tmp_path, monkeypatch)
        result = _exec("print('payload')")  # no session name: still its own tab
        assert result["success"] is True
        first = result["output"].splitlines()[0]
        assert "ws:unset url:unset" in first
        assert "marker:0" in first and "live:unset" in first
        assert "_hermes_live_own_tab" in result["output"]  # the fail-closed preamble, not the shared one
        assert "_hermes_ensure_own_tab" not in result["output"]
        assert result["output"].index("_hermes_live_own_tab") < result["output"].index("print('payload')")

    def test_no_cloud_autospawn_fallback(self, tmp_path, monkeypatch, chrome_running):
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        monkeypatch.setattr(bu_cli, "is_legacy_browser_use_cloud_config", lambda cfg: True)
        _cli(tmp_path, monkeypatch)
        assert "autospawn:unset" in _exec()["output"]

    def test_no_second_connection_for_the_supervisor(self, tmp_path, monkeypatch, chrome_running,
                                                     supervisor_attaches):
        """A second CDP connection would raise a second Allow popup."""
        _cli(tmp_path, monkeypatch)
        _exec(task_id="t1")
        assert supervisor_attaches == []
        from tools.browser_session_cdp import ensure_cdp_supervisor
        ensure_cdp_supervisor("t1")
        assert supervisor_attaches == []

    def test_description_tells_model_the_user_clicks_allow(self):
        text = bu_cli._dynamic_schema_overrides()["description"]
        assert "the USER clicks Allow" in text
        assert "Never try to approve it yourself" in text

    def test_harness_messages_pass_through_verbatim(self, tmp_path, monkeypatch, chrome_running):
        msg = ("remote-debugging-setup: opened chrome://inspect/#remote-debugging in Chrome -- ask the user to "
               "click Allow. Retry after the user confirms; do not retry before.")
        _cli(tmp_path, monkeypatch, f'cat > /dev/null\necho "{msg}" >&2\nexit 1\n')
        assert _exec()["stderr"] == msg


def _profile(tmp_path, *, running):
    base = tmp_path / "Chrome"
    base.mkdir()
    (base / "Local State").write_text("{}")
    if running:
        os.symlink(f"testhost-{os.getpid()}", base / "SingletonLock")
    return base


@pytest.fixture
def no_spawning(monkeypatch):
    """Any process start while routing is a test failure: Hermes must never launch Chrome."""
    def boom(*a, **kw):
        pytest.fail(f"process started: {a!r}")
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(os, "system", boom)


class TestChromeNotRunning:
    @pytest.fixture(autouse=True)
    def _on(self, config):
        config["live_chrome"] = True

    def test_error_and_no_launch(self, tmp_path, monkeypatch, no_spawning):
        monkeypatch.setattr(live, "_profile_dirs", lambda system: [_profile(tmp_path, running=False)])
        marker = tmp_path / "cli-ran"
        _cli(tmp_path, monkeypatch, f"touch {marker}\n")
        result = _exec()
        assert "no running Chrome" in result["error"]
        assert "chrome://inspect/#remote-debugging" in result["error"]
        assert not marker.exists()

    def test_stale_singleton_lock_is_not_running(self, tmp_path, monkeypatch):
        base = _profile(tmp_path, running=False)
        os.symlink("testhost-999999999", base / "SingletonLock")
        monkeypatch.setattr(live, "_profile_dirs", lambda system: [base])
        assert "no running Chrome" in live.preflight("Linux")

    def test_running_chrome_passes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(live, "_profile_dirs", lambda system: [_profile(tmp_path, running=True)])
        assert live.preflight("Linux") is None


class TestMacosAccess:
    @pytest.mark.parametrize("name", ["Local State", "DevToolsActivePort"])
    def test_permission_error_names_folder(self, tmp_path, monkeypatch, name):
        base = _profile(tmp_path, running=True)
        monkeypatch.setattr(live, "_profile_dirs", lambda system: [base])
        real = live._read_probe

        def probe(path):
            if path.name == name:
                raise PermissionError(1, "Operation not permitted", str(path))
            real(path)

        monkeypatch.setattr(live, "_read_probe", probe)
        err = live.preflight("Darwin")
        assert str(base) in err and "System Settings" in err and "Operation not permitted" in err

    @pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), reason="needs chmod")
    def test_unreadable_local_state(self, tmp_path, monkeypatch):
        base = _profile(tmp_path, running=True)
        (base / "Local State").chmod(0)
        monkeypatch.setattr(live, "_profile_dirs", lambda system: [base])
        assert str(base) in live.preflight("Darwin")

    def test_surfaces_through_browser_exec_without_workaround(self, tmp_path, monkeypatch, config, no_spawning):
        config["live_chrome"] = True
        base = _profile(tmp_path, running=True)
        monkeypatch.setattr(live, "_profile_dirs", lambda system: [base])
        monkeypatch.setattr(live, "_read_probe", lambda path: (_ for _ in ()).throw(PermissionError(1, "x")))
        marker = tmp_path / "cli-ran"
        _cli(tmp_path, monkeypatch, f"touch {marker}\n")
        assert str(base) in _exec()["error"]
        assert not marker.exists()


class TestTeardownStopsNoBrowser:
    def test_cleanup_after_live_session(self, tmp_path, config, chrome_running, monkeypatch, supervisor_attaches):
        config["live_chrome"] = True
        _cli(tmp_path, monkeypatch)
        stopped = []
        monkeypatch.setattr(browser_launcher, "stop_browser", lambda key, *a, **kw: stopped.append(key))
        monkeypatch.setattr(lifecycle, "release_session_backend", lambda *a, **kw: stopped.append(a))
        monkeypatch.setattr(bu_cli, "stop_harness_daemons", lambda *a, **kw: None)
        assert _exec(task_id="t1")["success"] is True
        assert "t1" not in bs._active_sessions  # a live session owns no launched/cloud browser to close
        lifecycle.cleanup_browser("t1")
        lifecycle.cleanup_all_browsers()
        assert stopped == []


class TestNeverApproves:
    @pytest.fixture(autouse=True)
    def _on(self, config):
        config["live_chrome"] = True

    @pytest.mark.parametrize("code", [
        "import subprocess; subprocess.run(['browser-harness', 'mac-approve'])",
        "from browser_harness import macos; macos.approve()",
        "import browser_harness.macos",
        "mac_approve()",
    ])
    def test_model_code_reaching_for_approval_is_refused(self, tmp_path, monkeypatch, chrome_running, code):
        marker = tmp_path / "cli-ran"
        _cli(tmp_path, monkeypatch, f"touch {marker}\n")
        assert "click Allow" in _exec(code)["error"]
        assert not marker.exists()

    def test_hermes_never_runs_mac_approve_on_any_path(self, tmp_path, monkeypatch, config, chrome_running):
        """Across a normal call, a harness 'run mac-approve' hint on stderr, and a timeout."""
        config["live_chrome"] = True
        argvs = []
        real_popen = subprocess.Popen

        def spy(cmd, *a, **kw):
            argvs.append(cmd)
            return real_popen(cmd, *a, **kw)

        monkeypatch.setattr(subprocess, "Popen", spy)
        monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **kw: argvs.append(cmd))
        hint = 'Chrome is asking "Allow remote debugging?" - run `browser-harness mac-approve`'
        _cli(tmp_path, monkeypatch, f'cat > /dev/null\necho \'{hint}\' >&2\n')
        assert "mac-approve" in _exec()["stderr"]  # relayed to the model, never executed
        _cli(tmp_path, monkeypatch, "cat > /dev/null\nsleep 30\n")
        assert "click Allow" in _exec(timeout_s=5)["error"]
        assert not any("mac-approve" in " ".join(map(str, c)) for c in argvs)
        assert "browser_harness.macos" not in sys.modules


_NAME_PROBE = 'cat > /dev/null\necho "name:${BU_NAME:-default} ws:${BU_CDP_WS:-unset}"\n'


class TestToggleDetaches:
    """The switch is read on every call: a daemon attached to the live Chrome must never serve a call made
    after it is turned off, and a warm packaged daemon must never serve a live call."""

    @pytest.fixture
    def reloaded(self, monkeypatch):
        calls = []
        monkeypatch.setattr(bu_cli, "_reload_daemons", lambda daemons: calls.extend(daemons))
        with bu_cli._driven_daemons_lock:
            bu_cli._driven_daemons.clear()
        yield calls
        with bu_cli._driven_daemons_lock:
            bu_cli._driven_daemons.clear()

    @pytest.mark.parametrize("session", ["", "r7k2"])
    def test_on_then_off_uses_another_daemon_and_stops_the_live_one(self, tmp_path, monkeypatch, config,
                                                                    chrome_running, reloaded, session):
        _cli(tmp_path, monkeypatch, _NAME_PROBE)
        config["live_chrome"] = True
        on = _exec(session=session)["output"]
        live_name = f"_live_{session or 'default'}"
        assert f"name:{live_name} ws:unset" in on
        assert reloaded == []  # still on: nothing stopped

        config["live_chrome"] = False
        off = _exec(session=session)["output"]
        assert f"name:{session or 'default'} ws:ws://127.0.0.1:47000/" in off
        assert [d[-1] for d in reloaded] == [live_name]  # the live daemon was stopped
        with bu_cli._driven_daemons_lock:
            assert all(d[-1] != live_name for d in bu_cli._driven_daemons)

    def test_off_then_on_never_reuses_the_packaged_daemon(self, tmp_path, monkeypatch, config, chrome_running,
                                                          reloaded):
        _cli(tmp_path, monkeypatch, _NAME_PROBE)
        off = _exec()["output"]
        assert "name:default ws:ws://127.0.0.1:47000/" in off
        config["live_chrome"] = True
        on = _exec()["output"]
        assert "name:_live_default ws:unset" in on
        assert reloaded == []  # the packaged daemon is left alone; it simply isn't used

    def test_off_stops_only_this_profiles_live_daemons(self, monkeypatch, config, reloaded):
        """A multi-profile gateway: one profile's switch never stops another profile's live daemon."""
        mine, theirs = ("dirs-a", "_live_default"), ("dirs-b", "_live_default")
        packaged = ("dirs-a", "default")
        with bu_cli._driven_daemons_lock:
            bu_cli._driven_daemons.update({mine: "home-a", theirs: "home-b", packaged: "home-a"})
        monkeypatch.setattr(bu_cli, "hermes_home_key", lambda: "home-a")
        bu_cli._stop_live_daemons_if_off()  # config has live_chrome unset: off for home-a
        assert reloaded == [mine]
        with bu_cli._driven_daemons_lock:
            assert theirs in bu_cli._driven_daemons and packaged in bu_cli._driven_daemons
            assert mine not in bu_cli._driven_daemons
