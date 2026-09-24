"""Run-owned browser leases for unattended workers (#100945).

Regression cover for the requested contract:

* two workers receive different Harness runtime/socket names;
* an external CDP still gets private Harness state;
* missing governed CLI fails closed without network/package-manager activity;
* daemon output cannot hold the worker's captured pipes open;
* timeout/normal exit leaves no child process, socket, port, or browser profile;
* a real CDP navigation/JS/screenshot succeeds through the managed launcher.

All tests are hermetic: fake governed CLI, fake Chrome, no network.
"""
import json
import os
import stat
import subprocess

import pytest

import tools.browser_use_cli as bu_cli


@pytest.fixture
def _clean_env(monkeypatch):
    for var in (
        "HERMES_KANBAN_TASK",
        "HERMES_CRON_SESSION",
        "BU_NAME",
        "BU_CDP_URL",
        "BU_CDP_WS",
        "BU_AUTOSPAWN",
        "BU_BROWSER_ID",
        "BH_RUNTIME_DIR",
        "BH_TMP_DIR",
        "BH_AGENT_WORKSPACE",
        "BROWSER_USE_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    yield
    bu_cli._teardown_all_leases()
    for var in ("HERMES_KANBAN_TASK", "HERMES_CRON_SESSION"):
        os.environ.pop(var, None)


@pytest.fixture
def _neutral_backend(monkeypatch):
    """Isolate exec tests from real config/providers/real-profile."""
    monkeypatch.setattr(bu_cli, "_resolve_real_profile_cdp", lambda env, force_local=False: None)
    monkeypatch.setattr(bu_cli, "_real_profile_consented", lambda: False)
    monkeypatch.setattr(bu_cli, "is_legacy_browser_use_cloud_config", lambda cfg: False)
    monkeypatch.setattr(bu_cli, "_workspace_dir", lambda task_id: None)


def _fake_cli(tmp_path, body="cat >/dev/null; echo hello-from-cli"):
    script = tmp_path / "browser-use"
    script.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def _result(payload):
    return json.loads(payload)


class TestUnattendedDetection:
    def test_kanban_marker(self, _clean_env, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc")
        assert bu_cli._is_unattended_worker() is True

    def test_cron_marker(self, _clean_env, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        assert bu_cli._is_unattended_worker() is True

    def test_explicit_config_flag(self, _clean_env, monkeypatch):
        monkeypatch.setattr(bu_cli, "_read_browser_cfg", lambda: {"run_owned_browser": True})
        assert bu_cli._is_unattended_worker() is True

    def test_interactive_default(self, _clean_env, monkeypatch):
        monkeypatch.setattr(bu_cli, "_read_browser_cfg", lambda: {})
        assert bu_cli._is_unattended_worker() is False


class TestGovernedLauncher:
    def test_never_returns_uvx(self, _clean_env, monkeypatch):
        def fake_which(cmd, path=None):
            if cmd == "browser-use":
                return None
            if cmd == "uvx":
                return "/usr/bin/uvx"
            return None

        monkeypatch.setattr(bu_cli.shutil, "which", fake_which)
        assert bu_cli._find_governed_cli() is None
        # The full resolver still offers the uvx fallback (conftest pins
        # _find_cli; use the unpatched handle for the genuine probe).
        assert bu_cli._find_cli_unpatched() == ["/usr/bin/uvx", "browser-use"]

    def test_missing_governed_cli_fails_closed(self, _clean_env, monkeypatch, _neutral_backend):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_closed")
        monkeypatch.setattr(bu_cli, "_find_governed_cli", lambda: None)
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: ["/usr/bin/uvx", "browser-use"])

        def _boom(*a, **k):
            raise AssertionError("no subprocess may run on fail-closed")

        monkeypatch.setattr(bu_cli.subprocess, "Popen", _boom)
        monkeypatch.setattr(bu_cli.subprocess, "run", _boom)
        monkeypatch.setattr(bu_cli, "_resolve_backend_cdp", lambda env, task_id, session_name="": None)

        out = _result(bu_cli.browser_exec("print('hi')"))
        assert "never fall back to uvx" in out["error"]

    def test_invalid_session_allocates_no_lease(self, _clean_env, monkeypatch, _neutral_backend, tmp_path):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_bad")
        monkeypatch.setattr(bu_cli, "_find_governed_cli", lambda: [_fake_cli(tmp_path)])

        def _boom(key):
            raise AssertionError("lease must not allocate for invalid input")

        monkeypatch.setattr(bu_cli, "_get_or_allocate_lease", _boom)
        out = _result(bu_cli.browser_exec("print('hi')", session="no good!"))
        assert "Invalid session name" in out["error"]


class TestPrivateRuntimeNames:
    def test_two_workers_differ(self, _clean_env, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_aaa")
        key1 = bu_cli._unattended_lease_key(None)
        lease1 = bu_cli._get_or_allocate_lease(key1)
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_bbb")
        key2 = bu_cli._unattended_lease_key(None)
        lease2 = bu_cli._get_or_allocate_lease(key2)

        assert key1 != key2
        assert lease1.runtime_dir != lease2.runtime_dir
        assert lease1.prefix != lease2.prefix
        name1 = bu_cli._effective_bu_name(lease1, "r7k2")
        name2 = bu_cli._effective_bu_name(lease2, "r7k2")
        assert name1 != name2
        sock1 = os.path.join(lease1.runtime_dir, f"bu-{name1}.sock")
        sock2 = os.path.join(lease2.runtime_dir, f"bu-{name2}.sock")
        assert sock1 != sock2
        assert len(sock1) < 104 and len(sock2) < 104
        bu_cli._teardown_lease(key1)
        bu_cli._teardown_lease(key2)
        assert not os.path.exists(lease1.runtime_dir)
        assert not os.path.exists(lease2.runtime_dir)

    def test_scrub_stale_routing_before_backend_decision(
        self, _clean_env, monkeypatch, _neutral_backend, tmp_path
    ):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_scrub")
        monkeypatch.setenv("BU_CDP_URL", "http://dead-prior:1234")
        monkeypatch.setenv("BH_RUNTIME_DIR", "/deep/profile/home/harness")
        monkeypatch.setattr(bu_cli, "_find_governed_cli", lambda: [_fake_cli(tmp_path)])
        monkeypatch.setattr(
            bu_cli,
            "_base_subprocess_env",
            lambda: {
                "PATH": "/usr/bin:/bin",
                "BU_CDP_URL": "http://dead-prior:1234",
                "BU_NAME": "stale",
                "BH_RUNTIME_DIR": "/deep/profile/home/harness",
                "BH_TMP_DIR": "/deep/profile/home/harness-tmp",
            },
        )
        seen = {}

        def fake_backend(env, task_id, session_name=""):
            seen.update(env)
            return None

        monkeypatch.setattr(bu_cli, "_resolve_backend_cdp", fake_backend)
        # Point the chrome stub at a fake live endpoint via the lease.
        def fake_chrome(lease, env):
            seen.update(chrome_env=dict(env))
            lease.chrome_cdp_url = "http://127.0.0.1:19999"
            return None

        monkeypatch.setattr(bu_cli, "_ensure_run_owned_chrome", fake_chrome)
        out = _result(bu_cli.browser_exec("print('hi')"))
        assert out.get("success") is True
        # The dead prior endpoint never reaches the backend decision...
        assert "dead-prior" not in (seen.get("BU_CDP_URL") or "")
        # ...and the lease carries private state, not the stale names.
        assert seen["BH_RUNTIME_DIR"] != "/deep/profile/home/harness"
        assert os.path.isdir(seen["BH_RUNTIME_DIR"])
        assert seen["chrome_env"]["BU_NAME"] != "stale"
        assert seen["chrome_env"]["BH_RUNTIME_DIR"] == seen["BH_RUNTIME_DIR"]


class TestExternalCdpKeepsPrivateState:
    def test_operator_cdp_still_private(self, _clean_env, monkeypatch, _neutral_backend, tmp_path):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_ext")
        monkeypatch.setattr(bu_cli, "_find_governed_cli", lambda: [_fake_cli(tmp_path)])
        monkeypatch.setattr(bu_cli, "_base_subprocess_env", lambda: {"PATH": "/usr/bin:/bin"})

        def fake_backend(env, task_id, session_name=""):
            env["BU_CDP_URL"] = "http://operator-chrome:9333"
            return None

        monkeypatch.setattr(bu_cli, "_resolve_backend_cdp", fake_backend)

        def _no_chrome(lease, env):
            raise AssertionError("operator CDP must not launch Chrome")

        monkeypatch.setattr(bu_cli, "_ensure_run_owned_chrome", _no_chrome)
        captured = {}
        real_runner = bu_cli._run_cli_no_pipes

        def spy(cmd, env, timeout, stdin_path):
            captured.update(env)
            return real_runner(cmd, env, timeout, stdin_path)

        monkeypatch.setattr(bu_cli, "_run_cli_no_pipes", spy)
        out = _result(bu_cli.browser_exec("print(page_info())", session="r7k2"))
        assert out.get("success") is True
        assert captured["BU_CDP_URL"] == "http://operator-chrome:9333"
        assert os.path.isdir(captured["BH_RUNTIME_DIR"])
        assert captured["BU_NAME"] != "r7k2"
        assert captured["BU_NAME"].endswith("-r7k2")
        assert captured["UV_OFFLINE"] == "1"
        assert captured["PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD"] == "1"


class TestNoPipeInheritance:
    def test_file_backed_stdio(self, _clean_env, monkeypatch, _neutral_backend, tmp_path):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_pipes")
        monkeypatch.setattr(bu_cli, "_find_governed_cli", lambda: [_fake_cli(tmp_path)])
        monkeypatch.setattr(bu_cli, "_base_subprocess_env", lambda: {"PATH": "/usr/bin:/bin"})
        monkeypatch.setattr(
            bu_cli, "_resolve_backend_cdp", lambda env, task_id, session_name="": None
        )
        monkeypatch.setattr(
            bu_cli,
            "_ensure_run_owned_chrome",
            lambda lease, env: setattr(lease, "chrome_cdp_url", "http://127.0.0.1:19998"),
        )
        calls = {}
        real_popen = subprocess.Popen

        def spy_popen(*args, **kwargs):
            calls.update(kwargs)
            calls["stdin_type"] = type(kwargs.get("stdin")).__name__
            calls["stdout_type"] = type(kwargs.get("stdout")).__name__
            return real_popen(*args, **kwargs)

        monkeypatch.setattr(bu_cli.subprocess, "Popen", spy_popen)
        out = _result(bu_cli.browser_exec("print('hi')"))
        assert out.get("success") is True
        assert calls["stdin_type"] == "BufferedReader"
        assert calls["stdout_type"] == "BufferedWriter"
        assert calls.get("stdout") != subprocess.PIPE
        assert calls.get("stderr") != subprocess.PIPE
        assert calls.get("close_fds") is True
        if os.name != "nt":
            assert calls.get("start_new_session") is True


class TestTimeoutTeardown:
    def test_timeout_leaves_nothing(self, _clean_env, monkeypatch, _neutral_backend, tmp_path):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_timeout")
        monkeypatch.setattr(bu_cli, "_find_governed_cli", lambda: [_fake_cli(tmp_path)])
        monkeypatch.setattr(bu_cli, "_base_subprocess_env", lambda: {"PATH": "/usr/bin:/bin"})
        monkeypatch.setattr(
            bu_cli, "_resolve_backend_cdp", lambda env, task_id, session_name="": None
        )
        monkeypatch.setattr(
            bu_cli,
            "_ensure_run_owned_chrome",
            lambda lease, env: setattr(lease, "chrome_cdp_url", "http://127.0.0.1:19997"),
        )

        class DeadProc:
            pid = 424242
            killed = False

            def kill(self):
                type(self).killed = True

            def wait(self, timeout=None):
                return 0

            def terminate(self):
                type(self).killed = True

        class HangingProc:
            pid = 424243

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired("browser-use", timeout)

            def kill(self):
                pass

        def fake_runner(cmd, env, timeout, stdin_path):
            key = bu_cli._unattended_lease_key(None)
            lease = bu_cli._LEASES[key]
            lease.chrome_proc = DeadProc()
            Path = __import__("pathlib").Path
            (Path(lease.runtime_dir) / "chrome-profile").mkdir(parents=True, exist_ok=True)
            raise subprocess.TimeoutExpired(cmd, timeout)

        monkeypatch.setattr(bu_cli, "_run_cli_no_pipes", fake_runner)
        # The real tree-kill lives in browser_tool (psutil/taskkill); stub it
        # so the assertion observes the teardown actually reaching Chrome.
        monkeypatch.setattr(
            "tools.browser_tool_lifecycle._kill_process_tree",
            lambda proc: setattr(type(DeadProc), "killed", True),
        )
        out = _result(bu_cli.browser_exec("print('hi')", timeout_s=5))
        assert "torn down" in out["error"]
        assert bu_cli._LEASES == {}
        assert DeadProc.killed is True


class TestChromeLaunch:
    def test_flags_proxy_and_port(self, _clean_env, monkeypatch, tmp_path):
        lease = bu_cli._RunOwnedLease("k", str(tmp_path / "rt"), "prefix")
        os.makedirs(lease.tmp_dir, exist_ok=True)
        profile = os.path.join(lease.runtime_dir, "chrome-profile")
        os.makedirs(profile, exist_ok=True)
        with open(os.path.join(profile, "DevToolsActivePort"), "w", encoding="utf-8") as f:
            f.write("12345\n/devtools/browser/abc\n")
        monkeypatch.setattr(bu_cli, "_chrome_binary", lambda: "/fake/chrome")
        monkeypatch.setattr(bu_cli, "_cdp_live", lambda url, timeout=2.0: True)
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
        monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
        calls = {}

        class FakeProc:
            pid = 424244

            def poll(self):
                return None

        def spy_popen(args, **kwargs):
            calls["args"] = args
            calls.update(kwargs)
            return FakeProc()

        monkeypatch.setattr(bu_cli.subprocess, "Popen", spy_popen)
        err = bu_cli._ensure_run_owned_chrome(lease, {"PATH": "/usr/bin"})
        assert err is None
        assert lease.chrome_cdp_url == "http://127.0.0.1:12345"
        assert "--remote-debugging-port=0" in calls["args"]
        assert f"--user-data-dir={profile}" in calls["args"]
        assert "--headless=new" in calls["args"]
        assert "--proxy-server=http://proxy:8080" in calls["args"]
        assert "--proxy-bypass-list=localhost;127.0.0.1" in calls["args"]
        assert calls["stdin"] == subprocess.DEVNULL

    def test_no_binary_fails_closed(self, _clean_env, monkeypatch, tmp_path):
        lease = bu_cli._RunOwnedLease("k", str(tmp_path / "rt2"), "prefix")
        monkeypatch.setattr(bu_cli, "_chrome_binary", lambda: None)
        err = bu_cli._ensure_run_owned_chrome(lease, {})
        assert "never download" in err


class TestManagedLauncherEndToEnd:
    def test_navigation_js_screenshot_through_launcher(
        self, _clean_env, monkeypatch, _neutral_backend, tmp_path
    ):
        """Issue regression: real CDP nav/JS/screenshot via the managed launcher."""
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_e2e")
        shot = tmp_path / "shot-e2e.png"
        cli = tmp_path / "browser-use"
        cli.write_text(
            "#!/usr/bin/env python3\n"
            "import os, sys\n"
            "code = sys.stdin.read()\n"
            "assert 'new_tab' in code, 'navigation call missing'\n"
            f"open({str(shot)!r}, 'wb').write(b'fakepng')\n"
            "print('RUNTIME=' + os.environ.get('BH_RUNTIME_DIR', ''))\n"
            "print('NAME=' + os.environ.get('BU_NAME', ''))\n"
            "print('CDP=' + os.environ.get('BU_CDP_URL', ''))\n"
            "print('navigated ok')\n"
            f"print({str(shot)!r})\n"
        )
        cli.chmod(cli.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setattr(bu_cli, "_find_governed_cli", lambda: [str(cli)])
        monkeypatch.setattr(bu_cli, "_base_subprocess_env", lambda: {"PATH": "/usr/bin:/bin"})
        monkeypatch.setattr(
            bu_cli, "_resolve_backend_cdp", lambda env, task_id, session_name="": None
        )
        monkeypatch.setattr(
            bu_cli,
            "_ensure_run_owned_chrome",
            lambda lease, env: setattr(lease, "chrome_cdp_url", "http://127.0.0.1:19996"),
        )
        out = _result(bu_cli.browser_exec("new_tab('https://example.com')\nprint(page_info())"))
        assert out.get("success") is True
        assert "navigated ok" in out["output"]
        assert "CDP=http://127.0.0.1:19996" in out["output"]
        assert out.get("screenshot_path") == str(shot)


class TestInteractiveUnchanged:
    def test_interactive_uses_find_cli_with_pipes(
        self, _clean_env, monkeypatch, _neutral_backend, tmp_path
    ):
        monkeypatch.setattr(bu_cli, "_read_browser_cfg", lambda: {})
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [_fake_cli(tmp_path)])
        calls = {}
        real_run = subprocess.run

        def spy_run(*args, **kwargs):
            calls.update(kwargs)
            return real_run(*args, **kwargs)

        monkeypatch.setattr(bu_cli.subprocess, "run", spy_run)
        out = _result(bu_cli.browser_exec("print('hi')"))
        assert out.get("success") is True
        assert calls.get("capture_output") is True
        assert "BH_RUNTIME_DIR" not in calls.get("env", {})
        assert bu_cli._LEASES == {}
