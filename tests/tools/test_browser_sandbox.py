"""Regression tests for preserving Chromium's sandbox in containers."""

import os
import shutil

import pytest

import tools.browser_tool as bt


class _FakeBrowserProcess:
    def __init__(self, stdout_fd):
        os.write(stdout_fd, b'{"success": true, "data": {}}')
        self.returncode = 0

    def wait(self, timeout=None):
        return 0


class TestSandboxOptIn:
    def test_force_sandbox_disables_automatic_bypass(self, monkeypatch):
        monkeypatch.setenv("AGENT_BROWSER_FORCE_SANDBOX", "1")
        monkeypatch.setattr(bt, "_running_in_docker", lambda: True)

        assert bt._chromium_sandbox_requested() is True
        assert bt._needs_chromium_sandbox_bypass() is False

    def test_force_sandbox_rejects_sandbox_bypass_flags(self, monkeypatch):
        monkeypatch.setenv("AGENT_BROWSER_ARGS", "--disable-dev-shm-usage,--no-sandbox")

        with pytest.raises(ValueError, match="AGENT_BROWSER_FORCE_SANDBOX"):
            bt._sandbox_chromium_args(os.environ)

    @pytest.mark.parametrize("flag", ["--single-process", "--in-process-gpu"])
    def test_force_sandbox_rejects_process_isolation_flags(self, monkeypatch, flag):
        monkeypatch.setenv("AGENT_BROWSER_ARGS", flag)

        with pytest.raises(ValueError, match="AGENT_BROWSER_FORCE_SANDBOX"):
            bt._sandbox_chromium_args(os.environ)

    def test_sandbox_command_keeps_control_flags_owned_by_hermes(self):
        command = bt._build_sandboxed_chromium_command(
            "/usr/bin/chromium",
            "/tmp/hermes-browser-profile",
            headed=False,
            extra_args=["--disable-dev-shm-usage"],
        )

        assert "--no-sandbox" not in command
        assert "--no-zygote-sandbox" not in command
        assert "--remote-debugging-address=127.0.0.1" in command
        assert "--remote-debugging-port=0" in command
        assert "--user-data-dir=/tmp/hermes-browser-profile" in command


class TestSandboxBrowserRouting:
    def test_sandbox_chromium_env_strips_inherited_launcher_flags(self, monkeypatch):
        captured = {}
        env = {
            key: "--single-process"
            for key in bt._SANDBOX_LAUNCHER_ENV_VARS
        }
        monkeypatch.setattr(bt.os, "makedirs", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            bt,
            "_read_sandbox_cdp_url",
            lambda _: "ws://127.0.0.1:4321/devtools/browser/test",
        )

        def fake_popen(_command, **kwargs):
            captured["env"] = kwargs["env"]
            return _FakeChromiumProcess()

        monkeypatch.setattr(bt.subprocess, "Popen", fake_popen)

        bt._launch_sandboxed_chromium(
            "/usr/bin/chromium",
            os.path.join(os.getcwd(), ".sandbox-chromium-profile"),
            env,
            headed=False,
        )

        assert all(key not in captured["env"] for key in bt._SANDBOX_LAUNCHER_ENV_VARS)
        assert all(key in env for key in bt._SANDBOX_LAUNCHER_ENV_VARS)

    def test_force_sandbox_connects_agent_browser_over_cdp(self, monkeypatch):
        captured = {}
        runtime_dir = os.path.join(os.getcwd(), ".sandbox-browser-runtime")
        os.makedirs(runtime_dir, exist_ok=True)
        session_info = {
            "session_name": "h_sandbox",
            "cdp_url": None,
            "features": {"local": True},
        }

        monkeypatch.setenv("AGENT_BROWSER_FORCE_SANDBOX", "1")
        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "/usr/bin/agent-browser")
        monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _: False)
        monkeypatch.setattr(bt, "_is_local_mode", lambda: True)
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt, "_is_headed_mode", lambda: False)
        monkeypatch.setattr(bt, "_get_browser_engine", lambda: "auto")
        monkeypatch.setattr(bt, "_chromium_installed", lambda: True)
        monkeypatch.setattr(bt, "_find_chromium_executable", lambda: "/usr/bin/chromium")
        monkeypatch.setattr(bt, "_get_session_info", lambda _: session_info)
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {"PATH": "/usr/bin"})
        monkeypatch.setattr(bt, "_merge_browser_path", lambda path: path)
        monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: runtime_dir)
        monkeypatch.setattr(bt, "_write_owner_pid", lambda *_: None)
        monkeypatch.setattr(bt, "_safe_command_timeout", lambda: 5)
        monkeypatch.setattr(bt, "_launch_sandboxed_chromium", self._fake_launch(captured))
        real_makedirs = bt.os.makedirs

        def safe_makedirs(path, mode=0o777, exist_ok=False):
            return real_makedirs(path, mode=0o777, exist_ok=exist_ok)

        monkeypatch.setattr(bt.os, "makedirs", safe_makedirs)

        def fake_popen(command, **kwargs):
            captured["agent_browser_command"] = command
            return _FakeBrowserProcess(kwargs["stdout"])

        monkeypatch.setattr(bt.subprocess, "Popen", fake_popen)

        try:
            result = bt._run_browser_command("task-1", "open", ["about:blank"])

            assert result["success"] is True
            command = captured["agent_browser_command"]
            assert "--cdp" in command
            assert "ws://127.0.0.1:4321/devtools/browser/test" in command
            assert "--session" not in command
            assert "--no-sandbox" not in captured["chromium_args"]
            assert "--no-zygote-sandbox" not in captured["chromium_args"]
        finally:
            shutil.rmtree(runtime_dir, ignore_errors=True)

    @staticmethod
    def _fake_launch(captured):
        def launch(*args, **kwargs):
            captured["chromium_args"] = bt._build_sandboxed_chromium_command(
                "/usr/bin/chromium",
                args[1],
                headed=kwargs["headed"],
                extra_args=bt._sandbox_chromium_args(args[2]),
            )
            return _FakeChromiumProcess(), "ws://127.0.0.1:4321/devtools/browser/test"

        return launch


class _FakeChromiumProcess:
    pid = 1234

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0
