"""Tests for browser first-open timeout and timeout diagnostics."""

from unittest.mock import patch

import pytest

import tools.browser_tool as bt


@pytest.fixture(autouse=True)
def _reset_browser_caches():
    bt._cached_command_timeout = None
    bt._command_timeout_resolved = False
    yield
    bt._cached_command_timeout = None
    bt._command_timeout_resolved = False


class TestOpenCommandTimeout:
    def test_first_open_uses_longer_floor(self, monkeypatch):
        monkeypatch.setattr(bt, "_get_command_timeout", lambda: 30)
        assert bt._get_open_command_timeout(first_open=True) == bt.MIN_FIRST_OPEN_TIMEOUT
        assert bt._get_open_command_timeout(first_open=False) == bt.MIN_OPEN_TIMEOUT

    def test_respects_config_above_floor(self, monkeypatch):
        monkeypatch.setattr(bt, "_get_command_timeout", lambda: 180)
        assert bt._get_open_command_timeout(first_open=True) == 180
        assert bt._get_open_command_timeout(first_open=False) == 180


class TestSandboxBypass:
    def test_docker_triggers_bypass(self, monkeypatch):
        monkeypatch.setattr(bt, "_running_in_docker", lambda: True)
        assert bt._needs_chromium_sandbox_bypass() is True

    def test_apparmor_userns_triggers_bypass(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt, "_running_in_docker", lambda: False)
        sysctl = tmp_path / "apparmor_restrict_unprivileged_userns"
        sysctl.write_text("1\n", encoding="utf-8")

        import builtins

        real_open = builtins.open

        def _open(path, *args, **kwargs):
            if "apparmor_restrict_unprivileged_userns" in str(path):
                return real_open(sysctl, *args, **kwargs)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", _open)
        assert bt._needs_chromium_sandbox_bypass() is True


class TestTimeoutErrorFormatting:
    def test_includes_stderr_detail(self):
        err = bt._format_browser_timeout_error(
            "open",
            120,
            "",
            "Daemon process exited during startup",
        )
        assert "120 seconds" in err
        assert "Daemon process exited" in err

    def test_sandbox_hint(self):
        err = bt._format_browser_timeout_error(
            "open",
            60,
            "",
            "No usable sandbox!",
        )
        assert "AGENT_BROWSER_ARGS" in err

    def test_local_install_hint(self, monkeypatch):
        monkeypatch.setattr(bt, "_is_local_mode", lambda: True)
        monkeypatch.setattr(bt, "_running_in_docker", lambda: False)
        err = bt._format_browser_timeout_error("open", 60, "", "")
        assert "agent-browser install --with-deps" in err


class TestReadCommandOutputFiles:
    def test_reads_stdout_and_stderr(self, tmp_path):
        stdout_path = tmp_path / "out"
        stderr_path = tmp_path / "err"
        stdout_path.write_text("ok", encoding="utf-8")
        stderr_path.write_text("warn", encoding="utf-8")
        stdout, stderr = bt._read_command_output_files(str(stdout_path), str(stderr_path))
        assert stdout == "ok"
        assert stderr == "warn"


class TestBrowserNavigateOpenTimeout:
    def test_first_navigation_uses_first_open_timeout(self, monkeypatch):
        captured: dict = {}

        def fake_run(task_id, command, args, timeout=None):
            if command == "open":
                captured["timeout"] = timeout
            return {"success": True, "data": {"title": "t", "url": args[0] if args else ""}}

        monkeypatch.setattr(bt, "_get_open_command_timeout", lambda first_open=False: 120 if first_open else 60)
        monkeypatch.setattr(bt, "_run_browser_command", fake_run)
        monkeypatch.setattr(bt, "_get_session_info", lambda key: {"_first_nav": True, "features": {}})
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt, "_is_local_backend", lambda: True)
        monkeypatch.setattr(bt, "_is_local_sidecar_key", lambda key: False)
        monkeypatch.setattr(bt, "_navigation_session_key", lambda task_id, url: task_id)
        monkeypatch.setattr(bt, "_maybe_start_recording", lambda *a, **kw: None)
        monkeypatch.setattr(bt, "check_website_access", lambda url: None)

        bt.browser_navigate("https://example.com", task_id="task-1")
        assert captured["timeout"] == 120
