"""Tests for native Windows SSH environment support."""

import base64
import json
import subprocess
import zipfile
from unittest.mock import MagicMock

import pytest

from tools.environments import ssh as ssh_env
from tools.environments import ssh_windows
from tools.environments.ssh_windows import WindowsSSHEnvironment


class _FakeSyncManager:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def sync(self, **kwargs):
        pass

    def sync_back(self):
        pass


@pytest.fixture
def windows_env(monkeypatch):
    monkeypatch.setattr(ssh_windows, "_ensure_ssh_available", lambda: None)
    monkeypatch.setattr(WindowsSSHEnvironment, "_establish_connection", lambda self: None)
    monkeypatch.setattr(WindowsSSHEnvironment, "_detect_remote_home", lambda self: "C:/Users/alice")
    monkeypatch.setattr(WindowsSSHEnvironment, "_detect_remote_temp", lambda self: "C:/Users/alice/AppData/Local/Temp")
    monkeypatch.setattr(WindowsSSHEnvironment, "_ensure_remote_dirs", lambda self: None)
    monkeypatch.setattr(ssh_windows, "FileSyncManager", _FakeSyncManager)
    return WindowsSSHEnvironment(host="win.example", user="alice")


def test_detect_windows_ssh_host_true(monkeypatch):
    monkeypatch.setattr(ssh_env.shutil, "which", lambda name: "/usr/bin/ssh")

    def fake_run(cmd, **kwargs):
        assert "powershell" in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="hermes-windows-ssh\n", stderr="")

    monkeypatch.setattr(ssh_env.subprocess, "run", fake_run)

    assert ssh_env.detect_windows_ssh_host("host", "user") is True


def test_detect_windows_ssh_host_false_on_probe_failure(monkeypatch):
    monkeypatch.setattr(ssh_env.shutil, "which", lambda name: "/usr/bin/ssh")
    monkeypatch.setattr(
        ssh_env.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 127, stdout="", stderr="powershell: not found"),
    )

    assert ssh_env.detect_windows_ssh_host("host", "user") is False


def test_create_environment_uses_windows_backend(monkeypatch):
    from tools import terminal_tool

    created = {}

    class FakeWindowsEnv:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(terminal_tool, "_detect_windows_ssh_host", lambda *args: True)
    monkeypatch.setattr(terminal_tool, "_WindowsSSHEnvironment", FakeWindowsEnv)

    env = terminal_tool._create_environment(
        env_type="ssh",
        image="",
        cwd="~",
        timeout=90,
        ssh_config={"host": "win.example", "user": "alice", "port": 2222, "key": "/key"},
    )

    assert isinstance(env, FakeWindowsEnv)
    assert created == {
        "host": "win.example",
        "user": "alice",
        "port": 2222,
        "key_path": "/key",
        "cwd": "~",
        "timeout": 90,
    }


def test_windows_paths_are_used_for_sync_sources(monkeypatch):
    captured = {}
    monkeypatch.setattr(ssh_windows, "_ensure_ssh_available", lambda: None)
    monkeypatch.setattr(WindowsSSHEnvironment, "_establish_connection", lambda self: None)
    monkeypatch.setattr(WindowsSSHEnvironment, "_detect_remote_home", lambda self: "C:/Users/alice")
    monkeypatch.setattr(WindowsSSHEnvironment, "_detect_remote_temp", lambda self: "C:/Users/alice/AppData/Local/Temp")
    monkeypatch.setattr(WindowsSSHEnvironment, "_ensure_remote_dirs", lambda self: None)
    monkeypatch.setattr(ssh_windows, "FileSyncManager", _FakeSyncManager)

    def fake_iter_sync_files(container_base):
        captured["container_base"] = container_base
        return []

    monkeypatch.setattr(ssh_windows, "iter_sync_files", fake_iter_sync_files)

    env = WindowsSSHEnvironment(host="win.example", user="alice")
    manager = env._sync_manager
    manager.kwargs["get_files_fn"]()

    assert captured["container_base"] == "C:/Users/alice/.hermes"
    assert callable(manager.kwargs["bulk_upload_fn"])
    assert manager.kwargs["bulk_download_fn"] is None


def test_powershell_upload_writes_base64_payload(windows_env, monkeypatch, tmp_path):
    source = tmp_path / "hello.txt"
    source.write_text("hello", encoding="utf-8")

    calls = []

    def fake_run(script, *, timeout=30, input_data=None):
        calls.append((script, input_data))
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(windows_env, "_run_powershell_script", fake_run)

    windows_env._powershell_upload(str(source), "C:/Users/alice/.hermes/skills/hello.txt")

    assert "WriteAllBytes" in calls[0][0]
    assert "New-Item -ItemType Directory -Force -Path" in calls[0][0]
    payload = json.loads(calls[0][1])
    assert payload["path"] == "C:/Users/alice/.hermes/skills/hello.txt"
    assert base64.b64decode(payload["content"]) == b"hello"


def test_powershell_delete_uses_literal_paths(windows_env, monkeypatch):
    calls = []

    def fake_run(script, *, timeout=30, input_data=None):
        calls.append((script, input_data))
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(windows_env, "_run_powershell_script", fake_run)

    windows_env._powershell_delete(["C:\\Users\\alice\\.hermes\\old.txt"])

    assert "Remove-Item -LiteralPath" in calls[0][0]
    assert json.loads(calls[0][1]) == ["C:/Users/alice/.hermes/old.txt"]


def test_powershell_bulk_upload_writes_zip_payload(windows_env, monkeypatch, tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("aaa", encoding="utf-8")
    second.write_text("bbb", encoding="utf-8")

    calls = []
    scp_calls = []

    def fake_run(script, *, timeout=30, input_data=None):
        calls.append((script, input_data, timeout))
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    def fake_scp(cmd, **kwargs):
        scp_calls.append(cmd)
        local_zip = cmd[-2]
        with zipfile.ZipFile(local_zip) as zf:
            assert sorted(zf.namelist()) == ["cache/b.txt", "skills/a.txt"]
            assert zf.read("skills/a.txt") == b"aaa"
            assert zf.read("cache/b.txt") == b"bbb"
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(windows_env, "_run_powershell_script", fake_run)
    monkeypatch.setattr(ssh_windows.subprocess, "run", fake_scp)

    windows_env._powershell_bulk_upload([
        (str(first), "C:/Users/alice/.hermes/skills/a.txt"),
        (str(second), "C:/Users/alice/.hermes/cache/b.txt"),
    ])

    script, input_data, timeout = calls[0]
    assert scp_calls
    assert scp_calls[0][-1].startswith("alice@win.example:C:/Users/alice/AppData/Local/Temp/hermes-sync-")
    assert "System.IO.Compression.ZipFile" in script
    assert timeout >= 120
    payload = json.loads(input_data)
    assert payload["destination"] == "C:/Users/alice/.hermes"
    assert "content" not in payload


def test_execute_wraps_command_and_strips_cwd_marker(windows_env, monkeypatch):
    captured = {}
    marker = windows_env._cwd_marker

    class FakeProc:
        stdout = MagicMock()
        returncode = 0

        def poll(self):
            return 0

        def kill(self):
            pass

        def wait(self, timeout=None):
            return 0

    def fake_run_powershell(script, *, stdin_data=None):
        captured["script"] = script
        return FakeProc()

    monkeypatch.setattr(windows_env, "_run_powershell", fake_run_powershell)
    monkeypatch.setattr(
        windows_env,
        "_wait_for_process",
        lambda proc, timeout=120: {
            "output": f"ok\n\n{marker}C:\\Users\\alice\\work{marker}\n",
            "returncode": 0,
        },
    )

    result = windows_env.execute("whoami", cwd="C:/Users/alice/work")

    assert "$__hermes_cwd = 'C:/Users/alice/work'" in captured["script"]
    assert "Set-Location -LiteralPath $__hermes_cwd" in captured["script"]
    assert "whoami" in captured["script"]
    assert result["output"] == "ok\n"
    assert windows_env.cwd == "C:\\Users\\alice\\work"
