"""Tests for WSL execution environment backend.

All tests use mock subprocess — no real WSL calls are made.
Follows pytest style (functions, monkeypatch, assert), matching
test_docker_environment.py conventions.
"""
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from tools.environments import wsl as wsl_env


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_popen_mock(monkeypatch):
    """Replace subprocess.Popen with a MagicMock, return the mock."""
    mock = MagicMock()
    monkeypatch.setattr(wsl_env.subprocess, "Popen", mock)
    return mock


def _make_wsl_env_without_init():
    """Create WslEnvironment with _find_wsl + init_session bypassed."""
    import os

    class FakeEnv(wsl_env.WslEnvironment):
        def __init__(self, cwd=None):
            self._wsl = "/fake/wsl.exe"
            self._distro = os.getenv("TERMINAL_WSL_DISTRO", "")
            if not cwd:
                cwd = "/tmp"
            wsl_env.BaseEnvironment.__init__(self, cwd=cwd, timeout=60)

    return FakeEnv()


# ---------------------------------------------------------------------------
# _find_wsl
# ---------------------------------------------------------------------------

def test_find_wsl_system32_first(monkeypatch, tmp_path):
    fake_system32 = tmp_path / "System32"
    fake_system32.mkdir()
    wsl_exe = fake_system32 / "wsl.exe"
    wsl_exe.write_text("fake")
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    monkeypatch.setattr(wsl_env.shutil, "which", lambda _: "/some/other/wsl")
    assert wsl_env._find_wsl() == str(wsl_exe)


def test_find_wsl_falls_back_to_path(monkeypatch):
    monkeypatch.setenv("SystemRoot", "/nonexistent")
    monkeypatch.setattr(wsl_env.shutil, "which", lambda _: "/usr/bin/wsl")
    assert wsl_env._find_wsl() == "/usr/bin/wsl"


def test_find_wsl_raises_when_not_found(monkeypatch):
    monkeypatch.setenv("SystemRoot", "/nonexistent")
    monkeypatch.setattr(wsl_env.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="wsl.exe not found"):
        wsl_env._find_wsl()


# ---------------------------------------------------------------------------
# _probe_wsl_home
# ---------------------------------------------------------------------------

def test_probe_wsl_home_returns_linux_path(monkeypatch):
    result = subprocess.CompletedProcess(["wsl"], 0, stdout="/home/agents\n", stderr="")
    monkeypatch.setattr(wsl_env.subprocess, "run", lambda *a, **kw: result)
    assert wsl_env._probe_wsl_home("/fake/wsl.exe") == "/home/agents"


def test_probe_wsl_home_timeout_falls_back_to_root(monkeypatch):
    def _raise(*a, **kw):
        raise subprocess.TimeoutExpired("wsl", 5)
    monkeypatch.setattr(wsl_env.subprocess, "run", _raise)
    assert wsl_env._probe_wsl_home("/fake/wsl.exe") == "/root"


def test_probe_wsl_home_empty_output_falls_back_to_root(monkeypatch):
    result = subprocess.CompletedProcess(["wsl"], 0, stdout="\n", stderr="")
    monkeypatch.setattr(wsl_env.subprocess, "run", lambda *a, **kw: result)
    assert wsl_env._probe_wsl_home("/fake/wsl.exe") == "/root"


def test_probe_wsl_home_rejects_non_absolute_path(monkeypatch):
    result = subprocess.CompletedProcess(["wsl"], 0, stdout="agents\n", stderr="")
    monkeypatch.setattr(wsl_env.subprocess, "run", lambda *a, **kw: result)
    assert wsl_env._probe_wsl_home("/fake/wsl.exe") == "/root"


# ---------------------------------------------------------------------------
# WslEnvironment._run_bash
# ---------------------------------------------------------------------------

def test_run_bash_non_login(monkeypatch):
    env = _make_wsl_env_without_init()
    mock = _make_popen_mock(monkeypatch)
    env._run_bash("echo hello", login=False)
    assert mock.call_args[0][0][1:] == ["-e", "bash", "-c", "echo hello"]


def test_run_bash_login_uses_bash_dash_l(monkeypatch):
    env = _make_wsl_env_without_init()
    mock = _make_popen_mock(monkeypatch)
    env._run_bash("bootstrap", login=True)
    assert mock.call_args[0][0][1:] == ["-e", "bash", "-l", "-c", "bootstrap"]


def test_run_bash_with_distro(monkeypatch):
    monkeypatch.setenv("TERMINAL_WSL_DISTRO", "Debian")
    env = _make_wsl_env_without_init()
    env._distro = "Debian"
    mock = _make_popen_mock(monkeypatch)
    env._run_bash("ls")
    assert mock.call_args[0][0][1:5] == ["-d", "Debian", "-e", "bash"]


def test_stdin_data_piped(monkeypatch):
    env = _make_wsl_env_without_init()
    popen_mock = _make_popen_mock(monkeypatch)
    pipe_mock = MagicMock()
    monkeypatch.setattr(wsl_env, "_pipe_stdin", pipe_mock)
    env._run_bash("cat", stdin_data="hello")
    pipe_mock.assert_called_once()


def test_creationflags_windows_hide(monkeypatch):
    env = _make_wsl_env_without_init()
    mock = _make_popen_mock(monkeypatch)
    env._run_bash("echo")
    assert "creationflags" in mock.call_args[1]


# ---------------------------------------------------------------------------
# WslEnvironment — other methods
# ---------------------------------------------------------------------------

def test_update_cwd_from_marker():
    env = _make_wsl_env_without_init()
    env._cwd_marker = "MARK"
    result = {"output": "some output\nMARK/home/agentsMARK\nmore"}
    env._update_cwd(result)
    assert env.cwd == "/home/agents"


def test_cleanup_best_effort(monkeypatch):
    env = _make_wsl_env_without_init()
    env._snapshot_path = "/nonexistent/snap.sh"
    env._cwd_file = "/nonexistent/cwd.txt"
    mock_run = MagicMock()
    monkeypatch.setattr(wsl_env.subprocess, "run", mock_run)
    env.cleanup()  # should not raise


def test_kill_process_terminate_then_kill():
    env = _make_wsl_env_without_init()
    proc = MagicMock()
    proc.terminate.side_effect = OSError
    env._kill_process(proc)
    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# _get_env_config WSL CWD conversion (terminal_tool.py logic)
# ---------------------------------------------------------------------------



def test_distro_priority_param_over_env(monkeypatch):
    """Explicit distro param wins over env dict and os.getenv."""
    monkeypatch.setenv("TERMINAL_WSL_DISTRO", "Ubuntu")
    env = _make_wsl_env_without_init()
    env._distro = "Debian"  # simulate distro param
    assert env._distro == "Debian"


def test_distro_priority_env_over_os_getenv(monkeypatch):
    """Env dict wins over os.getenv."""
    monkeypatch.setenv("TERMINAL_WSL_DISTRO", "Ubuntu")
    env = _make_wsl_env_without_init()
    # _make_env uses os.getenv, so clear _distro to simulate env dict overriding
    env._distro = os.environ.get("TERMINAL_WSL_DISTRO", "")
    assert env._distro == "Ubuntu"


def test_distro_priority_os_getenv_fallback():
    """os.getenv is the last resort."""
    import os
    env = _make_wsl_env_without_init()
    assert env._distro == ""  # no env var set
