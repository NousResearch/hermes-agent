"""Tests for the bubblewrap backend wired into the terminal tool: the
environment factory, the unknown-backend error string, the
host-path defaults, and file access through the host path.

Unit tests never spawn bwrap. Integration tests are skipped as a module
when bwrap is missing or its runtime probe fails, so CI without bwrap
stays green.
"""

import json
import os
import shutil
import subprocess
from unittest.mock import patch

import pytest

from tools.environments import bubblewrap
from tools.environments.bubblewrap import BubblewrapEnvironment
from tools.environments.local import LocalEnvironment
from tools.terminal_tool_backends import _create_environment


def _bwrap_usable() -> bool:
    if shutil.which("bwrap") is None:
        return False
    try:
        probe = subprocess.run(
            ["bwrap", "--unshare-user", "--ro-bind", "/", "/", "true"],
            capture_output=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


BWRAP_USABLE = _bwrap_usable()
needs_bwrap = pytest.mark.skipif(not BWRAP_USABLE, reason="bwrap missing or its namespace probe failed")


@pytest.fixture
def work_dir(tmp_path):
    d = tmp_path / "work"
    d.mkdir()
    return d


@pytest.fixture
def isolated_tool(tmp_path, work_dir, monkeypatch):
    """terminal_tool with backend=bubblewrap, cwd=work_dir, an isolated
    HERMES_HOME and a clean environment cache."""
    import tools.terminal_tool as tt

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    monkeypatch.setattr(tt, "_terminal_config_bridge_attempted", True)
    monkeypatch.setenv("TERMINAL_ENV", "bubblewrap")
    monkeypatch.setenv("TERMINAL_CWD", str(work_dir))

    def _clear():
        with tt._env_lock:
            for env in list(tt._active_environments.values()):
                try:
                    env.cleanup()
                except Exception:
                    pass
            tt._active_environments.clear()
            tt._last_activity.clear()

    _clear()
    yield tt
    _clear()


def _no_spawn(monkeypatch):
    monkeypatch.setattr(bubblewrap, "_probed_bwrap_path", shutil.which("bwrap") or "/usr/bin/bwrap")
    return patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None)


class TestFactory:
    def test_factory_returns_bubblewrap_environment_with_cwd_timeout_and_config(self, isolated_tool, work_dir, monkeypatch):
        monkeypatch.setenv("TERMINAL_BUBBLEWRAP_PROFILE", "restricted")
        monkeypatch.setenv("TERMINAL_BUBBLEWRAP_MEMORY_MB", "512")
        with _no_spawn(monkeypatch):
            env = _create_environment("bubblewrap", "", str(work_dir), 42)
        try:
            assert isinstance(env, BubblewrapEnvironment)
            assert env.cwd == str(work_dir)
            assert env.timeout == 42
            assert env._config.profile == "restricted"
            assert env._config.memory_mb == 512
        finally:
            env.cleanup()

    def test_unknown_backend_error_lists_bubblewrap(self, isolated_tool, work_dir):
        with pytest.raises(ValueError, match="bubblewrap") as excinfo:
            _create_environment("no-such-backend", "", str(work_dir), 10)
        assert "Unknown environment type: no-such-backend" in str(excinfo.value)

    def test_default_cwd_is_the_host_cwd(self, isolated_tool, monkeypatch):
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        config = isolated_tool._get_env_config()
        assert config["env_type"] == "bubblewrap"
        assert config["cwd"] == isolated_tool._safe_getcwd()

    def test_check_terminal_requirements_follows_the_probe(self, isolated_tool, monkeypatch):
        monkeypatch.setattr(bubblewrap, "run_probe", lambda: ("/usr/bin/bwrap", None))
        assert isolated_tool.check_terminal_requirements() is True
        monkeypatch.setattr(bubblewrap, "run_probe", lambda: (None, "bubblewrap (bwrap) is not on PATH"))
        assert isolated_tool.check_terminal_requirements() is False


@needs_bwrap
class TestToolIntegration:
    def test_terminal_creates_a_file_and_read_file_returns_it_via_the_host_path(self, isolated_tool, work_dir):
        from tools.file_tools import read_file_tool

        task_id = "host-path"
        r = json.loads(isolated_tool.terminal_tool(
            "printf hello > probe.txt && ls /proc | grep -c '^[0-9]'", task_id=task_id,
        ))
        assert r["exit_code"] == 0, r
        # A pid namespace with a handful of processes: the command ran sandboxed.
        assert int(r["output"].strip().splitlines()[-1]) < 10
        assert isinstance(isolated_tool.get_active_env(task_id), BubblewrapEnvironment)

        host_file = work_dir / "probe.txt"
        assert host_file.read_text() == "hello"
        assert "hello" in read_file_tool(str(host_file), task_id=task_id)

    def test_background_is_refused_and_never_spawned_unsandboxed(self, isolated_tool, monkeypatch):
        from tools.process_registry import process_registry

        def forbidden(*args, **kwargs):
            raise AssertionError("spawn_local must not run under bubblewrap")

        monkeypatch.setattr(process_registry, "spawn_local", forbidden)
        r = json.loads(isolated_tool.terminal_tool("sleep 30", background=True, task_id="background"))
        assert r["status"] == "error"
        assert r["exit_code"] == -1
        assert "background" in r["error"]
        assert "session_id" not in r
