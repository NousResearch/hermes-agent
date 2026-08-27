"""Tests for the bubblewrap backend wired into the terminal tool: the
environment factory, the unknown-backend error string, the
host-path defaults, file access through the host path, and execute_code
dispatch (the script runs inside the sandbox, never on the host).

Unit tests never spawn bwrap. Integration tests are skipped as a module
when bwrap is missing or its runtime probe fails, so CI without bwrap
stays green.
"""

import json
import os
import shutil
import subprocess
import textwrap
from unittest.mock import MagicMock, patch

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


class TestExecuteCodeDispatch:
    def test_execute_code_dispatches_bubblewrap_to_the_env_backed_path(self, isolated_tool):
        """Pins the dispatch: under bubblewrap execute_code must take
        the env.execute() file-RPC path, which runs the script under the bwrap
        prefix. The host UDS path would spawn the script unsandboxed, so
        routing bubblewrap beside local fails this test."""
        import tools.code_execution_tool as cet

        sentinel = json.dumps({"status": "success", "output": "via-env"})
        remote = MagicMock(return_value=sentinel)
        local_tmp = MagicMock()
        local_tmp.mkdtemp.side_effect = AssertionError("host UDS path must not run under bubblewrap")

        with patch.object(cet, "_execute_remote", remote), \
             patch.object(cet, "tempfile", local_tmp), \
             patch("tools.process_registry._is_supervised_gateway_process", return_value=False), \
             patch("tools.approval.check_execute_code_guard", return_value={"approved": True}):
            assert isolated_tool._get_env_config()["env_type"] == "bubblewrap"
            result = cet.execute_code("print(1)", task_id="dispatch", enabled_tools=["terminal"])

        assert result == sentinel
        remote.assert_called_once_with("print(1)", "dispatch", ["terminal"], reset=False)
        local_tmp.mkdtemp.assert_not_called()


@needs_bwrap
class TestExecuteCodeIntegration:
    def test_execute_code_runs_inside_the_sandbox_under_the_default_rlimits(self, isolated_tool, work_dir):
        from tools.code_execution_tool import execute_code

        task_id = "execute-code"
        code = textwrap.dedent("""
            import os
            print("PIDS", len([n for n in os.listdir("/proc") if n.isdigit()]))
            try:
                bytearray(400 * 1024 * 1024)
                print("ALLOC ok")
            except MemoryError:
                print("ALLOC MemoryError")
        """)
        r = json.loads(execute_code(code, task_id=task_id, enabled_tools=["terminal"]))
        assert r["status"] == "success", r
        lines = r["output"].strip().splitlines()
        pids = int(next(l for l in lines if l.startswith("PIDS ")).split()[1])
        # A pid namespace with a handful of processes: the script ran sandboxed.
        assert pids < 10, r["output"]
        # RLIMIT_AS from the default memory_mb applies to the script too.
        assert "ALLOC MemoryError" in lines, r["output"]
        env = isolated_tool.get_active_env(task_id)
        assert isinstance(env, BubblewrapEnvironment)
        # The RPC commands ran with cwd=/; the terminal's tracked cwd is untouched
        # and the next terminal write lands in the project directory.
        assert env.cwd == str(work_dir)
        t = json.loads(isolated_tool.terminal_tool("touch ./after-execute-code && pwd", task_id=task_id))
        assert t["exit_code"] == 0, t
        assert t["output"].strip() == str(work_dir)
        assert (work_dir / "after-execute-code").is_file()
