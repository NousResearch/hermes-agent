"""Remote ProcessRegistry transport contracts, separate from host process tests."""
import subprocess
import time
from unittest.mock import patch

import pytest
from tools.process_registry import ProcessRegistry, ProcessSession
from tests.waiting import wait_until


@pytest.fixture
def registry():
    return ProcessRegistry()


def _make_session(sid):
    return ProcessSession(id=sid, command="echo hello", task_id="t1", started_at=time.time())


@pytest.mark.platforms("linux")
def test_spawn_via_env_wrapper_writes_numeric_exit_status(registry, tmp_path):
    """A successful non-local wrapper must persist a parseable exit code."""
    class FakeEnv:
        def get_temp_dir(self):
            return str(tmp_path)

        def execute(self, command, **kwargs):
            completed = subprocess.run(
                ["bash", "-lc", command], text=True, capture_output=True, check=False,
            )
            return {"output": completed.stdout, "returncode": completed.returncode}

    with patch("tools.process_registry.threading.Thread"):
        session = registry.spawn_via_env(FakeEnv(), "printf remote-ok")

    exit_path = tmp_path / f"hermes_bg_{session.id}.exit"
    wait_until(lambda: exit_path.exists() and exit_path.read_text().strip() == "0",
               timeout=5, what="numeric wrapper exit artifact")
    assert exit_path.read_text().strip() == "0"


def test_env_poller_quotes_temp_paths_with_spaces(registry):
    session = _make_session(sid="proc_space")
    session.exited = False

    class FakeEnv:
        def __init__(self):
            self.commands = []
            self._responses = iter([
                {"output": "6 0\nhello\n"},
                {"output": "1\n"},
                {"output": "0\n"},
            ])

        def execute(self, command, **kwargs):
            self.commands.append((command, kwargs))
            return next(self._responses)

    env = FakeEnv()
    session.pid = 4242
    session.env_ref = env
    env._responses = iter([
        {"output": "6 0\nhello\n"},
        {"output": "1 1 S\n", "returncode": 0},
        {"output": "0\n"},
    ])

    with patch("tools.process_registry.time.sleep", return_value=None), \
        patch.object(registry, "_move_to_finished"):
        registry._env_poller_loop(
            session,
            env,
            "/path with spaces/hermes_bg.log",
            "/path with spaces/hermes_bg.pid",
            "/path with spaces/hermes_bg.exit",
        )

    assert "'/path with spaces/hermes_bg.log'" in env.commands[0][0]
    assert "cat '/path with spaces/hermes_bg.log'" not in env.commands[0][0]
    assert env.commands[1][0] == "LC_ALL=C ps -eo pid=,pgid=,stat="
    assert session.exit_code == 0
    assert "'/path with spaces/hermes_bg.exit'" in env.commands[2][0]



def test_remote_kill_targets_the_dedicated_process_group(registry):
    class FakeEnv:
        def __init__(self):
            self.calls = []

        def execute(self, command, **kwargs):
            self.calls.append((command, kwargs))
            if command.startswith("kill "):
                self.stopped = True
                return {"returncode": 0}
            return {"returncode": 0, "output": "1 1 S\n" + (
                "4242 4242 S\n" if not getattr(self, "stopped", False) else "")}

    session = _make_session(sid="proc_remote_group")
    session.pid = 4242
    session.env_ref = env = FakeEnv()

    assert registry._signal_kill(session, session.id, consume_output=False) is None
    assert ("kill -TERM -- -4242 2>/dev/null", {"timeout": 5}) in env.calls
    assert env.calls[-1][0] == "LC_ALL=C ps -eo pid=,pgid=,stat="


def test_remote_kill_propagates_backend_signal_failure(registry):
    class FakeEnv:
        def execute(self, command, **kwargs):
            if command.startswith("kill "):
                return {"returncode": 1, "output": "kill: Operation not permitted"}
            return {"returncode": 0, "output": "1 1 S\n4242 4242 S\n"}

    session = _make_session(sid="proc_remote_group_failure")
    session.pid = 4242
    session.env_ref = FakeEnv()

    session.pid_scope = "sandbox"
    registry._running[session.id] = session
    result = registry.kill_process(session.id, consume_output=False)
    assert result["status"] == "error"
    assert "unconfirmed" in result["error"]
    assert registry.get(session.id) is session
    assert not session.exited
