"""Late session cwd updates must stay in the SSH peer's namespace.

The creation path already translates the Hermes subprocess home to remote ~.
These regressions exercise an existing environment and a later command reading
the raw session record, neither of which creates a new environment.
"""

import pytest

import hermes_constants
from tools import terminal_tool
from tools.file_operations import ShellFileOperations


HOST_HOME = "/srv/hermes-host/home"


class RecordingEnvironment:
    def __init__(self, env_type):
        self.env_type = env_type
        self.cwd = "~/initial"
        self.calls = []

    def execute(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return {"output": "probe", "returncode": 0}


@pytest.fixture(autouse=True)
def isolated_session_state(monkeypatch):
    monkeypatch.setattr(hermes_constants, "get_subprocess_home", lambda: HOST_HOME)
    monkeypatch.setattr(hermes_constants, "get_real_home", lambda: "/srv/os-user")
    for name in ("_session_cwd", "_task_env_overrides", "_active_environments", "_container_aliases"):
        monkeypatch.setattr(terminal_tool, name, {})


CASES = [
    ("ssh", HOST_HOME, "~"),
    ("ssh", f"{HOST_HOME}/project", "~/project"),
    ("ssh", "/remote/project", "/remote/project"),
    ("ssh", "~other/project", "~other/project"),
    ("ssh", f"{HOST_HOME}work", f"{HOST_HOME}work"),
    ("local", HOST_HOME, HOST_HOME),
]


@pytest.mark.parametrize("env_type, raw_cwd, backend_cwd", CASES)
def test_late_registration_keeps_file_execution_in_backend_namespace(
    env_type, raw_cwd, backend_cwd
):
    env = RecordingEnvironment(env_type)
    task_id = "session-example"
    terminal_tool._active_environments[task_id] = env
    file_ops = ShellFileOperations(env)

    terminal_tool.register_task_env_overrides(task_id, {"cwd": raw_cwd})

    # The UI's workspace record stays raw. Only execution crosses namespaces.
    assert terminal_tool.get_session_cwd(task_id) == raw_cwd
    assert env.cwd == backend_cwd
    result = file_ops._exec("probe")
    assert result.exit_code == 0
    assert env.calls[-1][1]["cwd"] == backend_cwd


@pytest.mark.parametrize("env_type, raw_cwd, backend_cwd", CASES)
def test_recorded_session_cwd_is_coerced_without_changing_explicit_remote_workdir(
    env_type, raw_cwd, backend_cwd
):
    task_id = "session-example"
    terminal_tool.record_session_cwd(task_id, raw_cwd)

    assert terminal_tool._resolve_command_cwd(
        workdir=None,
        default_cwd="~/default",
        session_key=task_id,
        env_type=env_type,
    ) == backend_cwd
    assert terminal_tool.get_session_cwd(task_id) == raw_cwd
    assert terminal_tool._resolve_command_cwd(
        workdir="/remote/explicit",
        default_cwd="~/default",
        session_key=task_id,
        env_type=env_type,
    ) == "/remote/explicit"
