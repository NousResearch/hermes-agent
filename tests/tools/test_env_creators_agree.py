"""Whichever tool builds a task's sandbox first must not change what it is.

The terminal tool, the file tools, ``execute_code`` and ``ensure_task_env`` each create the
shared Docker container lazily. They must hand the backend the same target (image, workdir,
workspace mount, container config): a raw host cwd override used to reach ``docker run -w``
through ``execute_code`` alone (#100444, #54447).
"""

import json

import pytest

import tools.code_execution_tool as code_execution_tool
import tools.file_tools as file_tools
import tools.terminal_tool as terminal_tool
import tools.terminal_tool_backends as backends

TASK = "session-abc"


class _Env:
    """Minimal live env: only what the creators and ShellFileOperations touch."""

    def __init__(self, cwd):
        self.cwd = cwd
        self.env = {}

    def cleanup(self, **_):
        pass


@pytest.mark.parametrize("persistent", ["true", "false"])
def test_every_env_creator_targets_the_same_sandbox(monkeypatch, tmp_path, persistent):
    workspace = tmp_path / "proj"
    workspace.mkdir()
    for key, value in {
        "TERMINAL_ENV": "docker",
        "TERMINAL_CONTAINER_PERSISTENT": persistent,
        "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE": "true",
        "TERMINAL_DOCKER_ENV": json.dumps({"STATIC": "from-config"}),
        "TERMINAL_DOCKER_FORWARD_ENV": json.dumps(["FWD"]),
        "TERMINAL_CWD": str(workspace),
    }.items():
        monkeypatch.setenv(key, value)

    created = []

    def _fake_create(**kwargs):
        created.append(kwargs)
        return _Env(kwargs["cwd"])

    monkeypatch.setattr(backends, "_create_environment", _fake_create)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    for name in ("_active_environments", "_last_activity", "_creation_locks", "_task_env_overrides"):
        monkeypatch.setattr(terminal_tool, name, {})
    monkeypatch.setattr(file_tools, "_file_ops_cache", {})
    # What a gateway/TUI/ACP session registers: its workspace, as a raw HOST path.
    terminal_tool.register_task_env_overrides(TASK, {"cwd": str(workspace)})

    def _via_terminal():
        plan = terminal_tool._plan_execution(
            "echo hi", task_id=TASK, timeout=None, background=False, _host_local=False)
        terminal_tool._acquire_env(plan, TASK)

    creators = {
        "terminal": _via_terminal,
        "file_tools": lambda: file_tools._get_file_ops(TASK),
        "execute_code": lambda: code_execution_tool._get_or_create_env(TASK),
        "ensure_task_env": lambda: terminal_tool.ensure_task_env(TASK),
    }
    targets = {}
    for name, create in creators.items():
        for cache in (terminal_tool._active_environments, terminal_tool._last_activity,
                      terminal_tool._creation_locks, file_tools._file_ops_cache):
            cache.clear()
        created.clear()
        create()
        assert len(created) == 1, name
        kw = created[0]
        targets[name] = {k: kw[k] for k in ("image", "cwd", "host_cwd", "container_config", "task_id")}

    assert all(target == targets["terminal"] for target in targets.values()), targets
    # Equal is not enough: the host path is mounted, so the sandbox workdir is the mount.
    assert targets["terminal"]["cwd"] == "/workspace"
    assert targets["terminal"]["host_cwd"] == str(workspace)
