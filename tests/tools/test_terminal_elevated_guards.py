"""Terminal-level regression tests for elevated execution guard integration.

Verifies that elevated commands pass through the same _check_all_guards
decision path as ordinary terminal commands, and that the elevated executor
is never called when the guard returns blocked or pending-approval.
"""

import json

import pytest

import tools.terminal_tool as terminal_tool


def _make_minimal_config(**overrides) -> dict:
    """Return a minimal _get_env_config()-shaped dict for testing."""
    config = {
        "env_type": "local",
        "cwd": "/tmp",
        "timeout": 30,
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "mount_docker_cwd": False,
        "docker_forward_env": [],
        "docker_volumes": [],
        "docker_env": {},
        "docker_extra_args": [],
        "container_cpu": 1.0,
        "container_memory": 5120,
        "container_disk": 51200,
        "forward_env": [],
        "always_forward_env": [],
    }
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# Elevated command blocked by guard — executor not called
# ---------------------------------------------------------------------------


def test_elevated_blocked_does_not_call_executor(monkeypatch):
    """When _check_all_guards returns blocked, elevated executor is not called."""
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: _make_minimal_config(),
    )

    # Guard returns blocked
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda cmd, env, **kw: {
            "approved": False,
            "status": "blocked",
            "description": "rm -rf / is hardline blocked",
            "message": "Command denied: some safety rule",
        },
    )

    executor_calls = []
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda *a, **kw: executor_calls.append((a, kw)) or {"output": "", "exit_code": 0, "error": None},
    )

    result = terminal_tool.terminal_tool(
        command="rm -rf /",
        elevated=True,
    )
    data = json.loads(result)

    assert data["status"] == "blocked"
    assert data["exit_code"] == -1
    assert executor_calls == [], (
        f"execute_elevated should not have been called when blocked, "
        f"but was called {len(executor_calls)} time(s)"
    )


# ---------------------------------------------------------------------------
# Elevated command pending-approval — executor not called
# ---------------------------------------------------------------------------


def test_elevated_pending_approval_does_not_call_executor(monkeypatch):
    """When _check_all_guards returns pending_approval, executor is not called."""
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: _make_minimal_config(),
    )

    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda cmd, env, **kw: {
            "approved": False,
            "status": "pending_approval",
            "description": "command flagged",
            "command": cmd,
            "pattern_key": "dangerous_pattern",
        },
    )

    executor_calls = []
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda *a, **kw: executor_calls.append((a, kw)) or {"output": "", "exit_code": 0, "error": None},
    )

    result = terminal_tool.terminal_tool(
        command="sudo rm -rf /var/log",
        elevated=True,
    )
    data = json.loads(result)

    assert data["status"] == "pending_approval"
    assert data.get("approval_pending") is True
    assert executor_calls == [], (
        f"execute_elevated should not have been called when pending_approval, "
        f"but was called {len(executor_calls)} time(s)"
    )


# ---------------------------------------------------------------------------
# Elevated command approved — executor IS called
# ---------------------------------------------------------------------------


def test_elevated_approved_calls_executor(monkeypatch):
    """When _check_all_guards approves, the elevated executor is called."""
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: _make_minimal_config(),
    )

    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda cmd, env, **kw: {
            "approved": True,
            "status": "approved",
            "description": "",
        },
    )

    executor_calls = []
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda command, cwd=None, timeout=120: executor_calls.append({
            "command": command,
            "cwd": cwd,
            "timeout": timeout,
        }) or {"output": "admin result\n", "exit_code": 0, "error": None},
    )

    result = terminal_tool.terminal_tool(
        command="whoami /priv",
        elevated=True,
    )
    data = json.loads(result)

    assert data["exit_code"] == 0
    assert "admin result" in data["output"]
    assert len(executor_calls) == 1, (
        f"execute_elevated should have been called once when approved, "
        f"but was called {len(executor_calls)} time(s)"
    )
    assert executor_calls[0]["command"] == "whoami /priv"


# ---------------------------------------------------------------------------
# force=True skips guard but still dispatches elevated
# ---------------------------------------------------------------------------


def test_elevated_force_skips_guard_calls_executor(monkeypatch):
    """force=True should skip _check_all_guards and still dispatch elevated."""
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: _make_minimal_config(),
    )

    guard_calls = []
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda cmd, env, **kw: guard_calls.append((cmd, env, kw)) or {"approved": True},
    )

    executor_calls = []
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda command, cwd=None, timeout=120: executor_calls.append(command)
        or {"output": "forced\n", "exit_code": 0, "error": None},
    )

    result = terminal_tool.terminal_tool(
        command="format C: /fs:NTFS",
        elevated=True,
        force=True,
    )
    data = json.loads(result)

    assert data["exit_code"] == 0
    assert guard_calls == [], "guard should not be called when force=True"
    assert executor_calls == ["format C: /fs:NTFS"]
