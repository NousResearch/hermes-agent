"""execute_code's gateway-lifecycle identity probe shares the cell deadline (#111922).

The same ``_is_supervised_gateway_process`` probe that wedged ``terminal_tool`` runs
ahead of every ``execute_code`` cell; a probe that renders no verdict must fail closed
within the deadline instead of holding the tool call (and its cron slot) forever.
"""

from __future__ import annotations

import json
import time

import tools.code_execution_tool as cet
import tools.terminal_tool as terminal_module


def test_wedged_lifecycle_probe_returns_bounded_error_without_running(monkeypatch):
    import tools.process_registry as process_registry

    monkeypatch.setattr(cet, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(cet, "_load_config", lambda: {"timeout": 0.05})
    monkeypatch.setattr(terminal_module, "_PRE_EXEC_GUARD_MIN_TIMEOUT_S", 0)
    monkeypatch.setattr(
        process_registry,
        "_is_supervised_gateway_process",
        lambda: time.sleep(1),
    )

    ran: list[str] = []
    monkeypatch.setattr(
        terminal_module,
        "_get_env_config",
        lambda: ran.append("env") or {"env_type": "local"},
    )

    start = time.monotonic()
    result = json.loads(cet.execute_code("print('hi')"))
    elapsed = time.monotonic() - start

    assert elapsed < 0.5, f"lifecycle probe wedged execute_code for {elapsed:.2f}s"
    assert "did not finish" in result["error"]
    assert ran == [], "a probe with no verdict must not fail open into execution"


def test_interpreter_kill_in_execute_code_names_the_owned_process_route(monkeypatch):
    """Sibling surface of the terminal guard (#113667): an image-name kill inside a cell gets the same
    proc_* / explicit-PID rejection, not the generic lifecycle text."""
    import tools.process_registry as process_registry

    monkeypatch.setattr(
        process_registry,
        "_is_supervised_gateway_process",
        lambda: True,
    )

    result = json.loads(
        cet.execute_code(
            'import subprocess; subprocess.run(["pkill", "-9", "python3"])'
        )
    )
    assert "proc_" in result["error"] and "explicit PID" in result["error"]

    generic = json.loads(
        cet.execute_code('import os; os.system("hermes gateway restart")')
    )
    assert "proc_" not in generic["error"]
    assert "cannot restart or stop the gateway" in generic["error"]


def test_execute_code_entry_point_blocks_config_mutation_before_dispatch(monkeypatch):
    import tools.process_registry as process_registry

    monkeypatch.setattr(cet, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(cet, "_load_config", lambda: {"timeout": 1})
    monkeypatch.setattr(
        process_registry, "_is_supervised_gateway_process", lambda: False
    )
    monkeypatch.setattr(
        terminal_module,
        "_get_env_config",
        lambda: {"env_type": "vercel_sandbox"},
    )
    dispatched: list[str] = []

    def forbidden_dispatch(*args, **kwargs):
        dispatched.append("called")
        raise AssertionError("Blocked code must never reach execution dispatch")

    monkeypatch.setattr(cet, "_execute_remote", forbidden_dispatch)

    result = json.loads(
        cet.execute_code(
            "open('/tmp/.hermes/config.yaml', 'w').write('approvals: {}')"
        )
    )

    assert result["status"] == "error"
    assert "config" in result["error"].lower()
    assert dispatched == []


def test_execute_code_entry_point_allows_benign_code_to_reach_dispatch(monkeypatch):
    import tools.process_registry as process_registry

    monkeypatch.setattr(cet, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(cet, "_load_config", lambda: {"timeout": 1})
    monkeypatch.setattr(
        process_registry, "_is_supervised_gateway_process", lambda: False
    )
    monkeypatch.setattr(
        terminal_module,
        "_get_env_config",
        lambda: {"env_type": "vercel_sandbox"},
    )

    dispatched: list[str] = []

    def fake_remote(code, task_id, enabled_tools, reset=False):
        dispatched.append(code)
        return '{"status":"success","output":"10"}'

    monkeypatch.setattr(cet, "_execute_remote", fake_remote)
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_loop.record_execution_backend",
        lambda *args: args[-1],
    )

    result = json.loads(cet.execute_code("print(sum(range(5)))"))

    assert result["status"] == "success"
    assert dispatched == ["print(sum(range(5)))"]



def test_execute_code_entry_point_blocks_config_mutation_before_local_kernel_dispatch(
    monkeypatch,
):
    import tools.process_registry as process_registry
    import tools.code_kernel as code_kernel

    monkeypatch.setattr(cet, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(cet, "_load_config", lambda: {"timeout": 1})
    monkeypatch.setattr(
        process_registry, "_is_supervised_gateway_process", lambda: False
    )
    monkeypatch.setattr(
        terminal_module,
        "_get_env_config",
        lambda: {"env_type": "local"},
    )

    dispatched: list[str] = []

    def forbidden_dispatch(*args, **kwargs):
        dispatched.append("called")
        raise AssertionError(
            "Blocked code must never reach the local kernel"
        )

    monkeypatch.setattr(
        code_kernel, "execute_in_session_kernel", forbidden_dispatch
    )

    result = json.loads(
        cet.execute_code(
            "Path('/tmp/.hermes/config.yaml').open('w').write('approvals: {}')"
        )
    )

    assert result["status"] == "error"
    assert "config" in result["error"].lower()
    assert dispatched == []
