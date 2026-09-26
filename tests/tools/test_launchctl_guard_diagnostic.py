"""Diagnostics must distinguish policy rejection from inspected job properties."""

import json
import plistlib

from tools.terminal_tool_guards import gateway_lifecycle_block


def test_bootstrap_rejection_does_not_invent_keepalive(tmp_path, monkeypatch):
    from tools import process_registry

    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    plist = tmp_path / "com.example.schedule.plist"
    plist.write_bytes(plistlib.dumps({
        "Label": "com.example.schedule",
        "ProgramArguments": ["/bin/true"],
        "RunAtLoad": False,
        "StartCalendarInterval": {"Hour": 9, "Minute": 0},
    }))
    blocked = gateway_lifecycle_block(
        command=f"launchctl bootstrap gui/501 {plist}",
        env=None, env_type="local", cwd=str(tmp_path), workdir=None,
        session_key="diagnostic-test",
    )
    assert blocked is not None
    result = json.loads(blocked)
    assert result["exit_code"] == 1
    assert result["error"]


def test_interpreter_kill_rejection_names_the_owned_process_route(tmp_path, monkeypatch):
    """Scheduled-Task topology (#113667): only ``HERMES_SUPERVISED_CHILD`` marks the launch, and an
    image-name kill of the interpreter is refused with the ``proc_*`` / explicit-PID route named.
    Other images and explicit PIDs pass."""
    import os

    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setenv("HERMES_SUPERVISED_CHILD", "1")
    for marker in ("INVOCATION_ID", "XPC_SERVICE_NAME", "HERMES_S6_SUPERVISED_CHILD", "HERMES_GATEWAY_EXTERNAL_SUPERVISOR"):
        monkeypatch.delenv(marker, raising=False)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: os.getpid())

    def run(command):
        return gateway_lifecycle_block(
            command=command, env=None, env_type="local", cwd=str(tmp_path), workdir=None,
            session_key="diagnostic-test",
        )

    blocked = json.loads(run("taskkill /F /IM python.exe 2>/dev/null | head -2"))
    assert blocked["exit_code"] == 1
    assert "proc_" in blocked["error"] and "explicit PID" in blocked["error"]
    assert "proc_" in json.loads(run("pkill -9 python3"))["error"]
    assert run("taskkill /F /IM agent-browser.exe /T") is None
    assert run("taskkill /F /PID 46544") is None
    # Absence on the wrong side: a plain foreground `hermes gateway run` carries no launch marker.
    monkeypatch.delenv("HERMES_SUPERVISED_CHILD")
    assert run("pkill -9 python3") is None


def _run_outside_gateway(command, tmp_path):
    from tools import process_registry
    from tools.terminal_tool_guards import gateway_lifecycle_block

    orig = process_registry._is_supervised_gateway_process
    process_registry._is_supervised_gateway_process = lambda: False
    try:
        return gateway_lifecycle_block(
            command=command, env=None, env_type="local", cwd=str(tmp_path), workdir=None,
            session_key="unattended-test",
        )
    finally:
        process_registry._is_supervised_gateway_process = orig


def test_oneshot_launchctl_registration_blocked_outside_gateway(tmp_path, monkeypatch):
    """#122501: a one-shot session (``hermes chat -q``) is not the "separate shell" the
    supervised-gateway refusal points at — register verbs are refused there too."""
    for marker in ("HERMES_SINGLE_QUERY_SESSION", "HERMES_SESSION_SOURCE"):
        monkeypatch.delenv(marker, raising=False)
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    blocked = json.loads(_run_outside_gateway(
        "launchctl submit -l ai.hermes.wealthresume.170819 -- /usr/local/bin/hermes --yolo", tmp_path))
    assert blocked["exit_code"] == 1
    assert "agent sessions" in blocked["error"]
    blocked = json.loads(_run_outside_gateway("launchctl bootstrap gui/501 /tmp/job.plist", tmp_path))
    assert blocked["exit_code"] == 1


def test_human_shell_and_interactive_cli_still_pass(tmp_path, monkeypatch):
    """No unattended markers (human shell semantics — this guard only runs on the agent's
    own terminal tool — or an interactive CLI/TUI turn) → launchctl registration passes."""
    for marker in ("HERMES_SINGLE_QUERY_SESSION", "HERMES_SESSION_SOURCE"):
        monkeypatch.delenv(marker, raising=False)
    assert _run_outside_gateway("launchctl submit -l com.example.job -- /bin/true", tmp_path) is None


def test_sourced_integration_session_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "tool")
    blocked = json.loads(_run_outside_gateway("launchctl submit -l com.example.job -- /bin/true", tmp_path))
    assert blocked["exit_code"] == 1


def test_cron_session_contextvar_blocked(tmp_path, monkeypatch):
    import gateway.session_context as session_context

    real_get = session_context.get_session_env
    monkeypatch.setattr(
        session_context, "get_session_env",
        lambda name, default="": "1" if name == "HERMES_CRON_SESSION" else real_get(name, default),
    )
    blocked = json.loads(_run_outside_gateway("launchctl submit -l com.example.job -- /bin/true", tmp_path))
    assert blocked["exit_code"] == 1


def test_unattended_non_registration_launchctl_passes(tmp_path, monkeypatch):
    """Inspection/removal verbs stay available to an unattended session."""
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    assert _run_outside_gateway("launchctl list | grep hermes", tmp_path) is None
    assert _run_outside_gateway("launchctl remove ai.hermes.wealthresume.170819", tmp_path) is None


def test_unattended_oversized_command_fails_closed(tmp_path, monkeypatch):
    """Fail-closed on scan-budget exhaustion, mirroring the gateway branch (#122501)."""
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    oversized = "echo " + "a" * (2 * 1024 * 1024)
    blocked = json.loads(_run_outside_gateway(oversized, tmp_path))
    assert blocked["exit_code"] == 1
    assert "too large" in blocked["error"]

