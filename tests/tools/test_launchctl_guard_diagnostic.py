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
    assert "regardless of the job label" in result["error"]
    assert "does not inspect" in result["error"]
    assert "KeepAlive settings" in result["error"]
    assert "separate shell outside the gateway" in result["error"]


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
