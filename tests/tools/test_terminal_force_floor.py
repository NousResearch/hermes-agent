"""``terminal_tool(force=True)`` skips the approval layers, never the floor (#157918).

``force`` says the caller already confirmed the command. That cannot outrank what YOLO and
``approvals.mode: off`` cannot: the hardline floor, the ``sudo -S`` guard, the operator's
``approvals.deny`` rules, and the refusal of a command the hardline scan could not read to the
end. Command strings are only passed to the guards, never run.
"""
import json

import pytest

from tools import approval_context
from tools import terminal_tool

GREP = 'v=$(grep -o "[^\\"]*" f)'
LOCAL = {"env_type": "local"}
ISOLATED_DOCKER = {"env_type": "docker"}
BIND_MOUNTED_DOCKER = {"env_type": "docker", "docker_volumes": ["/home/u/project:/workspace"]}


@pytest.fixture(autouse=True)
def _no_approval_surface(monkeypatch):
    for var in ("HERMES_YOLO_MODE", "HERMES_GATEWAY_SESSION", "HERMES_INTERACTIVE",
                "HERMES_EXEC_ASK", "HERMES_CRON_SESSION", "SUDO_PASSWORD"):
        monkeypatch.delenv(var, raising=False)


def _forced_rejection(command: str, config: dict) -> dict:
    with pytest.raises(terminal_tool._Rejected) as exc:
        terminal_tool._run_approval_guards(command, config["env_type"], config, force=True)
    return json.loads(exc.value.result_json)


@pytest.mark.parametrize("command,config,reason", [
    pytest.param("rm -rf /", LOCAL, "hardline", id="hardline-local"),
    pytest.param("mkfs.ext4 /dev/sda1", BIND_MOUNTED_DOCKER, "hardline", id="hardline-docker-with-host-mounts"),
    pytest.param("echo hunter2 | sudo -S apt install -y x", LOCAL, "sudo", id="sudo-stdin"),
    pytest.param("\n".join([GREP] * 500), LOCAL, "too long or complex", id="unread-by-the-scan"),
])
def test_force_does_not_run_what_the_floor_refuses(command, config, reason):
    result = _forced_rejection(command, config)
    assert result["status"] == "blocked"
    assert reason in result["error"]


def test_force_keeps_the_operators_deny_rules_in_an_isolated_container(monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_config",
                        lambda: {"mode": "manual", "deny": ["git push --force*"]})
    result = _forced_rejection("git push --force origin main", ISOLATED_DOCKER)
    assert result["status"] == "blocked" and "approvals.deny" in result["error"]


@pytest.mark.parametrize("command,config", [
    pytest.param("ls -la", LOCAL, id="confirmed-command-local"),
    pytest.param("rm -rf /", ISOLATED_DOCKER, id="isolated-container-is-its-own-floor"),
    # Only the deny rules bind there, as for an unforced command: the unread refusal is part of the floor.
    pytest.param("\n".join([GREP] * 500), ISOLATED_DOCKER, id="isolated-container-unread-by-the-scan"),
])
def test_force_runs_a_confirmed_command_the_floor_allows(command, config):
    verdict = terminal_tool._run_approval_guards(command, config["env_type"], config, force=True)
    assert verdict.approved_run is True


def test_force_skips_the_approval_layers(monkeypatch):
    """Only the floor binds a forced command, not an approval layer that refuses it unforced (cron's default deny)."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    with pytest.raises(terminal_tool._Rejected):
        terminal_tool._run_approval_guards("rm -rf build/", "local", LOCAL, force=False)
    assert terminal_tool._run_approval_guards("rm -rf build/", "local", LOCAL, force=True).approved_run is True
