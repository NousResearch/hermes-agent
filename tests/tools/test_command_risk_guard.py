"""Focused tests for Hermes-native critical shell-command risk classification."""

import json
import shlex
from unittest.mock import MagicMock

import pytest

import tools.approval as approval
import tools.terminal_tool as terminal_tool


@pytest.mark.parametrize(
    "command,category",
    [
        ('command rm -rf "/"', "critical_filesystem_destruction"),
        ("/bin/rm -rf /", "critical_filesystem_destruction"),
        ("env MODE=safe command rm -rf /", "critical_filesystem_destruction"),
        ("command reboot", "critical_host_disruption"),
        ("/sbin/reboot", "critical_host_disruption"),
        ("bash -O extglob -c 'reboot'", "critical_host_disruption"),
        ("bash -o errexit -c 'reboot'", "critical_host_disruption"),
        ("bash -c 'echo safe; reboot'", "critical_host_disruption"),
        ('bash -c \'sh -c "echo safe; reboot"\'', "critical_host_disruption"),
        ("bash -lc 'rm -rf /'", "critical_filesystem_destruction"),
        ('sh -c "systemctl reboot"', "critical_host_disruption"),
        ("/usr/bin/env X=1 /sbin/reboot", "critical_host_disruption"),
        ("bash --rcfile /tmp/x -c 'reboot'", "critical_host_disruption"),
        ("command -p reboot", "critical_host_disruption"),
        ("command -- /sbin/reboot", "critical_host_disruption"),
        ("/usr/bin/env -i /bin/bash -c reboot", "critical_host_disruption"),
        ("nohup -- /sbin/reboot", "critical_host_disruption"),
    ],
)
def test_classifies_critical_intent_through_basic_shell_indirection(command, category):
    classification = approval.classify_command_risk(command)

    assert classification is not None
    assert classification["level"] == "critical"
    assert classification["category"] == category
    assert classification["reason"]


@pytest.mark.parametrize(
    "command",
    [
        "command rm -rf /tmp/hermes-cache",
        "bash -lc 'printf \"%s\\n\" \"rm -rf /\"'",
        "bash -c 'echo \"safe; reboot\"'",
        "echo mkfs.ext4",
        "echo 'mkfs.ext4 /dev/sda'",
        "printf '%s' 'dd if=/dev/zero of=/dev/sda'",
        "echo 'kill -1'",
        "printf '%s\\n' 'kill -9 -1'",
        "echo '> /dev/sda'",
        'bash -c "echo \\"safe; reboot\\""',
        "bash -c 'sh -c \"echo \\\"safe; reboot\\\"\"'",
        "bash -c 'sh -c \"printf \\\"%s\\\" \\\"dd if=/dev/zero of=/dev/sda\\\"\"'",
        "echo command reboot",
        "systemctl status reboot.target",
    ],
)
def test_benign_near_neighbors_are_not_classified_as_critical(command):
    assert approval.classify_command_risk(command) is None


@pytest.mark.parametrize(
    "command,category,reason",
    [
        (
            "bash -lc 'rm -rf /'",
            "critical_filesystem_destruction",
            "recursive delete of root filesystem",
        ),
        (
            "bash -O extglob -c 'reboot'",
            "critical_host_disruption",
            "system shutdown/reboot",
        ),
        (
            "bash -o errexit -c 'reboot'",
            "critical_host_disruption",
            "system shutdown/reboot",
        ),
    ],
)
def test_combined_guard_blocks_critical_category_even_when_approvals_are_off(
    monkeypatch, command, category, reason
):
    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "off")

    result = approval.check_all_command_guards(command, "local")

    assert result["approved"] is False
    assert result["hardline"] is True
    assert result["risk_category"] == category
    assert reason in result["message"]


def _nested_shell_command(depth):
    payload = "command reboot"
    for _ in range(depth):
        payload = f"bash -c {shlex.quote(payload)}"
    return payload


@pytest.mark.parametrize("depth", [4, 6, 8])
def test_classifies_critical_intent_at_arbitrary_literal_shell_depth(depth):
    command = _nested_shell_command(depth)

    risk = approval.classify_command_risk(command)

    assert risk is not None
    assert risk["category"] == "critical_host_disruption"
    assert approval.check_all_command_guards(command, "local")["approved"] is False


@pytest.mark.parametrize("option_count", [7, 16, 64])
def test_shell_payload_scan_has_no_fixed_option_count_cap(option_count):
    command = "bash " + " ".join(["-e"] * option_count) + " -c reboot"

    risk = approval.classify_command_risk(command)

    assert risk is not None
    assert risk["category"] == "critical_host_disruption"
    assert approval.check_all_command_guards(command, "local")["approved"] is False


@pytest.mark.parametrize("force", [False, True])
def test_terminal_result_preserves_critical_category_reason_without_execution(
    monkeypatch, tmp_path, force
):
    config = {
        "env_type": "local",
        "timeout": 30,
        "cwd": str(tmp_path),
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }
    fake_env = MagicMock()
    fake_env.execute.return_value = {"output": "SHOULD NOT RUN", "returncode": 0}
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setitem(terminal_tool._active_environments, "default", fake_env)
    monkeypatch.setitem(terminal_tool._last_activity, "default", 0.0)

    result = json.loads(
        terminal_tool.terminal_tool(command="command reboot", force=force)
    )

    assert result["status"] == "blocked"
    assert result["risk_category"] == "critical_host_disruption"
    assert "system shutdown/reboot" in result["error"]
    fake_env.execute.assert_not_called()
