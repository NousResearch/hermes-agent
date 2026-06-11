"""Tests for non-executing maintenance-action policy validation."""

import pytest

from tools.approval import detect_hardline_command
from tools.maintenance_actions import evaluate_maintenance_action


SAFE_ARGV = [
    "ssh",
    "-o",
    "BatchMode=yes",
    "caspian-inference-01",
    "sudo",
    "/usr/local/sbin/caspian-power-control",
    "restart",
]


def _policy(**overrides):
    policy = {
        "enabled": True,
        "require_interactive_user_approval": True,
        "unattended_policy": "none",
        "actions": {
            "caspian_inference_restart": {
                "enabled": True,
                "host_label": "caspian-inference-01",
                "command_id": "caspian_power_control_restart",
                "exact_argv": list(SAFE_ARGV),
                "preflight_profile": "caspian_inference_gpu_reset_required_v1",
                "postcheck_profile": "caspian_inference_qwen_recovery_v1",
            }
        },
    }
    policy.update(overrides)
    return policy


class TestMaintenanceActionDefaultDeny:
    def test_absent_policy_blocks(self):
        result = evaluate_maintenance_action({}, "caspian_inference_restart", SAFE_ARGV)

        assert result.allowed is False
        assert result.reason == "policy_absent"

    def test_global_disabled_blocks(self):
        result = evaluate_maintenance_action(
            _policy(enabled=False), "caspian_inference_restart", SAFE_ARGV
        )

        assert result.allowed is False
        assert result.reason == "policy_disabled"

    def test_action_disabled_blocks(self):
        policy = _policy()
        policy["actions"]["caspian_inference_restart"]["enabled"] = False

        result = evaluate_maintenance_action(policy, "caspian_inference_restart", SAFE_ARGV)

        assert result.allowed is False
        assert result.reason == "action_disabled"

    def test_unknown_action_blocks(self):
        result = evaluate_maintenance_action(_policy(), "missing_action", SAFE_ARGV)

        assert result.allowed is False
        assert result.reason == "unknown_action"


class TestMaintenanceActionExactArgv:
    def test_exact_argv_match_can_be_eligible(self):
        result = evaluate_maintenance_action(_policy(), "caspian_inference_restart", SAFE_ARGV)

        assert result.allowed is False
        assert result.reason == "requires_current_user_approval"
        assert result.eligible is True
        assert result.command_id == "caspian_power_control_restart"

    def test_argv_mismatch_blocks(self):
        changed = list(SAFE_ARGV)
        changed[-1] = "shutdown"

        result = evaluate_maintenance_action(_policy(), "caspian_inference_restart", changed)

        assert result.allowed is False
        assert result.reason == "argv_mismatch"
        assert result.eligible is False

    @pytest.mark.parametrize(
        "requested_argv",
        [
            "ssh caspian-inference-01 sudo /usr/local/sbin/caspian-power-control restart",
            ["sh", "-c", "ssh caspian-inference-01 sudo /usr/local/sbin/caspian-power-control restart"],
            ["bash", "-lc", "ssh caspian-inference-01 sudo /usr/local/sbin/caspian-power-control restart"],
        ],
    )
    def test_shell_string_or_interpreter_wrapping_blocks(self, requested_argv):
        result = evaluate_maintenance_action(
            _policy(), "caspian_inference_restart", requested_argv
        )

        assert result.allowed is False
        assert result.reason in {"argv_must_be_list", "shell_wrapping_forbidden"}
        assert result.eligible is False


class TestMaintenanceActionMalformedPolicy:
    def test_non_dict_policy_blocks(self):
        result = evaluate_maintenance_action("not-a-policy", "caspian_inference_restart", SAFE_ARGV)

        assert result.allowed is False
        assert result.reason == "policy_invalid"

    @pytest.mark.parametrize("actions", [None, [], "not-actions"])
    def test_missing_or_non_dict_actions_blocks(self, actions):
        result = evaluate_maintenance_action(
            _policy(actions=actions), "caspian_inference_restart", SAFE_ARGV
        )

        assert result.allowed is False
        assert result.reason == "actions_invalid"

    def test_non_dict_action_body_blocks(self):
        policy = _policy(actions={"caspian_inference_restart": "not-an-action"})

        result = evaluate_maintenance_action(policy, "caspian_inference_restart", SAFE_ARGV)

        assert result.allowed is False
        assert result.reason == "action_invalid"

    @pytest.mark.parametrize("exact_argv", [None, "ssh host command", [], ["ssh", ""]])
    def test_missing_or_invalid_exact_argv_blocks(self, exact_argv):
        policy = _policy()
        policy["actions"]["caspian_inference_restart"]["exact_argv"] = exact_argv

        result = evaluate_maintenance_action(policy, "caspian_inference_restart", SAFE_ARGV)

        assert result.allowed is False
        assert result.reason == "exact_argv_invalid"

    @pytest.mark.parametrize("requested_argv", [[], ["ssh", ""], ["ssh", object()]])
    def test_empty_or_invalid_requested_argv_blocks(self, requested_argv):
        result = evaluate_maintenance_action(
            _policy(), "caspian_inference_restart", requested_argv
        )

        assert result.allowed is False
        assert result.reason == "argv_invalid"


class TestMaintenanceActionContext:
    @pytest.mark.parametrize(
        "invocation_context",
        ["cron", "unattended", "background", "scheduler", "CrOn"],
    )
    def test_unattended_context_blocks_by_default(self, invocation_context):
        result = evaluate_maintenance_action(
            _policy(),
            "caspian_inference_restart",
            SAFE_ARGV,
            invocation_context=invocation_context,
        )

        assert result.allowed is False
        assert result.reason == "unattended_forbidden"
        assert result.eligible is False

    def test_current_user_approval_allows_when_required_gates_pass(self):
        result = evaluate_maintenance_action(
            _policy(),
            "caspian_inference_restart",
            SAFE_ARGV,
            current_user_approved=True,
        )

        assert result.allowed is True
        assert result.reason == "approved"
        assert result.eligible is True

    def test_missing_preflight_profile_blocks(self):
        policy = _policy()
        del policy["actions"]["caspian_inference_restart"]["preflight_profile"]

        result = evaluate_maintenance_action(
            policy,
            "caspian_inference_restart",
            SAFE_ARGV,
            current_user_approved=True,
        )

        assert result.allowed is False
        assert result.reason == "missing_preflight_profile"


class TestMaintenanceActionDoesNotWeakenHardline:
    def test_ordinary_reboot_remains_hardline_blocked(self):
        is_hardline, description = detect_hardline_command("sudo reboot")

        assert is_hardline is True
        assert "reboot" in description.lower() or "shutdown" in description.lower()
