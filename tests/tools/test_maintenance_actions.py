"""Tests for non-executing maintenance-action policy validation."""

import pytest

from tools.approval import detect_hardline_command
from tools.maintenance_actions import classify_maintenance_action, evaluate_maintenance_action


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

    @pytest.mark.parametrize("enabled", [None, "false", "true", "no", "yes", 1, 0, [], {}])
    def test_malformed_global_enabled_blocks(self, enabled):
        result = evaluate_maintenance_action(
            _policy(enabled=enabled),
            "caspian_inference_restart",
            SAFE_ARGV,
            current_user_approved=True,
        )

        assert result.allowed is False
        assert result.reason == "policy_enabled_invalid"

    def test_global_disabled_blocks(self):
        result = evaluate_maintenance_action(
            _policy(enabled=False), "caspian_inference_restart", SAFE_ARGV
        )

        assert result.allowed is False
        assert result.reason == "policy_disabled"

    @pytest.mark.parametrize("enabled", [None, "false", "true", "no", "yes", 1, 0, [], {}])
    def test_malformed_action_enabled_blocks(self, enabled):
        policy = _policy()
        policy["actions"]["caspian_inference_restart"]["enabled"] = enabled

        result = evaluate_maintenance_action(
            policy,
            "caspian_inference_restart",
            SAFE_ARGV,
            current_user_approved=True,
        )

        assert result.allowed is False
        assert result.reason == "action_enabled_invalid"

    def test_action_disabled_blocks(self):
        policy = _policy()
        policy["actions"]["caspian_inference_restart"]["enabled"] = False

        result = evaluate_maintenance_action(policy, "caspian_inference_restart", SAFE_ARGV)

        assert result.allowed is False
        assert result.reason == "action_disabled"

    @pytest.mark.parametrize("action_id", [None, "", [], {}])
    def test_malformed_action_id_blocks(self, action_id):
        result = evaluate_maintenance_action(_policy(), action_id, SAFE_ARGV)

        assert result.allowed is False
        assert result.reason == "invalid_action_id"

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

    @pytest.mark.parametrize(
        "exact_argv",
        [
            None,
            "ssh host command",
            [],
            ["ssh", ""],
            ["bash", "-lc", "ssh host command"],
            ["bash", "--noprofile", "--norc", "-c", "ssh host command"],
            ["/usr/bin/env", "bash", "-lc", "ssh host command"],
            ["/usr/bin/env", "FOO=bar", "bash", "-lc", "ssh host command"],
            ["/usr/bin/env", "-S", "bash -lc ssh host command"],
        ],
    )
    def test_missing_or_invalid_exact_argv_blocks(self, exact_argv):
        policy = _policy()
        policy["actions"]["caspian_inference_restart"]["exact_argv"] = exact_argv

        result = evaluate_maintenance_action(policy, "caspian_inference_restart", exact_argv)

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
        ["cron", "unattended", "background", "scheduler", "CrOn", " cron "],
    )
    @pytest.mark.parametrize("unattended_policy", [None, False, "none", "NONE", "false"])
    def test_unattended_context_blocks_by_default(
        self, invocation_context, unattended_policy
    ):
        result = evaluate_maintenance_action(
            _policy(unattended_policy=unattended_policy),
            "caspian_inference_restart",
            SAFE_ARGV,
            invocation_context=invocation_context,
        )

        assert result.allowed is False
        assert result.reason == "unattended_forbidden"
        assert result.eligible is False

    @pytest.mark.parametrize("unattended_policy", [True, "allow", "manual", [], {}])
    def test_malformed_unattended_policy_blocks_interactive_context(
        self, unattended_policy
    ):
        result = evaluate_maintenance_action(
            _policy(unattended_policy=unattended_policy),
            "caspian_inference_restart",
            SAFE_ARGV,
            invocation_context="interactive",
            current_user_approved=True,
        )

        assert result.allowed is False
        assert result.reason == "unattended_policy_invalid"
        assert result.eligible is False

    @pytest.mark.parametrize("unattended_policy", [True, "allow", "manual", [], {}])
    def test_malformed_unattended_policy_blocks_unattended_context(
        self, unattended_policy
    ):
        result = evaluate_maintenance_action(
            _policy(unattended_policy=unattended_policy),
            "caspian_inference_restart",
            SAFE_ARGV,
            invocation_context="cron",
            current_user_approved=True,
        )

        assert result.allowed is False
        assert result.reason == "unattended_policy_invalid"
        assert result.eligible is False

    @pytest.mark.parametrize(
        "require_interactive_user_approval",
        [None, "", [], {}, 0, "false"],
    )
    def test_malformed_approval_requirement_blocks(self, require_interactive_user_approval):
        result = evaluate_maintenance_action(
            _policy(require_interactive_user_approval=require_interactive_user_approval),
            "caspian_inference_restart",
            SAFE_ARGV,
            current_user_approved=False,
        )

        assert result.allowed is False
        assert result.reason == "invalid_approval_requirement"
        assert result.eligible is False

    def test_explicit_boolean_false_approval_requirement_allows_static_approval(self):
        result = evaluate_maintenance_action(
            _policy(require_interactive_user_approval=False),
            "caspian_inference_restart",
            SAFE_ARGV,
            current_user_approved=False,
        )

        assert result.allowed is True
        assert result.reason == "approved"
        assert result.eligible is True

    @pytest.mark.parametrize("current_user_approved", [None, "false", "true", 1, [], {}])
    def test_malformed_current_user_approved_blocks(self, current_user_approved):
        result = evaluate_maintenance_action(
            _policy(),
            "caspian_inference_restart",
            SAFE_ARGV,
            current_user_approved=current_user_approved,
        )

        assert result.allowed is False
        assert result.reason == "invalid_current_user_approval"
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

    @pytest.mark.parametrize("preflight_profile", [True, [], {}])
    def test_malformed_preflight_profile_blocks(self, preflight_profile):
        policy = _policy()
        policy["actions"]["caspian_inference_restart"]["preflight_profile"] = preflight_profile

        result = evaluate_maintenance_action(
            policy,
            "caspian_inference_restart",
            SAFE_ARGV,
            current_user_approved=True,
        )

        assert result.allowed is False
        assert result.reason == "invalid_preflight_profile"

    def test_missing_postcheck_profile_blocks(self):
        policy = _policy()
        del policy["actions"]["caspian_inference_restart"]["postcheck_profile"]

        result = evaluate_maintenance_action(
            policy,
            "caspian_inference_restart",
            SAFE_ARGV,
            current_user_approved=True,
        )

        assert result.allowed is False
        assert result.reason == "missing_postcheck_profile"

    @pytest.mark.parametrize("postcheck_profile", [True, [], {}])
    def test_malformed_postcheck_profile_blocks(self, postcheck_profile):
        policy = _policy()
        policy["actions"]["caspian_inference_restart"]["postcheck_profile"] = postcheck_profile

        result = evaluate_maintenance_action(
            policy,
            "caspian_inference_restart",
            SAFE_ARGV,
            current_user_approved=True,
        )

        assert result.allowed is False
        assert result.reason == "invalid_postcheck_profile"


class TestMaintenanceActionClassifier:
    def test_classifier_returns_stable_dict_for_blocked_action(self):
        result = classify_maintenance_action(
            _policy(), "caspian_inference_restart", ["ssh", "wrong-host"]
        )

        assert result == {
            "allowed": False,
            "eligible": False,
            "reason": "argv_mismatch",
            "action_id": "caspian_inference_restart",
            "command_id": "caspian_power_control_restart",
            "host_label": "caspian-inference-01",
        }

    def test_classifier_returns_stable_dict_for_eligible_action(self):
        result = classify_maintenance_action(_policy(), "caspian_inference_restart", SAFE_ARGV)

        assert result == {
            "allowed": False,
            "eligible": True,
            "reason": "requires_current_user_approval",
            "action_id": "caspian_inference_restart",
            "command_id": "caspian_power_control_restart",
            "host_label": "caspian-inference-01",
        }

    def test_classifier_returns_stable_dict_for_approved_action(self):
        result = classify_maintenance_action(
            _policy(),
            "caspian_inference_restart",
            SAFE_ARGV,
            current_user_approved=True,
        )

        assert result == {
            "allowed": True,
            "eligible": True,
            "reason": "approved",
            "action_id": "caspian_inference_restart",
            "command_id": "caspian_power_control_restart",
            "host_label": "caspian-inference-01",
        }


class TestMaintenanceActionDoesNotWeakenHardline:
    def test_ordinary_reboot_remains_hardline_blocked(self):
        is_hardline, description = detect_hardline_command("sudo reboot")

        assert is_hardline is True
        assert "reboot" in description.lower() or "shutdown" in description.lower()
