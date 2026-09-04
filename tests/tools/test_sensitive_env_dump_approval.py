"""PRD-279: broad environment dumps require explicit approval."""

import pytest

import tools.approval as approval


@pytest.mark.parametrize(
    "command",
    [
        "env",
        "/usr/bin/env",
        "printenv",
        "set",
        "export -p",
        "declare -x",
        "launchctl print gui/501/ai.hermes.gateway",
        "/bin/launchctl print system/ai.hermes.gateway",
    ],
)
def test_broad_environment_dump_is_classified_as_dangerous(command):
    dangerous, key, description = approval.detect_dangerous_command(command)

    assert dangerous is True, command
    assert key == description
    assert "environment" in description.lower()


def test_sensitive_environment_dump_runs_through_manual_approval(monkeypatch):
    prompts = []
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: False)
    monkeypatch.setattr(approval, "_command_matches_permanent_allowlist", lambda _command: False)
    monkeypatch.setattr(approval, "_get_approval_config", lambda: {"mode": "manual"})

    result = approval.check_dangerous_command(
        "printenv",
        "local",
        approval_callback=lambda command, description, **kwargs: (
            prompts.append((command, description)) or "deny"
        ),
    )

    assert result["approved"] is False
    assert prompts and prompts[0][0] == "printenv"
    assert "environment" in prompts[0][1].lower()


def test_metadata_only_launchd_status_filter_does_not_require_approval():
    command = (
        "launchctl print gui/501/ai.hermes.gateway | "
        "/usr/bin/grep -E '^[[:space:]]*(state|pid|last exit code) ='"
    )

    assert approval.detect_dangerous_command(command) == (False, None, None)


@pytest.mark.parametrize(
    "command",
    [
        "printenv PATH",
        "env FEATURE_FLAG=1 command --version",
        "set -o pipefail",
        "launchctl print-disabled gui/501",
        "echo launchctl print gui/501/example",
    ],
)
def test_narrow_or_nonexecuted_environment_commands_are_not_classified(command):
    assert approval.detect_dangerous_command(command) == (False, None, None)