"""Terminal-gate coverage for security-policy config keys (#81101, #104697).

`hermes config set`/`unset` on an approval/security/allowlist/hooks key changes the
effective security policy mid-session, and `hermes approvals suggest --apply` writes
command_allowlist outside the config writer entirely. Both spellings must surface to
the operator through the dangerous-command gate. Split out of ``test_approval.py``
(file-size gate) — the behavioural coverage is unchanged.
"""

import pytest

from tools.approval import detect_dangerous_command


class TestConfigSetSecurityPolicy:
    """`hermes config set` on a security-policy key must require approval.

    config.yaml IS the security policy (approvals.mode, command_allowlist,
    security.*) and the config cache is mtime-keyed, so a write takes effect
    mid-session. The CLI refuses these keys through the config API (config.py
    #81101); this terminal-side pattern gates the terminal path the same way
    sed/tee on config.yaml are gated, so the operator is always surfaced for
    approval — including alternate entrypoints like ``python -m
    hermes_cli.main`` (#81108).
    """

    def test_security_policy_keys_detected(self):
        for cmd in (
            "hermes config set approvals.mode off",
            "hermes config set --force approvals.mode off",
            "hermes config set approvals.cron_mode deny",
            "hermes config set security.redact_secrets false",
            "hermes -p ade config set command_allowlist git",
            "hermes config unset approvals.mode",
            "hermes config unset command_allowlist",
            "python -m hermes_cli.main config unset security.redact_secrets",
        ):
            dangerous, _, desc = detect_dangerous_command(cmd)
            assert dangerous is True, cmd
            assert "security-policy" in desc, cmd

    def test_quoted_security_policy_keys_detected(self):
        # The shell strips quotes before the CLI sees the key, so quoting the
        # key must not bypass the terminal gate either (triage finding).
        for cmd in (
            'hermes config set "approvals.mode" off',
            'hermes config set --force "approvals.mode" off',
            "hermes config set 'security.redact_secrets' false",
            'hermes -p ade config set --force "command_allowlist" git',
        ):
            dangerous, _, desc = detect_dangerous_command(cmd)
            assert dangerous is True, cmd
            assert "security-policy" in desc, cmd

    def test_python_dash_m_entrypoint_detected(self):
        """``python -m hermes_cli.main`` is an alternate supported CLI
        entrypoint reaching the same config writer (main.py exposes
        ``if __name__ == "__main__"``); it must hit the same gate (#81108)."""
        for cmd in (
            "python -m hermes_cli.main config set --force approvals.mode off",
            "python3 -m hermes_cli.main config set approvals.mode off",
            "python3.12 -m hermes_cli.main -p ade config set --force security.redact_secrets false",
        ):
            dangerous, _, desc = detect_dangerous_command(cmd)
            assert dangerous is True, cmd
            assert "security-policy" in desc, cmd

    def test_benign_python_dash_m_not_flagged(self):
        for cmd in (
            "python -m hermes_cli.main config set terminal.backend docker",
            "python -m hermes_cli.main config show",
        ):
            dangerous, _, _ = detect_dangerous_command(cmd)
            assert dangerous is False, cmd

    def test_benign_config_set_not_flagged(self):
        for cmd in (
            "hermes config set terminal.backend docker",
            "hermes config set model gpt-4o",
            "hermes config set display.skin mono",
            "hermes config unset terminal.backend",
        ):
            dangerous, _, _ = detect_dangerous_command(cmd)
            assert dangerous is False, cmd

    def test_executable_policy_keys_detected(self):
        """hooks.* / hooks_auto_accept are executable policy — the hook runner
        executes configured commands via subprocess.Popen and treats
        hooks_auto_accept as consent, so every CLI spelling must hit the same
        security-policy gate as approvals.* (#104697 review round 7, F2)."""
        for cmd in (
            "hermes config set hooks_auto_accept true",
            "hermes config set --force hooks_auto_accept true",
            'hermes config set "hooks_auto_accept" true',
            'hermes config set hooks.pre_tool_call [{"command": "/bin/true"}]',
            "hermes config set hooks off",
            "hermes config unset hooks_auto_accept",
            "hermes config unset hooks.pre_tool_call",
            "python -m hermes_cli.main config set hooks_auto_accept true",
            "python -m hermes_cli.main config unset hooks_auto_accept",
            "python3.12 -m hermes_cli.main config set --force hooks.pre_tool_call x",
        ):
            dangerous, _, desc = detect_dangerous_command(cmd)
            assert dangerous is True, cmd
            assert "security-policy" in desc, cmd

    def test_approvals_suggest_apply_detected(self):
        """`hermes approvals suggest --apply` persists command_allowlist via
        tools/approval.save_permanent_allowlist() — outside the
        set_config_value guard — so the terminal gate is the only surface
        (#104697 review round 7, F2 inventory)."""
        for cmd in (
            "hermes approvals suggest --apply 1,3",
            "hermes approvals suggest --json --apply 1",
            "hermes -p ade approvals suggest --apply 2",
            "python -m hermes_cli.main approvals suggest --apply 1",
            "python3 -m hermes_cli.main approvals suggest --apply 2,4",
        ):
            dangerous, _, desc = detect_dangerous_command(cmd)
            assert dangerous is True, cmd
            assert "command_allowlist" in desc, cmd

    def test_approvals_suggest_dry_run_not_flagged(self):
        for cmd in (
            "hermes approvals suggest",
            "hermes approvals suggest --days 30",
            "hermes approvals test --json",
        ):
            dangerous, _, _ = detect_dangerous_command(cmd)
            assert dangerous is False, cmd

    def test_mode_change_spelling_detected(self):
        """Prophylactic gate for the shell spelling of the /approvals slash
        command (no such argv subcommand exists today — `hermes approvals` has
        only suggest/test)."""
        for cmd in (
            "hermes approvals off",
            "hermes approvals smart",
            "hermes approvals manual",
        ):
            dangerous, _, desc = detect_dangerous_command(cmd)
            assert dangerous is True, cmd
            assert "mode change" in desc, cmd
