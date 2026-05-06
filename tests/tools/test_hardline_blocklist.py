"""Tests for the unconditional hardline command blocklist.

The hardline list is a floor below yolo: a small set of commands so
catastrophic they should never run via the agent, regardless of --yolo,
gateway /yolo, approvals.mode=off, or cron approve mode.

Inspired by Mercury Agent's permission-hardened blocklist.
"""
import os

import pytest

from tools.approval import (
    DANGEROUS_PATTERNS,
    HARDLINE_PATTERNS,
    check_all_command_guards,
    check_dangerous_command,
    detect_dangerous_command,
    detect_hardline_command,
    disable_session_yolo,
    enable_session_yolo,
    reset_current_session_key,
    set_current_session_key,
)


# -------------------------------------------------------------------------
# Pattern detection
# -------------------------------------------------------------------------

# Commands that MUST be hardline-blocked.
_HARDLINE_BLOCK = [
    # rm -rf targeting root / system dirs / home
    "rm -rf /",
    "rm -rf /*",
    "rm -rf /home",
    "rm -rf /home/*",
    "rm -rf /etc",
    "rm -rf /usr",
    "rm -rf /var",
    "rm -rf /boot",
    "rm -rf /bin",
    "rm --recursive --force /",
    "rm -fr /",
    "sudo rm -rf /",
    "rm -rf ~",
    "rm -rf ~/",
    "rm -rf ~/*",
    "rm -rf $HOME",
    # Filesystem format
    "mkfs.ext4 /dev/sda1",
    "mkfs /dev/sdb",
    "mkfs.xfs /dev/nvme0n1",
    # Raw block device overwrites
    "dd if=/dev/zero of=/dev/sda bs=1M",
    "dd if=/dev/urandom of=/dev/nvme0n1",
    "dd if=anything of=/dev/hda",
    "echo bad > /dev/sda",
    "cat /dev/urandom > /dev/sdb",
    # Fork bomb
    ":(){ :|:& };:",
    # System-wide kill
    "kill -9 -1",
    "kill -1",
    # Shutdown / reboot / halt
    "shutdown -h now",
    "shutdown -r now",
    "sudo shutdown now",
    "reboot",
    "sudo reboot",
    "halt",
    "poweroff",
    "init 0",
    "init 6",
    "telinit 0",
    "systemctl poweroff",
    "systemctl reboot",
    "systemctl halt",
    # Compound / subshell variants
    "ls; reboot",
    "echo done && shutdown -h now",
    "false || halt",
    "$(reboot)",
    "`shutdown now`",
    "sudo -E shutdown now",
    "env FOO=1 reboot",
    "exec shutdown",
    "nohup reboot",
    "setsid poweroff",
]


# Commands that look superficially similar but must NOT be hardline-blocked.
_HARDLINE_ALLOW = [
    # rm on non-protected paths
    "rm -rf /tmp/foo",
    "rm -rf /tmp/*",
    "rm -rf ./build",
    "rm -rf node_modules",
    "rm -rf /home/user/scratch",  # subpath of /home, not /home itself
    "rm -rf ~/Downloads/old",
    "rm -rf $HOME/tmp",
    "rm foo.txt",
    "rm -rf some/path",
    # dd to regular files
    "dd if=/dev/zero of=./image.bin",
    "dd if=./data of=./backup.bin",
    # Redirect to regular files / non-block devices
    "echo done > /tmp/flag",
    "echo test > /dev/null",
    # Reading devices is fine
    "ls /dev/sda",
    "cat /dev/urandom | head -c 10",
    # Unrelated commands that happen to contain the trigger word
    "grep 'shutdown' logs.txt",
    "echo reboot",
    "echo '# init 0 in comment'",
    "cat rebooting.log",
    "echo 'halt and catch fire'",
    "python3 -c 'print(\"shutdown\")'",
    "find . -name '*reboot*'",
    # Word-boundary protection
    "mkfs_helper --version",
    # systemctl non-destructive verbs
    "systemctl status nginx",
    "systemctl restart nginx",
    "systemctl stop nginx",
    "systemctl start nginx",
    # targeted kill
    "kill -9 12345",
    "kill -HUP 1234",
    "pkill python",
    # Ordinary ops
    "git status",
    "npm run build",
    "sudo apt update",
    "curl https://example.com | head",
]


@pytest.mark.parametrize("command", _HARDLINE_BLOCK)
def test_hardline_detection_blocks(command):
    is_hl, desc = detect_hardline_command(command)
    assert is_hl, f"expected hardline to match {command!r}"
    assert desc, "hardline match must provide a description"


@pytest.mark.parametrize("command", _HARDLINE_ALLOW)
def test_hardline_detection_allows(command):
    is_hl, desc = detect_hardline_command(command)
    assert not is_hl, f"expected hardline NOT to match {command!r} (got: {desc})"
    assert desc is None


# -------------------------------------------------------------------------
# Integration with the approval flow
# -------------------------------------------------------------------------

@pytest.fixture
def clean_session(monkeypatch):
    """Reset session-scoped approval state around each test."""
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    token = set_current_session_key("hardline_test")
    try:
        disable_session_yolo("hardline_test")
        yield
    finally:
        disable_session_yolo("hardline_test")
        reset_current_session_key(token)


def test_check_dangerous_command_blocks_hardline(clean_session):
    result = check_dangerous_command("rm -rf /", "local")
    assert result["approved"] is False
    assert result.get("hardline") is True
    assert "BLOCKED (hardline)" in result["message"]


def test_check_all_command_guards_blocks_hardline(clean_session):
    result = check_all_command_guards("rm -rf /", "local")
    assert result["approved"] is False
    assert result.get("hardline") is True
    assert "BLOCKED (hardline)" in result["message"]


def test_yolo_env_var_cannot_bypass_hardline(clean_session, monkeypatch):
    """HERMES_YOLO_MODE=1 must not bypass the hardline floor."""
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    for cmd in ["rm -rf /", "shutdown -h now", "mkfs.ext4 /dev/sda", "reboot"]:
        r1 = check_dangerous_command(cmd, "local")
        assert r1["approved"] is False, f"yolo leaked hardline on {cmd!r} (check_dangerous_command)"
        assert r1.get("hardline") is True

        r2 = check_all_command_guards(cmd, "local")
        assert r2["approved"] is False, f"yolo leaked hardline on {cmd!r} (check_all_command_guards)"
        assert r2.get("hardline") is True


def test_session_yolo_cannot_bypass_hardline(clean_session):
    """Gateway /yolo (session-scoped) must not bypass the hardline floor."""
    enable_session_yolo("hardline_test")

    result = check_dangerous_command("rm -rf /", "local")
    assert result["approved"] is False
    assert result.get("hardline") is True

    result = check_all_command_guards("rm -rf /", "local")
    assert result["approved"] is False
    assert result.get("hardline") is True


def test_approvals_mode_off_cannot_bypass_hardline(clean_session, monkeypatch, tmp_path):
    """config approvals.mode=off (yolo-equivalent) must not bypass hardline."""
    # _get_approval_mode() reads from hermes config; simplest path: monkeypatch the helper.
    import tools.approval as approval_mod
    monkeypatch.setattr(approval_mod, "_get_approval_mode", lambda: "off")

    result = check_all_command_guards("rm -rf /", "local")
    assert result["approved"] is False
    assert result.get("hardline") is True


def test_cron_approve_mode_cannot_bypass_hardline(clean_session, monkeypatch):
    """Cron sessions with cron_mode=approve must not bypass hardline."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    import tools.approval as approval_mod
    monkeypatch.setattr(approval_mod, "_get_cron_approval_mode", lambda: "approve")

    result = check_all_command_guards("rm -rf /", "local")
    assert result["approved"] is False
    assert result.get("hardline") is True


def test_container_backends_still_bypass(clean_session):
    """Containerized backends remain bypass-approved — they can't touch the host.

    Hardline only protects environments with real host impact (local, ssh).
    """
    for env in ("docker", "singularity", "modal", "daytona", "vercel_sandbox"):
        r1 = check_dangerous_command("rm -rf /", env)
        assert r1["approved"] is True, f"container {env} should still bypass"
        r2 = check_all_command_guards("rm -rf /", env)
        assert r2["approved"] is True, f"container {env} should still bypass"


def test_hardline_runs_before_dangerous_detection(clean_session):
    """Hardline command should return hardline block, not dangerous approval prompt."""
    # `rm -rf /` is both hardline AND matches DANGEROUS_PATTERNS. Hardline must win.
    is_dangerous, _, _ = detect_dangerous_command("rm -rf /")
    assert is_dangerous, "precondition: rm -rf / is also in DANGEROUS_PATTERNS"

    result = check_dangerous_command("rm -rf /", "local")
    assert result.get("hardline") is True


def test_recoverable_dangerous_commands_still_pass_yolo(clean_session, monkeypatch):
    """Yolo still bypasses the regular DANGEROUS_PATTERNS list.

    This confirms we haven't broken the yolo escape hatch — only narrowed it.
    """
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    # These are dangerous but NOT hardline — yolo should still pass them.
    for cmd in ["rm -rf /tmp/x", "chmod -R 777 .", "git reset --hard", "git push --force"]:
        # Sanity: still flagged as dangerous
        is_dangerous, _, _ = detect_dangerous_command(cmd)
        assert is_dangerous, f"precondition: {cmd!r} should be in DANGEROUS_PATTERNS"
        # But NOT hardline
        is_hl, _ = detect_hardline_command(cmd)
        assert not is_hl, f"{cmd!r} should not be hardline"
        # And yolo bypasses the dangerous check
        result = check_dangerous_command(cmd, "local")
        assert result["approved"] is True, f"yolo should have bypassed {cmd!r}"


def test_hardline_list_is_small():
    """Hardline list stays focused on unrecoverable commands only.

    If you're adding a 20th+ pattern, reconsider — it probably belongs in
    DANGEROUS_PATTERNS where yolo can still bypass it.
    """
    assert len(HARDLINE_PATTERNS) <= 20, (
        f"HARDLINE_PATTERNS has grown to {len(HARDLINE_PATTERNS)} entries; "
        "only truly unrecoverable commands belong here."
    )
