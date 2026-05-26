"""Tests for the unconditional hardline command blocklist.

The hardline list is a floor below yolo: a small set of commands so
catastrophic they should never run via the agent, regardless of --yolo,
gateway /yolo, approvals.mode=off, or cron approve mode.

Inspired by Mercury Agent's permission-hardened blocklist.
"""
import logging
import os

import pytest

from tools.approval import (
    DANGEROUS_PATTERNS,
    HARDLINE_PATTERNS,
    check_all_command_guards,
    check_dangerous_command,
    check_unbypassable_command_guards,
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


# =========================================================================
# Sudo stdin guard — blocks "sudo -S" without SUDO_PASSWORD
# =========================================================================

_SUDO_STDIN_BLOCK = [
    "sudo -S whoami",
    "echo hunter2 | sudo -S whoami",
    "sudo -S -u root whoami",
    "sudo -S apt-get install foo",
    "echo password | sudo -S systemctl restart nginx",
    "sudo -k && sudo -S whoami",
]

_SUDO_STDIN_ALLOW = [
    # Plain sudo without -S — goes through normal approval
    "sudo whoami",
    "sudo apt-get update",
    "sudo -u root whoami",
    # -S flag not attached to sudo
    "echo -S hello",
    "some_tool -S thing",
    # Literal text mention of sudo
    "echo 'use sudo -S to pipe passwords'",
]

_SUDO_STDIN_BLOCK_YOLO = [
    "sudo -S whoami",
    "echo hunter2 | sudo -S apt-get install",
]


def test_sudo_stdin_guard_detects_without_password():
    """sudo -S is dangerous when SUDO_PASSWORD is not configured."""
    import tools.approval as approval_mod

    for cmd in _SUDO_STDIN_BLOCK:
        is_blocked, desc = approval_mod._check_sudo_stdin_guard(cmd)
        assert is_blocked, f"expected sudo stdin guard to block {cmd!r}"
        assert "sudo" in desc.lower()


def test_sudo_stdin_guard_allows_benign_commands():
    """Commands without explicit sudo -S are not blocked."""
    import tools.approval as approval_mod

    for cmd in _SUDO_STDIN_ALLOW:
        is_blocked, desc = approval_mod._check_sudo_stdin_guard(cmd)
        assert not is_blocked, f"expected sudo stdin guard NOT to block {cmd!r}"


def test_sudo_stdin_guard_bypassed_when_password_configured(monkeypatch):
    """When SUDO_PASSWORD is set, sudo -S is legitimate (injected by transform)."""
    import tools.approval as approval_mod

    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    for cmd in _SUDO_STDIN_BLOCK:
        is_blocked, _ = approval_mod._check_sudo_stdin_guard(cmd)
        assert not is_blocked, f"with SUDO_PASSWORD set, {cmd!r} should NOT be blocked"


def test_sudo_stdin_guard_blocks_via_check_all_command_guards(clean_session):
    """Integration: check_all_command_guards returns block for sudo -S."""
    for cmd in _SUDO_STDIN_BLOCK:
        result = check_all_command_guards(cmd, "local")
        assert result["approved"] is False, f"expected block on {cmd!r}"
        # Should NOT be marked as hardline (it's sudo-specific)
        assert result.get("hardline") is not True
        assert "BLOCKED" in result["message"]
        assert "sudo -S" in result["message"].lower() or "sudo password" in result["message"].lower()


def test_sudo_stdin_guard_not_blocked_by_yolo(clean_session, monkeypatch):
    """yolo/approvals.mode=off must NOT bypass sudo stdin guard."""
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    for cmd in _SUDO_STDIN_BLOCK_YOLO:
        result = check_all_command_guards(cmd, "local")
        assert result["approved"] is False, f"yolo leaked sudo guard on {cmd!r}"


def test_sudo_stdin_guard_container_bypass(clean_session):
    """Containerized backends still bypass — they can't touch the host."""
    for env in ("docker", "singularity", "modal", "daytona", "vercel_sandbox"):
        for cmd in _SUDO_STDIN_BLOCK:
            result = check_all_command_guards(cmd, env)
            assert result["approved"] is True, f"container {env} should bypass sudo guard on {cmd!r}"


# =========================================================================
# Protected Hermes config guard — direct writes must go through config CLI
# =========================================================================

_PROTECTED_CONFIG_BLOCK = [
    "echo x > ~/.hermes/config.yaml",
    "echo TOKEN=x >> ~/.hermes/.env",
    "printf x | tee $HERMES_HOME/config.yaml",
    "sed -i 's/a/b/' ${HERMES_HOME}/.env",
    "python3 -c \"open('~/.hermes/config.yaml','w').write('x')\"",
    "/usr/bin/python3 -c \"open('~/.hermes/config.yaml','w').write('x')\"",
    "/usr/bin/node -e \"require('fs').writeFileSync('~/.hermes/config.yaml','x')\"",
    "/usr/bin/perl -e \"open(my $fh, '>', '~/.hermes/config.yaml')\"",
    "/usr/bin/ruby -e \"File.write('~/.hermes/config.yaml','x')\"",
    "cat >| ~/.hermes/config.yaml",
    "touch ~/.hermes/config.yaml",
    "/usr/bin/touch ~/.hermes/config.yaml",
    "dd if=/tmp/config.yaml of=~/.hermes/config.yaml",
    # dd overwriting protected files must be blocked just like redirection/tee.
    "dd if=/tmp/x of=~/.hermes/config.yaml",
    f"dd if=/tmp/x of={os.path.expanduser('~/.hermes/.env')}",
    f'echo x > "{os.path.expanduser("~")}"/.hermes/config.yaml',
    f'echo x > "{os.path.expanduser("~/.hermes")}"/config.yaml',
    "echo x > ~/.hermes/./config.yaml",
    "echo x > ~/.hermes/../.hermes/config.yaml",
    "echo x > ~/.hermes//config.yaml",
    "echo x > ~/.hermes/logs/../config.yaml",
    "echo x > ~/\\.hermes/con\\fig.yaml",
    "echo x > ~/\\.hermes/.\\env",
    f"echo x > ~/../{os.path.basename(os.path.expanduser('~'))}/.hermes/config.yaml",
    f"echo x > $HOME/../{os.path.basename(os.path.expanduser('~'))}/.hermes/.env",
    "cd ~/.hermes && echo x > config.yaml",
    "cd -- ~/.hermes && echo x > config.yaml",
    "cd ~/.hermes || exit; echo x > config.yaml",
    "cd ~/.hermes/logs && echo x > ../config.yaml",
    "cd ~/.hermes && python3 -c \"open('config.yaml','w').write('x')\"",
    "env python3 -c \"open('~/.hermes/config.yaml','w').write('x')\"",
    "python3.12 -c \"open('~/.hermes/config.yaml','w').write('x')\"",
    "cd ~/.hermes && python3 -c \"from pathlib import Path; Path('config.yaml').write_text('x')\"",
    "cd ~/.hermes && /usr/bin/env node -e \"require('fs').writeFileSync('config.yaml','x')\"",
    "cd ~/.hermes && /usr/bin/node -e \"require('fs').writeFileSync('config.yaml','x')\"",
    "cd ~/.hermes && /usr/bin/ruby -e \"File.write('config.yaml','x')\"",
    "cd ~/.hermes && /usr/bin/perl -e \"open(my $fh, '>', 'config.yaml')\"",
    "cd ~/.hermes && echo x > config.yaml; cd /tmp && true",
    "cd -- ~/.hermes && echo x > config.yaml; cd /tmp && true",
    "cd ~/.hermes || exit; echo x > config.yaml; cd /tmp && true",
    "cd ~/.hermes && python3 -c \"open('config.yaml','w').write('x')\"; cd /tmp && true",
    "cd ~/.hermes && python3 -c \"from pathlib import Path; Path('config.yaml').write_text('x')\"; cd /tmp && true",
    "(cd ~/.hermes && echo x > config.yaml)",
    "{ cd ~/.hermes && echo x > config.yaml; }",
    "cd ~/.hermes && (cd /tmp) && echo x > config.yaml",
    "bash -c 'echo x > ~/.hermes/config.yaml'",
    "bash -o pipefail -c 'echo x > ~/.hermes/config.yaml'",
    "bash --rcfile /tmp/r -c 'echo x > ~/.hermes/config.yaml'",
    "/bin/sh -c 'cd ~/.hermes && echo x > config.yaml'",
    "cd ~/.hermes && bash -c 'echo x > config.yaml'",
    "cd ~/.hermes && python3 -c \"p='config.yaml'; open(p,'w').write('x')\"",
    "cd ~/.hermes && python3 -c \"from pathlib import Path; p=Path('config.yaml'); p.write_text('x')\"",
    "cd ~/.hermes && ln -s config.yaml /tmp/hermes-review-link && echo x > /tmp/hermes-review-link",
    "cd ~/.hermes && ln -s config.yaml /tmp/hermes-review-link && printf x | tee /tmp/hermes-review-link",
    "cd ~/.hermes && ln -s config.yaml /tmp/hermes-review-link && truncate -s0 /tmp/hermes-review-link",
    "cd ~/.hermes && ln -s config.yaml /tmp/hermes-review-link && cp /tmp/x /tmp/hermes-review-link",
    "rm ~/.hermes/config.yaml",
    "mv /tmp/config.yaml ~/.hermes/config.yaml",
    "/bin/mv /tmp/config.yaml ~/.hermes/config.yaml",
    "/bin/cp /tmp/config.yaml ~/.hermes/config.yaml",
    "/bin/cp /tmp/config.yaml ~/.hermes/",
    "/bin/mv /tmp/.env ~/.hermes/",
    "/usr/bin/install /tmp/config.yaml ~/.hermes/config.yaml",
    "/usr/bin/install /tmp/config.yaml ~/.hermes/",
    "cp -t ~/.hermes /tmp/config.yaml",
    "cp -t~/.hermes /tmp/config.yaml",
    "cp -at ~/.hermes /tmp/config.yaml",
    "install -t ~/.hermes /tmp/.env",
    "install -t~/.hermes /tmp/.env",
    "install -Ct ~/.hermes /tmp/.env",
    "mv -t~/.hermes /tmp/config.yaml",
    "mv -vt ~/.hermes /tmp/config.yaml",
    "bash -euxo pipefail -c 'echo x > ~/.hermes/config.yaml'",
    "cd ~/.hermes && mv config.yaml /tmp/config.yaml.bak",
    "cd ~/.hermes && /usr/bin/python3 -c \"open('config.yaml','w').write('x')\"",
    "cd ~/.hermes && mv config.yaml /tmp/config.yaml.bak; cd /tmp && true",
]

_PROTECTED_CONFIG_ALLOW = [
    "hermes config set web.search_backend brave-free",
    "cat ~/.hermes/config.yaml | head -n 1",
    "grep web ~/.hermes/config.yaml",
    "touch /tmp/done && grep web ~/.hermes/config.yaml",
    "rm -rf node_modules && cat ~/.hermes/config.yaml",
    "cp ~/.hermes/config.yaml /tmp/config.yaml.bak",
    "echo '~/.hermes/config.yaml'",
    "echo 'echo x > ~/.hermes/config.yaml'",
    "echo 'foo && echo x > ~/.hermes/config.yaml'",
    "printf '%s\\n' 'printf x | tee ~/.hermes/config.yaml'",
    "echo 'touch ~/.hermes/config.yaml'",
    "echo 'python3 -c open(~/.hermes/config.yaml,w)'",
    "python3 -c \"print(1)\" ~/.hermes/config.yaml",
    "/usr/bin/env python3 -c \"print(1)\" ~/.hermes/config.yaml",
    "node -e \"console.log(1)\" ~/.hermes/config.yaml",
    "echo x > /tmp/config.yaml",
    # Read-only nested shell payloads under find/xargs must stay allowed.
    "find /tmp -maxdepth 0 -exec sh -c 'cat ~/.hermes/config.yaml' \\;",
    "xargs -I{} sh -c 'cat ~/.hermes/config.yaml' <<< foo",
    "find /usr/bin -name sh",
    # Symlink whose destination is OUTSIDE protected config (source protected,
    # link lands in /tmp) is not itself a write — stays allowed.
    "cd ~/.hermes && ln -s config.yaml /tmp/readonly-link",
]


def test_protected_config_guard_detects_direct_writes(clean_session):
    import tools.approval as approval_mod

    for cmd in _PROTECTED_CONFIG_BLOCK:
        is_blocked, desc = approval_mod._check_protected_config_write_guard(cmd)
        assert is_blocked, f"expected protected config guard to block {cmd!r}"
        assert "protected" in desc.lower()


def test_protected_config_guard_detects_literal_active_home(clean_session):
    from hermes_constants import get_hermes_home
    import tools.approval as approval_mod

    for name in ("config.yaml", ".env"):
        cmd = f"echo x > {get_hermes_home() / name}"
        is_blocked, desc = approval_mod._check_protected_config_write_guard(cmd)
        assert is_blocked, f"expected protected config guard to block active HERMES_HOME path {name!r}"
        assert "protected" in desc.lower()


def test_protected_config_guard_detects_default_root_in_profile_mode(clean_session, monkeypatch, tmp_path):
    root = tmp_path / "hermes-root"
    profile_home = root / "profiles" / "ops" / "home"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(profile_home))
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "ops"))

    for name in ("config.yaml", ".env"):
        cmd = f"echo x > {root / name}"
        result = check_all_command_guards(cmd, "local")
        assert result["approved"] is False, f"expected default Hermes root path to block {name!r}"
        assert result.get("protected_config") is True


def test_protected_config_guard_detects_relative_cwd_with_spaces(clean_session, monkeypatch, tmp_path):
    import tools.approval as approval_mod

    hermes_home = tmp_path / "hermes home with spaces"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    is_blocked, desc = approval_mod._check_protected_config_write_guard(
        "echo x > config.yaml",
        cwd=str(hermes_home),
    )
    assert is_blocked
    assert "protected" in desc.lower()

    absolute_cmd = f'echo x > "{hermes_home / "config.yaml"}"'
    is_blocked, desc = approval_mod._check_protected_config_write_guard(absolute_cmd)
    assert is_blocked
    assert "protected" in desc.lower()

    read_copy_cmd = f'cp "{hermes_home / "config.yaml"}" /tmp/config.yaml.bak'
    is_blocked, desc = approval_mod._check_protected_config_write_guard(read_copy_cmd)
    assert not is_blocked, f"protected source read should not be treated as direct write (got {desc})"

    quoted_cd_cmd = f'cd "{hermes_home}" && echo x > config.yaml'
    is_blocked, desc = approval_mod._check_protected_config_write_guard(quoted_cd_cmd)
    assert is_blocked
    assert "protected" in desc.lower()

    for cmd in (
        f'(cd "{hermes_home}"); echo x > config.yaml',
        f'(cd "{hermes_home}" && cat config.yaml); echo x > config.yaml',
    ):
        is_blocked, desc = approval_mod._check_protected_config_write_guard(cmd, cwd="/tmp")
        assert not is_blocked, f"subshell cd should not leak to parent cwd for {cmd!r} (got {desc})"


def test_protected_config_guard_detects_relative_guard_cwd(clean_session, monkeypatch, tmp_path):
    import tools.approval as approval_mod

    hermes_home = tmp_path / "hermes-home"
    repo_dir = hermes_home / "hermes-agent"
    repo_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.chdir(repo_dir)

    is_blocked, desc = approval_mod._check_protected_config_write_guard(
        "echo x > config.yaml",
        cwd="..",
    )
    assert is_blocked
    assert "protected" in desc.lower()


def test_protected_config_guard_detects_hermes_home_symbolic_parent(clean_session, monkeypatch, tmp_path):
    import tools.approval as approval_mod

    base = tmp_path / "base"
    hermes_home = base / "hermes-home"
    hermes_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    cmd = "echo x > $HERMES_HOME/../hermes-home/config.yaml"
    is_blocked, desc = approval_mod._check_protected_config_write_guard(cmd)
    assert is_blocked
    assert "protected" in desc.lower()


def test_protected_config_guard_allows_non_writes(clean_session):
    import tools.approval as approval_mod

    for cmd in _PROTECTED_CONFIG_ALLOW:
        is_blocked, desc = approval_mod._check_protected_config_write_guard(cmd)
        assert not is_blocked, f"expected protected config guard NOT to block {cmd!r} (got {desc})"


def test_protected_config_guard_below_yolo_and_off(clean_session, monkeypatch):
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")
    for cmd in _PROTECTED_CONFIG_BLOCK:
        r1 = check_dangerous_command(cmd, "local")
        assert r1["approved"] is False, f"yolo leaked protected config guard on {cmd!r}"
        assert r1.get("protected_config") is True

        r2 = check_all_command_guards(cmd, "local")
        assert r2["approved"] is False, f"yolo leaked protected config guard on {cmd!r}"
        assert r2.get("protected_config") is True


def test_protected_config_guard_below_approvals_off(clean_session, monkeypatch):
    import tools.approval as approval_mod

    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.setattr(approval_mod, "_get_approval_mode", lambda: "off")
    result = check_all_command_guards("echo x > ~/.hermes/config.yaml", "local")
    assert result["approved"] is False
    assert result.get("protected_config") is True


def test_unbypassable_guard_blocks_protected_config_direct_write(clean_session):
    result = check_unbypassable_command_guards("echo x > ~/.hermes/config.yaml", "local")
    assert result["approved"] is False
    assert result.get("protected_config") is True
    assert "Secret values were not inspected" in result["message"]


def test_protected_config_guard_log_does_not_leak_raw_command(clean_session, caplog):
    caplog.set_level(logging.WARNING, logger="tools.approval")

    result = check_unbypassable_command_guards(
        "echo SECRET_VALUE > ~/.hermes/config.yaml && sudo -S whoami",
        "local",
    )

    assert result["approved"] is False
    assert result.get("protected_config") is True
    assert "Protected config guard block" in caplog.text
    assert "SECRET_VALUE" not in caplog.text


def test_protected_config_guard_container_bypass(clean_session):
    for env in ("docker", "singularity", "modal", "daytona", "vercel_sandbox"):
        result = check_all_command_guards("echo x > ~/.hermes/config.yaml", env)
        assert result["approved"] is True, f"container {env} should bypass protected config guard"


# =========================================================================
# Independent review regressions — wrapper/stdin/argv bypasses
# =========================================================================

_REVIEW_SUDO_STDIN_BLOCK = [
    "sudo -nS whoami",
    "sudo -Sn whoami",
    "sudo --stdin whoami",
    "echo x | sudo -nS whoami",
    "/usr/bin/env -S 'sudo -S whoami'",
    "/usr/bin/env -S 'sh -c \"sudo -S whoami\"'",
    "/usr/bin/env --split-string='sudo --stdin whoami'",
]


def test_sudo_stdin_guard_blocks_combined_and_long_forms(clean_session, monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    for cmd in _REVIEW_SUDO_STDIN_BLOCK:
        result = check_unbypassable_command_guards(cmd, "local")
        assert result["approved"] is False, f"expected sudo stdin guard to block {cmd!r}"
        assert "sudo password guessing" in result["message"]


def test_sudo_stdin_guard_log_does_not_leak_raw_command(clean_session, monkeypatch, caplog):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    caplog.set_level(logging.WARNING, logger="tools.approval")

    result = check_unbypassable_command_guards(
        "printf SECRET_VALUE | sudo -S whoami",
        "local",
    )

    assert result["approved"] is False
    assert "Sudo stdin guard block" in caplog.text
    assert "SECRET_VALUE" not in caplog.text


_REVIEW_PROTECTED_CONFIG_BLOCK = [
    "nice truncate -s0 ~/.hermes/config.yaml",
    "time truncate -s0 ~/.hermes/.env",
    "sudo sh -c 'echo x > ~/.hermes/config.yaml'",
    "rsync /tmp/config.yaml ~/.hermes/config.yaml",
    "rsync /tmp/.env ~/.hermes/",
    "echo x > ~/.hermes/$'config.yaml'",
    "touch ~/.hermes/{config.yaml}",
    "python3 -c \"import sys; open(sys.argv[1], 'w').write('x')\" ~/.hermes/config.yaml",
    "python3 -c \"from pathlib import Path; import sys; Path(sys.argv[1]).write_text('x')\" ~/.hermes/config.yaml",
    "node -e \"require('fs').writeFileSync(process.argv[1], 'x')\" ~/.hermes/config.yaml",
    # env -S / --split-string must preserve the split-string flag payload and
    # recurse into nested shell writes.
    "/usr/bin/env -S 'sh -c \"echo x > ~/.hermes/config.yaml\"'",
    "/usr/bin/env --split-string='sh -c \"echo x > ~/.hermes/.env\"'",
    "env -S 'sh -c \"echo x > ~/.hermes/config.yaml\"'",
    # Nested shell payloads under xargs / find -exec — the inner `sh -c`
    # argument is opaque to the outer-command word scan, so the guard must
    # recurse into it.
    "find /tmp -maxdepth 0 -exec sh -c 'echo x > ~/.hermes/config.yaml' \\;",
    "find /tmp -maxdepth 0 -execdir sh -c 'echo x > ~/.hermes/.env' \\;",
    "xargs -I{} sh -c 'echo x > ~/.hermes/config.yaml' <<< foo",
    "echo foo | xargs sh -c 'echo x > ~/.hermes/config.yaml'",
    # Relative symlink whose destination IS the protected config after a cd —
    # creating the link overwrites config.yaml. The absolute dest form was
    # already blocked; this closes the relative-cwd gap.
    "cd ~/.hermes && ln -sf /tmp/x config.yaml",
    "cd ~/.hermes && ln -f /tmp/x .env",
    # Directory destinations must be treated like copy/move semantics: the
    # source basename becomes the link name inside the protected config dir.
    "ln -sf /tmp/config.yaml ~/.hermes/",
    "ln -sf /tmp/.env ~/.hermes/",
    "cd ~/.hermes && ln -sf /tmp/config.yaml .",
    "cd ~/.hermes && ln -sf /tmp/.env .",
]


def test_protected_config_guard_blocks_review_bypasses(clean_session):
    import tools.approval as approval_mod

    for cmd in _REVIEW_PROTECTED_CONFIG_BLOCK:
        is_blocked, desc = approval_mod._check_protected_config_write_guard(cmd)
        assert is_blocked, f"expected protected config guard to block {cmd!r} (got {desc})"


_REVIEW_HARDLINE_BLOCK = [
    "env -i reboot",
    "env -i PATH=/sbin reboot",
    "/usr/bin/env -S 'reboot'",
    "/usr/bin/env -S 'sh -c reboot'",
    "env --split-string='sh -c reboot'",
    "command reboot",
    "time -p reboot",
    "bash -c 'reboot'",
    "rm -rf '$HOME'",
    'rm -rf "$HOME"',
    "rm -rf '~'",
]


def test_hardline_guard_blocks_wrapper_forms(clean_session):
    for cmd in _REVIEW_HARDLINE_BLOCK:
        result = check_unbypassable_command_guards(cmd, "local")
        assert result["approved"] is False, f"expected hardline guard to block {cmd!r}"
        assert result.get("hardline") is True


def test_shell_stdin_and_shell_c_payloads_are_unbypassable(clean_session):
    import tools.approval as approval_mod

    hardline_cases = [
        "printf 'reboot' | sh",
        "bash <<< 'reboot'",
    ]
    for cmd in hardline_cases:
        approval = approval_mod.check_unbypassable_command_guards(cmd, "local")
        assert not approval["approved"], cmd

    protected_cases = [
        "printf 'echo x > ~/.hermes/config.yaml' | sh",
        "bash <<< 'echo x > ~/.hermes/config.yaml'",
    ]
    for cmd in protected_cases:
        assert approval_mod._check_protected_config_write_guard(cmd)[0], cmd

    sudo_cases = [
        "printf x | bash -c 'sudo -S whoami'",
        "sh -c 'sudo --stdin whoami'",
    ]
    for cmd in sudo_cases:
        approval = approval_mod.check_unbypassable_command_guards(cmd, "local")
        assert not approval["approved"], cmd


def test_export_alias_to_protected_config_is_blocked(clean_session):
    import tools.approval as approval_mod

    assert approval_mod._check_protected_config_write_guard(
        "export p=~/.hermes/config.yaml; echo x > $p"
    )[0]
    assert approval_mod._check_protected_config_write_guard(
        "readonly p=~/.hermes/.env; touch $p"
    )[0]


def test_process_stdin_wrapped_shell_and_repl_guards(tmp_path):
    from tools.process_registry import ProcessRegistry, ProcessSession

    class FakePty:
        def __init__(self):
            self.writes = []

        def write(self, data):
            self.writes.append(data)

    registry = ProcessRegistry()
    shell_session = ProcessSession(
        id="wrapped-shell",
        command="/usr/bin/env bash",
        cwd=str(tmp_path),
        started_at=0,
    )
    shell_session._pty = FakePty()
    registry._running[shell_session.id] = shell_session

    blocked = registry.submit_stdin(shell_session.id, "reboot")
    assert blocked["status"] == "blocked"
    assert shell_session._pty.writes == []

    repl_session = ProcessSession(
        id="py-repl",
        command="python3",
        cwd=str(tmp_path),
        started_at=0,
    )
    repl_session._pty = FakePty()
    registry._running[repl_session.id] = repl_session

    blocked = registry.submit_stdin(
        repl_session.id,
        "open('~/.hermes/config.yaml', 'w').write('x')",
    )
    assert blocked["status"] == "blocked"
    assert repl_session._pty.writes == []


@pytest.mark.parametrize("command", [
    "python3.12",
    "/usr/bin/python3.12",
    "node20",
    "nodejs20",
    "ruby3.2",
    "perl5.36",
])
def test_process_stdin_versioned_repl_names_are_guarded(tmp_path, command):
    from tools.process_registry import ProcessRegistry, ProcessSession

    class FakePty:
        def __init__(self):
            self.writes = []

        def write(self, data):
            self.writes.append(data)

    registry = ProcessRegistry()
    session = ProcessSession(
        id="versioned-repl-" + command.replace("/", "_"),
        command=command,
        cwd=str(tmp_path),
        started_at=0,
    )
    session._pty = FakePty()
    registry._running[session.id] = session

    blocked = registry.submit_stdin(
        session.id,
        "open('~/.hermes/config.yaml', 'w').write('x')",
    )
    assert blocked["status"] == "blocked"
    assert session._pty.writes == []


def test_process_stdin_split_writes_are_buffered(tmp_path):
    """Protected writes split across write_stdin chunks must still block.

    Each chunk is individually benign ("echo x > ~/.hermes/" is an incomplete
    path; "config.yaml" alone is harmless), but combined they overwrite the
    protected config. The stateful guard buffer reconstructs the pending line
    and blocks the completing chunk before its newline reaches the shell.
    """
    from tools.process_registry import ProcessRegistry, ProcessSession

    class FakePty:
        def __init__(self):
            self.writes = []

        def write(self, data):
            self.writes.append(data)

    registry = ProcessRegistry()
    shell_session = ProcessSession(
        id="split-shell",
        command="/usr/bin/env bash",
        cwd=str(tmp_path),
        started_at=0,
    )
    shell_session._pty = FakePty()
    registry._running[shell_session.id] = shell_session

    # Benign incomplete prefix — allowed and buffered.
    ok = registry.write_stdin(shell_session.id, "echo x > ~/.hermes/")
    assert ok["status"] == "ok"
    # Completing chunk + newline reconstructs the protected write → blocked,
    # and the dangerous tail must never reach the shell.
    blocked = registry.submit_stdin(shell_session.id, "config.yaml")
    assert blocked["status"] == "blocked"
    assert all(b"config.yaml" not in w for w in shell_session._pty.writes)

    # REPL variant: split an inline Python write across two chunks.
    repl_session = ProcessSession(
        id="split-repl",
        command="python3",
        cwd=str(tmp_path),
        started_at=0,
    )
    repl_session._pty = FakePty()
    registry._running[repl_session.id] = repl_session

    ok = registry.write_stdin(repl_session.id, "open('~/.hermes/")
    assert ok["status"] == "ok"
    blocked = registry.submit_stdin(repl_session.id, "config.yaml','w').write('x')")
    assert blocked["status"] == "blocked"
    assert all(b"config.yaml" not in w for w in repl_session._pty.writes)


def test_process_stdin_buffer_resets_after_newline(tmp_path):
    """A completed (newline-terminated) benign line must not leave stale state
    that falsely blocks a later unrelated write."""
    from tools.process_registry import ProcessRegistry, ProcessSession

    class FakePty:
        def __init__(self):
            self.writes = []

        def write(self, data):
            self.writes.append(data)

    registry = ProcessRegistry()
    session = ProcessSession(
        id="reset-shell",
        command="/usr/bin/env bash",
        cwd=str(tmp_path),
        started_at=0,
    )
    session._pty = FakePty()
    registry._running[session.id] = session

    first = registry.submit_stdin(session.id, "echo hello > /tmp/safe.txt")
    assert first["status"] == "ok"
    second = registry.submit_stdin(session.id, "echo world > /tmp/other.txt")
    assert second["status"] == "ok"


def test_eval_and_piped_interpreter_payloads_are_unbypassable(clean_session):
    import tools.approval as approval_mod

    guard_cases = [
        "eval reboot",
        'eval "sudo -S whoami"',
        'eval "echo x > ~/.hermes/config.yaml"',
    ]
    for cmd in guard_cases:
        approval = approval_mod.check_unbypassable_command_guards(cmd, "local")
        assert not approval["approved"], cmd

    interpreter_cases = [
        "printf \"open('~/.hermes/config.yaml','w').write('x')\" | python3",
        "printf \"import os; os.system('echo x > ~/.hermes/config.yaml')\" | python3",
        "python3 <<'PY'\nimport os; os.system('echo x > ~/.hermes/config.yaml')\nPY",
        "printf \"require('fs').writeFileSync('~/.hermes/config.yaml','x')\" | node",
    ]
    for cmd in interpreter_cases:
        assert approval_mod._check_protected_config_write_guard(cmd)[0], cmd


def test_process_command_metadata_is_redacted():
    from tools.process_registry import ProcessRegistry, ProcessSession, format_process_notification

    secret = "sk-" + "A" * 24
    command = f"python worker.py --api-key {secret} token={secret}"
    registry = ProcessRegistry()
    session = ProcessSession(id="redact-proc", command=command, cwd="/tmp", started_at=0)
    registry._running[session.id] = session

    listed = registry.list_sessions()
    polled = registry.poll(session.id)
    notice = format_process_notification({
        "type": "completion",
        "session_id": session.id,
        "command": command,
        "exit_code": 0,
        "output": "ok",
    })

    assert secret not in listed[0]["command"]
    assert secret not in polled["command"]
    assert secret not in notice
    assert "[REDACTED]" in listed[0]["command"]
    assert "[REDACTED]" in polled["command"]
    assert "[REDACTED]" in notice


def test_static_stdin_parser_handles_unquoted_echo_and_printf_format(clean_session):
    import tools.approval as approval_mod

    assert not approval_mod.check_unbypassable_command_guards(
        "echo reboot | sh",
        "local",
    )["approved"]
    assert not approval_mod.check_unbypassable_command_guards(
        "printf '%s\\n' reboot | sh",
        "local",
    )["approved"]
    assert approval_mod._check_protected_config_write_guard(
        "printf '%s\\n' \"open('~/.hermes/config.yaml','w').write('x')\" | python3"
    )[0]


def test_process_command_redaction_handles_prefixed_env_secret_names():
    from tools.process_registry import ProcessRegistry, ProcessSession

    value = "DUMMY_NOT_A_REAL_SECRET_12345"
    command = f"OPENAI_API_KEY={value} AWS_SECRET_ACCESS_KEY={value} python worker.py"
    registry = ProcessRegistry()
    session = ProcessSession(id="redact-env-proc", command=command, cwd="/tmp", started_at=0)
    registry._running[session.id] = session

    polled = registry.poll(session.id)["command"]
    listed = registry.list_sessions()[0]["command"]

    assert value not in polled
    assert value not in listed
    assert polled.count("[REDACTED]") >= 2


def test_process_command_redaction_handles_bearer_and_perplexity_tokens():
    from tools.process_registry import ProcessRegistry, ProcessSession, format_process_notification

    bearer = "pplx-" + "A" * 24
    command = f"curl -H 'Authorization: Bearer {bearer}' https://api.perplexity.ai/search"
    registry = ProcessRegistry()
    session = ProcessSession(id="redact-bearer-proc", command=command, cwd="/tmp", started_at=0)
    registry._running[session.id] = session

    rendered = [
        registry.poll(session.id)["command"],
        registry.list_sessions()[0]["command"],
        format_process_notification({
            "type": "completion",
            "session_id": session.id,
            "command": command,
            "exit_code": 0,
            "output": "ok",
        }),
    ]

    assert all(bearer not in item for item in rendered)
    assert all("[REDACTED]" in item for item in rendered)


def test_repl_stdin_nested_shell_and_eval_writes_are_guarded(clean_session, tmp_path):
    from tools.process_registry import ProcessRegistry, ProcessSession

    class FakePty:
        def __init__(self):
            self.writes = []

        def write(self, data):
            self.writes.append(data)

    cases = [
        "import os; os.system('echo x > ~/.hermes/config.yaml')",
        "__import__('os').system('echo x > ~/.hermes/config.yaml')",
        "import subprocess; subprocess.run('echo x > ~/.hermes/config.yaml', shell=True)",
        "import subprocess; subprocess.run(['sh','-c','echo x > ~/.hermes/config.yaml'])",
        "import subprocess; subprocess.run(['/bin/sh','-c','echo x > ~/.hermes/config.yaml'])",
        "import subprocess; subprocess.run(['bash','-lc','echo x > ~/.hermes/config.yaml'])",
        "from subprocess import run; run('echo x > ~/.hermes/config.yaml', shell=True)",
        "import subprocess; subprocess.run(args='echo x > ~/.hermes/config.yaml', shell=True)",
        "eval(\"open('~/.hermes/config.yaml','w').write('x')\")",
    ]
    for idx, command in enumerate(cases):
        registry = ProcessRegistry()
        session = ProcessSession(
            id=f"repl-nested-{idx}",
            command="python3",
            cwd=str(tmp_path),
            started_at=0,
        )
        session._pty = FakePty()
        registry._running[session.id] = session

        blocked = registry.submit_stdin(session.id, command)

        assert blocked["status"] == "blocked", command
        assert session._pty.writes == []
