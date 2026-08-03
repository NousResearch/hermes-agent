"""Tests for the unconditional hardline command blocklist.

The hardline list is a floor below yolo: a small set of commands so
catastrophic they should never run via the agent, regardless of --yolo,
gateway /yolo, approvals.mode=off, or cron approve mode.

Inspired by Mercury Agent's permission-hardened blocklist.
"""

import pytest

from tools.approval import (
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
    # Shell-equivalent spellings of "rm -rf /": repeated slashes and
    # current/parent-dir segments all collapse back to root, so they must
    # hit the hardline floor too (regression: these used to slip through the
    # root pattern's target group and fall to the softer DANGEROUS_PATTERNS
    # rule, which --yolo / approvals.mode=off / cron approve-mode bypass).
    "rm -rf //",
    "rm -rf /.",
    "rm -rf /./",
    "rm -rf /..",
    "rm -rf //*",
    "rm -fr /./",
    "ls && rm -rf //",
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
    # Quoted path idioms — the recommended shell form for paths with special
    # chars. These previously slipped past the floor because the surrounding
    # quote broke both the flag group and the (\s|$) terminator (regression
    # guard: catastrophic disk/home wipe under --yolo / approvals.mode=off).
    'rm -rf "/"',
    "rm -rf '/'",
    'rm -rf "/*"',
    'rm -rf "/etc"',
    "rm -rf '/etc'",
    'rm -rf "/home"',
    'rm -rf "/usr"',
    'rm -rf "$HOME"',
    "rm -rf '$HOME'",
    'rm -rf "$HOME/"',
    'rm -rf "~"',
    'sudo rm -rf "/"',
    'rm -rf "/" && echo done',
    # ${HOME} brace form (universally common, previously unmatched).
    "rm -rf ${HOME}",
    'rm -rf "${HOME}"',
    "rm -fr ${HOME}",
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
    # Bare subshell `(cmd)` and brace-group `{ cmd; }` openers put the trigger
    # at a real command position, so they must hit the floor just like `$(…)`.
    # These slipped through before the quote-aware command-start tokenizer
    # learned to recognize `(` / `{` (issue: (reboot) walked past --yolo).
    "(reboot)",
    "( reboot )",
    "(shutdown -h now)",
    "(poweroff)",
    "(halt)",
    "(init 0)",
    "(systemctl reboot)",
    "(sudo reboot)",
    "{ reboot; }",
    "{ shutdown -h now; }",
    "{ poweroff; }",
    "true && (reboot)",
    "echo hi; { reboot; }",
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
    # Literal root-level directories that only LOOK like root-collapse
    # spellings. Each inter-slash segment must be exactly "." or ".." to
    # count as a collapse back to "/" — "/..." is a dir literally named
    # "..." and "/.foo" is an ordinary root dotfile. These must NOT be
    # swept into the "recursive delete of root filesystem" hardline rule
    # (regression guard for the collapse-spelling tightening).
    "rm -rf /...",
    "rm -rf /....",
    "rm -rf /.foo",
    "rm -rf /.config/foo",
    # A dangerous-looking command embedded as a quoted *argument* to another
    # command must not trip the floor: the path is immediately followed by a
    # closing quote with no matching opening quote of its own, so the
    # quote-tolerant matcher must still ignore it (no new false positives).
    'git commit -m "rm -rf /"',
    'git commit -m "wipe with rm -rf /etc"',
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


# Commands written with the ordinary quoting / brace shell idioms that
# previously slipped past the floor. Kept as an explicit regression set so
# the intent (quoting `rm -rf "/"` must not be a disk-wipe bypass) survives
# any future refactor of the rm patterns.
_QUOTED_BRACE_BYPASS = [
    'rm -rf "/"',
    "rm -rf '/'",
    'rm -rf "/etc"',
    'rm -rf "/home"',
    'rm -rf "$HOME"',
    "rm -rf ${HOME}",
    'rm -rf "${HOME}"',
]


@pytest.mark.parametrize("command", _QUOTED_BRACE_BYPASS)
def test_quoted_and_brace_paths_are_hardline_blocked(command):
    """Quoted paths and ${HOME} must hit the floor (was a silent bypass)."""
    is_hl, desc = detect_hardline_command(command)
    assert is_hl, f"quoting/brace bypass leaked through hardline floor: {command!r}"
    assert desc


# Commands that carry the literal string "rm -rf /" (or a sibling) as DATA in
# another command's quoted argument — a PR title, a commit message, an echo /
# printf argument. The shell never executes that text as an rm command, so the
# hardline floor must NOT fire; otherwise the command cannot run at all (this
# blocked `gh pr create --title "…rm -rf /…"` outright). Regression guard for
# the command-position anchor on the rm rules.
_DATA_ARG_NOT_A_COMMAND = [
    'gh pr create --title "block rm -rf / spellings"',
    'git commit -m "fixes rm -rf / bypass"',
    'echo "run rm -rf / now"',
    'echo "rm -rf /"',
    'printf "%s" "rm -rf /"',
    'gh issue comment 1 --body "the fix blocks rm -rf //"',
    # A `(` or `{` INSIDE a quoted argument is prose, not a subshell/brace
    # opener — the trigger word after it is data. Naively adding `(` / `{` to
    # the flat command-position class blocked these (it broke our own
    # `gh pr create --title "…(reboot)…"` workflow); the quote-aware tokenizer
    # must leave them alone.
    'gh pr create --title "block (reboot) spellings"',
    'git commit -m "(rm -rf /) note"',
    'echo "(reboot)"',
    'echo "{ reboot; }"',
    "echo '(poweroff)'",
    "echo '{ rm -rf /; }'",
    'find . -name "*(reboot)*"',
]


@pytest.mark.parametrize("command", _DATA_ARG_NOT_A_COMMAND)
def test_root_wipe_string_as_data_arg_is_not_hardline(command):
    """"rm -rf /" as a quoted argument to another command is data, not a wipe."""
    is_hl, desc = detect_hardline_command(command)
    assert not is_hl, f"false positive: quoted data arg hit hardline floor: {command!r} ({desc})"


# Real root wipes at every command position — bare, chained after a separator,
# inside a command substitution ($()/backtick), or after sudo/env wrappers.
# The command-position anchor must keep catching all of these; the substitution
# forms exercise the shell-metacharacter terminator on the bare path branch.
_COMMAND_POSITION_ROOT_WIPES = [
    "rm -rf /",
    "ls && rm -rf /",
    "ls; rm -rf /",
    "echo x | rm -rf /",
    "sudo rm -rf /",
    "env X=1 rm -rf /",
    "$(rm -rf /)",
    "`rm -rf /`",
    'echo "$(rm -rf /)"',
    # Bare subshell / brace-group openers are real command positions too.
    "(rm -rf /)",
    "{ rm -rf /; }",
    "(rm -rf ~)",
    "(sudo rm -rf /)",
]


@pytest.mark.parametrize("command", _COMMAND_POSITION_ROOT_WIPES)
def test_root_wipe_at_command_position_is_hardline(command):
    """A real `rm -rf /` at any command position stays hardline-blocked."""
    is_hl, desc = detect_hardline_command(command)
    assert is_hl, f"real root wipe leaked past the floor: {command!r}"
    assert desc


# -------------------------------------------------------------------------
# Shell line-continuation bypass
# -------------------------------------------------------------------------
#
# A backslash immediately followed by a newline is a POSIX line
# continuation: the shell removes BOTH characters and joins the tokens, so
# `rm -rf \<newline>/` executes as `rm -rf /`. The normalizer used to strip
# only backslash-escapes of NON-newline characters (`\\([^\n])`), leaving the
# dangling backslash wedged between tokens — which broke the structured
# rm/dd/mkfs patterns and let a root wipe slip past the hardline floor.

# (command_with_continuation, description_substring) — each is the
# line-continuation form of a command already in _HARDLINE_BLOCK.
_HARDLINE_LINE_CONTINUATION = [
    ("rm -rf \\\n/", "root"),            # split before the path
    ("rm -r\\\nf /", "root"),            # split inside the flag bundle
    ("rm -rf \\\n~", "home"),            # home-directory wipe
    ("rm -rf \\\r\n/", "root"),          # CRLF line ending
    ("mkfs.ext4 \\\n/dev/sda1", "mkfs"),  # filesystem format
]


@pytest.mark.parametrize("command,desc_substr", _HARDLINE_LINE_CONTINUATION)
def test_hardline_blocks_line_continuation(command, desc_substr):
    is_hl, desc = detect_hardline_command(command)
    assert is_hl, f"line-continuation bypassed hardline detection: {command!r}"
    assert desc and desc_substr in desc.lower(), (
        f"unexpected description {desc!r} for {command!r}"
    )


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

    for cmd in ['rm -rf /', 'rm -rf "/"', 'rm -rf "$HOME"', "rm -rf ${HOME}",
                "shutdown -h now", "mkfs.ext4 /dev/sda", "reboot"]:
        r1 = check_dangerous_command(cmd, "local")
        assert r1["approved"] is False, f"yolo leaked hardline on {cmd!r} (check_dangerous_command)"
        assert r1.get("hardline") is True

        r2 = check_all_command_guards(cmd, "local")
        assert r2["approved"] is False, f"yolo leaked hardline on {cmd!r} (check_all_command_guards)"
        assert r2.get("hardline") is True


def test_root_collapse_forms_cannot_bypass_hardline(clean_session, monkeypatch):
    """Shell-equivalent spellings of "rm -rf /" stay blocked under yolo.

    "//", "/.", "/./", "/..", "//*" all collapse to the root filesystem in
    the shell. They previously matched only the softer DANGEROUS_PATTERNS
    rule, which yolo bypasses — leaving the hardline floor open to a full
    root wipe under --yolo / approvals.mode=off / cron approve-mode.
    """
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    for cmd in ["rm -rf //", "rm -rf /.", "rm -rf /./", "rm -rf /..", "rm -rf //*"]:
        is_hl, _ = detect_hardline_command(cmd)
        assert is_hl, f"{cmd!r} should be hardline-blocked"
        result = check_all_command_guards(cmd, "local")
        assert result["approved"] is False, f"yolo leaked hardline on {cmd!r}"
        assert result.get("hardline") is True


def test_root_collapse_pattern_leaves_real_paths_alone(clean_session):
    """The broadened root token must not over-match real trailing segments.

    A path with a real component after the root-collapse prefix (/tmp,
    /home/user/x, /.ssh, ./build) is recoverable-or-legitimate and must NOT
    be pulled onto the hardline floor by the "collapse to /" broadening.
    """
    for cmd in ["rm -rf /tmp", "rm -rf /home/user/x", "rm -rf /.ssh",
                "rm -rf /.config", "rm -rf ./build", "rm -rf /opt/foo",
                "rm -rf /...", "rm -rf /....", "rm -rf /.foo"]:
        is_hl, _ = detect_hardline_command(cmd)
        assert not is_hl, f"{cmd!r} must not be hardline-blocked (over-match)"


def test_subshell_brace_group_cannot_bypass_hardline(clean_session, monkeypatch):
    """Wrapping a catastrophic command in `(…)` or `{ …; }` must not bypass
    the floor, even under yolo. `(reboot)` / `{ shutdown -h now; }` walked
    straight past the guard before the command-start tokenizer recognized the
    subshell and brace-group openers.
    """
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    for cmd in ["(reboot)", "( reboot )", "(shutdown -h now)", "(poweroff)",
                "(systemctl reboot)", "(init 0)", "(sudo reboot)",
                "{ reboot; }", "{ shutdown -h now; }", "{ poweroff; }",
                "(rm -rf /)", "{ rm -rf /; }", "(rm -rf ~)",
                "true && (reboot)", "echo hi; { reboot; }"]:
        r1 = check_dangerous_command(cmd, "local")
        assert r1["approved"] is False, f"yolo leaked hardline on {cmd!r} (check_dangerous_command)"
        assert r1.get("hardline") is True

        r2 = check_all_command_guards(cmd, "local")
        assert r2["approved"] is False, f"yolo leaked hardline on {cmd!r} (check_all_command_guards)"
        assert r2.get("hardline") is True


def test_quoted_paren_brace_prose_not_blocked_under_yolo(clean_session, monkeypatch):
    """A `(` / `{` inside a quoted argument is prose, not a command opener.

    Regression guard: naively adding `(` / `{` to the flat command-position
    class blocked ordinary quoted arguments — including our own
    `gh pr create --title "…(reboot)…"` workflow. The quote-aware tokenizer
    must leave quoted text untouched, so these stay runnable.
    """
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    for cmd in ['gh pr create --title "block (reboot) spellings"',
                'git commit -m "(rm -rf /) note"',
                'echo "(reboot)"', 'echo "{ reboot; }"',
                "echo '(poweroff)'", 'find . -name "*(reboot)*"']:
        assert detect_hardline_command(cmd)[0] is False, (
            f"quoted prose false-positived on the hardline floor: {cmd!r}"
        )


def test_line_continuation_root_wipe_cannot_bypass_hardline(clean_session, monkeypatch):
    """A line-continuation root wipe must stay blocked even under yolo.

    `rm -rf \\<newline>/` runs as `rm -rf /`. Yolo bypasses the regular
    dangerous-command layer, so the hardline floor is the only thing left to
    catch it — it must hold.
    """
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    result = check_all_command_guards("rm -rf \\\n/", "local")
    assert result["approved"] is False, "yolo leaked a line-continuation root wipe"
    assert result.get("hardline") is True
    assert "BLOCKED (hardline)" in result["message"]


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
    for env in ("docker", "singularity", "modal", "daytona"):
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
    for env in ("docker", "singularity", "modal", "daytona"):
        for cmd in _SUDO_STDIN_BLOCK:
            result = check_all_command_guards(cmd, env)
            assert result["approved"] is True, f"container {env} should bypass sudo guard on {cmd!r}"


# Absolute-path invocations must not defeat the floor. _CMDPOS only accepted
# "start | separator | subshell opener | sudo/env-style wrappers" before the
# command word, so spelling the binary by path — the natural form on systems
# where PATH is unreliable, and the default an LLM produces for Windows
# tools — returned (False, None) for every hardline pattern. Regression set
# pins both directions: path-spelled commands at command position are
# blocked, the same strings as data arguments stay allowed.
_ABS_PATH_BLOCK = [
    "/sbin/shutdown -h now",
    "/usr/sbin/reboot",
    "/sbin/halt",
    "/sbin/init 6",
    "/bin/rm -rf /",
    "sudo /sbin/shutdown -h now",
    "env LC_ALL=C /sbin/reboot",
    "echo done; /sbin/shutdown -h now",
    "true && /usr/sbin/poweroff",
    r"C:\Windows\System32\shutdown.exe /s /t 0",
    "C:/Windows/System32/shutdown.exe /s /t 0",
    r'"C:\Program Files\Git\usr\bin\rm.exe" -rf /',
    r"C:\Windows\System32\shutdown.EXE /s",
    # path-spelled wrapper chains resolve in the single projection pass
    "/usr/bin/sudo /sbin/shutdown -h now",
    "/usr/bin/env /usr/bin/sudo /sbin/shutdown -h now",
    "exec /sbin/reboot",
    "( /sbin/shutdown -h now )",
    "/sbin/telinit 6",
    "./shutdown -h now",
    "shutdown.exe /s",
    # composed spellings reduce through the same collapsing as r\m detection
    "'/sbin/'shutdown -h now",
    "/sbin/shut\\down -h now",
    # a payload's executable can be path-spelled too
    "bash -c '/sbin/shutdown -h now'",
    "exec bash -c '/bin/rm -rf /'",
    # group/substitution closers stay outside the projected word
    "(/sbin/reboot)",
    "$(/sbin/reboot)",
    "`/sbin/reboot`",
    "{ /sbin/reboot; }",
    "env -u FOO /sbin/reboot",
    # `command` takes options too, and they sit before the executable
    "command -p /sbin/reboot",
    "command -- /sbin/reboot",
    "command -p -- /sbin/reboot",
    # `--` ends the option list of the wrappers whose options we model, so
    # the path after it is the program
    "exec -- /sbin/reboot",
    "nohup -- /sbin/reboot",
    "setsid -- /sbin/reboot",
    "time -- /sbin/reboot",
    "builtin -- command -p /sbin/reboot",
    # `exec -a NAME` consumes NAME; -c and -l take no operand
    "exec -a custom /sbin/reboot",
    "exec -c /sbin/reboot",
    "exec -l /sbin/reboot",
    # assignment prefixes are unbounded in shell grammar, so a fixed walk
    # budget must not run out and let the executable through unprojected
    " ".join(f"A{i}=1" for i in range(12)) + " /sbin/reboot",
    " ".join(f"A{i}=1" for i in range(40)) + " /sbin/reboot",
]

_ABS_PATH_ALLOW = [
    # `command -v` / `-V` only look the command up; nothing is executed, and
    # `command`'s short options are just p/v/V so a cluster carrying v is
    # lookup-only as well
    "command -v /sbin/reboot",
    "command -V /sbin/reboot",
    "command -pv /sbin/reboot",
    # a word outside a wrapper's option list is the program name, and the
    # shell fails to run it — blocking these would be a false positive
    # (time/setsid are the exception: their unknown options fail safe, see
    # the time/setsid section below)
    "exec -x /sbin/reboot",
    "exec -- -c /sbin/reboot",
    "nohup -x /sbin/reboot",
    "nohup -- -x /sbin/reboot",
    "builtin -x /sbin/reboot",
    "builtin -- -x /sbin/reboot",
    "echo /sbin/shutdown",
    "ls -la /sbin/shutdown",
    "grep 'shutdown' /var/log/syslog",
    "stat /usr/sbin/reboot",
    r"stat C:\Windows\System32\shutdown.exe",
    "cat /bin/rm",
    'echo "/sbin/init 6"',
    "md5sum /sbin/halt",
    # basename must match the pattern word exactly, not a near-miss
    "/usr/local/bin/rebooter --dry-run",
    "./deploy.sh shutdown",
    "cat /etc/init.d/reboot",
    # a bare assignment prefix is data — projecting its value manufactured
    # a False->True flip in the first cut of this fix (Sol round 2)
    "X=/sbin/shutdown echo ok",
    # a wrapper option's operand is data, not the command word
    "env --chdir /tmp/reboot /bin/echo ok",
    "env --chdir /tmp/reboot echo ok",
    "env --argv0 /tmp/reboot /bin/echo ok",
    "env -C /sbin/reboot /bin/echo ok",
    # sudo option operands are data; only the command after them executes.
    "sudo -D /sbin/reboot /bin/echo ok",
    "sudo -D/sbin/reboot /bin/echo ok",
    "sudo --chdir /sbin/reboot /bin/echo ok",
    "sudo --chroot=/sbin/reboot /bin/echo ok",
    # Exact `-h` only consumes a following non-option, non-assignment host.
    # With an option-looking/assignment next word it selects help mode, so no
    # later executable runs even when an option operand resembles one.
    "sudo -h -D /sbin/reboot /bin/echo ok",
    "sudo -h --chdir /sbin/reboot /bin/echo ok",
    "sudo -h -D/sbin/reboot /bin/echo ok",
    "sudo -h VAR=x /sbin/reboot",
    "sudo -h 1VAR=x /sbin/reboot",
    "sudo -h -- /sbin/reboot",
    "sudo -nh host /sbin/reboot",
    # sudo only treats an equals-bearing word as environment data when its
    # first byte is neither '/' nor '='; these are command words instead.
    "sudo /tmp=x /sbin/reboot",
    "sudo =x /sbin/reboot",
    "sudo -- 1VAR=x /sbin/reboot",
    # Shell assignment syntax does not restart after an ordinary wrapper.
    "exec VAR=x /sbin/reboot",
    "nohup VAR=x /sbin/reboot",
    # GNU env stops option parsing at the first NAME=VALUE operand. Later
    # option-shaped words are a command (or another assignment), not options.
    "env 1VAR=x -S /sbin/reboot",
    "env 1VAR=x --split-string /sbin/reboot",
    "env 1VAR=x --split-string=/sbin/reboot",
    "/usr/bin/env 1VAR=x -S /sbin/reboot",
    # `}` is not a shell metacharacter: it can end a legitimate word
    "/tmp/reboot}",
]


@pytest.mark.parametrize("command", _ABS_PATH_BLOCK)
def test_abs_path_invocation_is_hardline_blocked(command):
    is_hl, desc = detect_hardline_command(command)
    assert is_hl, f"absolute-path spelling bypassed the floor: {command!r}"
    assert desc, "hardline match must provide a description"


@pytest.mark.parametrize("command", _ABS_PATH_ALLOW)
def test_abs_path_as_data_is_not_hardline(command):
    is_hl, desc = detect_hardline_command(command)
    assert not is_hl, f"path-as-data false positive: {command!r} (got: {desc})"


def test_abs_path_hardline_not_bypassed_by_yolo(clean_session, monkeypatch):
    """The floor must hold for path-spelled commands under yolo too."""
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")
    result = check_all_command_guards("/sbin/shutdown -h now", "local")
    assert result["approved"] is False
    assert result.get("hardline") is True


_EXECUTABLE_WRAPPER_OPERAND_BLOCK = [
    # GNU env -S parses its operand into the command it executes. Until that
    # separate grammar is modeled, every split-string spelling must fail safe.
    "env -S /sbin/reboot",
    "env -S/sbin/reboot",
    "env -vS'/sbin/reboot -h now'",
    "env --split-string /sbin/reboot",
    "env --split-string=/sbin/reboot",
    "env --sp=/sbin/reboot",
    # sudo options below consume data before the command word. Short clusters,
    # attached operands, and unique long-option abbreviations use sudo's real
    # getopt grammar and must still expose the executable that follows.
    "sudo -D /tmp /sbin/reboot",
    "sudo -nD /tmp /sbin/reboot",
    "sudo -ED /tmp /sbin/reboot",
    "sudo -D/tmp /sbin/reboot",
    "sudo --chdir /tmp /sbin/reboot",
    "sudo --chd /tmp /sbin/reboot",
    "sudo -R /tmp /sbin/reboot",
    "sudo -r staff_r /sbin/reboot",
    "sudo -T 5 /sbin/reboot",
    "sudo -t staff_t /sbin/reboot",
    # sudo's historical separated host form applies only to an exact `-h`.
    # Attached hosts and a following ordinary host still leave a real command.
    "sudo -h host /sbin/reboot",
    "sudo -hhost /sbin/reboot",
    "sudo -h-D /sbin/reboot",
    "sudo -hD /sbin/reboot",
    "sudo -h /tmp=x /sbin/reboot",
    "sudo -h =x /sbin/reboot",
    # Wrapper-owned assignments are not limited to shell identifiers. sudo's
    # `is_envar` and GNU env both consume digit/dash-leading NAME=VALUE words
    # before resuming option/command parsing. Cover bare and path spellings so
    # both command-position walkers share the same state transition.
    "sudo 1VAR=x -D /tmp /sbin/reboot",
    "sudo name-with-dash=x -D /tmp /sbin/reboot",
    "/usr/bin/sudo 1VAR=x -D /tmp /sbin/reboot",
    "env 1VAR=x /sbin/reboot",
    "env name-with-dash=x /sbin/reboot",
    "env -- 1VAR=x /sbin/reboot",
    "/usr/bin/env 1VAR=x /sbin/reboot",
    # Uppercase -A is a flag, not lowercase -a with an operand.
    "sudo -A /sbin/reboot",
    # Ambiguous and unknown long options cannot place the command word safely.
    "sudo --ch /tmp /sbin/reboot",
    "sudo --not-a-sudo-option /sbin/reboot",
]


@pytest.mark.parametrize("command", _EXECUTABLE_WRAPPER_OPERAND_BLOCK)
def test_executable_wrapper_operand_is_hardline_blocked(command):
    is_hl, desc = detect_hardline_command(command)

    assert is_hl, f"wrapper operand bypassed the floor: {command!r}"
    assert desc, "hardline match must provide a description"


@pytest.mark.parametrize("command", _EXECUTABLE_WRAPPER_OPERAND_BLOCK)
def test_executable_wrapper_operand_not_bypassed_by_yolo(
    clean_session, command
):
    enable_session_yolo("hardline_test")

    result = check_all_command_guards(command, "local")

    assert result["approved"] is False, command
    assert result.get("hardline") is True, command


@pytest.mark.parametrize(
    "command,approved,hardline",
    [
        ("timeout 5 /bin/rm -rf /", False, True),
        ("timeout -k 3 5s rm -rf /", False, True),
        ("timeout -k3 5s rm -rf /", False, True),
        ("timeout -sTERM 5s rm -rf /", False, True),
        ("timeout -vk3 5s rm -rf /", False, True),
        ("timeout -vsTERM 5s rm -rf /", False, True),
        ("timeout --preserve-status 5 /bin/rm -rf /", False, True),
        ("timeout -p 5 /bin/rm -rf /", False, True),
        ("timeout -f 5 rm -rf /", False, True),
        ("timeout -pf 5 rm -rf /", False, True),
        ("timeout -vpf 5 rm -rf /", False, True),
        ("nice /bin/rm -rf /", False, True),
        ("nice -n 10 rm -rf /", False, True),
        ("nice -n10 rm -rf /", False, True),
        ("nice -10 rm -rf /", False, True),
        ("stdbuf -oL /bin/rm -rf /", False, True),
        ("stdbuf -o L rm -rf /", False, True),
        ("timeout 5 nice rm -rf /", False, True),
        ("echo timeout 5 rm -rf /", True, False),
        ("timeout 5 ls", True, False),
        ("nice -n 10 make build", True, False),
        ("timeout -vx 5s rm -rf /", True, False),
        ("timeout 5 -vsTERM rm -rf /", True, False),
        ("nice -vn10 rm -rf /", True, False),
        ("nice -10x rm -rf /", True, False),
    ],
)
def test_pass_through_launcher_wrappers_resolve_command_word_under_yolo(
    clean_session, monkeypatch, command, approved, hardline
):
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    result = check_all_command_guards(command, "local")

    assert result["approved"] is approved, command
    assert bool(result.get("hardline")) is hardline, command


# -------------------------------------------------------------------------
# time / setsid option grammar
# -------------------------------------------------------------------------
#
# `time` and `setsid` accept options of their own ahead of the program word
# (GNU time: -p/-a/-v/-q/-V, -f/-o with an operand; setsid: -c/-f/-w/-V/-h).
# Without a grammar for them the walker treats the option itself as the
# command word and never inspects the executable behind it.

_TIME_SETSID_BLOCK = [
    "time -p /sbin/reboot",
    "setsid -f /sbin/reboot",
    "setsid -w -c /sbin/reboot",
    "setsid -fw /sbin/reboot",
    "setsid --fork /sbin/reboot",
    "time --quiet /sbin/reboot",
    # an option's operand is skipped as data, then the executable resolves
    "time -o /tmp/x /sbin/reboot",
    "time -f '%e' /sbin/reboot",
    "time --format=%e /sbin/reboot",
    "time -p -- /sbin/reboot",
]


@pytest.mark.parametrize("command", _TIME_SETSID_BLOCK)
def test_time_setsid_options_resolve_command_word(command):
    is_hl, desc = detect_hardline_command(command)
    assert is_hl, f"time/setsid option hid the executable: {command!r}"
    assert desc, "hardline match must provide a description"


_TIME_SETSID_ALLOW = [
    # an option's operand is data, not the command word
    "time -o /sbin/reboot /bin/echo hi",
    "time -f /sbin/reboot /bin/echo hi",
    "time --output=/sbin/reboot /bin/echo hi",
    # ordinary launches keep working
    "time -p /bin/echo hi",
    "setsid -f /bin/echo hi",
    "setsid -w sleep 1",
    # a lone wrapper word has no command to resolve
    "time",
    "setsid",
]


@pytest.mark.parametrize("command", _TIME_SETSID_ALLOW)
def test_time_setsid_option_operands_are_data(command):
    is_hl, desc = detect_hardline_command(command)
    assert not is_hl, f"time/setsid false positive: {command!r} (got: {desc})"


# An option word we do NOT model (`time -Z`) means the walker cannot tell
# where the program word starts. That must fail safe (detected), never fall
# back to "the option is the program" — that misread is exactly how
# `time -p /sbin/reboot` walked past the floor.
_TIME_SETSID_UNKNOWN_OPTION = [
    "time -Z /sbin/reboot",
    "time -x /sbin/reboot",
    "time --bogus /sbin/reboot",
    "setsid -x /sbin/reboot",
    "setsid --bogus /sbin/reboot",
]


@pytest.mark.parametrize("command", _TIME_SETSID_UNKNOWN_OPTION)
def test_unknown_time_setsid_option_fails_safe(command):
    is_hl, desc = detect_hardline_command(command)
    assert is_hl, f"unknown time/setsid option fell open: {command!r}"
    assert desc


def test_time_setsid_hardline_under_yolo(clean_session, monkeypatch):
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")
    for cmd in ("time -p /sbin/reboot", "setsid -f /sbin/reboot"):
        result = check_all_command_guards(cmd, "local")
        assert result["approved"] is False, f"yolo leaked {cmd!r}"
        assert result.get("hardline") is True, cmd


# -------------------------------------------------------------------------
# Walker termination: no pass-through exit
# -------------------------------------------------------------------------
#
# The walk over the command prefix must end in exactly three ways: the
# command word resolves, the input ends, or the defensive word cap trips and
# the command is treated as unresolvable (detected). A fixed wrapper budget
# used to be a fourth exit that silently gave up and let the executable
# through uninspected.

def test_wrapper_chain_depth_is_not_a_bypass():
    import tools.approval as approval_mod

    capped = {approval_mod._PARSER_LIMIT_DESCRIPTION,
              approval_mod._MALFORMED_EXEC_DESCRIPTION}
    for depth in (11, 12, 13):
        command = "env " * depth + "/sbin/reboot"
        is_hl, desc = detect_hardline_command(command)
        assert is_hl, f"{depth} wrappers let the executable through"
        assert desc not in capped, (
            f"{depth} wrappers should resolve fully, not hit the cap: {desc!r}"
        )


def test_wrapper_chain_depth_bypass_closed_under_yolo(clean_session, monkeypatch):
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")
    command = "env " * 12 + "/sbin/reboot"
    result = check_all_command_guards(command, "local")
    assert result["approved"] is False, "12 wrappers leaked under yolo"
    assert result.get("hardline") is True


def test_walker_word_cap_fails_safe_never_open():
    import tools.approval as approval_mod

    capped = {approval_mod._PARSER_LIMIT_DESCRIPTION,
              approval_mod._MALFORMED_EXEC_DESCRIPTION}
    max_words = approval_mod._WALKER_MAX_WORDS
    for total_words, resolves in ((max_words - 1, True),
                                  (max_words, False),
                                  (max_words + 1, False)):
        command = "env " * (total_words - 1) + "/sbin/reboot"
        is_hl, desc = detect_hardline_command(command)
        assert is_hl, f"{total_words}-word walk fell open"
        if resolves:
            assert desc not in capped, (
                f"{total_words} words is under the cap and must resolve: {desc!r}"
            )
        else:
            assert desc in capped, (
                f"{total_words} words must trip the defensive cap: {desc!r}"
            )
