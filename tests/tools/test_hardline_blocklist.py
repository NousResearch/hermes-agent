"""Tests for the unconditional hardline command blocklist.

The hardline list is a floor below yolo: a small set of commands so
catastrophic they should never run via the agent, regardless of --yolo,
gateway /yolo, approvals.mode=off, or cron approve mode.

Inspired by Mercury Agent's permission-hardened blocklist.
"""

import pytest

from tools.approval import check_all_command_guards, check_dangerous_command, detect_dangerous_command, detect_hardline_command, disable_session_yolo, enable_session_yolo
from tools.approval_context import reset_current_session_key, set_current_session_key
from tools.approval_detection import HARDLINE_PATTERNS
from tools import approval_context


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
    # Command position via separator/wrapper still blocks (#93392)
    "true && mkfs.ext4 /dev/sda1",
    "sudo mkfs.ext4 /dev/sda1",
    "nohup mkfs /dev/sdb",
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
    # Quoted prose mentioning mkfs must not trip the hardline floor (#93392):
    # the word appears in an argument, not at a command position.
    'echo "does this workflow use mkfs anywhere?"',
    "echo 'run mkfs.ext4 on the backup disk later'",
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


# Multi-line QUOTED arguments are data, not command sequences: a newline
# inside quotes is part of the argument the shell passes to the program.
# These previously tripped the hardline floor because the flat command-start
# class treated every raw newline — even inside quotes — as a command
# boundary, blocking `hermes send` message bodies, multi-line
# `git commit -m` messages, and heredoc text that merely MENTION
# shutdown/reboot commands.
_QUOTED_NEWLINE_DATA_ALLOW = [
    # hermes send with a multi-line message body (the reported symptom)
    'hermes send -t telegram -s "spark1" "console output:\nsudo reboot\ndone"',
    'hermes send -t telegram "line1\nshutdown -h now\nline3"',
    # git commit -m with a multi-line message
    "git commit -m 'ops notes:\nreboot the box after the deploy'",
    'git commit -m "fix startup\nsystemctl reboot was flaky here"',
    # heredoc bodies quoting dangerous strings as data
    "python3 - <<'EOF'\nmsg = 'run sudo reboot later'\nprint(msg)\nEOF",
    "cat > /tmp/notes.txt <<'EOF'\nremember: shutdown -h now\nEOF",
    # rm hardline floor is anchored to the same class — quoted prose about it
    # across a line break must stay data too
    'git commit -m "docs:\nwarn about rm -rf / in the guide"',
]

# The masking must be strictly scoped to quoted data: real command
# boundaries around/inside those same shapes still hit the floor.
_QUOTED_NEWLINE_THREATS_BLOCK = [
    # unquoted newline is a real command separator
    "echo hi\nsudo reboot",
    'echo "a"\nsudo reboot',
    'git commit -m "safe message"\nshutdown -h now',
    # command substitution inside double quotes really executes
    'hermes send -t telegram "$(sudo reboot)"',
    'echo "`shutdown -h now`"',
    # multi-line quoted data followed by a REAL chained command
    'hermes send "line1\nline2" && sudo reboot',
    # a heredoc whose body is data, but the delivery command itself is hardline
    "sudo reboot <<'EOF'\nignored\nEOF",
]


@pytest.mark.parametrize("command", _QUOTED_NEWLINE_DATA_ALLOW)
def test_quoted_newline_data_not_blocked(command):
    """Newlines inside quoted arguments are data, not command starts."""
    is_hl, desc = detect_hardline_command(command)
    assert not is_hl, (
        f"multi-line quoted data false-positived the hardline floor: "
        f"{command!r} (got: {desc})"
    )


@pytest.mark.parametrize("command", _QUOTED_NEWLINE_THREATS_BLOCK)
def test_real_newline_separated_threats_still_blocked(command):
    """Unquoted newlines / $() / backticks remain real command boundaries."""
    is_hl, desc = detect_hardline_command(command)
    assert is_hl, f"real threat leaked through hardline floor: {command!r}"
    assert desc


def test_quoted_newline_data_not_blocked_by_full_guard_chain(clean_session):
    """End-to-end: the guard chain must not hardline-block a multi-line
    quoted message (yolo on, so only the unconditional floor can block)."""
    enable_session_yolo("hardline_test")
    command = 'hermes send -t telegram "status:\nsudo reboot happened at 3am"'
    result = check_all_command_guards(command, "local")
    assert result["approved"], (
        f"guard chain blocked multi-line quoted data: {result.get('message')}"
    )


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
# #93392 class regression: unanchored hardline patterns vs quoted prose
# -------------------------------------------------------------------------
# Every hardline rule that used a bare \b anchor (mkfs, dd, kill -1) fired on
# the token ANYWHERE in the command — including inside quoted prose handed to
# echo / git commit -m / gh --body — and the positionless rules (redirect to
# a block device, fork bomb) fired on quoted mentions too. Quoted prose must
# pass; every true-positive shape (bare, separators, sudo/env prefix, $(),
# backticks, sh -c/bash -c/eval payloads) must stay unconditionally blocked.

_QUOTED_PROSE_ALLOW_93392 = [
    # mkfs (the reported symptom)
    'echo "does this workflow use mkfs anywhere?"',
    'git commit -m "add mkfs.ext4 warning to the runbook"',
    'gh pr create --body "this PR anchors the mkfs pattern"',
    "grep 'mkfs' docs/runbook.md",
    # dd to block device
    'git commit -m "never run dd of=/dev/sda in prod"',
    'echo "dd if=/dev/zero of=/dev/sda wipes the disk"',
    "grep 'dd if=/dev/zero of=/dev/sda' notes.md",
    # kill -1
    'echo "kill -1 sends SIGHUP to every process"',
    'gh issue comment 7 --body "the agent must never run kill -1"',
    # redirect to block device (positionless rule -> quote-masked)
    'echo "cat file > /dev/sda destroys the disk"',
    "echo 'redirect > /dev/sdb1 is fatal'",
    'git commit -m "block > /dev/sda redirects"',
    'gh pr create --body "guards the > /dev/nvme0n1 redirect"',
    # fork bomb (positionless rule -> quote-masked)
    'git commit -m "document the fork bomb :(){ :|:& };: pattern"',
    'echo "classic fork bomb: :(){ :|:& };:"',
    "echo ':(){ :|:& };: is a fork bomb'",
]


@pytest.mark.parametrize("command", _QUOTED_PROSE_ALLOW_93392)
def test_quoted_prose_mentions_are_not_hardline(command):
    """Quoted prose mentioning a hardline trigger is data, not a command."""
    is_hl, desc = detect_hardline_command(command)
    assert not is_hl, (
        f"quoted prose false-positived the hardline floor: {command!r} "
        f"(got: {desc})"
    )


_TRUE_POSITIVES_93392 = [
    # mkfs at every command position
    "mkfs.ext4 /dev/sda1",
    "mkfs /dev/sdb",
    "sudo mkfs.xfs /dev/nvme0n1",
    "true && mkfs.ext4 /dev/sda1",
    "ls; mkfs /dev/sdb",
    "env FOO=1 mkfs.ext4 /dev/sda1",
    "$(mkfs.ext4 /dev/sda1)",
    "`mkfs /dev/sdb`",
    'bash -c "mkfs.ext4 /dev/sda1"',
    # dd to raw block device
    "dd if=/dev/zero of=/dev/sda bs=1M",
    "sudo dd if=/dev/urandom of=/dev/nvme0n1",
    "echo start && dd if=/dev/zero of=/dev/sdb",
    "ls; dd if=x of=/dev/mmcblk0",
    "env X=1 dd if=/dev/zero of=/dev/sda",
    "$(dd if=/dev/zero of=/dev/sda)",
    "`dd if=/dev/zero of=/dev/sda`",
    'sh -c "dd if=/dev/zero of=/dev/sda"',
    # redirect to raw block device (unquoted / carrier / substitution)
    "cat file > /dev/sda",
    "echo junk > /dev/sdb",
    "true && cat f > /dev/nvme0n1",
    'sh -c "cat f > /dev/sda"',
    'bash -c "echo x > /dev/sdb"',
    'eval "cat f > /dev/sda"',
    'echo "$(cat f > /dev/sda)"',
    'echo "`cat f > /dev/sdb`"',
    # kill -1
    "kill -1",
    "kill -9 -1",
    "sudo kill -1",
    "ls; kill -1",
    "true && kill -HUP -1",
    "$(kill -1)",
    'bash -c "kill -1"',
    # fork bomb
    ":(){ :|:& };:",
    "true && :(){ :|:& };:",
    'sh -c ":(){ :|:& };:"',
    "eval ':(){ :|:& };:'",
]


@pytest.mark.parametrize("command", _TRUE_POSITIVES_93392)
def test_true_positive_shapes_stay_hardline_blocked(command):
    """Every real destructive shape stays on the unconditional floor."""
    is_hl, desc = detect_hardline_command(command)
    assert is_hl, f"true positive leaked past the hardline floor: {command!r}"
    assert desc


# DANGEROUS-tier duplicates of the mkfs/dd rules must be anchored the same
# way: quoted prose must not even require approval, while real invocations
# stay flagged (yolo can still bypass this tier — that's what yolo is for).
_DANGEROUS_TIER_PROSE_ALLOW = [
    'echo "mkfs is a formatting tool"',
    'git commit -m "explain dd if=/dev/zero usage"',
]

_DANGEROUS_TIER_STILL_FLAGGED = [
    ("mkfs /dev/sdb1", "format filesystem"),
    ("sudo mkfs -t vfat /dev/sdc1", "format filesystem"),
    ("dd if=backup.img of=restore.img", "disk copy"),
    ("true && dd if=a.img of=b.img", "disk copy"),
]


@pytest.mark.parametrize("command", _DANGEROUS_TIER_PROSE_ALLOW)
def test_dangerous_tier_prose_not_flagged(command):
    is_dangerous, _, desc = detect_dangerous_command(command)
    assert not is_dangerous, (
        f"quoted prose tripped the dangerous tier: {command!r} (got: {desc})"
    )


@pytest.mark.parametrize("command,expected", _DANGEROUS_TIER_STILL_FLAGGED)
def test_dangerous_tier_real_commands_still_flagged(command, expected):
    is_dangerous, _, desc = detect_dangerous_command(command)
    assert is_dangerous, f"real command no longer dangerous-flagged: {command!r}"
    assert desc == expected


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
    from tools import approval_context
    from tools import approval_context
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "off")

    result = check_all_command_guards("rm -rf /", "local")
    assert result["approved"] is False
    assert result.get("hardline") is True


def test_cron_approve_mode_cannot_bypass_hardline(clean_session, monkeypatch):
    """Cron sessions with cron_mode=approve must not bypass hardline."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    import tools.approval as approval_mod
    monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: "approve")

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


def test_sudo_stdin_guard_container_bypass(clean_session):
    """Containerized backends still bypass — they can't touch the host."""
    for env in ("docker", "singularity", "modal", "daytona", "vercel_sandbox"):
        for cmd in _SUDO_STDIN_BLOCK:
            result = check_all_command_guards(cmd, env)
            assert result["approved"] is True, f"container {env} should bypass sudo guard on {cmd!r}"
