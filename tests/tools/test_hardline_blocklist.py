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
    "${UNSET:-reboot}",
    '"${UNSET:-reboot}"',
    "${SET/x/reboot}",
    "$(printf reboot)",
    "`printf reboot`",
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
    "echo '${UNSET:-reboot}'",
    "'${UNSET:-reboot}'",
    "echo '$(printf reboot)'",
    "'$(printf reboot)'",
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
# Command-carrying wrappers
# -------------------------------------------------------------------------
#
# A wrapper that runs a command string it is handed (`sh -c <string>`, GNU
# `env -S` / `--split-string`) or the `su`/`runuser` `-c` form puts the real
# verb inside an argument, not at a shell command position, so the anchored
# patterns cannot see it. The detector re-scans the carried string as its own
# command. Only literal strings are reachable: a value computed at runtime
# (`sh -c "$(...)"`, a pipe into a shell, a variable) is arbitrary execution no
# static scan can resolve, and stays out of scope (see issue for the ceiling).
_LONG_CARRIER_PREFIX = "env " + " ".join(f"A{i}=x" for i in range(21))

_CARRIED_HARDLINE_BYPASS = [
    "sh -c 'reboot'",
    "bash -c 'reboot'",
    "dash -c 'reboot'",
    "zsh -c 'reboot'",
    "ksh -c 'reboot'",
    "sh -c 'systemctl poweroff'",
    "bash -c 'shutdown -h now'",
    "sh -c 'rm -rf /'",
    "bash -c 'rm -rf ~'",
    "su -c 'reboot'",
    "runuser -c 'reboot'",
    "su root -c 'reboot'",
    "su --command 'reboot' root",
    "su --command='reboot' root",
    "su --session-command 'reboot' root",
    "su --session-command='reboot' root",
    "runuser --command 'reboot' root",
    "runuser --command='reboot' root",
    "runuser --session-command 'reboot' root",
    "runuser --session-command='reboot' root",
    "su --command=re'boot' root",
    "runuser --session-command re'boot' root",
    "su --command 'echo safe' --command reboot root",
    "runuser --session-command 'echo safe' --session-command reboot root",
    "su -c 'echo safe' --command reboot root",
    "runuser --command 'echo safe' -c reboot root",
    "su -c reboot -w --command root",
    "runuser -c reboot --whitelist-environment --command root",
    "su -c reboot -g --command root",
    "runuser -c reboot --shell --command root",
    "su -c reboot root # --command 'echo safe'",
    "runuser -c reboot root # --session-command 'echo safe'",
    "su -c reboot root\nprintf safe",
    "runuser -c reboot root\nrunuser --command 'echo safe' root",
    "su --comm reboot root",
    "runuser --sess reboot root",
    'env --split-string="reboot"',
    "env --split-string='rm -rf /'",
    "env --split-string reboot",
    "env --s='sh -c reboot'",
    "env --split 'sh -c reboot'",
    "env --split-strin='sh -c reboot'",
    "env --s''='sh -c reboot'",
    "env --sp''lit='sh -c reboot'",
    "env --s$''='sh -c reboot'",
    "env --$'split'='sh -c reboot'",
    r"env --$'\x73'='sh -c reboot'",
    r"env --s$'\0'='sh -c reboot'",
    r"env --s$'\c@'='sh -c reboot'",
    r"su --command re$'\0'boot root",
    r"env --s=sh\ -c\ reboot",
    r'''env -S "sh\_-c\_reboot"''',
    r'''env --s="sh\_-c\_reboot"''',
    "env --u sh -- sh -c reboot",
    "env --c /tmp -- sh -c reboot",
    "env --a sh -- sh -c reboot",
    "env --u FOO --s reboot",
    "env --c /tmp --s reboot",
    "env --a custom --s reboot",
    "env --b --s reboot",
    "env -Sreboot",
    "/bin/sh -c 'reboot'",
    "sudo sh -c 'reboot'",
    "env FOO=1 sh -c 'reboot'",
    "env FOO=1 BAR=2 bash -c 'rm -rf /'",
    "env A-B=1 sh -c reboot",
    "env -S 'A-B=1 sh -c reboot'",
    "sh -c 'sh -c reboot'",           # carrier nested in carrier
    "ls; sh -c 'reboot'",             # after a separator
    "sh -ec 'reboot'",                # clustered short options, -c is last
    "bash -ec 'reboot'",
    "dash -ec 'reboot'",
    "sh -xc 'rm -rf /'",
    "bash -exc 'rm -rf /'",
    "sh -lc 'reboot'",
    "sh -cx 'reboot'",                # letters after c are still shell options
    "sh -ce 'reboot'",
    "bash -cx 'reboot'",
    "bash -ce 'reboot'",
    # Shells keep parsing invocation options after seeing c; the first word
    # after the complete option prefix is still the command string.
    "sh -c -x 'reboot'",
    "bash -c -e 'rm -rf /'",
    "sh -ce -- 'reboot'",
    "bash -c -o nounset 'reboot'",
    "su -lc 'reboot'",
    # GNU env permits S after no-argument options in a short-option bundle and
    # appends operands after the split string to the resulting argv.
    "env -iS 'reboot'",
    "env -viS 'reboot'",
    "env -iS 'rm -rf /'",
    "env -iSreboot",
    "env -viS/bin/sh -c reboot",
    "env -vS/bin/sh -c 'rm -rf /'",
    r"env -S '/bin/sh\_-c\_reboot'",
    "env -S '-i sh -c reboot'",
    "env -iC /tmp sh -c reboot",
    "env -a custom -S 'sh -c reboot'",
    "env --argv0 custom -S 'sh -c reboot'",
    # A carrier reached behind wrapper OPTIONS, not just wrapper words and
    # NAME=VALUE assignments. The whole option prefix must be skipped.
    "env -i sh -c 'reboot'",
    "env --ignore-environment sh -c 'reboot'",
    "env -u FOO sh -c 'reboot'",
    "env --unset=FOO sh -c 'reboot'",
    "sudo -u root sh -c 'reboot'",
    "sudo --user root sh -c 'reboot'",
    "sudo -E sh -c 'reboot'",
    "sudo -n sh -c 'reboot'",
    "sudo -u root -E sh -c 'reboot'",
    "sudo -u root sh -c 'rm -rf /'",
    # Wrappers that carry no command of their own but still hide a carrier.
    "nice sh -c 'reboot'",
    "nice -n 10 sh -c 'reboot'",
    "ionice sh -c 'reboot'",
    "stdbuf -oL sh -c 'reboot'",
    "timeout 5 sh -c 'reboot'",
    "timeout -s KILL 5 sh -c 'reboot'",
    "timeout 1.5s sh -c 'reboot'",
    "timeout 5 bash -ec 'reboot'",
    "doas sh -c 'reboot'",
    "doas -u root sh -c 'reboot'",
    # Wrappers nested and path-prefixed with options in between.
    "sudo -u root env -i sh -c 'reboot'",
    "env -i sudo sh -c 'reboot'",
    "/usr/bin/env -i /bin/sh -c 'reboot'",
    "/bin/sudo -u root sh -c 'reboot'",
    # Options that really do take an operand still reach the carrier after it.
    "ionice -c 2 sh -c 'reboot'",
    "ionice -c best-effort sh -c 'reboot'",
    "stdbuf -o L sh -c 'reboot'",
    "timeout -s KILL 5 sh -c 'reboot'",
]


@pytest.mark.parametrize("command", _CARRIED_HARDLINE_BYPASS)
def test_command_carrying_wrappers_are_hardline_blocked(command):
    """A hardline verb inside a carried command string still hits the floor."""
    is_hl, desc = detect_hardline_command(command)
    assert is_hl, f"command-carrying wrapper leaked past the floor: {command!r}"
    assert desc


def test_carrier_scan_reaches_payload_beyond_former_token_boundary():
    """Legal wrapper prefixes cannot push a carried payload out of the scan."""
    for payload in ("reboot", "'rm -rf /'"):
        command = f"{_LONG_CARRIER_PREFIX} sh -c {payload}"
        is_hl, desc = detect_hardline_command(command)
        assert is_hl, f"long carrier prefix leaked past the floor: {command!r}"
        assert desc

    benign = f"{_LONG_CARRIER_PREFIX} sh -c 'echo reboot'"
    assert detect_hardline_command(benign) == (False, None)
    is_dangerous, pattern, description = detect_dangerous_command(benign)
    assert is_dangerous, f"long carrier prefix hid shell execution: {benign!r}"
    assert pattern
    assert description

    too_long_prefix = "env " + " ".join(f"A{i}=x" for i in range(61))
    truncated = f"{too_long_prefix} sh -c 'echo reboot'"
    is_hl, hl_description = detect_hardline_command(truncated)
    assert is_hl
    assert hl_description == "command parser limit exceeded"
    assert detect_dangerous_command(truncated) == (
        True,
        "command parser limit exceeded",
        "command parser limit exceeded",
    )

    ordinary = "echo " + " ".join(f"arg{i}" for i in range(100))
    assert detect_hardline_command(ordinary) == (False, None)
    assert detect_dangerous_command(ordinary) == (False, None, None)


# The verb sits at an ARGUMENT position inside the carried string (an echo/grep
# argument, a filename, a non-destructive subcommand), so re-scanning the string
# must keep the command-position anchor and leave these runnable. Same guard the
# top-level anchor gives, carried one level down.
_CARRIED_NOT_A_COMMAND = [
    "sh -c 'echo reboot'",
    "bash -c 'git commit -m reboot'",
    "sh -c 'systemctl status nginx'",
    "sh -c 'grep -r reboot /etc'",
    "bash -c 'ls -la'",
    "env --split-string='echo reboot'",
    "env --s='/bin/echo reboot'",
    "env --split '/bin/echo reboot'",
    "env --n --s reboot",
    "env --nu --s reboot",
    "env --nul --s reboot",
    "env FOO=1 --s reboot",
    "env FOO=1 --debug sh -c reboot",
    '''env --s='$(printf "reboot")' ''',
    '''env --s=A=1 '$(printf sh)' -c reboot''',
    r"env --unset=$'\U00110000' printf OK",
    "su --command 'echo reboot' root",
    "su --session-command='printf reboot' root",
    "runuser --command 'echo reboot' root",
    "runuser --session-command='printf reboot' root",
    "su --command reboot --command 'echo safe' root",
    "runuser --session-command reboot -c 'echo safe' root",
    "su -- root --command reboot",
    "runuser -- root --session-command reboot",
    "su --command 'echo safe' --command",
    "runuser --session-command 'echo safe' -c",
    "su -wFOO,creboot root",
    "runuser --whitelist-environment=FOO,creboot root",
    "su -s/bin/sh root",
    "runuser -u root --command reboot",
    "su -m -w FOO --command reboot root",
    "su --preserve-environment --whitelist-environment FOO --command reboot root",
    "runuser -p -w FOO --session-command reboot root",
    # Non-carrier programs whose own `-c` means something else entirely.
    "gcc -c main.c",
    "grep -c reboot access.log",
    "env EDITOR=vim git commit",
    "find . -name reboot.service",
    "sh -ec 'echo reboot'",           # clustered options, benign payload
    "sh -cx 'echo reboot'",           # options after c do not become its payload
    "sh -- -c reboot",                # option parsing already ended
    "sh /dev/null -c reboot",         # script path already selected
    "bash -lc 'ls -la'",
    "env -iS 'echo reboot'",
    "env -Svi reboot",                # vi is the split-string program
    "env -uS /bin/echo reboot",       # S is the operand of -u, not an option
    "env -S/bin/echo reboot",
    r"env -S '/bin/echo\_reboot'",
    "env -- /bin/echo -iS reboot",
    "env /bin/echo -viS reboot",
    "env -0S reboot",                 # --null is incompatible with a command
    # Wrapped commands with options, but no carrier or a benign payload, must
    # not be swept up by the option-skipping prefix walk.
    "sudo -u root ls",
    "sudo -u root sh",                # interactive shell, no -c payload
    "timeout 5 curl https://example.com",
    "nice -n 10 make -j4",
    "env -i printenv",
    "env -u FOO make",
    "stdbuf -oL grep reboot app.log",
    "sudo -E git push",
    "doas -u root ls",
    "timeout 5 sh -c 'echo reboot'",
    "sudo -u root sh -c 'echo reboot'",
    # Interpreter carriers run another language rather than a shell command
    # string, so they are the runtime-computed / arbitrary-code class and are
    # deliberately left to the softer guards, not the shell-carrier extractor.
    "python3 -c 'import os; os.system(\"reboot\")'",
    "perl -e 'system(\"reboot\")'",
    # A no-operand wrapper flag must not consume the wrapper's real program, so
    # its arguments (which merely look like a carrier) are not rescanned. Here
    # `echo` is the program and `sh -c reboot` are the words it prints.
    "sudo -E echo sh -c reboot",
    "sudo -n echo sh -c reboot",
    "timeout --foreground echo sh -c reboot",
    "timeout --preserve-status echo sh -c reboot",
    "sudo -H echo sh -c reboot",
    "env -v echo sh -c reboot",
    "stdbuf -oL echo sh -c reboot",
    # `nice -n` takes a numeric operand, so a non-numeric next token is the
    # program, not the niceness value.
    "nice -n echo sh -c reboot",
    # `env` must only parse its own option/assignment prefix for `-S` /
    # `--split-string`. After `--` or after the program word, later tokens are
    # data for that program, not an env-carried command string.
    "env -- echo --split-string reboot",
    "env echo --split-string reboot",
    "env -i echo --split-string reboot",
    "env FOO=1 echo --split-string reboot",
]


@pytest.mark.parametrize("command", _CARRIED_NOT_A_COMMAND)
def test_carried_arg_position_verb_is_not_hardline(command):
    """A verb used as data inside a carried string is not a command."""
    is_hl, desc = detect_hardline_command(command)
    assert not is_hl, f"false positive: carried arg-position verb blocked: {command!r} ({desc})"


def test_command_carrying_wrapper_blocked_under_yolo(clean_session, monkeypatch):
    """The carried root wipe cannot be waived by yolo, the floor runs first."""
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")
    for cmd in ("sh -c 'rm -rf /'", "env --split-string='rm -rf /'"):
        result = check_all_command_guards(cmd, "local")
        assert result["approved"] is False, f"yolo leaked carried hardline on {cmd!r}"


def test_command_carrying_wrapper_blocked_in_default_mode(clean_session):
    """The carried reboot (no dangerous backstop) is blocked in default mode."""
    for cmd in ("dash -c 'reboot'", "env --split-string='reboot'", "su -c 'reboot'"):
        result = check_all_command_guards(cmd, "local")
        assert result["approved"] is False, f"carried reboot approved with no prompt: {cmd!r}"


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
