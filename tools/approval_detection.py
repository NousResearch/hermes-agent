"""Dangerous-command detection: normalization, tokenizing, and pattern tables.

Pure command classification for :mod:`tools.approval` — no approval state, config reads, or
prompting live here.
"""

import functools
import logging
import os
import posixpath
import re
import shlex
import tempfile
import unicodedata

logger = logging.getLogger("tools.approval")

# Sensitive write targets, matched via ~ / $HOME / $HERMES_HOME spellings. The resolved absolute
# home is folded into these forms at detection time by _normalize_command_for_detection(), so no
# import-time path snapshot (stale once HERMES_HOME is set after import) lives in the patterns.
_SSH_SENSITIVE_PATH = r'(?:~|\$home|\$\{home\})/\.ssh(?:/|$)'
_HERMES_ENV_PATH = (
    r'(?:~\/\.hermes/|(?:\$home|\$\{home\})/\.hermes/|(?:\$hermes_home|\$\{hermes_home\})/)' r'\.env\b'
)
# ~/.hermes/config.yaml IS the security policy (approvals.mode, yolo, allowlist) and the config cache is mtime-keyed,
# so a write takes effect mid-session. Terminal-side coverage (sed -i, tee, >, cp) pairs the file_tools deny.
_HERMES_CONFIG_PATH = (
    r'(?:~\/\.hermes/|(?:\$home|\$\{home\})/\.hermes/|(?:\$hermes_home|\$\{hermes_home\})/)' r'config\.yaml\b'
)
_PROJECT_ENV_PATH = r'(?:(?:/|\.{1,2}/)?(?:[^\s/"\'`]+/)*\.env(?:\.[^/\s"\'`]+)*)'
_PROJECT_CONFIG_PATH = r'(?:(?:/|\.{1,2}/)?(?:[^\s/"\'`]+/)*config\.yaml)'
_SHELL_RC_FILES = r'(?:~|\$home|\$\{home\})/\.' r'(?:bashrc|zshrc|profile|bash_profile|zprofile)\b'
_CREDENTIAL_FILES = r'(?:~|\$home|\$\{home\})/\.' r'(?:netrc|pgpass|npmrc|pypirc)\b'
# macOS: /etc, /var, /tmp, /home are symlinks to /private/*, so /private/etc/sudoers would bypass a plain
# "/etc/" check. Match both forms.
_MACOS_PRIVATE_SYSTEM_PATH = r'/private/(?:etc|var|tmp|home)/'
_SYSTEM_CONFIG_PATH = rf'(?:/etc/|{_MACOS_PRIVATE_SYSTEM_PATH})'
_SENSITIVE_WRITE_TARGET = (
    rf'(?:{_SYSTEM_CONFIG_PATH}|/dev/sd|{_SSH_SENSITIVE_PATH}|{_HERMES_ENV_PATH}|{_HERMES_CONFIG_PATH}|'
    rf'{_SHELL_RC_FILES}|{_CREDENTIAL_FILES})'
)
_USER_SENSITIVE_WRITE_TARGET = rf'(?:{_SSH_SENSITIVE_PATH}|{_SHELL_RC_FILES}|{_CREDENTIAL_FILES})'
_PROJECT_SENSITIVE_WRITE_TARGET = rf'(?:{_PROJECT_ENV_PATH}|{_PROJECT_CONFIG_PATH})'
# cp/mv/install: the sensitive path is a write target only as the LAST argument (destination), so
# `cp config.yaml backup.yaml` (config.yaml as SOURCE) stays out.
_COMMAND_TAIL = r'(?:\s*(?:&&|\|\||;).*)?$'
# `>`/`>>`/tee: the path is ALWAYS a write target regardless of what follows, so only require a
# shell word boundary (_COMMAND_TAIL let `echo x > .env extra` / `echo x > .env # note` slip past).
# `#` is deliberately NOT a boundary: a glued `#` is part of the filename (`.env#backup`).
_WRITE_TARGET_BOUNDARY = r'(?=[\s;&|<>"\']|$)'

# ---- Hardline (unconditional) blocklist ---------------------------------------------------
# Commands that NEVER run via the agent, regardless of --yolo, approvals.mode=off, or cron approve
# mode — a floor below yolo. Applies only to environments that can damage the host (local, ssh,
# container-host cron); containerized backends already bypass the dangerous-command layer.
# Deliberately tiny: only things with no recovery path (root wipe, raw block device writes,
# shutdown, DoS). Recoverable operations (git reset --hard, chmod -R 777, curl|sh) stay in
# DANGEROUS_PATTERNS.

# Start-of-command position: start of string, newline, subshell opener ($( or backtick), optionally consuming
# sudo/env/exec/nohup/setsid/time wrappers. Keeps shutdown/reboot rules from firing on "echo reboot" / "grep
# 'shutdown' log". Real ;/&/| separators are converted to newlines by the quote-aware _mark_command_starts pass;
# keeping them here mistakes quoted data (grep '(safe|rm -rf /)') for commands.
_CMDPOS = (
    r'(?:^|[\n`]|\$\()' r'\s*'  # start position, optional whitespace
    r'(?:sudo\s+(?:-[^\s]+\s+)*)?' r'(?:env\s+(?:\w+=\S*\s+)*)?'  # optional sudo with flags, env VAR=VAL pairs
    r'(?:(?:exec|nohup|setsid|time)\s+)*' r'\s*'  # optional wrapper commands
)


# Destructive-path matcher for the rm hardline rules: accept the path fully wrapped in a matching
# quote pair (`rm -rf "/"`, `rm -rf "$HOME"`) OR bare with a terminator (whitespace, end, or
# `) ` ; | &` so `$(rm -rf /)` is caught). A bare-token-only anchor lets quoted spellings slip past the floor.
def _hardline_rm_path(path_alt: str, tail: str = r'(?:\s|$|[)`;|&])') -> str:
    return rf'(?:["\'](?:{path_alt})["\']|(?:{path_alt}){tail})'


# Protected system roots whose recursive deletion has no recovery path.
_HARDLINE_SYSTEM_DIRS = (r'/home|/home/\*|/root|/root/\*|/etc|/etc/\*|/usr|/usr/\*|'
                         r'/var|/var/\*|/bin|/bin/\*|/sbin|/sbin/\*|/boot|/boot/\*|/lib|/lib/\*')

# `rm` plus flag group, shared by the rm hardline rules (plain concatenation, not f-string:
# backslashes in replacement fields are unsupported on the 3.11 floor). _CMDPOS-anchored so `rm`
# must be an actual command word — "rm -rf /" as DATA in `git commit -m "…rm -rf /…"` must not trip the floor.
_RM_FLAG_PREFIX = _CMDPOS + r'rm\s+(-[^\s]*\s+)*'
# Package-manager global options, each optionally taking ONE non-dash operand.
_PKG_OPTS = r'(?:-[^\s]+(?:\s+[^-\s][^\s]*)?\s+)*'

HARDLINE_PATTERNS = [
    # Root path: any root-anchored path whose components collapse to "/" in the shell ("/", "//",
    # "/.", "/./", "/../..", optional trailing glob). Each inter-slash segment must be exactly "."
    # or "..", so "/tmp", "/.ssh", even "/..." are literal dirs that fall through to the softer
    # DANGEROUS rules. The "/ \*" alt covers `rm -rf / *` (two args: "/" plus the glob).
    (_RM_FLAG_PREFIX + _hardline_rm_path(r'/(?:(?:\.\.?)?/)*(?:\.\.?)?\**|/ \*'), "recursive delete of root filesystem"),
    (_RM_FLAG_PREFIX + _hardline_rm_path(_HARDLINE_SYSTEM_DIRS), "recursive delete of system directory"),
    (_RM_FLAG_PREFIX + _hardline_rm_path(r'(?:~|\$\{?HOME\}?)(?:/?|/\*)?'), "recursive delete of home directory"),
    # Command-name rules (mkfs, dd, kill, shutdown...) are _CMDPOS-anchored so quoted prose
    # (`echo "does this use mkfs?"`) cannot trip the floor.
    # See #93392.
    (_CMDPOS + r'mkfs(\.[a-z0-9]+)?\b', "format filesystem (mkfs)"),
    # `dd` is a command-name token, so anchor it to command position like mkfs/rm/shutdown (#93392): quoted
    # prose such as `git commit -m "never dd of=/dev/sda"` is an argument, not a command. The argument tail
    # ([^\n]*of=/dev/...) is kept so flag order doesn't matter.
    (_CMDPOS + r'dd\b[^\n]*\bof=/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*', "dd to raw block device"),
    # Positionless rules (no command-name token: `>` sits mid-command, the fork bomb is a function
    # definition) are matched against a QUOTE-MASKED variant (_QUOTE_MASKED_HARDLINE_DESCRIPTIONS /
    # _mask_quoted_prose) so quoted prose cannot trip them; sh -c / bash -c / eval payloads still scan raw.
    # The redirect rule has no command-name token to anchor (`>` appears mid-command: `cat f > /dev/sda`),
    # so command-position anchoring is the wrong tool. It is instead matched against a QUOTE-MASKED variant
    # of the command (see _QUOTE_MASKED_HARDLINE / _mask_quoted_strings) so quoted prose (`echo "cat f >
    # /dev/sda"`) cannot trip it, while shell-carrying wrappers (sh -c / bash -c / eval) still surface their
    # payload as a raw detection variant — quoting is not a bypass (#93392).
    (r'>\s*/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*\b', "redirect to raw block device"),
    (r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:', "fork bomb"),
    # Kill every process on the system — anchor the command-name token so `echo "kill -1 sends SIGHUP to
    # everything"` doesn't trip (#93392).
    (_CMDPOS + r'kill\s+(-[^\s]+\s+)*-1\b', "kill all processes"),
    (_CMDPOS + r'(shutdown|reboot|halt|poweroff)\b', "system shutdown/reboot"),
    (_CMDPOS + r'init\s+[06]\b', "init 0/6 (shutdown/reboot)"),
    (_CMDPOS + r'systemctl\s+(poweroff|reboot|halt|kexec)\b', "systemctl poweroff/reboot"),
    (_CMDPOS + r'telinit\s+[06]\b', "telinit 0/6 (shutdown/reboot)"),
    # In-place edits of the Hermes security policy / credential files are
    # hardline too, but they are NOT expressible as a regex over the raw
    # command text: deciding them correctly needs the real command word and
    # the editor's option grammar. See _detect_hermes_inplace_edit() below,
    # which detect_hardline_command() consults alongside these patterns.
]

# Pre-compiled at module load so the hot-path matcher never pays the cold re.compile fan-out
# (re._cache can be evicted by unrelated regex work).
_RE_FLAGS = re.IGNORECASE | re.DOTALL
# Positionless hardline rules matched against quote-masked variants (see above).
_QUOTE_MASKED_HARDLINE_DESCRIPTIONS = frozenset({"redirect to raw block device", "fork bomb"})
HARDLINE_PATTERNS_COMPILED = [
    (re.compile(p, _RE_FLAGS), d, d in _QUOTE_MASKED_HARDLINE_DESCRIPTIONS) for p, d in HARDLINE_PATTERNS
]
# Commands that hand a quoted argument to another shell to EXECUTE: quoted text is code, not
# prose, so quote-masked hardline rules scan the raw string.
_SHELL_CARRIER_NAMES = frozenset({"eval", "sh", "bash", "zsh", "ksh", "dash", "source", "."})


def _contains_shell_carrier(command: str) -> bool:
    """Return whether any command-position word is a shell-carrying command."""
    return any(
        os.path.basename(_deobfuscate_shell_word_for_detection(word)).lower() in _SHELL_CARRIER_NAMES
        for _, _, word in _iter_shell_command_word_spans(command)
    )


def _mask_quoted_prose(command: str) -> str:
    """Blank out quoted string CONTENT for positionless hardline matching (detection-only).
    Quote characters stay; inside double quotes `$(...)` and backtick spans are kept RAW because
    the shell really executes them. An unclosed quote masks to end-of-string, which cannot hide a
    runnable command (the shell would not run it either).

    Detection-only rewrite used by the quote-masked hardline rules (redirect-to-block-device, fork bomb):
    text inside single or double quotes is data the shell passes as an argument, so `echo "cat f >
    /dev/sda"` must not trip the unconditional floor (#93392). Unquoted text is untouched.
    """
    return "".join(
        command[i:j] if quote is None or kind in ("quote", "subst") else " " * (j - i)
        for kind, i, j, quote in _scan_shell(command, subst="q", naive_backtick=True)
    )


# ---- Sudo stdin guard: without SUDO_PASSWORD configured, an explicit "sudo -S" is the LLM piping
# a guessed password via stdin (brute-force vector). Unconditional block.
_SUDO_STDIN_RE = re.compile(r'(?:^|[;&|`\n]|&&|\|\||\$\()\s*sudo\s+-S\b', re.IGNORECASE)


def _check_sudo_stdin_guard(command: str) -> tuple:
    """Detect ``sudo -S`` without configured SUDO_PASSWORD -> (is_blocked, description). When
    SUDO_PASSWORD is set, ``_transform_sudo_command`` injects ``-S`` itself, so this guard only
    fires when the LLM wrote it explicitly."""
    if "SUDO_PASSWORD" not in os.environ and _SUDO_STDIN_RE.search(_normalize_command_for_detection(command).lower()):
        return (True, "sudo password guessing via stdin (sudo -S)")
    return (False, None)


# =========================================================================
# In-place edits of the Hermes security policy / credential files
# =========================================================================
#
# ~/.hermes/config.yaml holds approvals.mode, yolo, and the permanent
# allowlist; ~/.hermes/.env holds credentials.  Both are write-protected on
# the file_tools side, and the smart-approval adjudicator has approved a
# `sed -i` on config.yaml even while describing it correctly — so without a
# terminal-side floor the agent can rewrite its own guardrails.
#
# This is deliberately NOT a regex over the command text.  A text search
# gets both directions wrong, as the review of #60523 demonstrated:
#
#   * bypasses — `sed -e 's/a/b/' -i F` (the `i` is not in the first option
#     token), `command sed -i F`, `env -i sed -i F`, `/usr/bin/sed -i F`.
#     A pattern that hardcodes one spelling of the editor and one position
#     for `-i` misses every equivalent spelling.
#   * false positives — `sed -i 's|~/.hermes/config.yaml|config.yml|'
#     README.md` (the protected path is inside the sed PROGRAM, the target
#     is README.md) and `sed -i 's/a/b/' ~/.hermes/config.yaml.bak` (a
#     backup copy, not the policy file).  Searching for the path anywhere
#     after the options cannot tell an operand from program text.
#
# So resolve the command word for real, walk the editor's option grammar to
# decide whether in-place mode is actually on, and compare the resulting
# FILE OPERANDS against the protected paths as whole tokens.  Same
# reasoning as _WRITE_TARGET_BOUNDARY above, which already keeps
# `config.yaml.bak` out of the redirection deny.
#
# This runs its OWN wrapper resolution (_resolve_command_word below) rather
# than reusing _iter_shell_command_word_spans' _COMMAND_WRAPPER_WORDS table:
# that table is a best-effort heuristic for the softer execution-flag
# detection (unrecognized options are skipped and guessed past), which is
# the right tradeoff there but wrong for an unconditional floor — an
# unrecognized wrapper option must never silently misresolve which word is
# the editor. _resolve_command_word instead FAILS CLOSED (raises
# _WrapperResolutionFailed) on any option shape it does not recognize; see
# _unresolvable_segment_is_inplace_threat for how that exception is turned
# into a block rather than a silent pass-through.
#
# Out of scope on purpose (unchanged from the DANGEROUS/smart level, and
# matching how the xargs/find rm rules are drawn): indirect path delivery
# via xargs, find -exec, or shell variable expansion.  Those never name the
# target at a command position we can resolve statically.
#
# One precise instance of the same class: the `cd`-relative operand.
#   cd ~/.hermes && sed -i 's/a/b/' config.yaml
# After `cd` the relative operand `config.yaml` names the very same policy
# file, but the guard sees only the bare relative token — it has no `.hermes/`
# prefix and no CWD to resolve against, so whole-token matching cannot bind it
# to the protected file. Closing that would require tracking the `cd` target
# across the `&&` boundary (working-directory state), which is exactly the
# indirect-delivery class this guard deliberately does not model.

# Wrapper commands that prefix a real command without being one.
_EDITOR_WRAPPERS = frozenset({
    "sudo", "doas", "env", "exec", "nohup", "setsid", "time",
    "command", "builtin", "stdbuf", "nice", "ionice",
    # Command-position wrappers whose first operand is a mandatory POSITIONAL
    # (not an option), so the command word sits after a fixed number of
    # positionals. Covered by _WRAPPER_POSITIONAL below, not _WRAPPER_OPTS_WITH_ARG.
    "timeout", "flock", "taskset", "chrt",
})
# Wrapper options that consume the FOLLOWING token as their argument, so we
# do not mistake that argument for the command word (`env -u PATH sed ...`).
# Every entry verified against the tool's man page (2026-08-25: sudo 1.9.18,
# doas OpenBSD, util-linux 2.43, coreutils 9.11, stdbuf(1)). The `=`-form
# (`--user=root`) needs no separate-arg entry: the resolver already skips the
# single token, which carries its own value. Attached short forms (`-n5`,
# `-uPATH`) are skipped as a single token too. Only the separate-argument
# spellings must be listed here.
#
# NOTE: for wrappers in _WRAPPER_POSITIONAL, an option listed here must be
# recognized as argument-taking or its argument is consumed as the wrapper's
# POSITIONAL and the resolver returns the wrong command word (the P2-1(a)
# bypass class). The fail-closed rule in _resolve_command_word() refuses to
# guess when an unrecognized option precedes a positional, but the tables
# must still be complete.
_WRAPPER_OPTS_WITH_ARG = {
    "sudo": frozenset({"-u", "-g", "-h", "-p", "-C", "-U", "-D", "-R", "-T",
                       "--user", "--group", "--host", "--prompt",
                       "--close-from", "--chdir", "--chroot", "--other-user",
                       "--command-timeout"}),
    "doas": frozenset({"-u", "-C", "-a"}),
    "env": frozenset({"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "time": frozenset({"-f", "--format", "-o", "--output"}),
    "ionice": frozenset({"-c", "-n", "-p", "-P", "-u",
                         "--class", "--classdata", "--pid", "--pgid", "--uid"}),
    "stdbuf": frozenset({"-i", "-o", "-e", "--input", "--output", "--error"}),
    # timeout(1): -s/--signal, -k/--kill-after take separate arguments.
    # -f/-p/-v are flags.
    "timeout": frozenset({"-s", "--signal", "-k", "--kill-after"}),
    # flock(1): -w/--wait/--timeout, -E/--conflict-exit-code, -c/--command
    # take separate arguments. -n/--nb/--nonblocking are aliases; -s/-e/-x/-F/-o/-u
    # are flags. --start/--length take arguments too but only apply with --fcntl.
    "flock": frozenset({"-w", "--wait", "--timeout", "-E", "--conflict-exit-code",
                        "-c", "--command", "--start", "--length"}),
    # taskset(1): -c/--cpu-list, -p/--pid take separate arguments. -a is a flag.
    "taskset": frozenset({"-c", "--cpu-list", "-p", "--pid"}),
    # chrt(1): -p/--pid, -T/--sched-runtime, -P/--sched-period,
    # -D/--sched-deadline take separate arguments. Policy flags
    # (-o/-f/-r/-b/-i/-d/-e) take none.
    "chrt": frozenset({"-p", "--pid", "-T", "--sched-runtime",
                       "-P", "--sched-period", "-D", "--sched-deadline"}),
}
# Wrapper options that take NO argument (pure flags). Listed here so the
# option loop in _resolve_command_word() recognizes them as options rather
# than tripping the fail-closed rule that refuses to guess about
# unrecognized option shapes. Verified against the same man pages.
_WRAPPER_NOARG_FLAGS = {
    "sudo": frozenset({"-A", "-B", "-b", "-E", "-e", "-H", "-i", "-K", "-k",
                       "-l", "-N", "-n", "-P", "-S", "-s", "-V", "-v"}),
    "doas": frozenset({"-L", "-n", "-s"}),
    "env": frozenset({"-i", "-0", "--null", "--ignore-environment"}),
    "nice": frozenset(),
    # time(1): -a (append), -p (portability) are pure flags.
    "time": frozenset({"-a", "--append", "-p", "--portability"}),
    "ionice": frozenset({"-t", "--ignore"}),
    "stdbuf": frozenset({"-L", "-q", "-h", "--line-buffered",
                         "--quiet", "--help"}),
    "timeout": frozenset({"-f", "--foreground", "-p", "--preserve-status",
                          "-v", "--verbose"}),
    "flock": frozenset({"-F", "--no-fork", "-e", "-x", "--exclusive",
                        "-n", "--nb", "--nonblocking", "-o", "--close",
                        "-s", "--shared", "-u", "--unlock", "--fcntl",
                        "--verbose"}),
    "taskset": frozenset({"-a", "--all-tasks"}),
    "chrt": frozenset({"-o", "--other", "-f", "--fifo", "-r", "--rr",
                       "-b", "--batch", "-i", "--idle", "-d", "--deadline",
                       "-e", "--ext", "-G", "--reclaim-grub", "-O",
                       "--deadline-overrun", "-R", "--reset-on-fork",
                       "-a", "--all-tasks", "-m", "--max", "-v", "--verbose"}),
    # setsid(1): -c/--ctty, -f/--fork, -w/--wait, -h/--help, -V/--version.
    # nohup(1): --help, --version.  exec, command, builtin: no options.
    "setsid": frozenset({"-c", "--ctty", "-f", "--fork", "-w", "--wait",
                         "-h", "--help", "-V", "--version"}),
    "nohup": frozenset({"--help", "--version"}),
    # time(1): -p, -a are flags; -f, -o take arguments (see _WRAPPER_OPTS_WITH_ARG).
    "time": frozenset({"-p", "-a", "--portability", "--append"}),
    "exec": frozenset(),
    "command": frozenset(),
    "builtin": frozenset(),
}
# Command-position wrappers and how many POSITIONAL operands each consumes
# before the command word. Verified against the man pages listed above.
_WRAPPER_POSITIONAL = {
    "timeout": 1,    # timeout DURATION command …
    "flock": 1,      # flock FILE command …   (FILE is positional; the `flock 9` fd form is out of scope)
    "taskset": 1,    # taskset MASK command …
    "chrt": 1,       # chrt [OPTS] [SCHED_TYPE] PRIORITY command … (PRIORITY is the one positional)
}
# Wrappers whose single positional may be SUPPLIED BY AN OPTION ARGUMENT
# instead: taskset(1) `taskset --cpu-list 0 command` — the cpu-list IS the
# mask, so no separate mask positional follows. When such an option is
# consumed by the option loop, the positional count drops by one.
_WRAPPER_POSITIONAL_SUPPLIED_BY = {
    "taskset": frozenset({"-c", "--cpu-list"}),
}


class _WrapperResolutionFailed(Exception):
    """A wrapper in the command uses an option or operand shape the resolver
    cannot account for.

    Raised by `_resolve_command_word()` when it must refuse to guess which
    token is the command word. Deliberately fail-closed: a wrong guess
    returns the wrong editor and is exactly the hardline bypass class this
    guard exists to prevent (P2-1(a)). `_detect_hermes_inplace_edit()` turns
    this into an unconditional hardline block for any in-place editor in the
    command.
    """


# Editors with an in-place mode, and the short options that take a separate
# argument.  Case matters: ruby/perl `-I` (include dir, takes an argument)
# must not be confused with `-i` (in-place), so option parsing runs on the
# original-case text.
_INPLACE_EDITOR_ARG_OPTS = {
    "sed": frozenset({"e", "f", "l"}),
    "perl": frozenset({"e", "E", "F", "I", "M", "m"}),
    "ruby": frozenset({"e", "I", "r", "C", "F", "K", "E"}),
}
# Long options taking a separate argument when written without "=".
_INPLACE_EDITOR_LONG_ARG_OPTS = {
    "sed": frozenset({"--expression", "--file", "--line-length"}),
    "perl": frozenset(),
    "ruby": frozenset({"--encoding"}),
}
# Long options that select a script inline, so the first operand is a FILE
# rather than the program text.
_INPLACE_SCRIPT_OPTS = {
    "sed": frozenset({"e", "f"}),
    "perl": frozenset({"e", "E"}),
    "ruby": frozenset({"e"}),
}
_INPLACE_SCRIPT_LONG_OPTS = frozenset({"--expression", "--file"})
# The protected files, as the path token a shell would hand the editor.
_HERMES_PROTECTED_BASENAMES = frozenset({"config.yaml", ".env"})
_HERMES_HOME_PREFIXES = (
    "~/.hermes/",
    "$home/.hermes/",
    "${home}/.hermes/",
    "$hermes_home/",
    "${hermes_home}/",
)


def _shell_word_split(segment: str) -> list:
    """Split one command segment into shell words, dropping quote marks.

    Forgiving by design — this runs on adversarial and half-normalized text,
    so an unterminated quote must yield tokens rather than raise.  Quotes are
    removed because the operand comparison wants the path the shell would
    actually pass ("~/.hermes/config.yaml" -> ~/.hermes/config.yaml), and a
    quoted sed program collapses to a single token, which is exactly how it
    reaches the editor.
    """
    tokens: list = []
    current: list = []
    quote = None
    for ch in segment:
        if quote is not None:
            if ch == quote:
                quote = None
            else:
                current.append(ch)
        elif ch in "'\"":
            quote = ch
        elif ch.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(ch)
    if current:
        tokens.append("".join(current))
    return tokens


def _wrapper_operand_span(tokens: list, index: int, wrapper: str) -> int:
    """Advance `index` past `wrapper`'s own options and mandatory positionals.

    `tokens[index - 1]` is the wrapper. Returns the index of the next token
    (the command word, or the following wrapper in a chain).

    Raises _WrapperResolutionFailed when the option/positional shape is
    ambiguous (unrecognized option, missing mandatory positional) — see the
    module-level note above on why this guard fails closed rather than
    reusing the softer _COMMAND_WRAPPER_WORDS heuristic.
    """
    limit = len(tokens)
    takes_arg = _WRAPPER_OPTS_WITH_ARG.get(wrapper, frozenset())
    noarg_flags = _WRAPPER_NOARG_FLAGS.get(wrapper, frozenset())
    # --help / --version are universal GNU noarg options (every man page lists
    # them); treat them as recognized noarg so they never trip the fail-closed
    # rule.
    noarg_flags = noarg_flags | {"--help", "--version"}
    # Short letters that take an argument (`-u` -> 'u') and short letters that
    # are pure flags (`-n` -> 'n').
    arg_letters = {o[1] for o in takes_arg
                   if len(o) == 2 and o.startswith("-") and not o.startswith("--")}
    flag_letters = {o[1] for o in noarg_flags
                    if len(o) == 2 and o.startswith("-") and not o.startswith("--")}
    # How much of the wrapper's positional requirement an option argument
    # already satisfies (taskset --cpu-list).
    supplied_by = _WRAPPER_POSITIONAL_SUPPLIED_BY.get(wrapper, frozenset())
    supplied = 0
    # Pure flags actually seen (for the chrt optional-priority rule).
    noarg_flags_seen = set()
    # Consume the wrapper's own options.
    while index < limit:
        opt = tokens[index]
        if not opt.startswith("-") or opt == "-":
            break
        if opt == "--":
            index += 1
            break
        name = opt.split("=", 1)[0]
        if name in takes_arg:
            # `--user root`, `--user=root`, `-u root`, `-u=root`.
            index += 1
            if "=" not in opt and index < limit:
                index += 1
                if name in supplied_by:
                    supplied += 1
            continue
        if name in noarg_flags:
            noarg_flags_seen.add(name)
            index += 1
            continue
        # Attached short form: `-n5` (nice), `-uPATH` (env), `-oL` (stdbuf).
        # The option is recognized and its argument is glued to it, so only
        # this one token is consumed.
        if (len(name) > 2 and name.startswith("-") and not name.startswith("--")
                and name[1] in arg_letters):
            index += 1
            if name[1] in supplied_by:  # defensive: not used by current tables
                supplied += 1
            continue
        # A bundle of recognized pure flags (`-xn`): no arguments. Any bundle
        # member that takes an argument makes the argument count ambiguous —
        # refuse to guess (fail closed).
        if (len(name) > 2 and name.startswith("-") and not name.startswith("--")
                and all(ch in flag_letters for ch in name[1:])):
            index += 1
            continue
        # Unrecognized shape (unknown option, unknown bundle member, or a
        # malformed flag). We cannot tell how many tokens the wrapper
        # consumes, so refusing to resolve is the only non-bypassable choice:
        # a wrong guess returns the wrong command word and is the P2-1(a)
        # bypass class. Fail closed.
        raise _WrapperResolutionFailed(
            f"unrecognized option {opt!r} for wrapper {wrapper!r}"
        )
    # Wrappers in _WRAPPER_POSITIONAL take a mandatory POSITIONAL before the
    # command word (minus whatever an option argument already supplied, e.g.
    # `taskset --cpu-list 0 command`).
    required = _WRAPPER_POSITIONAL.get(wrapper, 0) - supplied
    # chrt(1) is the one wrapper whose single positional (PRIORITY) is
    # OPTIONAL once a policy flag (-o/-f/-r/-b/-i/-d/-e) is used: `chrt -b 80
    # command` and `chrt -b command` are both legal. Disambiguation: the
    # priority is numeric (0-99), the command word is not.
    chrt_policy_flags = {"-o", "--other", "-f", "--fifo", "-r", "--rr",
                         "-b", "--batch", "-i", "--idle", "-d", "--deadline",
                         "-e", "--ext"}
    chrt_policy_seen = wrapper == "chrt" and bool(
        chrt_policy_flags & noarg_flags_seen
    )
    for _ in range(required):
        if index >= limit:
            if wrapper == "chrt" and chrt_policy_seen:
                # Priority was optional; the command word simply was not
                # given — nothing more to consume.
                break
            # The wrapper is missing its mandatory positional (e.g. bare
            # `flock`): there is no resolvable command word behind it, so
            # refuse rather than guess.
            raise _WrapperResolutionFailed(
                f"missing positional operand for wrapper {wrapper!r}"
            )
        if tokens[index].startswith("-") and tokens[index] != "-":
            # A flag where the wrapper's mandatory positional was expected
            # (e.g. `flock -w` with no seconds): the shell would hand the flag
            # to the wrapper as the positional, so the command word is not
            # where a positional would be. Refuse to resolve (fail closed).
            raise _WrapperResolutionFailed(
                f"option {tokens[index]!r} where wrapper {wrapper!r} "
                "needs its positional operand"
            )
        if wrapper == "chrt" and chrt_policy_seen:
            if tokens[index].replace(".", "", 1).isdigit():
                # Numeric: the optional PRIORITY. The command word is the
                # next token.
                index += 1
                if index >= limit:
                    raise _WrapperResolutionFailed(
                        "chrt: no command word after the priority"
                    )
            # Non-numeric (or after skipping the priority): the command word
            # is at index — stop here.
            break
        index += 1
    return index


def _resolve_command_word(tokens: list) -> tuple:
    """Return (command_word, remaining_args) after stripping wrappers.

    Resolves `/usr/bin/sed` to `sed` and steps over `sudo`/`env`/`command`
    and friends, including their own options and `VAR=VAL` assignments, so
    the caller sees the program that actually runs.

    Raises _WrapperResolutionFailed when a wrapper uses an option we do not
    recognize (or a bundle / malformed shape whose argument count is
    ambiguous). Failing closed here — rather than guessing which token is
    the command word — is deliberate: a wrong guess returns the wrong
    editor and is exactly the hardline bypass class this guard exists to
    prevent. The caller turns the exception into an unconditional block.
    """
    index = 0
    limit = len(tokens)
    while index < limit:
        token = tokens[index]
        if not token:
            index += 1
            continue
        token = token.lstrip("(`{")
        if not token:
            index += 1
            continue
        if "=" in token and not token.startswith("-"):
            name = token.split("=", 1)[0]
            if name and (name[0].isalpha() or name[0] == "_") and all(
                c.isalnum() or c == "_" for c in name
            ):
                index += 1
                continue
        base = token.replace("\\", "/").rsplit("/", 1)[-1]
        if base in _EDITOR_WRAPPERS:
            index = _wrapper_operand_span(tokens, index + 1, base)
            continue
        return (base, tokens[index + 1:])
    return ("", [])


def _is_protected_hermes_operand(token: str) -> bool:
    """True when this operand names ~/.hermes/config.yaml or ~/.hermes/.env.

    Whole-token comparison with an exact filename match, so `config.yaml.bak`
    and `config.yaml.orig` — distinct files that carry no policy — stay out.

    Before the match, canonicalize the path with pure string/``posixpath``
    normalization (no filesystem access, no ``os.path.realpath`` — this runs
    on adversarial input inside a guard): backslash-fold to ``/``, collapse
    repeated separators, resolve ``.`` and ``..`` segments, and strip a
    trailing separator. So the aliases `~/.hermes/./config.yaml`,
    `~/.hermes//config.yaml`, `~/.hermes/sub/../config.yaml`, and the
    trailing-separator form all name the very same policy file and are caught,
    while `config.yaml.bak` / `config.yaml.orig` remain allowed.
    """
    candidate = token.strip().rstrip(")`;&|").replace("\\", "/").lower()
    if not candidate:
        return False
    candidate = posixpath.normpath(candidate)
    for prefix in _HERMES_HOME_PREFIXES:
        if candidate.startswith(prefix):
            if candidate[len(prefix):] in _HERMES_PROTECTED_BASENAMES:
                return True
    # Spelled-out home directory (/home/u/.hermes/config.yaml, and the
    # /Users/u form on macOS). The tilde/$HOME spellings above are the ones
    # an agent writes, but an absolute path mutates the very same file.
    for basename in _HERMES_PROTECTED_BASENAMES:
        if candidate.endswith("/.hermes/" + basename):
            return True
    return False


def _editor_targets_protected_file(editor: str, args: list) -> bool:
    """Walk an in-place editor's options; report a protected FILE operand.

    Returns True only when in-place mode is genuinely enabled AND one of the
    operands the editor would rewrite is a protected file.
    """
    short_arg_opts = _INPLACE_EDITOR_ARG_OPTS[editor]
    long_arg_opts = _INPLACE_EDITOR_LONG_ARG_OPTS[editor]
    script_opts = _INPLACE_SCRIPT_OPTS[editor]
    in_place = False
    have_script_option = False
    operands: list = []
    index = 0
    limit = len(args)
    end_of_options = False
    while index < limit:
        token = args[index]
        index += 1
        if not token:
            continue
        if end_of_options or not token.startswith("-") or token == "-":
            operands.append(token)
            continue
        if token == "--":
            end_of_options = True
            continue
        if token.startswith("--"):
            name = token.split("=", 1)[0]
            if name in ("--in-place", "--inplace"):
                in_place = True
                continue
            if name in _INPLACE_SCRIPT_LONG_OPTS:
                have_script_option = True
            if name in long_arg_opts and "=" not in token and index < limit:
                index += 1
            continue
        # Short option token, possibly a bundle: -ri, -i.bak, -ne, -pi
        chars = token[1:]
        position = 0
        while position < len(chars):
            letter = chars[position]
            if letter == "i":
                # Everything after `i` is the attached backup suffix.
                in_place = True
                break
            if letter in short_arg_opts:
                if letter in script_opts:
                    have_script_option = True
                # Attached argument, or the next token if nothing is attached.
                if position + 1 >= len(chars) and index < limit:
                    index += 1
                break
            position += 1
    if not in_place:
        return False
    # Without an explicit script option the FIRST operand is the program
    # text (`sed -i 's/a/b/' file`), not a file the editor rewrites.
    files = operands if have_script_option else operands[1:]
    return any(_is_protected_hermes_operand(operand) for operand in files)


def _unresolvable_segment_is_inplace_threat(tokens: list) -> bool:
    """Fail-closed fallback for a segment whose wrapper the resolver refused.

    When `_resolve_command_word()` raises `_WrapperResolutionFailed`, we
    cannot determine the real command word — so we cannot rule out that the
    wrapper is hiding an in-place edit of a protected file. Block the segment
    (return True) when it still names BOTH an in-place editor (sed/perl/ruby)
    and a protected operand, and stay silent otherwise. This keeps the
    fail-closed posture where it matters while not false-positiving on a read
    (`flock --bogus /tmp/lock cat ~/.hermes/config.yaml`), an unrelated
    command, or `--help`.
    """
    editor_words = {"sed", "perl", "ruby"}
    has_editor = False
    has_protected = False
    for token in tokens:
        base = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if base in editor_words:
            has_editor = True
        if _is_protected_hermes_operand(token):
            has_protected = True
    return has_editor and has_protected


def _detect_hermes_inplace_edit(command_variant: str) -> bool:
    """True when this variant runs an in-place editor on a protected file."""
    for segment in command_variant.split("\n"):
        if not segment.strip():
            continue
        tokens = _shell_word_split(segment)
        if not tokens:
            continue
        try:
            editor, args = _resolve_command_word(tokens)
        except _WrapperResolutionFailed:
            # The wrapper's command word is not determinable. Fail closed:
            # block this segment if it still plausibly edits a protected file
            # in place (see _unresolvable_segment_is_inplace_threat).
            if _unresolvable_segment_is_inplace_threat(tokens):
                return True
            continue
        if editor in _INPLACE_EDITOR_ARG_OPTS and _editor_targets_protected_file(
            editor, args
        ):
            return True
    return False


_HERMES_INPLACE_DESCRIPTION = (
    "in-place edit of the Hermes approval policy / credential file "
    "(~/.hermes/config.yaml, ~/.hermes/.env)"
)


def detect_hardline_command(command: str) -> tuple:
    """Check hardline patterns (NEVER bypassable, even in YOLO) -> (is_hardline, description)."""
    if _command_parser_limit_exceeded(command):
        return (True, _PARSER_LIMIT_DESCRIPTION)
    # The malformed-quoting verdict needs the author's quote state. Normalization strips escapes
    # (`\"` -> `"`), so a shell-valid pattern like `grep -o "[^\"]*"` lexed as unterminated and was
    # reported as a hardline block (118 of 125 hardline blocks in one week of real use, every one a
    # benign grep). Only quoted newlines are masked: they are data, and masking keeps quoting intact.
    _, malformed_grep = _grep_safe_detection_variant(_mask_quoted_newlines(command))
    if malformed_grep:
        return (True, _MALFORMED_EXEC_DESCRIPTION)
    for command_variant in _command_detection_variants(command):
        variant_lower = command_variant.lower()
        masked_lower: str | None = None
        for pattern_re, description, quote_masked in HARDLINE_PATTERNS_COMPILED:
            if quote_masked and masked_lower is None:
                # Positionless rules see quoted prose as DATA, except under shell carriers
                # (sh -c, eval, source) whose quoted argument is code — those scan raw. bash -c
                # payloads also surface as their own raw variants via _execution_flag_findings.
                masked_lower = (
                    variant_lower if _contains_shell_carrier(command_variant)
                    else _mask_quoted_prose(command_variant).lower()
                )
            if pattern_re.search(masked_lower if quote_masked else variant_lower):
                return (True, description)
        # Option-grammar check, not a text search — see the block comment
        # above _shell_word_split(). Runs on the original-case variant
        # because ruby/perl `-I` and `-i` mean different things.
        if _detect_hermes_inplace_edit(command_variant):
            return (True, _HERMES_INPLACE_DESCRIPTION)
    return (False, None)


# ---- Dangerous command patterns -----------------------------------------------------------
DANGEROUS_PATTERNS = [
    (r'\brm\s+(-[^\s]*\s+)*/', "delete in root path"),
    (r'\brm\s+-[^\s]*r', "recursive delete"),
    (r'\brm\s+--recursive\b', "recursive delete (long flag)"),
    # GNU rm permutes options, so flags may FOLLOW operands (`rm build/ -rf`). The operand run
    # cannot cross a command separator (so `rm foo | grep -r` is not attributed to rm), a quote,
    # or a bare ` -- ` end-of-options (after which `-rf` is a literal filename). The flag token
    # must follow whitespace so the `r` in long options like `--registry` does not count.
    (r'\brm\s+(?!--(?:\s|$))(?:(?!\s--(?:\s|$))[^\n"\';|&])*\s' r'(?:-[a-z]*r[a-z]*\b|--recursive\b)',
     # GNU rm permutes options, so a recursive flag group may legally FOLLOW the operands: `rm build/ -rf`,
     # `rm build/ -r -f`, and `rm build/ --recursive --force` are all equivalent to the flags-first
     # spellings the two patterns above catch — without this rule they run with no approval prompt at all.
     # Port of openai/codex#33464 ("recognize force options when they follow operands").
     "recursive delete (flags after operands)"),
    # Windows cmd/powershell destructive built-ins: gate only when executed through the shell so
    # prose/filenames containing "del"/"rd" do not trip.
    (r'\bcmd(?:\.exe)?\s+/(?:c|k)\s+.*\b(?:del|erase|rd|rmdir)\b', "Windows cmd destructive delete"),
    # PowerShell runs the verb as default positional arg (no -Command needed); anchor the verb to command
    # position (after leading -Flag switches and optional -Command/-c) so `-File c:\del-logs\run.ps1` is not caught.
    (r'\b(?:powershell|pwsh)(?:\.exe)?\b(?:\s+-\S+)*\s+(?:-(?:command|c)\s+)?["\']?(?:remove-item|rmdir|erase|del|rd|ri|rm)\b', "Windows PowerShell destructive delete"),
    (r'\b(?:powershell|pwsh)(?:\.exe)?\b.*\s-(?:encodedcommand|enc|e)\b', "PowerShell encoded command execution"),
    # ── Windows destructive tier: native Windows EXEs/cmdlets reachable from ANY backend on a
    # Windows host (incl. git-bash). Input is lowercased by the variant loop, so patterns are
    # lowercase. Each requires the destructive flag/verb so benign usage (`taskkill /IM app.exe`,
    # `reg query`, `icacls file`) does NOT prompt. Bare Remove-Item form (ACP clients, pwsh-default
    # SSH hosts, or compound commands where `powershell` appeared earlier).
    # See #69472.
    (r'\bremove-item\b[^\n;|&]*\s-(?:recurse|force)\b', "PowerShell destructive delete (Remove-Item)"),
    # Bare cmd builtins with /s (recurse) or /q (quiet); plain `del file.txt` is covered only by the prefixed rule.
    (r'\b(?:del|erase|rd|rmdir)\s+(?:/[a-z]\s+)*/[sq]\b', "Windows destructive delete (recursive/quiet switch)"),
    # Remote content piped to Invoke-Expression — PowerShell's `curl | sh`.
    (r'\b(?:iwr|invoke-webrequest|invoke-restmethod|irm|curl|wget)\b[^\n]*\|\s*(?:iex|invoke-expression)\b', "pipe remote content to PowerShell (iwr | iex)"),
    (r'\b(?:iex|invoke-expression)\s*\(\s*(?:iwr|invoke-webrequest|invoke-restmethod|irm)\b', "execute remote content via Invoke-Expression"),
    # Force process kills — Windows analogue of pkill -9.
    (r'\btaskkill\b[^\n]*\s/f\b', "force kill processes (taskkill /F)"),
    (r'\bstop-process\b[^\n]*\s-force\b', "force kill processes (Stop-Process -Force)"),
    # Volume/disk destruction — Windows analogue of mkfs / dd.
    (r'\bformat-volume\b', "format filesystem (Format-Volume)"),
    (r'\bclear-disk\b', "wipe disk (Clear-Disk)"),
    (r'\bdiskpart\b', "disk partitioning (diskpart)"),
    (r'\bformat(?:\.com)?\s+[a-z]:', "format drive (format.com)"),
    (r'\bcipher\s+/w\b', "wipe free space (cipher /w)"),
    # ACL destruction — Windows analogue of chmod 777.
    (r'\bicacls\b[^\n]*\s/grant\b[^\n]*\b(?:everyone|todos|jeder|tout\s+le\s+monde|\*s-1-1-0)\b', "grant Everyone access (icacls)"),
    (r'\bicacls\b[^\n]*\s/reset\b', "reset ACLs recursively (icacls /reset)"),
    # Backup/recovery destruction — classic ransomware prep.
    (r'\bvssadmin\b[^\n]*\bdelete\s+shadows\b', "delete volume shadow copies (vssadmin)"),
    (r'\bwbadmin\b[^\n]*\bdelete\b', "delete backups (wbadmin)"),
    (r'\bbcdedit\b[^\n]*\s/set\b', "modify boot configuration (bcdedit /set)"),
    # Registry deletion with force flag.
    (r'\breg(?:\.exe)?\s+delete\b', "registry delete (reg delete)"),
    (r'\bremove-itemproperty\b[^\n]*\s-force\b', "registry value delete (Remove-ItemProperty -Force)"),
    # Windows service/system stop — analogue of systemctl stop.
    (r'\bstop-service\b[^\n]*\s-force\b', "force stop service (Stop-Service -Force)"),
    (r'\bsc(?:\.exe)?\s+(?:stop|delete)\b', "stop/delete service (sc)"),
    # Windows-form credential paths; the POSIX ~/.ssh patterns never match drive-letter or backslash spellings.
    (r'\busers[\\/][^\\/\s]+[\\/]\.ssh\b', "access to SSH keys (Windows path)"),
    (r'\bappdata[\\/](?:local|roaming)[\\/]hermes[^\n]*\.env\b', "access to Hermes secrets (Windows path)"),
    # ── end of Windows tier
    (r'\bchmod\s+(-[^\s]*\s+)*(777|666|o\+[rwx]*w|a\+[rwx]*w)\b', "world/other-writable permissions"),
    (r'\bchmod\s+--recursive\b.*(777|666|o\+[rwx]*w|a\+[rwx]*w)', "recursive world/other-writable (long flag)"),
    (r'\bchown\s+(-[^\s]*)?R\s+root', "recursive chown to root"),
    (r'\bchown\s+--recur[a-z]*\b.*root', "recursive chown to root (long flag)"),
    # _CMDPOS-anchored like the hardline twins: quoted prose mentioning mkfs/dd must not require approval to echo.
    # See #93392.
    (_CMDPOS + r'mkfs\b', "format filesystem"),
    (_CMDPOS + r'dd\s+.*if=', "disk copy"),
    (r'>\s*/dev/sd', "write to block device"),
    (r'\bDROP\s+(TABLE|DATABASE)\b', "SQL DROP"),
    # [^\n]* not .*: under DOTALL a WHERE on the *next* line would satisfy the lookahead and
    # silently allow DELETE without WHERE.
    (r'\bDELETE\s+FROM\b(?![^\n]*\bWHERE\b)', "SQL DELETE without WHERE"),
    (r'\bTRUNCATE\s+(TABLE)?\s*\w', "SQL TRUNCATE"),
    (rf'>\s*{_SYSTEM_CONFIG_PATH}', "overwrite system config"),
    (r'\bsystemctl\s+(-[^\s]+\s+)*(stop|restart|disable|mask)\b', "stop/restart system service"),
    (r'\bkill\s+-9\s+-1\b', "kill all processes"),
    (r'\bpkill\s+-9\b', "force kill processes"),
    # killall with SIGKILL (-9 / -KILL / -s KILL / -SIGKILL) and `killall -r <regex>` broad sweeps
    # that can wipe unrelated processes.
    (r'\bkillall\s+(-[^\s]*\s+)*-(9|KILL|SIGKILL)\b', "force kill processes (killall -KILL)"),
    (r'\bkillall\s+(-[^\s]*\s+)*-s\s+(KILL|SIGKILL|9)\b', "force kill processes (killall -s KILL)"),
    (r'\bkillall\s+(-[^\s]*\s+)*-r\b', "kill processes by regex (killall -r)"),
    (r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:', "fork bomb"),
    # Shell -c is parsed structurally by _execution_flag_findings(); a regex searching a dash-token
    # for "c" also matched --norc/--rcfile/--restricted.
    (r'\b(curl|wget)\b.*\|\s*(?:[/\w]*/)?(?:ba)?sh(?:\s|$|-c)', "pipe remote content to shell"),
    (r'\b(bash|sh|zsh|ksh)\s+<\s*<?\s*\(\s*(curl|wget)\b', "execute remote script via process substitution"),
    # eval/source/. $(curl ...) — equivalent to piping remote content to a shell.
    (r'(?:\beval\b|\bsource\b|\.)\s*(?:\$\(\s*|`\s*)(?:curl|wget)\b', "execute remote content via command substitution"),
    # Cloud instance-metadata (IMDS) credential endpoints — deterministic containment-escape
    # detection. On a cloud VM these serve live IAM/service-account credentials to ANY local
    # process with no auth, so a fetch is credential exfiltration unless the operator expects it.
    # The host literals have no other use, so their appearance ANYWHERE in the command (any HTTP
    # client, env assignment, or script argument) is the signal; lookarounds keep other 169.254.x.x
    # link-local addresses and longer dotted strings out. This prompts for approval (legit uses
    # exist on real cloud VMs) — it is NOT a hardline block. Covers the link-local IPv4 endpoint
    # (AWS/Azure/GCP/OpenStack), its AWS IPv6 form fd00:ec2::254, the GCP hostname, and Alibaba
    # Cloud's 100.100.100.200.
    (r'(?<![\d.])(?:169\.254\.169\.254|100\.100\.100\.200)(?![\d.])'
     r'|(?<![\w.-])metadata\.google\.internal(?![\w.-])'
     r'|fd00:ec2::254',
     "cloud metadata endpoint access (instance credentials)"),
    # Decode-and-execute: `echo <base64> | base64 -d | bash` carries no dangerous keywords in the
    # raw text yet runs arbitrary commands.
    (r'\b(base64|base32|base16)\s+(?:-[dD]|--decode)\b.*\|\s*\b(bash|sh|zsh|ksh|dash)\b', "pipe decoded content to shell (possible command obfuscation)"),
    # xxd uses -r for decode, not -d.
    (r'\bxxd\s+-r\b.*\|\s*\b(bash|sh|zsh|ksh|dash)\b', "pipe xxd-decoded content to shell (possible command obfuscation)"),
    # `echo 'eq -pe v/' | tr 'eqv' 'rmf' | bash` decodes to `rm -rf /`.
    (r'\becho\b[^|]*\|\s*\btr\b[^|]*\|\s*\b(bash|sh|zsh|ksh|dash)\b', "pipe tr-transformed output to shell (possible command obfuscation)"),
    (r'\bopenssl\b.*\b(?:base64|enc)\b[^|]*\s+-[dD]\b[^|]*\|\s*\b(bash|sh|zsh|ksh|dash)\b',
     "pipe openssl-decoded content to shell (possible command obfuscation)"),
    (rf'\btee\b.*["\']?{_SENSITIVE_WRITE_TARGET}', "overwrite system file via tee"),
    (rf'>>?\s*["\']?{_SENSITIVE_WRITE_TARGET}', "overwrite system file via redirection"),
    (rf'\btee\b.*["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_WRITE_TARGET_BOUNDARY}', "overwrite project env/config via tee"),
    (rf'>>?\s*["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_WRITE_TARGET_BOUNDARY}', "overwrite project env/config via redirection"),
    (r'\bxargs\s+.*\brm\b', "xargs with rm"),
    # -execdir has the same semantics as -exec (runs in each match's directory).
    (r'\bfind\b.*-exec(?:dir)?\s+(/\S*/)?rm\b', "find -exec/-execdir rm"),
    (r'\bfind\b.*-delete\b', "find -delete"),
    # Gateway lifecycle: stopping/restarting the gateway kills all running agents. Global flags
    # between `hermes` and `gateway` (`hermes -p ade gateway restart`) are allowed so a profile flag can't slip past.
    (r'\bhermes\s+(?:-{1,2}\S+(?:\s+\S+)?\s+)*gateway\s+(stop|restart)\b', "stop/restart hermes gateway (kills running agents)"),
    (r'\bhermes\s+update\b', "hermes update (restarts gateway, kills running agents)"),
    # Docker/Podman daemon redirect — global flags or env that point the CLI at a DIFFERENT (often remote) daemon:
    # `docker -H ssh://prod stop app` looks local but operates on remote infra, so any redirect requires approval
    # regardless of subcommand. The flag must be in global position (before the subcommand) and -H/--host/--context
    # must carry a value, keeping `docker -h` and `docker run -h <hostname>` out. Listed BEFORE the lifecycle rules so
    # a redirected lifecycle command surfaces the more specific reason.
    (r'\bdocker\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*(?:-h|--host)[=\s]+\S+', "docker with remote daemon redirect (-H/--host)"),
    (r'\bdocker\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*(?:-c|--context)[=\s]+\S+', "docker with daemon redirect (--context: alternate daemon)"),
    (r'\bdocker\s+context\s+use\b', "docker context use (switches default daemon for future commands)"),
    (r'\bpodman\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*(?:--url|--connection|--identity)[=\s]+\S+', "podman with remote daemon redirect (--url/--connection/--identity)"),
    (r'\bpodman\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*(?:-r\b|--remote\b)', "podman remote mode (-r/--remote: remote daemon)"),
    (r'\b(?:docker_host|docker_context|container_host|container_connection)=\S+', "docker/podman daemon redirect via environment (DOCKER_HOST/CONTAINER_HOST)"),
    # Container lifecycle (docker.sock mounts let the agent stop/kill containers) always needs
    # consent. Global flags between docker/compose and the verb and the legacy `docker-compose`
    # binary are allowed so a flag can't slip past.
    (r'\bdocker(?:-compose|\s+compose)\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*(restart|stop|kill|down)\b', "docker compose restart/stop/kill/down (container lifecycle)"),
    (r'\bdocker\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*(restart|stop|kill)\b', "docker restart/stop/kill (container lifecycle)"),
    # Gateway protection: never start gateway outside systemd management
    (r'gateway\s+run\b.*(&\s*$|&\s*;|\bdisown\b|\bsetsid\b)', "start gateway outside systemd (use 'systemctl --user restart hermes-gateway')"),
    (r'\bnohup\b.*gateway\s+run\b', "start gateway outside systemd (use 'systemctl --user restart hermes-gateway')"),
    # Self-termination protection: prevent agent from killing its own process
    (r'\b(pkill|killall)\b.*\b(hermes|gateway|cli\.py)\b', "kill hermes/gateway process (self-termination)"),
    # Self-termination via kill + $(pgrep/pidof): the substitution is opaque to the name-based
    # pattern above, so catch the structural form.
    (r'\bkill\b.*\$\(\s*(pgrep|pidof)\b', "kill process via pgrep/pidof expansion (self-termination)"),
    (r'\bkill\b.*`\s*(pgrep|pidof)\b', "kill process via backtick pgrep/pidof expansion (self-termination)"),
    # launchctl-driven gateway stop/restart on macOS (label `ai.hermes.gateway`). Two independent lookaheads, NOT a
    # sequential match: a for-loop building the label from a list defined EARLIER (`for item in 'ai.hermes...'; do
    # launchctl bootout "$label"`) never has "hermes" after the verb, and that slipped past and restarted 4 gateways
    # with zero approval. Erring broad is correct for an approval gate: an extra prompt is cheap.
    (r'(?=[\s\S]*\blaunchctl\s+(?:stop|kickstart|bootout|unload|kill|disable|remove)\b)(?=[\s\S]*\b(?:hermes|ai\.hermes)\b)', "stop/restart hermes launchd service (kills running agents)"),
    (rf'\b(cp|mv|install)\b.*\s{_SYSTEM_CONFIG_PATH}', "copy/move file into system config path"),
    (rf'\b(cp|mv|install)\b.*\s["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_COMMAND_TAIL}', "overwrite project env/config file"),
    # cp/mv/install OVERWRITING a credential/SSH/shell-rc/Hermes file (key implant, login-time
    # injection) — pairs the tee/redirection coverage. Anchored to the command tail so only the
    # DESTINATION fires; reading OUT of a sensitive path (`cp ~/.ssh/config /tmp/x`) stays safe.
    # The trailing `[^\s"\']*` consumes the rest of the destination filename.
    # The tee/redirection patterns above already gate _SENSITIVE_WRITE_TARGET (~/.ssh/*,
    # ~/.netrc/.pgpass/.npmrc/.pypirc, shell rc files, ~/.hermes/config.yaml/.env), but cp/mv/install was
    # only paired for /etc and project-relative env/config — so `cp evil ~/.ssh/authorized_keys` (key
    # implant), `cp creds ~/.netrc`, and `cp evil ~/.bashrc` (login-time command injection) slipped through
    # with auto-approve. Same unpaired-door rationale as #14639 / the sed-tee-redirect pairing on these
    # targets. `authorized_keys` after the `~/.ssh/` fragment).
    (rf'\b(cp|mv|install)\b.*\s["\']?{_SENSITIVE_WRITE_TARGET}[^\s"\']*["\']?{_COMMAND_TAIL}', "copy/move file into sensitive credential/SSH/shell-rc path"),
    # In-place edits mutate the file directly, bypassing redirection/tee/cp coverage; gate the same
    # startup/credential files.
    (rf'\bsed\s+-[^\s]*i.*(?:{_USER_SENSITIVE_WRITE_TARGET})[^\s"\']*', "in-place edit of sensitive credential/SSH/shell-rc path"),
    (rf'\bsed\s+--in-place\b.*(?:{_USER_SENSITIVE_WRITE_TARGET})[^\s"\']*', "in-place edit of sensitive credential/SSH/shell-rc path (long flag)"),
    (rf'\b(?:perl|ruby)\b.*(?:^|\s)-[^\s]*i\b.*(?:{_USER_SENSITIVE_WRITE_TARGET})[^\s"\']*', "in-place edit of sensitive credential/SSH/shell-rc path (perl/ruby)"),
    (rf'\bsed\s+-[^\s]*i.*\s{_SYSTEM_CONFIG_PATH}', "in-place edit of system config"),
    (rf'\bsed\s+--in-place\b.*\s{_SYSTEM_CONFIG_PATH}', "in-place edit of system config (long flag)"),
    # sed -i on Hermes config/.env bypasses the redirection/tee rules; pairs the file_tools
    # write_file/patch deny so the terminal side is not an open door.
    # In-place edit of a Hermes-managed security file (~/.hermes/config.yaml or .env). sed -i bypasses the
    # redirection/tee patterns above because it mutates the file directly. See #14639.
    (rf'\bsed\s+-[^\s]*i.*(?:{_HERMES_CONFIG_PATH}|{_HERMES_ENV_PATH})', "in-place edit of Hermes config/env"),
    (rf'\bsed\s+--in-place\b.*(?:{_HERMES_CONFIG_PATH}|{_HERMES_ENV_PATH})', "in-place edit of Hermes config/env (long flag)"),
    # perl/ruby -i: the flag may be its own token after other flags (`-p -i -e`), combined (`-pi`), or carry a backup
    # suffix (`-i.bak`), so match any flag token containing `i` anywhere; `perl -e '...'` (no -i) does not trip.
    # perl -i and ruby -i perform the same in-place mutation as sed -i but are not caught by the -e/-c
    # script-execution pattern above (which targets code evaluation, not file mutation). Pairs the sed -i
    # coverage from #14639.
    (rf'\b(?:perl|ruby)\b.*(?:^|\s)-[^\s]*i\b.*(?:{_HERMES_CONFIG_PATH}|{_HERMES_ENV_PATH})', "in-place edit of Hermes config/env (perl/ruby)"),
    # Interpreter heredocs are handled by _execution_flag_findings(); only shell heredocs stay
    # regex-based. `bash <<'EOF'` runs arbitrary commands without triggering the `bash -c` path.
    (r'\b(bash|sh|zsh|ksh)\s+<<', "shell execution via heredoc"),
    # Git destructive operations. `git reset --hard` accepts any unambiguous long-flag prefix (--h,
    # --ha, --har): --hard is the only reset mode starting with "h", and `--help` is special-cased
    # by git before mode resolution.
    (r'\bgit\s+reset\s+--h(?:a(?:r(?:d)?)?)?\b', "git reset --hard (destroys uncommitted changes)"),
    (r'\bgit\s+push\b.*--forc[a-z]*\b', "git force push (rewrites remote history)"),
    (r'\bgit\s+push\b.*-f\b', "git force push short flag (rewrites remote history)"),
    (r'\bgit\s+clean\s+-[^\s]*f', "git clean with force (deletes untracked files)"),
    (r'\bgit\s+branch\s+-D\b', "git branch force delete"),
    # `-D` = `-d --force`; the long spellings are different tokens, so match delete+force in either order, bounded to
    # one command segment (no `;`/`|`/`&`/newline) so an unrelated later command isn't contaminated.
    (r'\bgit\s+branch\b[^;|&\n]*?(?:-d\b|--delete\b)[^;|&\n]*?(?:-f\b|--force\b)', "git branch force delete (long flags)"),
    (r'\bgit\s+branch\b[^;|&\n]*?(?:-f\b|--force\b)[^;|&\n]*?(?:-d\b|--delete\b)', "git branch force delete (long flags, force-first)"),
    # chmod +x then immediate run: the script content may hold dangerous commands individual patterns miss.
    (r'\bchmod\s+\+x\b.*[;&|]+\s*\./', "chmod +x followed by immediate execution"),
    # Sudo stdin/askpass/shell/list-privs flags. The agent has no TTY, so sudo invocations that succeed
    # non-interactively read the password from stdin (-S) or askpass (-A); -s (shell) and -a (list) are gated as
    # privilege chains (read SUDO_PASSWORD from .env -> sudo -S -s). Plain `sudo cmd` is TTY-bound and excluded. Input
    # is lowercased, so S/s and A/a collapse. Lazy `[^;|&\n]*?` allows flag args without spanning separators. sudo
    # resolves unambiguous long-flag prefixes: `--stdin` is the only long option starting with "st", `--askpass` the
    # only one starting with "a".
    (r'\bsudo\b[^;|&\n]*?\s+(?:-s\b|--st[a-z]*\b|-a\b|--a[a-z]*\b)', "sudo with privilege flag (stdin/askpass/shell/list)"),
    # Combined short-flag form (-nS, -sa, -las).
    (r'\bsudo\b[^;|&\n]*?\s+-[a-z]*[sa][a-z]*\b', "sudo with combined-flag privilege escalation"),
    # Package-manager uninstall commands can remove installed software outside
    # the current project (notably `npm uninstall -g`). Treat their destructive
    # subcommands like other state-removing operations while leaving installs
    # and updates alone.
    # _CMDPOS-anchored (quoted prose like `git commit -m "npm uninstall docs"` is data); the
    # option group also swallows one operand (`--prefix DIR`, `--proxy URL`, `--cwd DIR`).
    (_CMDPOS + r'npm\s+' + _PKG_OPTS + r'(?:uninstall|unlink|remove|rm|r|un)\b', "package manager uninstall"),
    (_CMDPOS + r'pnpm\s+' + _PKG_OPTS + r'(?:uninstall|remove|rm|un)\b', "package manager uninstall"),
    (_CMDPOS + r'yarn\s+' + _PKG_OPTS + r'(?:global\s+)?(?:uninstall|remove)\b', "package manager uninstall"),
    (_CMDPOS + r'pip(?:3)?\s+' + _PKG_OPTS + r'uninstall\b', "package manager uninstall"),
    (_CMDPOS + r'brew\s+' + _PKG_OPTS + r'(?:uninstall|remove|rm)\b', "package manager uninstall"),
]


DANGEROUS_PATTERNS_COMPILED = [(re.compile(p, _RE_FLAGS), d) for p, d in DANGEROUS_PATTERNS]

# Preserve approvals stored under the removed interpreter regex rules.
_REMOVED_PATTERN_KEY_ALIASES = {
    "script execution via -e/-c flag": "(python[23]?|perl|ruby|node)\\s+-[ec]\\s+",
    "script execution via heredoc": "(python[23]?|perl|ruby|node)\\s+<<",
}
# description <-> legacy regex-derived key (the old approval key, kept for backwards compatibility
# with stored allowlist/session entries), both ways.
_PATTERN_KEY_ALIASES: dict[str, set[str]] = {}
for _canonical_key, _legacy_key in [
    (d, p.split(r'\b')[1] if r'\b' in p else p[:20]) for p, d in DANGEROUS_PATTERNS
] + list(_REMOVED_PATTERN_KEY_ALIASES.items()):
    _PATTERN_KEY_ALIASES.setdefault(_canonical_key, set()).update({_canonical_key, _legacy_key})
    _PATTERN_KEY_ALIASES.setdefault(_legacy_key, set()).update({_legacy_key, _canonical_key})


def _approval_key_aliases(pattern_key: str) -> set[str]:
    """All approval keys for this pattern: the description plus the historical regex-derived key
    older allowlist/session entries may still use."""
    return _PATTERN_KEY_ALIASES.get(pattern_key, {pattern_key})


# ---- Detection ----------------------------------------------------------------------------
def _normalize_command_for_detection(command: str) -> str:
    """Normalize a command before pattern matching so ANSI escapes, null bytes, Unicode fullwidth
    forms, and shell splicing tricks cannot bypass detection."""
    from tools.ansi_strip import strip_ansi
    command = unicodedata.normalize('NFKC', strip_ansi(command).replace('\x00', ''))
    # Collapse backslash-newline continuations (`rm -rf \<newline>/` runs as `rm -rf /`). MUST
    # precede the generic escape strip below, whose [^\n] class skips newlines and would leave the
    # backslash wedged between tokens, defeating the structured rm/mkfs/dd patterns incl. the HARDLINE floor.
    command = re.sub(r'\\\r?\n', '', command)
    # Fold absolute user/Hermes home prefixes to ~/ and ~/.hermes/ so the static patterns catch /home/alice/.bashrc
    # and C:\Users\alice\.bashrc. Resolved at detection time (not import time) so it tracks HOME/HERMES_HOME set
    # later. MUST run before the backslash strip (which would dissolve C:\Users\alice to C:Usersalice). Hermes home
    # first: on Windows it nests under the user home, and folding the user home first would eat the prefix it needs.
    command = _rewrite_resolved_hermes_home(command)
    command = _rewrite_resolved_user_home(command)
    # Strip backslash-escapes (r\m -> rm) and empty-string literals (r''m -> rm).
    command = re.sub(r'\\([^\n])', r'\1', command)
    command = re.sub(r"''|\"\"", '', command)
    # Collapse $IFS / ${IFS...} (incl. `${IFS:0:1}`) to a space: IFS defaults to whitespace, so `rm${IFS}-rf${IFS}/`
    # runs as `rm -rf /`, and every pattern — incl. the hardline floor — anchors on literal \s between tokens.
    return re.sub(r'\$\{IFS\b[^}]*\}|\$IFS\b', ' ', command)


# Shell metacharacters, quotes, and whitespace that terminate a path token.
_PATH_TOKEN_STOP = r"""\s'"`;|&<>()"""
_PATH_TAIL = r"(?P<tail>(?:[/\\][^/\\" + _PATH_TOKEN_STOP + r"]*)+)"


@functools.lru_cache(maxsize=64)
def _home_prefix_fold_regex(path: str):
    """Compile a regex matching *path* as an absolute directory prefix.
    Components match with either separator so native Windows, forward-slash, and mixed forms all
    fold; the caller normalizes the tail's backslashes to ``/``. A non-empty tail is required, so a
    bare home is never folded. Returns ``None`` for an unset/degenerate path (fewer than two
    components: ``/``, ``C:\\``, ``""``) so a stray HOME cannot rewrite unrelated prefixes."""
    components = [c for c in re.split(r"[/\\]+", path) if c] if path else []
    if len(components) < 2:
        return None
    # Optional leading root separator; a Windows drive letter is a component.
    return re.compile(r"[/\\]*" + r"[/\\]+".join(re.escape(c) for c in components) + _PATH_TAIL)


def _fold_home_prefixes(command: str, paths, replacement: str) -> str:
    """Fold each resolved home prefix in *command* to *replacement* (no trailing separator; the tail
    supplies it). Longest first so a deeper home folds before a shorter overlapping one that would clobber it."""
    for path in dict.fromkeys(sorted((p for p in paths if p), key=len, reverse=True)):
        pattern = _home_prefix_fold_regex(path)
        if pattern is not None:
            command = pattern.sub(lambda m: replacement + m.group("tail").replace("\\", "/"), command)
    return command


def _rewrite_resolved_user_home(command: str) -> str:
    """User home (expanduser / realpath / $HOME) -> ``~/``; no-op when unset, degenerate, or unresolvable."""
    try:
        # expanduser, realpath, and an explicit HOME — Windows expanduser uses USERPROFILE, not HOME.
        home = os.path.expanduser("~")
        paths = [home, os.path.realpath(home), os.environ.get("HOME", "")]
    except Exception:
        return command
    return _fold_home_prefixes(command, paths, "~")


def _rewrite_resolved_hermes_home(command: str) -> str:
    """Resolved HERMES_HOME (and its realpath) -> ``~/.hermes/`` so the _HERMES_CONFIG_PATH /
    _HERMES_ENV_PATH rules match Docker/gateway deployments that spell the absolute path."""
    try:
        from hermes_constants import get_hermes_home
        home = get_hermes_home().expanduser()
        paths = [str(home), str(home.resolve(strict=False))]
    except Exception:
        return command
    return _fold_home_prefixes(command, paths, "~/.hermes")


_PARAM_REPLACEMENT_RE = re.compile(r"\$\{[^}/\s]+/[^}/]*/(?P<replacement>[^}]*)\}")
_PARAM_DEFAULT_RE = re.compile(r"\$\{[^}:}\s]+:-(?P<default>[^}]*)\}")
_SIMPLE_SHELL_LITERAL_RE = re.compile(r"^[A-Za-z0-9_./:@%+=,-]+$")
_ENV_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
_COMMAND_WRAPPER_WORDS = {"sudo", "doas", "env", "exec", "nohup", "setsid", "time", "command", "builtin",
                          "nice", "timeout", "stdbuf", "ionice", "chrt", "taskset", "chroot"}
_SUDO_OPTIONS_WITH_ARG = {"-c", "--close-from", "-g", "--group", "-h", "--host", "-p", "--prompt", "-u", "--user"}
# doas (OpenBSD) options that take a separate argument, for the same reason
# sudo's are listed: so a wrapped `-c/-e` payload extraction (below) does not
# mistake the option's own argument for the wrapped command word.
_DOAS_OPTIONS_WITH_ARG = {"-u", "-C", "-a"}
# Adapted from embwl0x's command-position work in #76063. Option operands are
# data, not executable positions; option spelling remains case-sensitive.
_COMMAND_WRAPPER_OPTIONS_WITH_ARG = {
    "chroot": {"--groups", "--userspec"},
    "sudo": _SUDO_OPTIONS_WITH_ARG,
    "doas": _DOAS_OPTIONS_WITH_ARG,
    "env": {"-a", "--argv0", "-C", "--chdir", "-S", "--split-string", "-u", "--unset"},
    "exec": {"-a"}, "nice": {"-n", "--adjustment"},
    "time": {"-f", "--format", "-o", "--output"},
    "timeout": {"-k", "--kill-after", "-s", "--signal"},
    "stdbuf": {"-e", "--error", "-i", "--input", "-o", "--output"},
    "ionice": {"-c", "--class", "-n", "--classdata"},
}
_COMMAND_WRAPPER_NON_EXECUTING_OPTIONS = {
    "command": {"-v", "-V"}, "chrt": {"-p", "--pid"},
    "ionice": {"-p", "--pid", "--pgid", "--uid"}, "taskset": {"-p", "--pid"},
}
_COMMAND_WRAPPER_POSITIONAL_ARGS = {"chroot": 1, "chrt": 1, "taskset": 1, "timeout": 1}
_SHELL_COMMAND_TRANSITIONS = {"if", "then", "else", "elif", "do", "while", "until", "!"}
_SHELL_REDIRECTION_RE = re.compile(r"(?:[0-9]+)?(?:>>|<<|<>|>&|<&|>\||[<>])")

_INTERPRETER_NAME_RES = tuple((family, re.compile(pattern)) for family, pattern in (
    ("python", r"py(?:\.exe)?|python[23]?(?:\.\d+)*(?:\.exe)?"), ("node", r"node(?:js)?(?:\.exe)?"),
    ("perl", r"perl[0-9]*(?:\.\d+)*(?:\.exe)?"), ("ruby", r"ruby[0-9.]*(?:\.exe)?"), ("php", r"php(?:\.exe)?"),
    ("powershell", r"powershell(?:\.exe)?|pwsh(?:\.exe)?"),
))
_INTERPRETER_EXEC_FLAGS = {
    "python": {"-c"}, "node": {"-e", "--eval", "-p", "--print"}, "perl": {"-e", "--eval"}, "ruby": {"-e"},
    "php": {"-r"}, "powershell": {"-command", "-c", "-file", "-f"},
}
_INTERPRETER_WITH_ARG = {
    "python": {"-W", "-X", "--check-hash-based-pycs"},
    "node": {"-C", "--conditions", "--cpu-prof-dir", "--diagnostic-dir", "--icu-data-dir", "--import", "--loader",
             "--openssl-config", "--require", "--title"},
    "perl": {"-0", "-F", "-I", "-M", "-m", "-x"}, "ruby": {"-C", "-E", "-F", "-I", "-K", "-r"},
    "php": {"-c", "-d", "-z"},
    "powershell": {"-configurationname", "-custompipename", "-executionpolicy", "-inputformat", "-outputformat",
                   "-settingsfile", "-version", "-windowstyle", "-workingdirectory"},
}
_READ_TOOL_EXEC_FLAGS = {
    "sort": {"--compress-program"}, "rg": {"--pre", "--hostname-bin"}, "ag": {"--pager"},
    "man": {"--pager", "--html", "-P", "-H"},
}
# Required-argument options are ownership boundaries: an option-looking next token is data, not another option. These
# sets mirror the invocation grammar of the supported binaries (ripgrep 14, GNU sort, man-db, and ag 2.2).
_READ_TOOL_LONG_OPTIONS_WITH_ARG = {
    "rg": {
        "--after-context", "--before-context", "--color", "--colors", "--context", "--context-separator",
        "--dfa-size-limit", "--encoding", "--engine", "--field-context-separator", "--field-match-separator",
        "--file", "--generate", "--glob", "--hostname-bin", "--hyperlink-format", "--iglob", "--ignore-file",
        "--max-columns", "--max-count", "--max-depth", "--max-filesize", "--path-separator", "--pre", "--pre-glob",
        "--regex-size-limit", "--regexp", "--replace", "--sort", "--sortr", "--threads", "--type", "--type-add",
        "--type-clear", "--type-not",
    },
    "sort": {
        "--batch-size", "--buffer-size", "--compress-program", "--field-separator", "--files0-from", "--key",
        "--output", "--parallel", "--random-source", "--sort", "--temporary-directory",
    },
    "man": {
        "--config-file", "--encoding", "--extension", "--locale", "--manpath", "--pager", "--preprocessor",
        "--prompt", "--recode", "--sections", "--systems",
    },
    "ag": {
        "--ackmate-dir-filter", "--color-line-number", "--color-match", "--color-path", "--depth",
        "--filename-pattern", "--file-search-regex", "--ignore", "--ignore-dir", "--max-count", "--pager",
        "--path-to-ignore", "--width", "--workers",
    },
}
_READ_TOOL_SHORT_OPTIONS_WITH_ARG = {
    "rg": frozenset("efEmjgdtTABCMr"), "sort": frozenset("koStT"), "man": frozenset("CRLmMSserEPp"),
    "ag": frozenset("gGmpW"),
}
_GREP_OPTIONS_WITH_ARG = {
    "--after-context", "--before-context", "--binary-files", "--context", "--directories", "--devices", "--exclude",
    "--exclude-dir", "--exclude-from", "--include", "--label", "--max-count", "--regexp", "--file",
}
_GREP_SHORT_OPTIONS_WITH_ARG = {"A", "B", "C", "D", "d", "e", "f", "m"}
_BASH_OPTIONS_WITH_ARG = {"-O", "+O", "-o", "+o", "--init-file", "--rcfile"}
_BASH_SHORT_OPTION_LETTERS = frozenset("ilrsDcabefhkmnptuvxBCEHPTOo")
_MAX_DETECTION_COMMAND_CHARS, _MAX_SEPARATOR_FREE_COMMAND_CHARS, _MAX_DETECTION_SEGMENTS = 128_000, 4_096, 25_000
_PARSER_LIMIT_DESCRIPTION = "command parser limit exceeded"
_MALFORMED_EXEC_DESCRIPTION = "command parser limit or malformed executable payload"
_GATEWAY_LIFECYCLE_SPLICE_DESCRIPTION = "stop/restart hermes gateway via shell-spliced verb (kills running agents)"


def _command_parser_limit_exceeded(command: str) -> bool:
    """Bound all parser work before normalization/tokenization. Separator counting is deliberately
    conservative: quoted separators over-count, but crossing the ceiling fails closed rather than
    letting an uninspected suffix execute."""
    if len(command) > _MAX_DETECTION_COMMAND_CHARS:
        return True
    # Long separator-free input has no compound-command utility and makes every regex inspect one giant token.
    if len(command) > _MAX_SEPARATOR_FREE_COMMAND_CHARS and not any(char in command for char in ";&|\n"):
        return True
    return sum(command.count(char) for char in ";&|\n") >= _MAX_DETECTION_SEGMENTS


def _backtick_end_from(segment: str, i: int) -> int | None:
    """Index of the backtick closing the one opened at ``i``, or None when none follows."""
    j = segment.find("`", i + 1)
    while j != -1 and segment[j - 1] == "\\":
        j = segment.find("`", j + 1)
    return None if j == -1 else j


def _shell_tokens_with_spans(segment: str, start: int):
    """Return shell words as ``(value, start, end, quoted)`` or ``None`` on malformed quoting.
    Deliberately small lexer that never expands shell syntax; it exists to keep source spans (which
    ``shlex`` does not expose) for deciding which quoted grep operand is data, not another command.

    Lexing stops at the end of the simple command that begins at *start*: an unquoted ``;``, ``|``,
    ``&`` or newline, or the ``)`` / backtick that closes the substitution the command sits inside.
    Without that, a grep nested as ``"$(grep … | cut …)"`` was lexed together with the enclosing
    command's closing quote, read as unbalanced quoting, and reported as a hardline block (546
    blocked turns in one run, every one a false positive; ``sed -n "$(grep -n X f | cut -d: -f1),+3p" f``
    is the canonical shape)."""
    tokens, value, token_start, quote = [], [], None, None
    depth = 0  # $(...) nesting opened AFTER start; a closer at depth 0 ends the enclosing substitution
    # A backtick opened AFTER start is an operand substitution (``grep -e `cmd` f``); the matching
    # closer belongs to it, not to an enclosing backtick the command might sit inside.
    in_backtick = False

    def flush(end: int) -> None:
        raw = segment[token_start:end]
        # Only a wholly single-quoted operand is inert shell data. Double quotes still execute $()
        # and backticks; unquoted substitutions do too.
        inert = (raw.startswith("'") and raw.endswith("'")) or ("='" in raw and raw.endswith("'"))
        tokens.append(("".join(value), token_start, end, inert))

    end_at = len(segment)
    for kind, i, _, _ in _scan_shell(segment, start):
        ch = segment[i]
        if kind == "char" and not quote:
            if ch.isspace() and ch != "\n":
                if token_start is not None:
                    flush(i)
                    value, token_start = [], None
                continue
            if segment.startswith("$(", i):
                depth += 1
            elif ch == "`":
                if in_backtick:
                    in_backtick = False
                elif depth == 0 and _backtick_end_from(segment, i) is None:
                    end_at = i  # unmatched: it closes the substitution this command sits inside
                    break
                else:
                    in_backtick = True
            elif ch == ")":
                if depth == 0:
                    end_at = i
                    break
                depth -= 1
            elif ch in ";|&\n":
                end_at = i
                break
        if token_start is None:
            token_start = i
        if kind == "quote":
            quote = None if quote else ch
        elif kind == "esc":
            value.append(segment[i + 1])
        elif ch == "\\" and not quote:
            return None  # dangling backslash
        else:
            value.append(ch)
    if quote:
        return None
    if token_start is not None:
        flush(end_at)
    return tokens


def _quoted_grep_pattern_spans(command: str) -> tuple[list[tuple[int, int]], bool]:
    """Structurally locate quoted grep PCRE operands -> (spans, malformed). On an ambiguous or
    malformed grep parse callers fail closed and use the original command: no text is hidden on
    an uncertain parse."""
    spans: list[tuple[int, int]] = []
    offset = 0
    for segment in _iter_top_level_shell_segments(command):
        segment_at = command.find(segment, offset)
        offset = segment_at + len(segment)
        for start, _, word in _iter_shell_command_word_spans(segment):
            if os.path.basename(_deobfuscate_shell_word_for_detection(word)).lower() not in {"grep", "egrep"}:
                continue
            tokens = _shell_tokens_with_spans(segment, start)
            if tokens is None:
                return [], True
            args, pattern_indexes = tokens[1:], []
            pcre = explicit_patterns = False
            operand_index, i, options = None, 0, True
            while i < len(args):
                token = args[i][0]
                if options and token == "--":
                    options = False
                elif options and token.startswith("--"):
                    option, equals, _ = token.partition("=")
                    pcre = pcre or option == "--perl-regexp"
                    explicit_patterns = explicit_patterns or option in {"--regexp", "--file"}
                    takes_next = option in _GREP_OPTIONS_WITH_ARG and not equals
                    if takes_next and i + 1 >= len(args):
                        return [], True
                    if option == "--regexp":
                        pattern_indexes.append(i + 1 if takes_next else i)
                    i += 1 if takes_next else 0
                elif options and token.startswith("-") and token != "-":
                    chars = token[1:]
                    for j, char in enumerate(chars):
                        pcre = pcre or char == "P"
                        explicit_patterns = explicit_patterns or char in {"e", "f"}
                        if char in _GREP_SHORT_OPTIONS_WITH_ARG:
                            # The first argument-taking short option owns the rest of the bundle,
                            # or the next token when it comes last.
                            attached = j + 1 < len(chars)
                            if not attached and i + 1 >= len(args):
                                return [], True
                            if char == "e":
                                pattern_indexes.append(i if attached else i + 1)
                            i += 0 if attached else 1
                            break
                elif operand_index is None:
                    operand_index = i
                i += 1
            if not explicit_patterns:
                if operand_index is None:
                    return [], pcre
                pattern_indexes.append(operand_index)
            if pcre:
                spans.extend(
                    (segment_at + token_start, segment_at + token_end)
                    for _, token_start, token_end, quoted in map(args.__getitem__, pattern_indexes) if quoted
                )
    return spans, False


def _splice(command: str, edits) -> str:
    """Apply sorted, non-overlapping ``(start, end, text)`` edits to *command* in one pass
    (re-slicing per edit is quadratic on 10k+ segments)."""
    parts, previous = [], 0
    for start, end, text in edits:
        parts.extend((command[previous:start], text))
        previous = end
    return "".join(parts) + command[previous:]


def _grep_safe_detection_variant(command: str) -> tuple[str, bool]:
    spans, malformed = _quoted_grep_pattern_spans(command)
    if malformed or not spans:
        return command, malformed
    return _splice(command, [(start, end, " " * (end - start)) for start, end in spans]), False


def _interpreter_family(executable: str) -> str | None:
    name = os.path.basename(executable).lower()
    return next((family for family, name_re in _INTERPRETER_NAME_RES if name_re.fullmatch(name)), None)


def _shell_segment_tokens(segment: str, start: int) -> list[str] | None:
    """Tokenize an already-bounded command segment. ``None`` distinguishes malformed quoting from
    an empty segment so callers can fail closed for a program-bearing option rather than silently
    skip it."""
    try:
        lexer = shlex.shlex(segment[start:], posix=True, punctuation_chars="<>")
        lexer.whitespace_split, lexer.commenters = True, ""
        return list(lexer)
    except ValueError:
        return None


def _iter_top_level_shell_segments(command: str):
    """Yield top-level command segments in one left-to-right pass."""
    start = 0
    for kind, i, j, quote in _scan_shell(command, comments=True):
        if kind == "comment" or (kind == "char" and quote is None and command[i] in ";&|\n"):
            if start < i:
                yield command[start:i]
            start = j
    if start < len(command):
        yield command[start:]


def _interpreter_exec_flag(family: str, args: list[str]) -> str | None:
    """Return an execution-bearing interpreter option, if present."""
    flags, with_arg = _INTERPRETER_EXEC_FLAGS[family], _INTERPRETER_WITH_ARG[family]
    powershell = family == "powershell"
    skip_value = False
    for token in args:
        if skip_value:
            skip_value = False
            continue
        if token == "--" or (not powershell and not token.startswith("-")):
            break
        option, equals, _ = token.partition("=")
        comparable = option.lower() if powershell else option
        if comparable in flags:
            return comparable
        # `-Wonce` and `ruby -rjson` attach an option value; they are not short-option bundles containing an execution
        # flag. PowerShell's normal long options also use one dash, so bundle parsing never applies to that family.
        has_attached_option_value = any(
            option.startswith(short) and len(option) > len(short)
            for short in with_arg if short.startswith("-") and not short.startswith("--")
        )
        if not powershell and not option.startswith("--") and len(option) > 2 and not has_attached_option_value:
            bundled = next((f"-{char}" for char in option[1:] if f"-{char}" in flags), None)
            if bundled:
                return bundled
        skip_value = comparable in with_arg and not equals
    return None


def _bash_exec_payload(args: list[str]) -> tuple[bool, str | None]:
    """Return whether Bash ``-c`` occurs and the command string it owns.
    Bash's O/o options consume the following argument even when they precede a later ``-c`` or
    share its short-option bundle; the two startup-file long options own their next token.
    Parsing those first prevents both missed payloads and false ``-c`` hits."""
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--" or not token.startswith(("-", "+")):
            break
        if token in _BASH_OPTIONS_WITH_ARG:
            index += 2
            continue
        chars = token[1:]
        # Bash option letters are case-sensitive; restricting to the documented alphabet
        # preserves invalid controls such as `-Wc`.
        if token.startswith("--") or not set(chars) <= _BASH_SHORT_OPTION_LETTERS:
            index += 1
            continue
        consumed_option_arg = int("O" in chars or "o" in chars)
        if "c" in chars:
            payload_index = index + 1 + consumed_option_arg
            return True, (args[payload_index] if payload_index < len(args) else None)
        index += 1 + consumed_option_arg
    return False, None


def _flock_exec_payload(args: list[str]) -> tuple[bool, str | None]:
    """Return whether flock's own ``-c`` occurs and the payload it owns.

    flock(1) ``flock FILE [-c COMMAND]`` runs COMMAND through a shell itself
    (``sh -c COMMAND``), so the payload is a shell string exactly like bash's
    ``-c``. ``-c/--command`` and ``--command=…`` both own their payload.
    Argument-taking flock options (``-w``, ``-E``, ``-c``, ``--start``,
    ``--length``) consume the following token, so they are skipped first to
    avoid mistaking their argument for the command word.
    """
    arg_options = {"-w", "--wait", "--timeout", "-E", "--conflict-exit-code",
                   "--start", "--length"}
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            break
        option, equals, payload = token.partition("=")
        payload = payload if equals else None
        if option in {"-c", "--command"}:
            if payload is None and index + 1 < len(args):
                payload = args[index + 1]
            return payload is not None, payload
        if option in arg_options and payload is None:
            index += 2
            continue
        index += 1
    return False, None


def _read_tool_exec_flag(tool: str, args: list[str]) -> tuple[str, str] | None:
    """Return (option, program) for a read-only tool's program-running flag."""
    flags = _READ_TOOL_EXEC_FLAGS[tool]
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            break
        option, equals, payload = token.partition("=")
        payload = payload if equals else None
        matched = option if option in flags else None
        if tool == "man" and token.startswith(("-P", "-H")) and len(token) > 2:
            matched, payload = token[:2], token[2:]
        if matched:
            if payload is None and index + 1 < len(args):
                payload = args[index + 1]
            # The option owns its program argument regardless of spelling; the real binaries
            # execute a '-'-prefixed payload rather than reparsing it.
            if payload:
                return matched, payload
            index += 2 if payload is not None and "=" not in token else 1
        elif option in _READ_TOOL_LONG_OPTIONS_WITH_ARG[tool] and payload is None:
            index += 2
        elif token.startswith("-") and not token.startswith("--") and len(token) > 1:
            # In a short bundle, the first argument-taking option owns the rest of the token, or
            # the following token when it occurs last.
            shorts = _READ_TOOL_SHORT_OPTIONS_WITH_ARG[tool]
            owner = next((k for k, char in enumerate(token[1:], start=1) if char in shorts), None)
            index += 2 if owner == len(token) - 1 else 1
        else:
            index += 1
    return None


def _execution_flag_findings(command: str):
    """Yield scoped execution mechanisms and any executable payloads."""
    for segment in _iter_top_level_shell_segments(command):
        for start, _, word in _iter_shell_command_word_spans(segment):
            executable = _deobfuscate_shell_word_for_detection(word)
            tokens = _shell_segment_tokens(segment, start)
            executable_name = os.path.basename(executable).lower()
            family = _interpreter_family(executable)
            if tokens is None:
                if family is not None or executable_name in _READ_TOOL_EXEC_FLAGS or executable_name == "flock":
                    yield (_MALFORMED_EXEC_DESCRIPTION, None)
                continue
            if not tokens:
                continue
            args = tokens[1:]
            if family and _interpreter_exec_flag(family, args):
                yield ("script execution via -e/-c flag", None)
            elif family and any(token.startswith("<<") for token in args):
                yield ("script execution via heredoc", None)
            else:
                if executable_name in {"bash", "sh", "zsh", "ksh"}:
                    found, payload = _bash_exec_payload(args)
                    if found:
                        yield ("shell command via -c/-lc flag", payload)
                if executable_name == "flock":
                    found, payload = _flock_exec_payload(args)
                    if found:
                        yield ("shell command via flock -c flag", payload)
                if executable_name in _READ_TOOL_EXEC_FLAGS:
                    finding = _read_tool_exec_flag(executable_name, args)
                    if finding:
                        yield (f"arbitrary program execution via {executable_name} {finding[0]}", finding[1])


def _skip_shell_whitespace(command: str, pos: int) -> int:
    while pos < len(command) and command[pos].isspace():
        pos += 1
    return pos


def _scan_shell(text: str, start: int = 0, end: int | None = None, *, subst: str = "",
                brace: bool = False, stop_unterminated: bool = False, naive_backtick: bool = False,
                comments: bool = False):
    """Yield ``(kind, i, j, quote)`` lexical steps over ``text[start:end]`` without expanding.

    The single quote/escape state machine behind every detection scanner. ``kind`` is ``"char"``
    (one char), ``"esc"`` (backslash + the char it escapes; never inside single quotes), ``"quote"``
    (an opening/closing quote char) or ``"subst"`` (a ``$(...)`` / backtick / ``${...}`` span);
    With *comments*, ``"comment"`` spans are skipped without interpreting their quote syntax.
    ``quote`` is the state the step was read in (``None``, ``'`` or ``"``). Substitutions are
    recognized unquoted when ``"u"`` is in *subst*, inside double quotes when ``"q"`` is; *brace*
    adds unquoted ``${...}``. An unterminated substitution falls through as plain chars unless
    *stop_unterminated*, which yields ``("subst", i, None, quote)`` and ends the scan (the caller
    descends to *end* itself). *naive_backtick* closes a backtick at the next backtick even if
    escaped (the quoted-prose masker's historical behavior)."""
    n = len(text) if end is None else end
    quote: str | None = None
    i = start
    while i < n:
        ch = text[i]
        kind, j = "char", i + 1
        if comments and quote is None and _is_shell_comment_start(text, i):
            kind, j = "comment", text.find("\n", i, n)
            if j < 0:
                j = n
        elif quote != "'" and ch == "\\" and i + 1 < n:
            kind, j = "esc", i + 2
        elif ch == quote or (quote is None and ch in "'\""):
            kind = "quote"
        elif quote != "'" and ("q" if quote else "u") in subst and (
            ch == "`" or text.startswith("$(", i) or (brace and not quote and text.startswith("${", i))
        ):
            if ch == "`":
                close = text.find("`", i + 1) + 1 or None if naive_backtick else _scan_backtick_end(text, i)
            elif text[i + 1] == "(":
                close = _scan_dollar_paren_end(text, i)
            else:
                close = text.find("}", i + 2) + 1 or None
            if close is not None:
                kind, j = "subst", close
            elif stop_unterminated:
                yield ("subst", i, None, quote)
                return
        yield (kind, i, j, quote)
        if kind == "quote":
            quote = None if quote else ch
        i = j


def _scan_dollar_paren_end(command: str, start: int) -> int | None:
    """Return the offset after a balanced ``$(...)`` command substitution."""
    depth = 1
    for kind, i, _, quote in _scan_shell(command, start + 2):
        if kind == "char" and not quote:
            depth += command.startswith("$(", i) - (command[i] == ")")
            if depth == 0:
                return i + 1
    return None


def _scan_backtick_end(command: str, start: int) -> int | None:
    # Backticks have no quote awareness: only a backslash escapes the next char.
    match = re.compile(r"(?:\\.|[^`\\])*`", re.DOTALL).match(command, start + 1)
    return match.end() if match else None


def _read_shell_word(command: str, pos: int) -> tuple[int, int, str]:
    """Read one shell word without executing expansions."""
    start = end = _skip_shell_whitespace(command, pos)
    for kind, i, j, quote in _scan_shell(command, start, subst="u", brace=True):
        if kind == "char" and quote is None and (command[i].isspace() or command[i] in ";&|<>()"):
            break
        end = j
    return (start, end, command[start:end])


def _literal_command_substitution_output(script: str) -> str | None:
    """Resolve tiny literal command substitutions without executing a shell."""
    try:
        tokens = shlex.split(script, posix=True)
    except ValueError:
        tokens = []
    if not tokens:
        return None
    command, args = tokens[0].lower(), tokens[1:]
    if command == "echo":
        while args and re.fullmatch(r"-[nEe]+", args[0]):
            args = args[1:]
    elif command != "printf":
        return None
    if len(args) == 1 and _SIMPLE_SHELL_LITERAL_RE.fullmatch(args[0]):
        return args[0]
    if command == "printf" and len(args) == 2 and args[0] == "%s" and _SIMPLE_SHELL_LITERAL_RE.fullmatch(args[1]):
        return args[1]
    return None


def _replace_simple_command_substitutions(word: str) -> str:
    chars: list[str] = []
    i = 0
    while i < len(word):
        opener = 2 if word.startswith("$(", i) else 1 if word[i] == "`" else 0
        end = (_scan_dollar_paren_end if opener == 2 else _scan_backtick_end)(word, i) if opener else None
        replacement = _literal_command_substitution_output(word[i + opener:end - 1]) if end is not None else None
        if replacement is None:
            replacement, end = word[i], i + 1
        chars.append(replacement)
        i = end
    return "".join(chars)


def _replace_simple_shell_expansions(word: str) -> str:
    word = _replace_simple_command_substitutions(word)
    word = _PARAM_REPLACEMENT_RE.sub(lambda match: match.group("replacement"), word)
    return _PARAM_DEFAULT_RE.sub(lambda match: match.group("default"), word)


def _strip_shell_word_syntax(word: str) -> str:
    return "".join(
        word[i + 1] if kind == "esc" else word[i]
        for kind, i, _, _ in _scan_shell(word) if kind != "quote"
    )


def _deobfuscate_shell_word_for_detection(word: str) -> str:
    """Approximate how shell syntax can spell a command word: collapses quoting/escaping plus
    simple literal command substitutions in the word itself. Intentionally narrow and non-executing."""
    for _ in range(2):
        previous = word
        word = _strip_shell_word_syntax(_replace_simple_shell_expansions(word))
        if word == previous:
            break
    return word


def _is_shell_comment_start(command: str, index: int) -> bool:
    return command[index] == "#" and (index == 0 or command[index - 1].isspace()
                                      or command[index - 1] in ";&|()<>")


def _iter_shell_command_starts(command: str):
    starts = [0]

    def scan(start: int, end: int) -> None:
        skip = -1
        for kind, i, j, quote in _scan_shell(command, start, end, subst="uq", stop_unterminated=True,
                                            comments=True):
            if kind == "subst":
                # Record a nested $(...)/backtick command start and scan its body.
                inner = i + (1 if command[i] == "`" else 2)
                starts.append(inner)
                scan(inner, end if j is None else j - 1)
            elif kind == "char" and quote is None and i != skip:
                # `${` opens a parameter expansion, not a brace group: a start marked inside it would
                # split `${IFS}` and defeat the IFS collapse in normalization.
                if command[i] in "({;\n" and not (command[i] == "{" and i > 0 and command[i - 1] == "$"):
                    starts.append(i + 1)
                elif command[i] in "&|":
                    repeated = i + 1 < end and command[i + 1] == command[i]
                    skip = i + 1 if repeated else skip
                    starts.append(i + 1 + repeated)

    scan(0, len(command))
    seen = set()
    for start in starts:
        start = _skip_shell_whitespace(command, start)
        if start >= len(command) or start in seen or _is_shell_comment_start(command, start):
            continue
        seen.add(start)
        yield start
        _, end, word = _read_shell_word(command, start)
        if word in _SHELL_COMMAND_TRANSITIONS:
            starts.append(end)


def _mark_command_starts(command: str, marker: str = "\n") -> str:
    """Insert *marker* (a newline) before each real (quote-aware) command start.
    ``\\n`` is already a ``_CMDPOS`` separator, so this exposes subshell ``(cmd)`` and brace-group
    ``{ cmd; }`` openers — which the flat pattern class omits — to the anchored patterns WITHOUT the
    quoted-prose false positives that adding ``(`` / ``{`` to ``_CMDPOS`` would cause: starts inside
    quotes are never produced, so ``--title "block (reboot)"`` is left as-is."""
    offsets = sorted(o for o in _iter_shell_command_starts(command) if o > 0)
    return _splice(command, [(o, o, marker) for o in offsets]) if offsets else command


def _mask_quoted_newlines(command: str) -> str:
    """Replace raw newlines inside single/double quotes with a space (detection-only).
    A quoted newline is DATA to the shell, yet the flat ``_CMDPOS`` class treats every raw ``\\n``
    as a command start, so multi-line quoted arguments (commit messages, heredoc text) tripped the
    hardline blocklist when a data line began with e.g. ``sudo reboot``. Quote tracking mirrors
    ``_iter_shell_command_starts``. Unquoted newlines pass through and ``_mark_command_starts``
    still re-inserts newlines at genuine starts; an unclosed quote absorbs following newlines
    exactly as the shell would, so masking them cannot hide a runnable command."""
    if "\n" not in command:
        return command
    return _mask_quoted_newlines_span(command, 0, len(command))


def _mask_quoted_newlines_span(command: str, start: int, end: int) -> str:
    """``_mask_quoted_newlines`` over ``command[start:end]``. A ``$(...)`` / backtick substitution
    inside double quotes is EXECUTABLE, not data: its body is re-scanned with a fresh quote state so a
    newline that separates commands inside it survives as a command boundary. Masking it as quoted
    data turned ``"$(grep x f\nreboot)"`` into ``... f reboot)``, which no later stage can tell from an
    operand; the pre-fix scanner only caught it by refusing the whole command as malformed."""
    out: list[str] = []
    for kind, i, j, quote in _scan_shell(command, start, end, subst="q"):
        if kind == "subst":
            # j is the index just past the closer; keep the opener and closer, recurse into the body.
            body_start = i + (2 if command.startswith("$(", i) else 1)
            body_end = j - 1
            out.append(command[i:body_start])
            out.append(_mask_quoted_newlines_span(command, body_start, body_end))
            out.append(command[body_end:j])
        elif quote and kind == "char" and command[i] == "\n":
            out.append(" ")
        else:
            out.append(command[i:j])
    return "".join(out)


def _iter_shell_command_word_spans(command: str):
    """Yield command-position words that may be executable names."""
    for pos in _iter_shell_command_starts(command):
        wrapper, positionals = None, 0
        options, skip_arg = True, False
        while pos < len(command):
            redirect = _SHELL_REDIRECTION_RE.match(command, _skip_shell_whitespace(command, pos))
            if redirect:
                _, pos, _ = _read_shell_word(command, redirect.end())
                continue
            word_start, word_end, word = _read_shell_word(command, pos)
            if word_start == word_end:
                break
            pos = word_end
            deobfuscated = _deobfuscate_shell_word_for_detection(word)
            name = os.path.basename(deobfuscated).lower()
            if skip_arg:
                skip_arg = False
                continue
            if wrapper and options and deobfuscated == "--":
                options = False
                continue
            if wrapper and options and deobfuscated.startswith("-"):
                option = deobfuscated.split("=", 1)[0]
                if wrapper == "env" and (option == "--split-string" or deobfuscated.startswith("-S")):
                    # The split string and remaining argv form ONE command, handled
                    # by _env_split_payload; the suffix is not a new executable.
                    break
                queries = _COMMAND_WRAPPER_NON_EXECUTING_OPTIONS.get(wrapper, set())
                if option in queries or (wrapper == "command" and not option.startswith("--")
                                         and set(option[1:]) & {"v", "V"}):
                    break
                skip_arg = "=" not in deobfuscated and option in _COMMAND_WRAPPER_OPTIONS_WITH_ARG.get(wrapper, set())
                continue
            if positionals:
                positionals -= 1
                continue
            if _ENV_ASSIGNMENT_RE.fullmatch(word):
                continue
            yield (word_start, word_end, word)
            if name not in _COMMAND_WRAPPER_WORDS:
                break
            wrapper, options = name, True
            positionals = _COMMAND_WRAPPER_POSITIONAL_ARGS.get(name, 0)


def _shell_command_segment(command: str, start: int) -> str:
    """Bound a candidate to its command, preserving quoted argument bytes."""
    end = len(command)
    for kind, i, _, quote in _scan_shell(command, start, subst="uq", brace=True, comments=True):
        if kind == "comment" or (kind == "char" and quote is None and command[i] in ";&|\n)`"):
            end = i
            break
    return command[start:end].strip()


def _split_env_string(payload: str) -> list[str] | None:
    r"""Project GNU env -S literal argv, not POSIX shell words.

    Dynamic ${NAME} expansion is deliberately not evaluated: the execution
    backend's environment need not be this process's environment.
    """
    escapes = {"f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v",
               "#": "#", "$": "$", "\"": "\"", "'": "'", "\\": "\\"}
    args, word = [], []
    quote, started, index = None, False, 0
    while index < len(payload):
        char = payload[index]
        index += 1
        if char == "\\":
            if index == len(payload):
                return None
            escaped = payload[index]
            if quote == "'" and escaped not in ("'", "\\"):
                word.append(char)
                started = True
                continue
            index += 1
            if escaped == "c":
                if quote:
                    return None
                break
            if escaped == "_" and quote is None:
                if started:
                    args.append("".join(word))
                word, started = [], False
                continue
            if escaped not in escapes and escaped != "_":
                return None
            word.append(" " if escaped == "_" else escapes[escaped])
            started = True
            continue
        if char in ("'", '"') and (quote is None or char == quote):
            quote = char if quote is None else None
            started = True
            continue
        if quote is None and char in " \t\n\r\v\f":
            if started:
                args.append("".join(word))
            word, started = [], False
            continue
        if quote is None and char == "#" and not started:
            break
        if char == "$" and quote != "'":
            return None
        word.append(char)
        started = True
    if quote:
        return None
    if started:
        args.append("".join(word))
    return args


def _env_split_payload(tokens: list[str]) -> str | None:
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--" or not token.startswith("-"):
            return None
        option, equals, value = token.partition("=")
        if option == "--split-string" or token.startswith("-S"):
            attached = equals if option == "--split-string" else len(token) > 2
            if not attached:
                index += 1
            payload = (value if option == "--split-string" else token[2:]) if attached else (
                tokens[index] if index < len(tokens) else "")
            args = _split_env_string(payload)
            # Protect literal separators when reusing command-position detection;
            # only a real shell -c carrier may turn these argv bytes into code.
            return shlex.join(args + tokens[index + 1:]) if args is not None else None
        index += 2 if not equals and option in _COMMAND_WRAPPER_OPTIONS_WITH_ARG["env"] else 1
    return None


def _deny_command_variants(command: str):
    """Add executable projections without reparsing normalized argument data.

    Whole-input matching is retained for existing globs. New projections parse
    the original quote state, preserve path-specific rules, and fold only the
    executable basename (never arbitrary argument paths).
    """
    yield from _command_detection_variants(command)
    pending, seen = [command], set()
    while pending:
        source = pending.pop()
        if source in seen:
            continue
        seen.add(source)
        for start, end, word in _iter_shell_command_word_spans(source):
            segment = _shell_command_segment(source, start)
            executable = _deobfuscate_shell_word_for_detection(word)
            tail = segment[end - start:]
            # Collapse only unquoted inter-word whitespace; quoted prose is data.
            parts = []
            for kind, i, j, quote in _scan_shell(tail):
                if kind == "char" and quote is None and tail[i].isspace():
                    if not parts or parts[-1] != " ":
                        parts.append(" ")
                else:
                    parts.append(tail[i:j])
            tail = "".join(parts)
            for name in dict.fromkeys((executable, os.path.basename(executable))):
                candidate = name + tail
                yield candidate
                # Apply the existing text matching semantics only AFTER locating
                # executable positions; never parse its rewritten quotes again.
                yield _normalize_command_for_detection(candidate)
            if os.path.basename(executable) == "env":
                tokens = _shell_segment_tokens(segment, 0)
                if tokens:
                    payload = _env_split_payload(tokens)
                    if payload:
                        pending.append(payload)
        for _, payload in _execution_flag_findings(source):
            if payload:
                pending.append(payload)


def _command_detection_variants(command: str):
    # Mask quoted newlines BEFORE normalization: normalization strips escapes (\" -> ") and ""
    # pairs, corrupting quote tracking (`echo "a\""` becomes an unterminated quote) so masking
    # afterwards could swallow a REAL unquoted newline separator. The raw command carries faithful quote state.
    normalized = _normalize_command_for_detection(_mask_quoted_newlines(command))
    # Quote-aware grep parsing hides only structurally identified pattern operands; malformed or
    # ambiguous input stays byte-for-byte intact.
    grep_safe, _ = _grep_safe_detection_variant(normalized)
    seen = {grep_safe}
    yield grep_safe

    def fresh(variant: str) -> bool:
        if not variant or variant in seen:
            return False
        seen.add(variant)
        return True

    # Windows-path variant: normalization strips backslashes as shell escapes, so `del C:\Users\me\.ssh\id_rsa`
    # reaches the patterns as `del C:Usersme.sshid_rsa`. When the RAW command has a drive-letter or UNC backslash
    # path, also yield a variant with backslashes flattened to `/` BEFORE normalization. Gated on a real path shape so
    # POSIX escape semantics (`echo a\"b`) are untouched elsewhere.
    # See #69472.
    if re.search(r"(?:[A-Za-z]:|\\\\)[\\\\]", command) or re.search(r"[A-Za-z]:\\", command):
        win_variant = _normalize_command_for_detection(_mask_quoted_newlines(command.replace("\\", "/")))
        if fresh(win_variant):
            yield win_variant
    # Program-bearing options are parsed in their owning command's context; surfacing only the payload lets the
    # hardline floor inspect what will actually run without promoting similar flags or quoted prose.
    pending = [normalized]
    while pending:
        for _, payload in _execution_flag_findings(pending.pop()):
            if fresh(payload):
                yield payload
                # A payload may start with an option-looking program and then invoke a hardline command
                # after a separator; mark its starts.
                marked_payload = _mark_command_starts(payload)
                if marked_payload != payload and fresh(marked_payload):
                    yield marked_payload
                pending.append(payload)
    # Subshell `(cmd)` / brace-group `{ cmd; }` openers put `cmd` at a real command position the flat `_CMDPOS`
    # patterns can't see (adding `(`/`{` there would match quoted prose like `--title "(reboot)"`). Insert a newline
    # at each start the QUOTE-AWARE tokenizer found instead; this covers every `_CMDPOS` rule in one place.
    marked = _mark_command_starts(grep_safe)
    if marked != grep_safe and fresh(marked):
        yield marked
    # Every variant above tracks quotes on NORMALIZED text, where `\"` has already become `"`. That
    # flips quote parity, so in `cat "f\"n.txt"; rm -rf /` the `; rm` start sat "inside" a phantom
    # quote, no start was marked, and the hardline floor let it through. Mark starts on the RAW
    # command (only quoted newlines masked), then normalize; the leading space keeps the marker
    # from being eaten as a `\<newline>` continuation when the preceding text ends in a backslash.
    faithful = _normalize_command_for_detection(_mark_command_starts(_mask_quoted_newlines(command), marker=" \n"))
    if fresh(faithful):
        yield faithful
    # Quoting/escaping can spell an executable in pieces (r\m, r''m). Keep that deobfuscation scoped
    # to command words so arguments don't false-positive.
    for word_start, word_end, word in _iter_shell_command_word_spans(normalized):
        deobfuscated = _deobfuscate_shell_word_for_detection(word)
        if deobfuscated and deobfuscated != word:
            variant = normalized[:word_start] + deobfuscated + normalized[word_end:]
            if fresh(variant):
                yield variant


def _is_verification_artifact_cleanup(command: str) -> bool:
    """Return whether *command* only removes one Hermes ad-hoc temp script."""
    try:
        argv = shlex.split(command, posix=True)
    except ValueError:
        return False
    if len(argv) != 3 or argv[0] != "rm" or argv[1] != "-f":
        return False
    operand = argv[2]
    temp_dir = os.path.realpath(tempfile.gettempdir())
    basename = os.path.basename(operand)
    return (
        operand == os.path.join(temp_dir, basename)
        and os.path.dirname(os.path.realpath(operand)) == temp_dir
        and re.fullmatch(r"hermes-(?:verify|ad-hoc)-[A-Za-z0-9_.-]+", basename) is not None
    )


def _is_shell_token_spliced_gateway_lifecycle(command: str) -> bool:
    """Catch gateway-lifecycle verbs spelled with quote splicing.
    Backslash splicing (``kick\\start``) is undone by normalization, but quote splicing is not:
    ``_deobfuscate_shell_word_for_detection`` is deliberately scoped to command-position words
    (widening it would let quoted prose like ``git commit -m "rm -rf /"`` match), and the spliced
    verb is an ARGUMENT, so ``launchctl kick"start" -k gui/501/ai.hermes.gateway`` auto-approved.
    Delegates to ``cron.lifecycle_guard`` (shlex-tokenized, anchored on a hermes-gateway
    identifier). Runs last so an ordinary pattern match keeps its more specific reason; this layer
    only prompts — the non-bypassable block still lives in ``cron.lifecycle_guard``.

    ``_normalize_command_for_detection`` strips backslash escapes, so ``kick\\start`` already reaches the
    launchctl pattern above. See #80269.
    """
    try:
        from cron.lifecycle_guard import contains_gateway_lifecycle_command
    except Exception:
        return False
    return contains_gateway_lifecycle_command(command)


def detect_dangerous_command(command: str) -> tuple:
    """Check dangerous patterns -> (is_dangerous, pattern_key, description)."""
    if _command_parser_limit_exceeded(command):
        return (True, _PARSER_LIMIT_DESCRIPTION, _PARSER_LIMIT_DESCRIPTION)
    if _is_verification_artifact_cleanup(command):
        return (False, None, None)
    for command_variant in _command_detection_variants(command):
        command_lower = command_variant.lower()
        for pattern_re, description in DANGEROUS_PATTERNS_COMPILED:
            if pattern_re.search(command_lower):
                return (True, description, description)
    normalized = _normalize_command_for_detection(command)
    for description, _ in _execution_flag_findings(normalized):
        return (True, description, description)
    if _is_shell_token_spliced_gateway_lifecycle(command):
        return (True, _GATEWAY_LIFECYCLE_SPLICE_DESCRIPTION, _GATEWAY_LIFECYCLE_SPLICE_DESCRIPTION)
    return (False, None, None)
