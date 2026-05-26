"""Dangerous command approval -- detection, prompting, and per-session state.

This module is the single source of truth for the dangerous command system:
- Pattern detection (DANGEROUS_PATTERNS, detect_dangerous_command)
- Per-session approval state (thread-safe, keyed by session_key)
- Approval prompting (CLI interactive + gateway async)
- Smart approval via auxiliary LLM (auto-approve low-risk commands)
- Permanent allowlist persistence (config.yaml)
"""

import contextvars
import fnmatch
import logging
import os
import re
import shlex
import sys
import threading
import time
import unicodedata
from typing import Optional
from hermes_cli.config import cfg_get

from utils import env_var_enabled, is_truthy_value

logger = logging.getLogger(__name__)

# Freeze YOLO mode at module import time. Reading os.environ on every call
# would allow any skill running inside the process to set this variable and
# instantly bypass all approval checks — a prompt-injection escalation path.
_YOLO_MODE_FROZEN: bool = is_truthy_value(os.getenv("HERMES_YOLO_MODE", ""))

# Per-thread/per-task gateway session identity.
# Gateway runs agent turns concurrently in executor threads, so reading a
# process-global env var for session identity is racy. Keep env fallback for
# legacy single-threaded callers, but prefer the context-local value when set.
_approval_session_key: contextvars.ContextVar[str] = contextvars.ContextVar(
    "approval_session_key",
    default="",
)


def _fire_approval_hook(hook_name: str, **kwargs) -> None:
    """Invoke a plugin lifecycle hook for the approval system.

    Lazy-imports the plugin manager to avoid circular imports (approval.py is
    imported very early, long before plugins are discovered). Never raises --
    plugin errors are logged and swallowed.

    Only fires for the two approval-specific hooks in VALID_HOOKS:
    pre_approval_request, post_approval_response.
    """
    try:
        from hermes_cli.plugins import invoke_hook
    except Exception:
        # Plugin system not available in this execution context
        # (e.g. bare tool-only imports, minimal test environments).
        return
    try:
        invoke_hook(hook_name, **kwargs)
    except Exception as exc:
        # invoke_hook() already swallows per-callback errors, so reaching here
        # means the dispatch layer itself failed. Log and move on -- approval
        # flow is safety-critical, plugin observability is not.
        logger.debug("Approval hook %s dispatch failed: %s", hook_name, exc)



def set_current_session_key(session_key: str) -> contextvars.Token[str]:
    """Bind the active approval session key to the current context."""
    return _approval_session_key.set(session_key or "")


def reset_current_session_key(token: contextvars.Token[str]) -> None:
    """Restore the prior approval session key context."""
    _approval_session_key.reset(token)


def get_current_session_key(default: str = "default") -> str:
    """Return the active session key, preferring context-local state.

    Resolution order:
    1. approval-specific contextvars (set by gateway before agent.run)
    2. session_context contextvars (set by _set_session_env)
    3. os.environ fallback (CLI, cron, tests)
    """
    session_key = _approval_session_key.get()
    if session_key:
        return session_key
    from gateway.session_context import get_session_env
    return get_session_env("HERMES_SESSION_KEY", default)


def _get_session_platform() -> str:
    """Return the current gateway platform from contextvars/env fallback."""
    try:
        from gateway.session_context import get_session_env

        return get_session_env("HERMES_SESSION_PLATFORM", "") or ""
    except Exception:
        return os.getenv("HERMES_SESSION_PLATFORM", "") or ""


def _is_gateway_approval_context() -> bool:
    """True when this call is inside a gateway/API session.

    Legacy gateway integrations set HERMES_GATEWAY_SESSION in process env.
    Newer concurrent gateway paths bind HERMES_SESSION_PLATFORM via
    contextvars so approval mode does not depend on process-global flags.

    Cron jobs are NEVER gateway-approval contexts even when they originate
    from a gateway platform (cron binds HERMES_SESSION_PLATFORM via
    contextvars for delivery routing). Cron approvals are governed by
    ``approvals.cron_mode`` config, not interactive resolve — letting cron
    fall through to the gateway branch would submit a pending approval
    with no listener and block the job indefinitely.
    """
    if env_var_enabled("HERMES_CRON_SESSION"):
        return False
    if env_var_enabled("HERMES_GATEWAY_SESSION"):
        return True
    return bool(_get_session_platform())

# Sensitive write targets that should trigger approval even when referenced
# via shell expansions like $HOME or $HERMES_HOME.
_SSH_SENSITIVE_PATH = r'(?:~|\$home|\$\{home\})/\.ssh(?:/|$)'
_HERMES_ENV_PATH = (
    r'(?:~\/\.hermes/|'
    r'(?:\$home|\$\{home\})/\.hermes/|'
    r'(?:\$hermes_home|\$\{hermes_home\})/)'
    r'\.env\b'
)
_PROJECT_ENV_PATH = r'(?:(?:/|\.{1,2}/)?(?:[^\s/"\'`]+/)*\.env(?:\.[^/\s"\'`]+)*)'
_PROJECT_CONFIG_PATH = r'(?:(?:/|\.{1,2}/)?(?:[^\s/"\'`]+/)*config\.yaml)'
_SHELL_RC_FILES = (
    r'(?:~|\$home|\$\{home\})/\.'
    r'(?:bashrc|zshrc|profile|bash_profile|zprofile)\b'
)
_CREDENTIAL_FILES = (
    r'(?:~|\$home|\$\{home\})/\.'
    r'(?:netrc|pgpass|npmrc|pypirc)\b'
)
# macOS: /etc, /var, /tmp, /home are symlinks to /private/{etc,var,tmp,home}.
# A command written to target /private/etc/sudoers works identically to
# /etc/sudoers on macOS but bypasses a plain "/etc/" pattern check. Match
# both forms. Inspired by Claude Code 2.1.113's "dangerous path protection".
_MACOS_PRIVATE_SYSTEM_PATH = r'/private/(?:etc|var|tmp|home)/'
# System-config paths that should trigger approval for any write/edit,
# collapsing /etc, its macOS /private/etc mirror, and /etc/sudoers.d/ into
# one shared fragment so new DANGEROUS_PATTERNS stay consistent.
_SYSTEM_CONFIG_PATH = (
    rf'(?:/etc/|{_MACOS_PRIVATE_SYSTEM_PATH})'
)
_SENSITIVE_WRITE_TARGET = (
    rf'(?:{_SYSTEM_CONFIG_PATH}|/dev/sd|'
    rf'{_SSH_SENSITIVE_PATH}|'
    rf'{_HERMES_ENV_PATH}|'
    rf'{_SHELL_RC_FILES}|'
    rf'{_CREDENTIAL_FILES})'
)
_PROJECT_SENSITIVE_WRITE_TARGET = rf'(?:{_PROJECT_ENV_PATH}|{_PROJECT_CONFIG_PATH})'
_COMMAND_TAIL = r'(?:\s*(?:&&|\|\||;).*)?$'

# =========================================================================
# Hardline (unconditional) blocklist
# =========================================================================
#
# Commands so catastrophic they should NEVER run via the agent, regardless
# of --yolo, /yolo, approvals.mode=off, or cron approve mode.  This is a
# floor below yolo: opting into yolo is the user trusting the agent with
# their files and services, not trusting it to wipe the disk or power the
# box off.
#
# Hardline only applies to environments that can actually damage the host
# (local, ssh, container-host cron).  Containerized backends (docker,
# singularity, modal, daytona) already bypass the dangerous-command layer
# because nothing they do can touch the host, so we leave that behavior
# alone.
#
# The list is deliberately tiny — only things with no recovery path:
# filesystem destruction rooted at /, raw block device overwrites, kernel
# shutdown/reboot, and denial-of-service commands that take the host down.
# Recoverable-but-costly operations (git reset --hard, rm -rf /tmp/x,
# chmod -R 777, curl|sh) stay in DANGEROUS_PATTERNS where yolo can pass
# them through — that's what yolo is for.
#
# Inspired by Mercury Agent's permission-hardened blocklist
# (https://github.com/cosmicstack-labs/mercury-agent).

# Regex fragment matching the *start* of a command (i.e. positions where
# a shell would begin parsing a new command).  Used by shutdown/reboot
# patterns so they don't fire on "echo reboot" or "grep 'shutdown' log".
# Matches: start of string, after command separators (; && || | newline),
# after subshell openers ( `$(` or backtick ), optionally consuming
# leading wrapper commands (sudo, env VAR=VAL, exec, nohup, setsid).
_CMDPOS = (
    r'(?:^|[;&|\n`]|\$\()'         # start position
    r'\s*'                          # optional whitespace
    r'(?:sudo\s+(?:-[^\s]+\s+)*)?'  # optional sudo with flags
    r'(?:env\s+(?:\w+=\S*\s+)*)?'   # optional env with VAR=VAL pairs
    r'(?:(?:exec|nohup|setsid|time)\s+)*'  # optional wrapper commands
    r'\s*'
)

HARDLINE_PATTERNS = [
    # rm recursive targeting the root filesystem or protected roots
    (r'\brm\s+(-[^\s]*\s+)*(/|/\*|/ \*)(\s|$)', "recursive delete of root filesystem"),
    (r'\brm\s+(-[^\s]*\s+)*(/home|/home/\*|/root|/root/\*|/etc|/etc/\*|/usr|/usr/\*|/var|/var/\*|/bin|/bin/\*|/sbin|/sbin/\*|/boot|/boot/\*|/lib|/lib/\*)(\s|$)', "recursive delete of system directory"),
    (r'\brm\s+(-[^\s]*\s+)*(~|\$HOME)(/?|/\*)?(\s|$)', "recursive delete of home directory"),
    # Filesystem format
    (r'\bmkfs(\.[a-z0-9]+)?\b', "format filesystem (mkfs)"),
    # Raw block device overwrites (dd + redirection)
    (r'\bdd\b[^\n]*\bof=/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*', "dd to raw block device"),
    (r'>\s*/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*\b', "redirect to raw block device"),
    # Fork bomb (classic shell form)
    (r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:', "fork bomb"),
    # Kill every process on the system
    (r'\bkill\s+(-[^\s]+\s+)*-1\b', "kill all processes"),
    # System shutdown / reboot — anchor to command position (start of line,
    # after a command separator, or after sudo/env wrappers) so we don't
    # false-positive on "echo reboot" or "grep 'shutdown' logs".
    # _CMDPOS matches start-of-command positions.
    (_CMDPOS + r'(shutdown|reboot|halt|poweroff)\b', "system shutdown/reboot"),
    (_CMDPOS + r'init\s+[06]\b', "init 0/6 (shutdown/reboot)"),
    (_CMDPOS + r'systemctl\s+(poweroff|reboot|halt|kexec)\b', "systemctl poweroff/reboot"),
    (_CMDPOS + r'telinit\s+[06]\b', "telinit 0/6 (shutdown/reboot)"),
]

# Pre-compiled variant used by the hot-path matcher. Building these at module
# load eliminates the ~2.6 ms cold-cache re.compile fan-out on the first
# terminal() call per process (12 HARDLINE + 47 DANGEROUS patterns, each
# potentially evicted from Python's 512-entry ``re._cache`` by unrelated
# regex work elsewhere in the agent). DANGEROUS_PATTERNS_COMPILED is built
# at the end of this module after DANGEROUS_PATTERNS is defined.
_RE_FLAGS = re.IGNORECASE | re.DOTALL
HARDLINE_PATTERNS_COMPILED = [
    (re.compile(pattern, _RE_FLAGS), description)
    for pattern, description in HARDLINE_PATTERNS
]


# =========================================================================
# Sudo stdin guard — block password guessing via "sudo -S"
# =========================================================================
# When SUDO_PASSWORD is not configured, any explicit "sudo -S" in the
# command is the LLM piping a guessed password via stdin.  This is a
# brute-force attack vector: the model iterates through candidate
# passwords, inspects sudo's "Sorry, try again" output, and refines.
# Treat this as an unconditional block — there is never a legitimate
# reason for the agent to pipe passwords to sudo -S when no password
# has been configured.
_SUDO_STDIN_RE = re.compile(
    # Command-position sudo invocation with any stdin-reading form:
    #   sudo -S, sudo -nS, sudo -Sn, sudo --stdin
    # Stop at shell separators so a quoted/later mention does not bleed into
    # the current sudo command. The input is normalized/lowercased before use.
    r'(?:^|[;&|`\n]|&&|\|\||\$\()\s*sudo\b[^;&|`\n]*?(?:--stdin\b|-[a-z]*s[a-z]*\b)',
    re.IGNORECASE)


def _extract_static_shell_stdin_payloads(command: str) -> list[str]:
    """Return statically visible payloads fed to shell/interpreter stdin.

    Conservative parser for literal echo/printf pipelines plus here-strings and
    heredocs.  It does not emulate shell expansion; it only unwraps payload text
    that is already present in the command.
    """
    raw = _normalize_command_for_detection(command or "")
    payloads: list[str] = []
    stdin_consumers = r"(?:bash|sh|zsh|ksh|dash|python|python3|node|nodejs|ruby|perl)"

    # Generic pipeline parser for: echo payload | sh, printf '%s\n' payload | python3.
    # Split pipes with quote awareness first; regexes such as [^;&|] miss
    # semicolons inside quoted interpreter payloads.
    pipe_segments: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    escaped = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if escaped:
            buf.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\" and quote != "'":
            buf.append(ch)
            escaped = True
            i += 1
            continue
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in {'"', "'"}:
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "|" and not raw.startswith("||", i) and (i == 0 or raw[i - 1] != ">"):
            pipe_segments.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    pipe_segments.append("".join(buf).strip())

    for left, right in zip(pipe_segments, pipe_segments[1:]):
        right_words = _shlex_words(right)
        if not right_words or not re.fullmatch(stdin_consumers, _command_basename(right_words[0]), _RE_FLAGS):
            continue
        words = _shlex_words(left)
        if not words:
            continue
        cmd = _command_basename(words[0]).lower()
        if cmd == "echo":
            idx = 1
            while idx < len(words) and re.fullmatch(r"-[A-Za-z]+", words[idx]):
                idx += 1
            if idx < len(words):
                payloads.append(" ".join(words[idx:]))
        elif cmd == "printf":
            idx = 1
            while idx < len(words) and words[idx].startswith("--"):
                idx += 1
            if idx >= len(words):
                continue
            fmt = words[idx]
            args = words[idx + 1:]
            if args:
                # For guard purposes, the argument text is the dangerous part in
                # common `printf '%s\n' payload | sh/python` forms.
                payloads.append("\n".join(args) if "%" in fmt else " ".join(args))
            else:
                payloads.append(fmt)

    for match in re.finditer(
        rf"\b(?:\S*/)?{stdin_consumers}\b[^;&|`\n]*<<<\s*(?P<q>['\"])(?P<payload>.*?)(?P=q)",
        raw,
        _RE_FLAGS,
    ):
        payloads.append(match.group("payload"))
    heredoc_re = re.compile(
        rf"\b(?:\S*/)?{stdin_consumers}\b[^\n]*<<-?\s*['\"]?(?P<tag>[A-Za-z_][A-Za-z0-9_]*)['\"]?\n(?P<body>.*?)\n(?P=tag)(?:\n|$)",
        _RE_FLAGS,
    )
    payloads.extend(match.group("body") for match in heredoc_re.finditer(raw))
    return [payload for payload in payloads if payload.strip()]


def _shell_c_payloads_from_command(command: str) -> list[str]:
    """Return statically visible payloads passed to shell -c after wrappers."""
    payloads: list[str] = []
    normalized = _normalize_command_for_detection(command or "")
    for segment in _split_command_segments(normalized):
        words = _shlex_words(segment)
        if not words:
            continue
        for idx in _command_word_indices(words):
            if _command_basename(words[idx]).lower() in {"bash", "sh", "zsh", "ksh", "dash"}:
                payload = _shell_c_payload_from_words(words, idx)
                if payload:
                    payloads.append(payload)
    return payloads


def _eval_payloads_from_command(command: str) -> list[str]:
    """Return simple literal payloads passed to shell eval."""
    payloads: list[str] = []
    normalized = _normalize_command_for_detection(command or "")
    for segment in _split_command_segments(normalized):
        words = _shlex_words(segment)
        if not words:
            continue
        for idx in _command_word_indices(words):
            if _command_basename(words[idx]).lower() == "eval" and idx + 1 < len(words):
                payloads.append(" ".join(words[idx + 1:]))
    return [payload for payload in payloads if payload.strip()]


def _env_split_string_payloads_from_command(command: str) -> list[str]:
    """Return statically visible `/usr/bin/env -S/--split-string` payloads.

    Keep the original option spelling while matching the `env` command word
    case-insensitively. This prevents callers that need exact flag case (notably
    the sudo-stdin guard) from losing `-S` by lowercasing the whole command.
    """
    payloads: list[str] = []
    normalized = _normalize_command_for_detection(command or "")
    for segment in _split_command_segments(normalized):
        words = _shlex_words(segment)
        if not words:
            continue
        for idx in _command_word_indices(words):
            if _command_basename(words[idx]).lower() == "env":
                payloads.extend(_env_split_string_payloads(words, idx))
    return [payload for payload in payloads if payload.strip()]


def _check_sudo_stdin_guard(command: str) -> tuple:
    """Detect ``sudo -S`` (stdin password) without configured SUDO_PASSWORD.

    When SUDO_PASSWORD is set, ``_transform_sudo_command`` injects ``-S``
    internally — that path is legitimate and handled elsewhere.  This guard
    only fires when SUDO_PASSWORD is *not* set, meaning the LLM explicitly
    wrote ``sudo -S`` to pipe a guessed password.

    Returns:
        (is_blocked: bool, description: str | None)
    """
    if "SUDO_PASSWORD" in os.environ:
        return (False, None)
    normalized_preserving_case = _normalize_command_for_detection(command)
    normalized = normalized_preserving_case.lower()
    if _SUDO_STDIN_RE.search(normalized):
        return (True, "sudo password guessing via stdin (sudo -S)")
    for payload in (
        _shell_c_payloads_from_command(normalized_preserving_case)
        + _extract_static_shell_stdin_payloads(normalized_preserving_case)
        + _eval_payloads_from_command(normalized_preserving_case)
        + _env_split_string_payloads_from_command(normalized_preserving_case)
    ):
        is_blocked, desc = _check_sudo_stdin_guard(payload)
        if is_blocked:
            return (True, desc)
    return (False, None)


def detect_hardline_command(command: str) -> tuple:
    """Check if a command matches the unconditional hardline blocklist.

    Returns:
        (is_hardline, description) or (False, None)
    """
    normalized = _normalize_command_for_detection(command)
    normalized_lower = normalized.lower()
    for pattern_re, description in HARDLINE_PATTERNS_COMPILED:
        if pattern_re.search(normalized_lower):
            return (True, description)
    structural = _detect_hardline_structural(normalized)
    if structural:
        return (True, structural)
    return (False, None)




def _detect_hardline_structural(normalized: str) -> Optional[str]:
    """Detect hardline commands after simple wrapper/shell unwrapping.

    Regexes above intentionally stay fast and broad; this structural pass closes
    bypasses such as ``env -i reboot``, ``command reboot``, ``time -p reboot``,
    and ``bash -c 'reboot'`` without trying to execute or expand the command.
    """
    shutdown_verbs = {"reboot", "shutdown", "halt", "poweroff", "kexec"}
    for payload in _extract_static_shell_stdin_payloads(normalized) + _eval_payloads_from_command(normalized):
        nested = _detect_hardline_structural(_normalize_command_for_detection(payload))
        if nested:
            return nested
    for segment in _split_command_segments(normalized):
        words = _shlex_words(segment)
        if not words:
            continue
        for idx in _command_word_indices(words):
            cmd_word = _command_basename(words[idx]).lower()
            if cmd_word in shutdown_verbs:
                return f"{cmd_word} command"
            if cmd_word == "rm":
                args = words[idx + 1:]
                has_recursive = any(w.startswith("-") and "r" in w for w in args)
                if has_recursive:
                    for arg in args:
                        if arg.startswith("-"):
                            continue
                        expanded = _expand_protected_path_token(arg).lower()
                        if expanded in {"~", os.path.expanduser("~").lower()}:
                            return "recursive delete of home directory"
            if cmd_word in {"bash", "sh", "zsh", "ksh", "dash"}:
                payload = _shell_c_payload_from_words(words, idx)
                if payload:
                    nested = _detect_hardline_structural(_normalize_command_for_detection(payload))
                    if nested:
                        return nested
            if cmd_word == "env":
                for payload in _env_split_string_payloads(words, idx):
                    nested = _detect_hardline_structural(_normalize_command_for_detection(payload))
                    if nested:
                        return nested
    return None


def _hardline_block_result(description: str) -> dict:
    """Build the standard block result for a hardline match."""
    return {
        "approved": False,
        "hardline": True,
        "message": (
            f"BLOCKED (hardline): {description}. "
            "This command is on the unconditional blocklist and cannot "
            "be executed via the agent — not even with --yolo, /yolo, "
            "approvals.mode=off, or cron approve mode. If you genuinely "
            "need to run it, run it yourself in a terminal outside the "
            "agent."
        ),
    }


def _sudo_stdin_block_result(description: str) -> dict:
    """Build the standard block result for sudo stdin guard."""
    return {
        "approved": False,
        "message": (
            f"BLOCKED: {description}. "
            "Do not pipe passwords to 'sudo -S' — this is a brute-force "
            "attack vector. Set SUDO_PASSWORD in your .env file if the "
            "agent needs passwordless sudo, or run the sudo command "
            "manually in your own terminal."
        ),
    }


# =========================================================================
# Protected Hermes config guard — unbypassable direct-write block
# =========================================================================
# The supported mutation path for Hermes configuration is `hermes config set`.
# Direct shell writes to ~/.hermes/.env or ~/.hermes/config.yaml are blocked
# below yolo/approvals.mode=off so rules cannot be bypassed by a convenience
# mode. This guard does not inspect or log secret values; it only detects the
# command shape and protected path names.

def _protected_hermes_path_pattern() -> str:
    """Return a regex matching active/default Hermes .env and config.yaml paths."""
    symbolic_roots = [
        r'~\/\.hermes',
        r'\$home\/\.hermes',
        r'\$\{home\}\/\.hermes',
        r'\$hermes_home',
        r'\$\{hermes_home\}',
    ]

    literal_roots = {os.path.expanduser("~/.hermes")}
    active_home = os.getenv("HERMES_HOME")
    if active_home:
        literal_roots.add(os.path.expanduser(active_home))
    try:
        from hermes_constants import get_default_hermes_root

        literal_roots.add(str(get_default_hermes_root()))
    except Exception:
        pass

    literal_paths = []
    for root in literal_roots:
        root = os.path.realpath(root).lower()
        literal_paths.extend([
            re.escape(os.path.join(root, ".env")),
            re.escape(os.path.join(root, "config.yaml")),
        ])

    symbolic = rf'(?:{"|".join(symbolic_roots)})\/(?:\.env|config\.yaml)'
    literal = rf'(?:{"|".join(sorted(set(literal_paths)))})' if literal_paths else r'(?!)'
    return rf'(?:{symbolic}|{literal})(?=$|[\s"\'`,;&|<>)])'


def _normalize_protected_config_command(command: str) -> str:
    """Normalize shell path variants before protected config matching."""
    normalized = _normalize_command_for_detection(command).lower()
    normalized = normalized.replace('"', "").replace("'", "")
    # Shell backslashes can escape ordinary path characters
    # (``~/\.hermes/con\fig.yaml`` resolves to ``~/.hermes/config.yaml``).
    normalized = re.sub(r"\\([^\s])", r"\1", normalized)
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    while "/./" in normalized:
        normalized = normalized.replace("/./", "/")
    while re.search(r"/[^/\s;&|<>]+/\.\./", normalized):
        normalized = re.sub(r"/[^/\s;&|<>]+/\.\./", "/", normalized)
    return normalized


def _protected_config_resolved_paths() -> set[str]:
    """Return resolved protected Hermes config/env paths for path-token checks."""
    roots = {os.path.expanduser("~/.hermes")}
    active_home = os.getenv("HERMES_HOME")
    if active_home:
        roots.add(os.path.expanduser(active_home))
    try:
        from hermes_constants import get_default_hermes_root

        roots.add(str(get_default_hermes_root()))
    except Exception:
        pass
    protected = set()
    for root in roots:
        root = os.path.realpath(root)
        protected.add(os.path.realpath(os.path.join(root, ".env")).lower())
        protected.add(os.path.realpath(os.path.join(root, "config.yaml")).lower())
    return protected


def _expand_protected_path_token(token: str) -> str:
    """Expand shell-ish path tokens enough for protected-path comparison."""
    token = (token or "").strip().strip("() ").strip('"\'')
    # Bash ANSI-C quoted path fragments: ~/.hermes/$'config.yaml'
    token = re.sub(r"\$'([^']*)'", r"\1", token)
    token = token.replace("/$config.yaml", "/config.yaml").replace("/$.env", "/.env")
    # Deterministic brace expansion used in write targets such as
    # ``~/.hermes/{config.yaml}``.  Leave multi-value/glob braces unresolved
    # so they fail closed through the wildcard guard below rather than being
    # mistaken for a literal safe path.
    token = re.sub(r"\{(config\.yaml|\.env)\}", r"\1", token, flags=re.IGNORECASE)
    token = re.sub(r"\\([^\s])", r"\1", token)
    home = os.path.expanduser("~")
    hermes_home = os.path.expanduser(os.getenv("HERMES_HOME") or "~/.hermes")
    replacements = (
        (r"\$\{home\}|\$home", home),
        (r"\$\{hermes_home\}|\$hermes_home", hermes_home),
    )
    for pattern, new in replacements:
        token = re.sub(pattern, new, token, flags=re.IGNORECASE)
    return os.path.expanduser(token)


def _expand_known_shell_vars(token: str, shell_vars: dict[str, str]) -> str:
    """Expand simple shell variables assigned earlier in the same command."""
    if not shell_vars:
        return token

    def replace_braced(match: re.Match) -> str:
        return shell_vars.get((match.group(1) or "").lower(), match.group(0) or "")

    def replace_plain(match: re.Match) -> str:
        return shell_vars.get((match.group(1) or "").lower(), match.group(0) or "")

    token = re.sub(r"\$\{([A-Za-z_]\w*)\}", replace_braced, token)
    token = re.sub(r"\$([A-Za-z_]\w*)", replace_plain, token)
    return token


def _resolve_protected_path_token(token: str, cwd: Optional[str] = None) -> Optional[str]:
    """Resolve a candidate shell path token without reading the path."""
    token = _expand_protected_path_token(token)
    if not token or any(ch in token for ch in "*?[]{}"):
        return None
    if not os.path.isabs(token):
        if not cwd:
            return None
        token = os.path.join(cwd, token)
    return os.path.realpath(token).lower()


def _token_glob_matches_protected_path(token: str, cwd: Optional[str]) -> bool:
    """Return True when a shell glob token can match a protected config path."""
    expanded = _expand_protected_path_token(token)
    if not expanded or not any(ch in expanded for ch in "*?[]"):
        return False
    if "{" in expanded or "}" in expanded:
        return False
    if not os.path.isabs(expanded):
        if not cwd:
            return False
        expanded = os.path.join(cwd, expanded)
    pattern = os.path.realpath(expanded).lower()
    return any(fnmatch.fnmatch(path, pattern) for path in _protected_config_resolved_paths())


def _infer_cd_cwd(normalized_command: str) -> Optional[str]:
    """Infer obvious shell cwd changes before relative write checks.

    This intentionally covers only simple, deterministic forms; it is a
    pre-exec guard, not a full shell interpreter.  Supported separators include
    `&&`, `;`, and the common defensive `cd PATH || exit; ...` shape.
    """
    cwd = None
    cd_pattern = r"(?:^|[;&|]\s*)cd\s+(?:--\s+)?([^\s;&|]+)\s*(?=&&|;|\|\|\s*exit\s*;)"
    for match in re.finditer(cd_pattern, normalized_command):
        cwd = _resolve_protected_path_token(match.group(1))
    return cwd


def _split_command_segments(normalized_command: str) -> list[str]:
    """Split shell command separators while preserving quoted literals."""
    segments = []
    buf: list[str] = []
    quote: str | None = None
    escaped = False
    i = 0
    while i < len(normalized_command):
        ch = normalized_command[i]
        if escaped:
            buf.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\" and quote != "'":
            buf.append(ch)
            escaped = True
            i += 1
            continue
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in {'"', "'"}:
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if normalized_command.startswith(("&&", "||"), i):
            cleaned = "".join(buf).strip()
            if cleaned:
                segments.append(cleaned)
            buf = []
            i += 2
            continue
        if ch == ";" or (ch == "|" and (i == 0 or normalized_command[i - 1] != ">")):
            cleaned = "".join(buf).strip()
            if cleaned:
                segments.append(cleaned)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    cleaned = "".join(buf).strip()
    if cleaned:
        segments.append(cleaned)
    return segments


def _shlex_words(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _command_basename(word: str) -> str:
    """Normalize a shell command word for guard verb comparisons."""
    return os.path.basename((word or "").lstrip("({"))


def _is_inline_script_interpreter(cmd_word: str) -> bool:
    """Return True for interpreter command names that support inline code."""
    cmd_word = _command_basename(cmd_word).lower()
    return bool(
        re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", cmd_word)
        or re.fullmatch(r"perl(?:\d+(?:\.\d+)?)?", cmd_word)
        or re.fullmatch(r"ruby(?:\d+(?:\.\d+)?)?", cmd_word)
        or re.fullmatch(r"node(?:js)?(?:\d+(?:\.\d+)?)?", cmd_word)
    )


def _is_shell_assignment(word: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_]\w*=.*", word or ""))


def _command_word_indices(words: list[str]) -> list[int]:
    """Return word indexes that are actual command positions in a simple segment."""
    indices: list[int] = []
    idx = 0
    while idx < len(words) and _is_shell_assignment(words[idx]):
        idx += 1
    while idx < len(words):
        if idx not in indices:
            indices.append(idx)
        cmd_word = _command_basename(words[idx]).lower()
        if cmd_word == "env":
            j = idx + 1
            while j < len(words):
                word = words[j]
                if word == "--":
                    j += 1
                    break
                if _is_shell_assignment(word):
                    j += 1
                    continue
                # Upstream `.lower()` collapses env's short flags (-S -> -s,
                # -C -> -c). env defines no lowercase -s/-c of its own, so
                # matching value-taking flags case-insensitively keeps the
                # split-string / chdir argument from being mistaken for the
                # wrapped command word.
                lower = word.lower()
                if lower in {"-i", "-0", "--ignore-environment", "--null"}:
                    j += 1
                    continue
                if lower in {"-u", "--unset", "-c", "--chdir", "-s", "--split-string"} and j + 1 < len(words):
                    j += 2
                    continue
                if any(lower.startswith(prefix) for prefix in ("--unset=", "--chdir=", "--split-string=")):
                    j += 1
                    continue
                if word.startswith("-"):
                    j += 1
                    continue
                break
            if j < len(words):
                idx = j
                continue
        if cmd_word == "sudo":
            j = idx + 1
            value_options = {"-u", "--user", "-g", "--group", "-h", "--host", "-p", "--prompt", "-C", "--close-from", "-T", "--command-timeout"}
            while j < len(words):
                word = words[j]
                if word == "--":
                    j += 1
                    break
                if word in value_options and j + 1 < len(words):
                    j += 2
                    continue
                if any(word.startswith(opt + "=") for opt in value_options if opt.startswith("--")):
                    j += 1
                    continue
                if word.startswith("-"):
                    j += 1
                    continue
                break
            if j < len(words):
                idx = j
                continue
        if cmd_word == "time":
            j = idx + 1
            while j < len(words) and words[j].startswith("-"):
                j += 1
            if j < len(words):
                idx = j
                continue
        if cmd_word == "nice":
            j = idx + 1
            while j < len(words):
                word = words[j]
                if word in {"-n", "--adjustment"} and j + 1 < len(words):
                    j += 2
                    continue
                if word.startswith("--adjustment=") or re.fullmatch(r"-\d+", word):
                    j += 1
                    continue
                if word.startswith("-") and word != "-":
                    j += 1
                    continue
                break
            if j < len(words):
                idx = j
                continue
        if cmd_word == "stdbuf":
            j = idx + 1
            while j < len(words):
                word = words[j]
                if word in {"-i", "-o", "-e"} and j + 1 < len(words):
                    j += 2
                    continue
                if re.fullmatch(r"-[ioe].+", word) or any(word.startswith(prefix) for prefix in ("--input=", "--output=", "--error=")):
                    j += 1
                    continue
                if word.startswith("-") and word != "-":
                    j += 1
                    continue
                break
            if j < len(words):
                idx = j
                continue
        if cmd_word in {"command", "builtin", "exec", "nohup", "setsid"} and idx + 1 < len(words):
            idx += 1
            continue
        break
    return indices


def _redirection_targets_from_words(words: list[str]) -> list[str]:
    """Return shell redirection targets from shlex words without scanning quotes."""
    targets: list[str] = []
    redirect_ops = {">", ">>", ">|", "&>", "&>>"}
    for idx, word in enumerate(words):
        if word in redirect_ops or re.fullmatch(r"\d*(?:>>?|>\|)", word):
            if idx + 1 < len(words):
                targets.append(words[idx + 1])
            continue
        match = re.fullmatch(r"(?:\d*(?:>>?|>\|)|&>>?)(.+)", word)
        if match:
            targets.append(match.group(1))
    return targets


def _dd_of_targets_from_words(words: list[str]) -> list[str]:
    """Return dd(1) output-file targets from shlex words."""
    return [word.split("=", 1)[1] for word in words if word.startswith("of=") and len(word) > 3]


def _token_is_protected_path(token: str, cwd: Optional[str]) -> bool:
    resolved = _resolve_protected_path_token(token, cwd)
    return bool((resolved and resolved in _protected_config_resolved_paths()) or _token_glob_matches_protected_path(token, cwd))


def _token_or_alias_is_protected_path(token: str, cwd: Optional[str], aliases: set[str]) -> bool:
    resolved = _resolve_protected_path_token(token, cwd)
    return bool(
        (resolved and (resolved in aliases or resolved in _protected_config_resolved_paths()))
        or _token_glob_matches_protected_path(token, cwd)
    )


def _protected_config_root_paths() -> set[str]:
    roots = {os.path.expanduser("~/.hermes")}
    active_home = os.getenv("HERMES_HOME")
    if active_home:
        roots.add(os.path.expanduser(active_home))
    try:
        from hermes_constants import get_default_hermes_root

        roots.add(str(get_default_hermes_root()))
    except Exception:
        pass
    return {os.path.realpath(root).lower() for root in roots}


def _token_is_protected_config_root(token: str, cwd: Optional[str]) -> bool:
    resolved = _resolve_protected_path_token(token, cwd)
    return bool(resolved and resolved in _protected_config_root_paths())


def _source_basename_is_protected_config(token: str) -> bool:
    return os.path.basename((token or "").strip().strip("{}() ").strip('"\'')).lower() in {"config.yaml", ".env"}


def _copy_like_operands(args: list[str]) -> tuple[list[str], Optional[str]]:
    operands: list[str] = []
    target_dir: Optional[str] = None
    options_with_values = {
        "-t", "--target-directory", "-m", "--mode", "-o", "--owner", "-g", "--group",
        "-e", "--rsh", "--exclude", "--include", "--filter",
    }
    short_options_with_values = {"t", "m", "o", "g", "e"}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--target-directory="):
            target_dir = arg.split("=", 1)[1]
            i += 1
            continue
        if arg.startswith("--") and "=" in arg:
            i += 1
            continue
        if arg.startswith("-t") and len(arg) > 2:
            target_dir = arg[2:]
            i += 1
            continue
        if arg in {"-t", "--target-directory"} and i + 1 < len(args):
            target_dir = args[i + 1]
            i += 2
            continue
        if arg.startswith("-") and not arg.startswith("--") and len(arg) > 2:
            consumed_next = False
            cluster = arg[1:]
            for pos, opt in enumerate(cluster):
                if opt not in short_options_with_values:
                    continue
                value = cluster[pos + 1:]
                if opt == "t":
                    if value:
                        target_dir = value
                    elif i + 1 < len(args):
                        target_dir = args[i + 1]
                        consumed_next = True
                elif not value and i + 1 < len(args):
                    consumed_next = True
                break
            i += 2 if consumed_next else 1
            continue
        if arg in options_with_values and i + 1 < len(args):
            i += 2
            continue
        if arg.startswith("-"):
            i += 1
            continue
        operands.append(arg)
        i += 1
    return operands, target_dir


def _copy_like_hits_protected_path(args: list[str], cwd: Optional[str], aliases: set[str] | None = None) -> bool:
    aliases = aliases or set()
    operands, target_dir = _copy_like_operands(args)
    if target_dir:
        return _token_is_protected_config_root(target_dir, cwd) and any(_source_basename_is_protected_config(src) for src in operands)
    if not operands:
        return False
    dest = operands[-1]
    sources = operands[:-1]
    if _token_or_alias_is_protected_path(dest, cwd, aliases):
        return True
    return _token_is_protected_config_root(dest, cwd) and any(_source_basename_is_protected_config(src) for src in sources)


def _shell_c_payload_from_words(words: list[str], start: int) -> Optional[str]:
    """Return the payload passed to a shell ``-c`` option, if present."""
    idx = start + 1
    value_options = {"-o", "+o", "-O", "+O", "--rcfile", "--init-file"}
    while idx < len(words):
        word = words[idx]
        if word in {"--"}:
            return None
        if word in value_options and idx + 1 < len(words):
            idx += 2
            continue
        if word == "-c" and idx + 1 < len(words):
            if words[idx + 1] == "--" and idx + 2 < len(words):
                return words[idx + 2]
            return words[idx + 1]
        if any(word.startswith(prefix + "=") for prefix in value_options if prefix.startswith("--")):
            idx += 1
            continue
        if word.startswith(("-", "+")) and not word.startswith(("--", "++")) and len(word) > 1:
            consumed_next = False
            cluster = word[1:]
            for pos, opt in enumerate(cluster):
                rest = cluster[pos + 1:]
                if word[0] == "-" and opt == "c":
                    if rest:
                        return rest
                    if idx + 1 < len(words) and words[idx + 1] == "--" and idx + 2 < len(words):
                        return words[idx + 2]
                    return words[idx + 1] if idx + 1 < len(words) else None
                if opt in {"o", "O"}:
                    if not rest and idx + 1 < len(words):
                        consumed_next = True
                    break
            idx += 2 if consumed_next else 1
            continue
        if not word.startswith("-"):
            return None
        idx += 1
    return None


def _nested_shell_c_payloads_after_command(words: list[str], start: int) -> list[str]:
    """Return shell ``-c`` payloads nested under xargs / find -exec invocations.

    ``xargs -I{} sh -c '<payload>'`` and ``find ... -exec sh -c '<payload>' \\;``
    run a fresh shell whose ``-c`` argument is opaque to the outer-command
    guards (xargs/find are not command positions a shell payload is parsed
    from). Pull those payloads out so the protected-config / write checks can
    recurse into them. Read-only nested commands (no shell ``-c`` argument, or
    no write primitive inside it) yield nothing and stay allowed — quoted
    literals and read-only operands do not match.
    """
    payloads: list[str] = []
    j = start + 1
    while j < len(words):
        if _command_basename(words[j]).lower() in {"bash", "sh", "zsh", "ksh", "dash"}:
            payload = _shell_c_payload_from_words(words, j)
            if payload:
                payloads.append(payload)
        j += 1
    return payloads


def _env_split_string_payloads(words: list[str], start: int) -> list[str]:
    """Return `/usr/bin/env -S/--split-string` payloads for recursive inspection.

    Callers should pass tokens with their original option spelling. The matcher
    also accepts lowercase ``-s`` for older guard paths that normalized before
    tokenizing; ``env`` has no lowercase ``-s`` split-string alternative in real
    execution, so this is conservative only for static detection.
    """
    payloads: list[str] = []
    idx = start + 1
    while idx < len(words):
        word = words[idx]
        lower = word.lower()
        if lower in {"-s", "--split-string"} and idx + 1 < len(words):
            payloads.append(words[idx + 1])
            idx += 2
            continue
        if lower.startswith("--split-string="):
            payloads.append(word.split("=", 1)[1])
            idx += 1
            continue
        if word == "--":
            break
        idx += 1
    return payloads


def _inline_payload_and_argv_from_words(words: list[str], start: int) -> tuple[Optional[str], list[str]]:
    """Return inline code passed to interpreter -c/-e plus post-payload argv."""
    idx = start + 1
    while idx < len(words):
        word = words[idx]
        if word in {"-c", "-e"} and idx + 1 < len(words):
            return words[idx + 1], words[idx + 2:]
        if word.startswith(("-c", "-e")) and len(word) > 2:
            return word[2:], words[idx + 1:]
        idx += 1
    return None, []


def _inline_payload_from_words(words: list[str], start: int) -> Optional[str]:
    """Return inline code passed to interpreter -c/-e, if present."""
    payload, _argv = _inline_payload_and_argv_from_words(words, start)
    return payload


def _inline_payload_argv_writes_protected_path(payload: str, argv: list[str], cwd: Optional[str]) -> bool:
    """Detect inline code writing to a protected path supplied as argv.

    Read-only argv references such as ``python -c 'print(1)' ~/.hermes/config.yaml``
    must remain allowed, but write primitives targeting ``sys.argv`` /
    ``process.argv`` / ``ARGV`` are direct writes to the post-payload operand.
    """
    if not argv:
        return False
    normalized = payload.lower()
    has_argv_reference = bool(re.search(r"\b(?:sys\.argv|process\.argv|argv)\b", normalized))
    if not has_argv_reference:
        return False
    # Conservative: once inline code contains a write primitive and references
    # argv, any protected argv operand is considered a protected write target.
    for candidate in argv:
        if _token_is_protected_path(candidate, cwd):
            return True
    return False


def _inline_script_nested_payloads(payload: str) -> list[str]:
    """Return literal code/shell strings nested inside inline-language payloads.

    This catches guarded REPL/stdin cases such as ``os.system('echo x >
    ~/.hermes/config.yaml')`` or ``eval("open(..., 'w')")``.  It is a
    deliberately small static extractor: only literal quoted first arguments
    are recursed into, so read-only mentions remain allowed while obvious
    self-executing payloads cannot hide protected writes.
    """
    nested: list[str] = []
    call_patterns = [
        r"\b(?:os\.)?system\s*\(\s*(['\"])(?P<system>.*?)(?<!\\)(?:\1)",
        r"\b(?:subprocess\s*\.\s*)?(?:run|call|check_call|check_output|popen|Popen)\s*\(\s*(['\"])(?P<subprocess>.*?)(?<!\\)(?:\1)",
        r"\b(?:subprocess\s*\.\s*)?(?:run|call|check_call|check_output|popen|Popen)\s*\(\s*\[\s*['\"][^'\"]*(?:sh|bash|zsh|dash|ksh)['\"]\s*,\s*['\"]-[lc]+['\"]\s*,\s*(['\"])(?P<subprocess_shell_list>.*?)(?<!\\)(?:\1)",
        r"\b(?:subprocess\s*\.\s*)?(?:run|call|check_call|check_output|popen|Popen)\s*\([^)]*\bargs\s*=\s*(['\"])(?P<subprocess_args>.*?)(?<!\\)(?:\1)[^)]*\bshell\s*=\s*True\b",
        r"\b(?:subprocess\s*\.\s*)?(?:run|call|check_call|check_output|popen|Popen)\s*\([^)]*\bshell\s*=\s*True\b[^)]*\bargs\s*=\s*(['\"])(?P<subprocess_args_after_shell>.*?)(?<!\\)(?:\1)",
        r"\b(?:eval|exec)\s*\(\s*(['\"])(?P<eval>.*?)(?<!\\)(?:\1)",
        r"\b__import__\s*\(\s*(['\"])os(?:\1)\s*\)\s*\.\s*system\s*\(\s*(['\"])(?P<import_os>.*?)(?<!\\)(?:\2)",
    ]
    for pattern in call_patterns:
        for match in re.finditer(pattern, payload, _RE_FLAGS):
            for value in match.groupdict().values():
                if value:
                    nested.append(value)
                    break
    return nested


def _inline_script_payload_writes_protected_path(payload: Optional[str], cwd: Optional[str],
                                                 argv: Optional[list[str]] = None,
                                                 _depth: int = 0) -> bool:
    """Detect protected writes inside inline Python/Node/Ruby/Perl payloads."""
    if not payload:
        return False
    normalized = payload.lower()
    if _script_write_target_hits_protected_path(normalized, cwd):
        return True
    if _depth < 3:
        for nested_payload in _inline_script_nested_payloads(payload):
            nested_normalized = nested_payload.lower()
            if (
                _write_target_hits_protected_path_preserving_quotes(nested_payload, cwd)
                or _relative_write_target_hits_protected_path(nested_normalized, cwd)
                or _inline_script_payload_writes_protected_path(nested_payload, cwd, _depth=_depth + 1)
            ):
                return True
    has_write_primitive = bool(re.search(
        r"\b(?:open|write_text|write_bytes|writefilesync|appendfilesync|createwritestream|file\.write|file\.open)\b",
        normalized,
        _RE_FLAGS,
    ))
    if not has_write_primitive:
        return False
    if argv and _inline_payload_argv_writes_protected_path(payload, argv, cwd):
        return True
    for quoted in re.findall(r"['\"]([^'\"]+)['\"]", payload):
        if _token_is_protected_path(quoted, cwd):
            return True
    path_pat = _protected_hermes_path_pattern()
    return bool(re.search(path_pat, _normalize_protected_config_command(payload), _RE_FLAGS))


def _strip_shell_quoted_content(command: str) -> str:
    """Remove quoted literal contents before broad fallback regex checks."""
    out: list[str] = []
    quote: str | None = None
    escaped = False
    for ch in command:
        if escaped:
            if quote is None:
                out.append(ch)
            escaped = False
            continue
        if ch == "\\" and quote != "'":
            if quote is None:
                out.append(ch)
            escaped = True
            continue
        if quote:
            if ch == quote:
                quote = None
                out.append(ch)
            else:
                out.append(" ")
            continue
        if ch in {'"', "'"}:
            quote = ch
            out.append(ch)
            continue
        out.append(ch)
    return "".join(out)


def _write_target_hits_protected_path_preserving_quotes(command: str, cwd: Optional[str]) -> bool:
    """Detect write targets while preserving quoted path tokens with spaces."""
    raw = _normalize_command_for_detection(command).lower()
    # Bash ANSI-C quoted fragments can appear in paths before shlex sees them:
    # ~/.hermes/$'config.yaml' -> ~/.hermes/config.yaml
    raw = re.sub(r"\$'([^']*)'", r"\1", raw)
    raw = re.sub(r"\\([^\s])", r"\1", raw)
    for payload in _extract_static_shell_stdin_payloads(raw) + _eval_payloads_from_command(raw):
        if (
            _inline_script_payload_writes_protected_path(payload, cwd)
            or _write_target_hits_protected_path_preserving_quotes(payload, cwd)
            or _relative_write_target_hits_protected_path(payload.lower(), cwd)
        ):
            return True
    for segment in _split_command_segments(raw):
        words = _shlex_words(segment)
        if not words:
            continue
        for target in _redirection_targets_from_words(words):
            if _token_is_protected_path(target, cwd):
                return True
        for idx in _command_word_indices(words):
            word = words[idx]
            cmd_word = _command_basename(word)
            if cmd_word == "env":
                for payload in _env_split_string_payloads(words, idx):
                    if (
                        _write_target_hits_protected_path_preserving_quotes(payload, cwd)
                        or _relative_write_target_hits_protected_path(payload.lower(), cwd)
                    ):
                        return True
            if cmd_word in {"xargs", "find"}:
                for payload in _nested_shell_c_payloads_after_command(words, idx):
                    if (
                        _write_target_hits_protected_path_preserving_quotes(payload, cwd)
                        or _relative_write_target_hits_protected_path(payload.lower(), cwd)
                    ):
                        return True
            if _is_inline_script_interpreter(cmd_word):
                payload, argv = _inline_payload_and_argv_from_words(words, idx)
                if _inline_script_payload_writes_protected_path(payload, cwd, argv):
                    return True
            if cmd_word == "dd":
                for target in _dd_of_targets_from_words(words[idx + 1:]):
                    if _token_is_protected_path(target, cwd):
                        return True
            if cmd_word == "tee":
                for candidate in [w for w in words[idx + 1:] if not w.startswith("-")]:
                    if _token_is_protected_path(candidate, cwd):
                        return True
            if cmd_word in {"sed", "perl", "ruby"}:
                has_in_place = any(w == "--in-place" or (w.startswith("-") and "i" in w) for w in words[idx + 1:])
                if has_in_place:
                    for candidate in [w for w in words[idx + 1:] if not w.startswith("-")]:
                        if _token_is_protected_path(candidate, cwd):
                            return True
            if cmd_word in {"rm", "unlink", "truncate", "touch", "nano", "vim", "vi", "nvim", "emacs", "code"}:
                for candidate in [w for w in words[idx + 1:] if not w.startswith("-")]:
                    if _token_is_protected_path(candidate, cwd):
                        return True
            if cmd_word == "mv":
                args = words[idx + 1:]
                for candidate in [w for w in args if not w.startswith("-")]:
                    if _token_is_protected_path(candidate, cwd):
                        return True
                if _copy_like_hits_protected_path(args, cwd):
                    return True
            if cmd_word in {"cp", "install", "ln", "rsync"}:
                args = words[idx + 1:]
                candidates = [w for w in args if not w.startswith("-")]
                if candidates and _token_is_protected_path(candidates[-1], cwd):
                    return True
                # ln SOURCE... DIR/ with DIR == protected config root and a
                # source basename of config.yaml/.env overwrites the protected
                # file via the link created inside DIR (absolute-dest form of
                # `ln -sf /tmp/config.yaml ~/.hermes/`).
                if (
                    cmd_word == "ln"
                    and len(candidates) >= 2
                    and _token_is_protected_config_root(candidates[-1], cwd)
                    and any(_source_basename_is_protected_config(src) for src in candidates[:-1])
                ):
                    return True
                if cmd_word in {"cp", "install", "rsync"} and _copy_like_hits_protected_path(args, cwd):
                    return True
            if cmd_word in {"bash", "sh", "zsh", "ksh", "dash"}:
                payload = _shell_c_payload_from_words(words, idx)
                if payload and (
                    _write_target_hits_protected_path_preserving_quotes(payload, cwd)
                    or _relative_write_target_hits_protected_path(payload.lower(), cwd)
                ):
                    return True
    return False


def _script_write_target_hits_protected_path(normalized: str, cwd: Optional[str]) -> bool:
    """Detect simple inline-language writes to protected paths.

    The command has already had shell quotes removed by
    `_normalize_protected_config_command`, so match conservative token shapes
    such as `open(config.yaml,w)` and `Path(config.yaml).write_text(...)`.
    """
    script_write_patterns = [
        r"\bopen\(\s*([^,\s)]+)\s*,\s*['\"]?[r]?['\"]?\s*['\"]?[wxa+]",
        r"\bpath\(\s*([^,\s)]+)\s*\)\s*\.\s*(?:write_text|write_bytes)",
        r"\b(?:writefilesync|appendfilesync|createwritestream)\(\s*([^,\s)]+)",
        r"\bfile\s*\.\s*write\(\s*([^,\s)]+)",
        r"\bfile\s*\.\s*open\(\s*([^,\s)]+)\s*,\s*['\"]?[wa]",
        r"\bopen\(\s*[^,]+,\s*['\"]?>{1,2}['\"]?\s*,\s*([^,\s)]+)",
    ]
    for pattern in script_write_patterns:
        for target in re.findall(pattern, normalized, _RE_FLAGS):
            if _token_is_protected_path(target, cwd):
                return True
    return False


def _cd_target_from_words(words: list[str]) -> Optional[str]:
    """Return the path argument from a simple ``cd`` command segment."""
    if not words:
        return None
    command_word = words[0].lstrip("({")
    args = words[1:]
    if command_word != "cd" and words[0] in {"{", "("} and len(words) > 1:
        command_word = words[1]
        args = words[2:]
    if command_word != "cd":
        return None
    if args and args[0] == "--":
        args = args[1:]
    return args[0] if args else "~"


def _record_script_protected_vars(segment: str, cwd: Optional[str], vars_: set[str]) -> None:
    """Remember simple script variables assigned to protected relative paths."""
    for name, value in re.findall(r"\b([a-z_]\w*)\s*=\s*([^,;\s)]+)", segment, _RE_FLAGS):
        if _token_is_protected_path(value, cwd):
            vars_.add(name.lower())
    for name, value in re.findall(r"\b([a-z_]\w*)\s*=\s*path\(\s*([^,\s)]+)\s*\)", segment, _RE_FLAGS):
        if _token_is_protected_path(value, cwd):
            vars_.add(name.lower())


def _script_write_uses_protected_var(segment: str, vars_: set[str]) -> bool:
    """Detect simple script writes whose target is a known protected variable."""
    if not vars_:
        return False
    script_write_patterns = [
        r"\bopen\(\s*([^,\s)]+)\s*,\s*['\"]?[r]?['\"]?\s*['\"]?[wxa+]",
        r"\bpath\(\s*([^,\s)]+)\s*\)\s*\.\s*(?:write_text|write_bytes)",
        r"\b([a-z_]\w*)\s*\.\s*(?:write_text|write_bytes)\s*\(",
        r"\b(?:writefilesync|appendfilesync|createwritestream)\(\s*([a-z_]\w*)",
        r"\bfile\s*\.\s*(?:write|open)\(\s*([a-z_]\w*)",
        r"\bopen\(\s*[^,]+,\s*['\"]?>{1,2}['\"]?\s*,\s*([a-z_]\w*)",
    ]
    for pattern in script_write_patterns:
        for target in re.findall(pattern, segment, _RE_FLAGS):
            if target.lower() in vars_:
                return True
    return False


def _relative_write_target_hits_protected_path(normalized: str, cwd: Optional[str]) -> bool:
    """Resolve write-target tokens relative to shell cwd and compare protected paths.

    This is intentionally a small pre-exec shell model, not a full interpreter:
    it walks simple command segments left-to-right, carrying forward obvious
    ``cd`` state so a later harmless ``cd`` cannot hide an earlier relative
    write to ``config.yaml``/``.env`` under Hermes home.  Subshell ``cd`` state
    is scoped to the parenthesized command and does not leak back to the parent.
    """
    current_cwd = cwd
    subshell_cwd: Optional[str] = None
    in_subshell = False
    script_cwd = None
    script_protected_vars: set[str] = set()
    protected_aliases: set[str] = set()
    shell_vars: dict[str, str] = {}
    for segment in _split_command_segments(normalized):
        stripped_segment = segment.strip()
        starts_subshell = stripped_segment.startswith("(")
        ends_subshell = stripped_segment.endswith(")")
        if starts_subshell:
            in_subshell = True
            subshell_cwd = current_cwd
            script_cwd = None
            script_protected_vars.clear()
        active_cwd = subshell_cwd if in_subshell else current_cwd
        words = _shlex_words(segment)
        if not words:
            if ends_subshell:
                in_subshell = False
                subshell_cwd = None
            continue
        for word in words:
            if not _is_shell_assignment(word):
                break
            name, value = word.split("=", 1)
            shell_vars[name.lower()] = _expand_known_shell_vars(value, shell_vars)
        words = [_expand_known_shell_vars(word, shell_vars) for word in words]

        for idx in _command_word_indices(words):
            if _command_basename(words[idx]).lower() not in {"export", "declare", "typeset", "readonly", "local"}:
                continue
            for assignment in words[idx + 1:]:
                if not _is_shell_assignment(assignment):
                    continue
                name, value = assignment.split("=", 1)
                expanded_value = _expand_known_shell_vars(value, shell_vars)
                shell_vars[name.lower()] = expanded_value
                resolved_alias = _resolve_protected_path_token(expanded_value, active_cwd)
                if resolved_alias and resolved_alias in _protected_config_resolved_paths():
                    protected_aliases.add(resolved_alias)

        cd_target = _cd_target_from_words(words)
        if cd_target is not None:
            resolved_cd = _resolve_protected_path_token(cd_target, active_cwd)
            if resolved_cd:
                if in_subshell:
                    subshell_cwd = resolved_cd
                else:
                    current_cwd = resolved_cd
            script_cwd = None
            script_protected_vars.clear()
            if ends_subshell:
                in_subshell = False
                subshell_cwd = None
            continue

        if script_cwd is not None:
            _record_script_protected_vars(segment, script_cwd, script_protected_vars)
            if (
                _script_write_target_hits_protected_path(segment, script_cwd)
                or _script_write_uses_protected_var(segment, script_protected_vars)
            ):
                return True

        for target in _redirection_targets_from_words(words):
            if _token_or_alias_is_protected_path(target, active_cwd, protected_aliases):
                return True

        for idx in _command_word_indices(words):
            word = words[idx]
            cmd_word = _command_basename(word)
            if cmd_word == "env":
                for payload in _env_split_string_payloads(words, idx):
                    if (
                        _write_target_hits_protected_path_preserving_quotes(payload, active_cwd)
                        or _relative_write_target_hits_protected_path(payload.lower(), active_cwd)
                    ):
                        return True
            if cmd_word in {"xargs", "find"}:
                for payload in _nested_shell_c_payloads_after_command(words, idx):
                    if (
                        _write_target_hits_protected_path_preserving_quotes(payload, active_cwd)
                        or _relative_write_target_hits_protected_path(payload.lower(), active_cwd)
                    ):
                        return True
            if _is_inline_script_interpreter(cmd_word):
                payload, argv = _inline_payload_and_argv_from_words(words, idx)
                if _inline_script_payload_writes_protected_path(payload, active_cwd, argv):
                    return True
            if cmd_word == "dd":
                for target in _dd_of_targets_from_words(words[idx + 1:]):
                    if _token_or_alias_is_protected_path(target, active_cwd, protected_aliases):
                        return True
            if cmd_word == "tee":
                for candidate in [w for w in words[idx + 1:] if not w.startswith("-")]:
                    if _token_or_alias_is_protected_path(candidate, active_cwd, protected_aliases):
                        return True
            if cmd_word in {"sed", "perl", "ruby"}:
                has_in_place = any(w == "--in-place" or (w.startswith("-") and "i" in w) for w in words[idx + 1:])
                if has_in_place:
                    for candidate in [w for w in words[idx + 1:] if not w.startswith("-")]:
                        if _token_or_alias_is_protected_path(candidate, active_cwd, protected_aliases):
                            return True
            if cmd_word in {"rm", "unlink"}:
                for candidate in [w for w in words[idx + 1:] if not w.startswith("-")]:
                    if _token_is_protected_path(candidate, active_cwd):
                        return True
            if cmd_word in {"truncate", "touch", "nano", "vim", "vi", "nvim", "emacs", "code"}:
                for candidate in [w for w in words[idx + 1:] if not w.startswith("-")]:
                    if _token_or_alias_is_protected_path(candidate, active_cwd, protected_aliases):
                        return True
            if cmd_word == "mv":
                args = words[idx + 1:]
                candidates = [w for w in args if not w.startswith("-")]
                for candidate in candidates[:-1]:
                    if _token_is_protected_path(candidate, active_cwd):
                        return True
                if candidates and _token_or_alias_is_protected_path(candidates[-1], active_cwd, protected_aliases):
                    return True
                if _copy_like_hits_protected_path(args, active_cwd, protected_aliases):
                    return True
            if cmd_word in {"cp", "install", "ln", "rsync"}:
                candidates = [w for w in words[idx + 1:] if not w.startswith("-")]
                if cmd_word == "ln":
                    # Creating a link AT a protected config path overwrites it
                    # (cd ~/.hermes && ln -sf /tmp/x config.yaml). The absolute
                    # form is caught by the quote-preserving pass; this closes
                    # the relative-cwd gap.
                    if candidates and _token_or_alias_is_protected_path(candidates[-1], active_cwd, protected_aliases):
                        return True
                    # ln SOURCE... DIR/ creates the link inside DIR named after
                    # the source basename. If DIR is the protected config root
                    # and a source basename is config.yaml/.env, the resulting
                    # link overwrites the protected file
                    # (ln -sf /tmp/config.yaml ~/.hermes/ ;
                    #  cd ~/.hermes && ln -sf /tmp/config.yaml .).
                    if (
                        len(candidates) >= 2
                        and _token_is_protected_config_root(candidates[-1], active_cwd)
                        and any(_source_basename_is_protected_config(src) for src in candidates[:-1])
                    ):
                        return True
                    # ln SOURCE LINK where SOURCE is protected: the link name
                    # becomes an alias for later writes through it.
                    if len(candidates) >= 2 and _token_is_protected_path(candidates[-2], active_cwd):
                        alias = _resolve_protected_path_token(candidates[-1], active_cwd)
                        if alias:
                            protected_aliases.add(alias)
                elif _copy_like_hits_protected_path(words[idx + 1:], active_cwd, protected_aliases):
                    return True
            if cmd_word in {"bash", "sh", "zsh", "ksh", "dash"}:
                payload = _shell_c_payload_from_words(words, idx)
                if payload and (
                    _write_target_hits_protected_path_preserving_quotes(payload, active_cwd)
                    or _relative_write_target_hits_protected_path(payload.lower(), active_cwd)
                ):
                    return True
            if cmd_word == "eval" and idx + 1 < len(words):
                payload = " ".join(words[idx + 1:])
                if (
                    _write_target_hits_protected_path_preserving_quotes(payload, active_cwd)
                    or _relative_write_target_hits_protected_path(payload.lower(), active_cwd)
                ):
                    return True
            if _is_inline_script_interpreter(cmd_word):
                _record_script_protected_vars(segment, active_cwd, script_protected_vars)
                if (
                    _script_write_target_hits_protected_path(segment, active_cwd)
                    or _script_write_uses_protected_var(segment, script_protected_vars)
                ):
                    return True
                if any(word in {"-c", "-e"} for word in words[idx + 1:]) or "<<" in segment:
                    script_cwd = active_cwd
        if _is_inline_script_interpreter(words[0]):
            _record_script_protected_vars(segment, active_cwd, script_protected_vars)
            if (
                _script_write_target_hits_protected_path(segment, active_cwd)
                or _script_write_uses_protected_var(segment, script_protected_vars)
            ):
                return True
            if any(word in {"-c", "-e"} for word in words[1:]) or "<<" in segment:
                script_cwd = active_cwd
        if ends_subshell:
            in_subshell = False
            subshell_cwd = None
            script_cwd = None
            script_protected_vars.clear()
    return False


def _check_protected_config_write_guard(command: str, cwd: Optional[str] = None) -> tuple:
    """Detect direct writes/edits to protected Hermes config files.

    Returns:
        (is_blocked: bool, description: str | None)
    """
    effective_cwd = _resolve_protected_path_token(cwd, os.getcwd()) if cwd else None
    if _write_target_hits_protected_path_preserving_quotes(command, effective_cwd):
        return (True, "direct write to protected Hermes config/env")
    quote_preserving_command = _normalize_command_for_detection(command).lower()
    if _relative_write_target_hits_protected_path(quote_preserving_command, effective_cwd):
        return (True, "direct write to protected Hermes config/env")

    return (False, None)


def _protected_config_block_result(description: str) -> dict:
    """Build the standard block result for protected config direct writes."""
    return {
        "approved": False,
        "protected_config": True,
        "message": (
            f"BLOCKED: {description}. "
            "Do not edit ~/.hermes/.env or ~/.hermes/config.yaml directly "
            "from the agent. Use `hermes config set <key> <value>` for "
            "configuration changes, and do not bypass this guard. Secret "
            "values were not inspected or printed."
        ),
    }


def check_unbypassable_command_guards(command: str, env_type: str,
                                      cwd: Optional[str] = None) -> dict:
    """Run command guards that must apply even under yolo/force/off modes."""
    if env_type in {"docker", "singularity", "modal", "daytona", "vercel_sandbox"}:
        return {"approved": True, "message": None}

    is_protected_config_write, protected_desc = _check_protected_config_write_guard(command, cwd=cwd)
    if is_protected_config_write:
        logger.warning("Protected config guard block: %s", protected_desc)
        return _protected_config_block_result(protected_desc)

    is_hardline, hardline_desc = detect_hardline_command(command)
    if is_hardline:
        logger.warning("Hardline block: %s", hardline_desc)
        return _hardline_block_result(hardline_desc)

    is_sudo_guess, sudo_guess_desc = _check_sudo_stdin_guard(command)
    if is_sudo_guess:
        logger.warning("Sudo stdin guard block: %s", sudo_guess_desc)
        return _sudo_stdin_block_result(sudo_guess_desc)

    return {"approved": True, "message": None}


# =========================================================================
# Dangerous command patterns
# =========================================================================
DANGEROUS_PATTERNS = [
    (r'\brm\s+(-[^\s]*\s+)*/', "delete in root path"),
    (r'\brm\s+-[^\s]*r', "recursive delete"),
    (r'\brm\s+--recursive\b', "recursive delete (long flag)"),
    (r'\bchmod\s+(-[^\s]*\s+)*(777|666|o\+[rwx]*w|a\+[rwx]*w)\b', "world/other-writable permissions"),
    (r'\bchmod\s+--recursive\b.*(777|666|o\+[rwx]*w|a\+[rwx]*w)', "recursive world/other-writable (long flag)"),
    (r'\bchown\s+(-[^\s]*)?R\s+root', "recursive chown to root"),
    (r'\bchown\s+--recursive\b.*root', "recursive chown to root (long flag)"),
    (r'\bmkfs\b', "format filesystem"),
    (r'\bdd\s+.*if=', "disk copy"),
    (r'>\s*/dev/sd', "write to block device"),
    (r'\bDROP\s+(TABLE|DATABASE)\b', "SQL DROP"),
    # Use [^\n]* instead of .* so DOTALL mode does not cause a WHERE clause on the
    # *next* line to satisfy the negative lookahead, silently allowing DELETE without WHERE.
    (r'\bDELETE\s+FROM\b(?![^\n]*\bWHERE\b)', "SQL DELETE without WHERE"),
    (r'\bTRUNCATE\s+(TABLE)?\s*\w', "SQL TRUNCATE"),
    (rf'>\s*{_SYSTEM_CONFIG_PATH}', "overwrite system config"),
    (r'\bsystemctl\s+(-[^\s]+\s+)*(stop|restart|disable|mask)\b', "stop/restart system service"),
    (r'\bkill\s+-9\s+-1\b', "kill all processes"),
    (r'\bpkill\s+-9\b', "force kill processes"),
    # killall with SIGKILL (parallel to pkill -9). Catches -9 / -KILL /
    # -s KILL / -SIGKILL forms, and also `killall -r <regex>` broad sweeps
    # that can wipe out unrelated processes by accident.
    # Inspired by Claude Code 2.1.113 expanded deny rules.
    (r'\bkillall\s+(-[^\s]*\s+)*-(9|KILL|SIGKILL)\b', "force kill processes (killall -KILL)"),
    (r'\bkillall\s+(-[^\s]*\s+)*-s\s+(KILL|SIGKILL|9)\b', "force kill processes (killall -s KILL)"),
    (r'\bkillall\s+(-[^\s]*\s+)*-r\b', "kill processes by regex (killall -r)"),
    (r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:', "fork bomb"),
    # Any shell invocation via -c or combined flags like -lc, -ic, etc.
    (r'\b(bash|sh|zsh|ksh)\s+-[^\s]*c(\s+|$)', "shell command via -c/-lc flag"),
    (r'\b(python[23]?|perl|ruby|node)\s+-[ec]\s+', "script execution via -e/-c flag"),
    (r'\b(curl|wget)\b.*\|\s*(?:[/\w]*/)?(?:ba)?sh(?:\s|$|-c)', "pipe remote content to shell"),
    (r'\b(bash|sh|zsh|ksh)\s+<\s*<?\s*\(\s*(curl|wget)\b', "execute remote script via process substitution"),
    (rf'\btee\b.*["\']?{_SENSITIVE_WRITE_TARGET}', "overwrite system file via tee"),
    (rf'>>?\s*["\']?{_SENSITIVE_WRITE_TARGET}', "overwrite system file via redirection"),
    (rf'\btee\b.*["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_COMMAND_TAIL}', "overwrite project env/config via tee"),
    (rf'>>?\s*["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_COMMAND_TAIL}', "overwrite project env/config via redirection"),
    (r'\bxargs\s+.*\brm\b', "xargs with rm"),
    # find -exec rm / -execdir rm — the -execdir variant (same semantics,
    # runs in the directory of each match) was previously missed. Claude
    # Code 2.1.113 tightened their equivalent find rule to stop auto-
    # approving -exec / -delete flags.
    (r'\bfind\b.*-exec(?:dir)?\s+(/\S*/)?rm\b', "find -exec/-execdir rm"),
    (r'\bfind\b.*-delete\b', "find -delete"),
    # Gateway lifecycle protection: prevent the agent from killing its own
    # gateway process.  These commands trigger a gateway restart/stop that
    # terminates all running agents mid-work.
    (r'\bhermes\s+gateway\s+(stop|restart)\b', "stop/restart hermes gateway (kills running agents)"),
    (r'\bhermes\s+update\b', "hermes update (restarts gateway, kills running agents)"),
    # Gateway protection: never start gateway outside systemd management
    (r'gateway\s+run\b.*(&\s*$|&\s*;|\bdisown\b|\bsetsid\b)', "start gateway outside systemd (use 'systemctl --user restart hermes-gateway')"),
    (r'\bnohup\b.*gateway\s+run\b', "start gateway outside systemd (use 'systemctl --user restart hermes-gateway')"),
    # Self-termination protection: prevent agent from killing its own process
    (r'\b(pkill|killall)\b.*\b(hermes|gateway|cli\.py)\b', "kill hermes/gateway process (self-termination)"),
    # Self-termination via kill + command substitution (pgrep/pidof).
    # The name-based pattern above catches `pkill hermes` but not
    # `kill -9 $(pgrep -f hermes)` because the substitution is opaque
    # to regex at detection time. Catch the structural pattern instead.
    (r'\bkill\b.*\$\(\s*pgrep\b', "kill process via pgrep expansion (self-termination)"),
    (r'\bkill\b.*`\s*pgrep\b', "kill process via backtick pgrep expansion (self-termination)"),
    # File copy/move/edit into sensitive system paths (/etc/ and macOS
    # /private/etc/ mirror).
    (rf'\b(cp|mv|install)\b.*\s{_SYSTEM_CONFIG_PATH}', "copy/move file into system config path"),
    (rf'\b(cp|mv|install)\b.*\s["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_COMMAND_TAIL}', "overwrite project env/config file"),
    (rf'\bsed\s+-[^\s]*i.*\s{_SYSTEM_CONFIG_PATH}', "in-place edit of system config"),
    (rf'\bsed\s+--in-place\b.*\s{_SYSTEM_CONFIG_PATH}', "in-place edit of system config (long flag)"),
    # Script execution via heredoc — bypasses the -e/-c flag patterns above.
    # `python3 << 'EOF'` feeds arbitrary code via stdin without -c/-e flags.
    (r'\b(python[23]?|perl|ruby|node)\s+<<', "script execution via heredoc"),
    # Git destructive operations that can lose uncommitted work or rewrite
    # shared history. Not captured by rm/chmod/etc patterns.
    (r'\bgit\s+reset\s+--hard\b', "git reset --hard (destroys uncommitted changes)"),
    (r'\bgit\s+push\b.*--force\b', "git force push (rewrites remote history)"),
    (r'\bgit\s+push\b.*-f\b', "git force push short flag (rewrites remote history)"),
    (r'\bgit\s+clean\s+-[^\s]*f', "git clean with force (deletes untracked files)"),
    (r'\bgit\s+branch\s+-D\b', "git branch force delete"),
    # Script execution after chmod +x — catches the two-step pattern where
    # a script is first made executable then immediately run. The script
    # content may contain dangerous commands that individual patterns miss.
    (r'\bchmod\s+\+x\b.*[;&|]+\s*\./', "chmod +x followed by immediate execution"),
    # Sudo with stdin / askpass / shell / list-privs flags. An LLM-driven
    # agent has no TTY, so sudo invocations that succeed without human
    # interaction are those reading the password from stdin (-S/--stdin)
    # or via an askpass helper (-A/--askpass). The shell-launch (-s) and
    # list-privileges (-a) flags are also gated since they are
    # privilege-relevant invocations the agent can chain after acquiring
    # the password (e.g. read SUDO_PASSWORD from .env -> sudo -S -s ->
    # root shell). Plain `sudo cmd` (no flag) is TTY-bound and excluded.
    # `_normalize_command_for_detection` lowercases input before pattern
    # matching, so case variants of S/s and A/a collapse — both forms
    # are gated below. Lazy `[^;|&\n]*?` allows flag arguments (e.g.
    # `sudo -u root -S whoami`) without spanning command separators. See
    # #17873 category 4.
    (r'\bsudo\b[^;|&\n]*?\s+(?:-s\b|--stdin\b|-a\b|--askpass\b)',
     "sudo with privilege flag (stdin/askpass/shell/list)"),
    # Combined short-flag form: -nS, -ns, -sa, -las — sudo flags packed
    # into a single -X token. Catches the same threat class.
    (r'\bsudo\b[^;|&\n]*?\s+-[a-z]*[sa][a-z]*\b',
     "sudo with combined-flag privilege escalation"),
]


# Pre-compiled variant (same rationale as HARDLINE_PATTERNS_COMPILED above).
DANGEROUS_PATTERNS_COMPILED = [
    (re.compile(pattern, _RE_FLAGS), description)
    for pattern, description in DANGEROUS_PATTERNS
]


def _legacy_pattern_key(pattern: str) -> str:
    """Reproduce the old regex-derived approval key for backwards compatibility."""
    return pattern.split(r'\b')[1] if r'\b' in pattern else pattern[:20]


_PATTERN_KEY_ALIASES: dict[str, set[str]] = {}
for _pattern, _description in DANGEROUS_PATTERNS:
    _legacy_key = _legacy_pattern_key(_pattern)
    _canonical_key = _description
    _PATTERN_KEY_ALIASES.setdefault(_canonical_key, set()).update({_canonical_key, _legacy_key})
    _PATTERN_KEY_ALIASES.setdefault(_legacy_key, set()).update({_legacy_key, _canonical_key})


def _approval_key_aliases(pattern_key: str) -> set[str]:
    """Return all approval keys that should match this pattern.

    New approvals use the human-readable description string, but older
    command_allowlist entries and session approvals may still contain the
    historical regex-derived key.
    """
    return _PATTERN_KEY_ALIASES.get(pattern_key, {pattern_key})


# =========================================================================
# Detection
# =========================================================================

def _normalize_command_for_detection(command: str) -> str:
    """Normalize a command string before dangerous-pattern matching.

    Strips ANSI escape sequences (full ECMA-48 via tools.ansi_strip),
    null bytes, and normalizes Unicode fullwidth characters so that
    obfuscation techniques cannot bypass the pattern-based detection.
    """
    from tools.ansi_strip import strip_ansi

    # Strip all ANSI escape sequences (CSI, OSC, DCS, 8-bit C1, etc.)
    command = strip_ansi(command)
    # Strip null bytes
    command = command.replace('\x00', '')
    # Normalize Unicode (fullwidth Latin, halfwidth Katakana, etc.)
    command = unicodedata.normalize('NFKC', command)
    return command


def detect_dangerous_command(command: str) -> tuple:
    """Check if a command matches any dangerous patterns.

    Returns:
        (is_dangerous, pattern_key, description) or (False, None, None)
    """
    command_lower = _normalize_command_for_detection(command).lower()
    for pattern_re, description in DANGEROUS_PATTERNS_COMPILED:
        if pattern_re.search(command_lower):
            pattern_key = description
            return (True, pattern_key, description)
    return (False, None, None)


# =========================================================================
# Per-session approval state (thread-safe)
# =========================================================================

_lock = threading.Lock()
_pending: dict[str, dict] = {}
_session_approved: dict[str, set] = {}
_session_yolo: set[str] = set()
_permanent_approved: set = set()

# =========================================================================
# Blocking gateway approval (mirrors CLI's synchronous input() flow)
# =========================================================================
# Per-session QUEUE of pending approvals.  Multiple threads (parallel
# subagents, execute_code RPC handlers) can block concurrently — each gets
# its own threading.Event.  /approve resolves the oldest, /approve all
# resolves every pending approval in the session.


class _ApprovalEntry:
    """One pending dangerous-command approval inside a gateway session."""
    __slots__ = ("event", "data", "result")

    def __init__(self, data: dict):
        self.event = threading.Event()
        self.data = data          # command, description, pattern_keys, …
        self.result: Optional[str] = None  # "once"|"session"|"always"|"deny"


_gateway_queues: dict[str, list] = {}        # session_key → [_ApprovalEntry, …]
_gateway_notify_cbs: dict[str, object] = {}  # session_key → callable(approval_data)


def register_gateway_notify(session_key: str, cb) -> None:
    """Register a per-session callback for sending approval requests to the user.

    The callback signature is ``cb(approval_data: dict) -> None`` where
    *approval_data* contains ``command``, ``description``, and
    ``pattern_keys``.  The callback bridges sync→async (runs in the agent
    thread, must schedule the actual send on the event loop).
    """
    with _lock:
        _gateway_notify_cbs[session_key] = cb


def unregister_gateway_notify(session_key: str) -> None:
    """Unregister the per-session gateway approval callback.

    Signals ALL blocked threads for this session so they don't hang forever
    (e.g. when the agent run finishes or is interrupted).
    """
    with _lock:
        _gateway_notify_cbs.pop(session_key, None)
        entries = _gateway_queues.pop(session_key, [])
    for entry in entries:
        entry.event.set()


def resolve_gateway_approval(session_key: str, choice: str,
                             resolve_all: bool = False) -> int:
    """Called by the gateway's /approve or /deny handler to unblock
    waiting agent thread(s).

    When *resolve_all* is True every pending approval in the session is
    resolved at once (``/approve all``).  Otherwise only the oldest one
    is resolved (FIFO).

    Returns the number of approvals resolved (0 means nothing was pending).
    """
    with _lock:
        queue = _gateway_queues.get(session_key)
        if not queue:
            return 0
        if resolve_all:
            targets = list(queue)
            queue.clear()
        else:
            targets = [queue.pop(0)]
        if not queue:
            _gateway_queues.pop(session_key, None)

    for entry in targets:
        entry.result = choice
        entry.event.set()
    return len(targets)


def has_blocking_approval(session_key: str) -> bool:
    """Check if a session has one or more blocking gateway approvals waiting."""
    with _lock:
        return bool(_gateway_queues.get(session_key))


def submit_pending(session_key: str, approval: dict):
    """Store a pending approval request for a session."""
    with _lock:
        _pending[session_key] = approval


def approve_session(session_key: str, pattern_key: str):
    """Approve a pattern for this session only."""
    with _lock:
        _session_approved.setdefault(session_key, set()).add(pattern_key)


def enable_session_yolo(session_key: str) -> None:
    """Enable YOLO bypass for a single session key."""
    if not session_key:
        return
    with _lock:
        _session_yolo.add(session_key)


def disable_session_yolo(session_key: str) -> None:
    """Disable YOLO bypass for a single session key."""
    if not session_key:
        return
    with _lock:
        _session_yolo.discard(session_key)


def clear_session(session_key: str) -> None:
    """Remove all approval and yolo state for a given session."""
    if not session_key:
        return
    with _lock:
        _session_approved.pop(session_key, None)
        _session_yolo.discard(session_key)
        _pending.pop(session_key, None)
        entries = _gateway_queues.pop(session_key, [])
    for entry in entries:
        # Session-boundary cleanup should cancel any blocked approval waits
        # immediately so the old run can unwind instead of idling until timeout.
        entry.result = "deny"
        entry.event.set()


def is_session_yolo_enabled(session_key: str) -> bool:
    """Return True when YOLO bypass is enabled for a specific session."""
    if not session_key:
        return False
    with _lock:
        return session_key in _session_yolo


def is_current_session_yolo_enabled() -> bool:
    """Return True when the active approval session has YOLO bypass enabled."""
    return is_session_yolo_enabled(get_current_session_key(default=""))


def is_approved(session_key: str, pattern_key: str) -> bool:
    """Check if a pattern is approved (session-scoped or permanent).

    Accept both the current canonical key and the legacy regex-derived key so
    existing command_allowlist entries continue to work after key migrations.
    """
    aliases = _approval_key_aliases(pattern_key)
    with _lock:
        if any(alias in _permanent_approved for alias in aliases):
            return True
        session_approvals = _session_approved.get(session_key, set())
        return any(alias in session_approvals for alias in aliases)


def approve_permanent(pattern_key: str):
    """Add a pattern to the permanent allowlist."""
    with _lock:
        _permanent_approved.add(pattern_key)


def load_permanent(patterns: set):
    """Bulk-load permanent allowlist entries from config."""
    with _lock:
        _permanent_approved.update(patterns)



# =========================================================================
# Config persistence for permanent allowlist
# =========================================================================

def load_permanent_allowlist() -> set:
    """Load permanently allowed command patterns from config.

    Also syncs them into the approval module so is_approved() works for
    patterns added via 'always' in a previous session.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        patterns = set(config.get("command_allowlist", []) or [])
        if patterns:
            load_permanent(patterns)
        return patterns
    except Exception as e:
        logger.warning("Failed to load permanent allowlist: %s", e)
        return set()


def save_permanent_allowlist(patterns: set):
    """Save permanently allowed command patterns to config."""
    try:
        from hermes_cli.config import load_config, save_config
        config = load_config()
        config["command_allowlist"] = list(patterns)
        save_config(config)
    except Exception as e:
        logger.warning("Could not save allowlist: %s", e)


# =========================================================================
# Approval prompting + orchestration
# =========================================================================

def prompt_dangerous_approval(command: str, description: str,
                              timeout_seconds: int | None = None,
                              allow_permanent: bool = True,
                              approval_callback=None) -> str:
    """Prompt the user to approve a dangerous command (CLI only).

    Args:
        allow_permanent: When False, hide the [a]lways option (used when
            tirith warnings are present, since broad permanent allowlisting
            is inappropriate for content-level security findings).
        approval_callback: Optional callback registered by the CLI for
            prompt_toolkit integration. Signature:
            (command, description, *, allow_permanent=True) -> str.

    Returns: 'once', 'session', 'always', or 'deny'
    """
    if timeout_seconds is None:
        timeout_seconds = _get_approval_timeout()

    if approval_callback is not None:
        try:
            return approval_callback(command, description,
                                     allow_permanent=allow_permanent)
        except Exception as e:
            logger.error("Approval callback failed: %s", e, exc_info=True)
            return "deny"

    # Fail-closed guard: if prompt_toolkit owns the terminal (interactive
    # CLI session) and no approval callback is registered on this thread,
    # the input() fallback below would spawn a daemon thread whose read
    # can never see Enter -- the user's keystrokes go to prompt_toolkit,
    # not input(), producing an invisible 60s deadlock (issue #15216).
    # Deny fast and log loudly instead so the caller can surface a real
    # error to the agent. Any thread that needs interactive approval must
    # install a callback via tools.terminal_tool.set_approval_callback()
    # before reaching this point (see delegate_tool.py, run_agent.py
    # _execute_tool_calls_concurrent / _spawn_background_review for the
    # established pattern).
    try:
        from prompt_toolkit.application.current import get_app_or_none
        if get_app_or_none() is not None:
            logger.warning(
                "Dangerous-command approval requested on a thread with no "
                "approval callback while prompt_toolkit is active; denying "
                "to avoid stdin deadlock. command=%r description=%r",
                command, description,
            )
            return "deny"
    except Exception:
        # prompt_toolkit not installed, or detection failed -- fall through
        # to the legacy input() path (safe in non-TUI contexts: scripts,
        # tests, sshd, etc.).
        pass

    os.environ["HERMES_SPINNER_PAUSE"] = "1"
    try:
        # Resolve the active UI language once per prompt so we don't re-read
        # config/YAML inside the retry loop below.
        from agent.i18n import t
        while True:
            print()
            print(f"  {t('approval.dangerous_header', description=description)}")
            print(f"      {command}")
            print()
            if allow_permanent:
                print(t("approval.choose_long"))
            else:
                print(t("approval.choose_short"))
            print()
            sys.stdout.flush()

            result = {"choice": ""}

            def get_input():
                try:
                    prompt = t("approval.prompt_long") if allow_permanent else t("approval.prompt_short")
                    result["choice"] = input(prompt).strip().lower()
                except (EOFError, OSError):
                    result["choice"] = ""

            thread = threading.Thread(target=get_input, daemon=True)
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                print("\n" + t("approval.timeout"))
                return "deny"

            choice = result["choice"]
            if choice in {'o', 'once'}:
                print(t("approval.allowed_once"))
                return "once"
            elif choice in {'s', 'session'}:
                print(t("approval.allowed_session"))
                return "session"
            elif choice in {'a', 'always'}:
                if not allow_permanent:
                    print(t("approval.allowed_session"))
                    return "session"
                print(t("approval.allowed_always"))
                return "always"
            else:
                print(t("approval.denied"))
                return "deny"

    except (EOFError, KeyboardInterrupt):
        print("\n" + t("approval.cancelled"))
        return "deny"
    finally:
        if "HERMES_SPINNER_PAUSE" in os.environ:
            del os.environ["HERMES_SPINNER_PAUSE"]
        print()
        sys.stdout.flush()


def _normalize_approval_mode(mode) -> str:
    """Normalize approval mode values loaded from YAML/config.

    YAML 1.1 treats bare words like `off` as booleans, so a config entry like
    `approvals:\n  mode: off` is parsed as False unless quoted. Treat that as the
    intended string mode instead of falling back to manual approvals.
    """
    if isinstance(mode, bool):
        return "off" if mode is False else "manual"
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        return normalized or "manual"
    return "manual"


def _get_approval_config() -> dict:
    """Read the approvals config block. Returns a dict with 'mode', 'timeout', etc."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        return config.get("approvals", {}) or {}
    except Exception as e:
        logger.warning("Failed to load approval config: %s", e)
        return {}


def _get_approval_mode() -> str:
    """Read the approval mode from config. Returns 'manual', 'smart', or 'off'."""
    mode = _get_approval_config().get("mode", "manual")
    return _normalize_approval_mode(mode)


def _get_approval_timeout() -> int:
    """Read the approval timeout from config. Defaults to 60 seconds."""
    try:
        return int(_get_approval_config().get("timeout", 60))
    except (ValueError, TypeError):
        return 60


def _get_cron_approval_mode() -> str:
    """Read the cron approval mode from config. Returns 'deny' or 'approve'."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        mode = str(cfg_get(config, "approvals", "cron_mode", default="deny")).lower().strip()
        if mode in {"approve", "off", "allow", "yes"}:
            return "approve"
        return "deny"
    except Exception:
        return "deny"


def _smart_approve(command: str, description: str) -> str:
    """Use the auxiliary LLM to assess risk and decide approval.

    Returns 'approve' if the LLM determines the command is safe,
    'deny' if genuinely dangerous, or 'escalate' if uncertain.

    Inspired by OpenAI Codex's Smart Approvals guardian subagent
    (openai/codex#13860).
    """
    try:
        from agent.auxiliary_client import call_llm

        prompt = f"""You are a security reviewer for an AI coding agent. A terminal command was flagged by pattern matching as potentially dangerous.

Command: {command}
Flagged reason: {description}

Assess the ACTUAL risk of this command. Many flagged commands are false positives — for example, `python -c "print('hello')"` is flagged as "script execution via -c flag" but is completely harmless.

Rules:
- APPROVE if the command is clearly safe (benign script execution, safe file operations, development tools, package installs, git operations, etc.)
- DENY if the command could genuinely damage the system (recursive delete of important paths, overwriting system files, fork bombs, wiping disks, dropping databases, etc.)
- ESCALATE if you're uncertain

Respond with exactly one word: APPROVE, DENY, or ESCALATE"""

        response = call_llm(
            task="approval",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=16,
        )

        answer = (response.choices[0].message.content or "").strip().upper()

        if answer == "APPROVE":
            return "approve"
        elif answer == "DENY":
            return "deny"
        else:
            return "escalate"

    except Exception as e:
        logger.debug("Smart approvals: LLM call failed (%s), escalating", e)
        return "escalate"


def check_dangerous_command(command: str, env_type: str,
                            approval_callback=None,
                            cwd: Optional[str] = None) -> dict:
    """Check if a command is dangerous and handle approval.

    This is the main entry point called by terminal_tool before executing
    any command. It orchestrates detection, session checks, and prompting.

    Args:
        command: The shell command to check.
        env_type: Terminal backend type ('local', 'ssh', 'docker', etc.).
        approval_callback: Optional CLI callback for interactive prompts.

    Returns:
        {"approved": True/False, "message": str or None, ...}
    """
    unbypassable = check_unbypassable_command_guards(command, env_type, cwd=cwd)
    if not unbypassable["approved"]:
        return unbypassable
    if env_type in {"docker", "singularity", "modal", "daytona", "vercel_sandbox"}:
        return {"approved": True, "message": None}

    # --yolo: bypass regular approval prompts. Gateway /yolo is session-scoped;
    # CLI --yolo remains process-scoped via the env var for local use.
    if _YOLO_MODE_FROZEN or is_current_session_yolo_enabled():
        return {"approved": True, "message": None}

    is_dangerous, pattern_key, description = detect_dangerous_command(command)
    if not is_dangerous:
        return {"approved": True, "message": None}

    session_key = get_current_session_key()
    if is_approved(session_key, pattern_key):
        return {"approved": True, "message": None}

    is_cli = env_var_enabled("HERMES_INTERACTIVE")
    is_gateway = _is_gateway_approval_context()

    if not is_cli and not is_gateway:
        # Cron sessions: respect cron_mode config
        if env_var_enabled("HERMES_CRON_SESSION"):
            if _get_cron_approval_mode() == "deny":
                return {
                    "approved": False,
                    "message": (
                        f"BLOCKED: Command flagged as dangerous ({description}) "
                        "but cron jobs run without a user present to approve it. "
                        "Find an alternative approach that avoids this command. "
                        "To allow dangerous commands in cron jobs, set "
                        "approvals.cron_mode: approve in config.yaml."
                    ),
                }
        logger.warning(
            "AUTO-APPROVED dangerous command in non-interactive non-gateway context "
            "(pattern: %s): %s — set HERMES_INTERACTIVE or HERMES_GATEWAY_SESSION to require approval.",
            description, command[:200],
        )
        return {"approved": True, "message": None}

    if is_gateway or env_var_enabled("HERMES_EXEC_ASK"):
        submit_pending(session_key, {
            "command": command,
            "pattern_key": pattern_key,
            "description": description,
        })
        return {
            "approved": False,
            "pattern_key": pattern_key,
            "status": "approval_required",
            "command": command,
            "description": description,
            "message": (
                f"⚠️ This command is potentially dangerous ({description}). "
                f"Asking the user for approval.\n\n**Command:**\n```\n{command}\n```"
            ),
        }

    choice = prompt_dangerous_approval(command, description,
                                       approval_callback=approval_callback)

    if choice == "deny":
        return {
            "approved": False,
            "message": f"BLOCKED: User denied this potentially dangerous command (matched '{description}' pattern). Do NOT retry this command - the user has explicitly rejected it.",
            "pattern_key": pattern_key,
            "description": description,
        }

    if choice == "session":
        approve_session(session_key, pattern_key)
    elif choice == "always":
        approve_session(session_key, pattern_key)
        approve_permanent(pattern_key)
        save_permanent_allowlist(_permanent_approved)

    return {"approved": True, "message": None}


# =========================================================================
# Combined pre-exec guard (tirith + dangerous command detection)
# =========================================================================

def _format_tirith_description(tirith_result: dict) -> str:
    """Build a human-readable description from tirith findings.

    Includes severity, title, and description for each finding so users
    can make an informed approval decision.
    """
    findings = tirith_result.get("findings") or []
    if not findings:
        summary = tirith_result.get("summary") or "security issue detected"
        return f"Security scan: {summary}"

    parts = []
    for f in findings:
        severity = f.get("severity", "")
        title = f.get("title", "")
        desc = f.get("description", "")
        if title and desc:
            parts.append(f"[{severity}] {title}: {desc}" if severity else f"{title}: {desc}")
        elif title:
            parts.append(f"[{severity}] {title}" if severity else title)
    if not parts:
        summary = tirith_result.get("summary") or "security issue detected"
        return f"Security scan: {summary}"

    return "Security scan — " + "; ".join(parts)


def check_all_command_guards(command: str, env_type: str,
                             approval_callback=None,
                             cwd: Optional[str] = None) -> dict:
    """Run all pre-exec security checks and return a single approval decision.

    Gathers findings from tirith and dangerous-command detection, then
    presents them as a single combined approval request. This prevents
    a gateway force=True replay from bypassing one check when only the
    other was shown to the user.
    """
    # Skip containers for both checks
    unbypassable = check_unbypassable_command_guards(command, env_type, cwd=cwd)
    if not unbypassable["approved"]:
        return unbypassable
    if env_type in {"docker", "singularity", "modal", "daytona", "vercel_sandbox"}:
        return {"approved": True, "message": None}

    # --yolo or approvals.mode=off: bypass regular approval prompts.
    # Gateway /yolo is session-scoped; CLI --yolo remains process-scoped.
    approval_mode = _get_approval_mode()
    if _YOLO_MODE_FROZEN or is_current_session_yolo_enabled() or approval_mode == "off":
        return {"approved": True, "message": None}

    is_cli = env_var_enabled("HERMES_INTERACTIVE")
    is_gateway = _is_gateway_approval_context()
    is_ask = env_var_enabled("HERMES_EXEC_ASK")

    # Preserve the existing non-interactive behavior: outside CLI/gateway/ask
    # flows, we do not block on approvals and we skip external guard work.
    if not is_cli and not is_gateway and not is_ask:
        # Cron sessions: respect cron_mode config
        if env_var_enabled("HERMES_CRON_SESSION"):
            if _get_cron_approval_mode() == "deny":
                # Run detection to get a description for the block message
                is_dangerous, _pk, description = detect_dangerous_command(command)
                if is_dangerous:
                    return {
                        "approved": False,
                        "message": (
                            f"BLOCKED: Command flagged as dangerous ({description}) "
                            "but cron jobs run without a user present to approve it. "
                            "Find an alternative approach that avoids this command. "
                            "To allow dangerous commands in cron jobs, set "
                            "approvals.cron_mode: approve in config.yaml."
                        ),
                    }
        return {"approved": True, "message": None}

    # --- Phase 1: Gather findings from both checks ---

    # Tirith check — wrapper guarantees no raise for expected failures.
    # Only catch ImportError (module not installed).
    tirith_result = {"action": "allow", "findings": [], "summary": ""}
    try:
        from tools.tirith_security import check_command_security
        tirith_result = check_command_security(command)
    except ImportError:
        pass  # tirith module not installed — allow

    # Dangerous command check (detection only, no approval)
    is_dangerous, pattern_key, description = detect_dangerous_command(command)

    # --- Phase 2: Decide ---

    # Collect warnings that need approval
    warnings = []  # list of (pattern_key, description, is_tirith)

    session_key = get_current_session_key()

    # Tirith block/warn → approvable warning with rich findings.
    # Previously, tirith "block" was a hard block with no approval prompt.
    # Now both block and warn go through the approval flow so users can
    # inspect the explanation and approve if they understand the risk.
    if tirith_result["action"] in {"block", "warn"}:
        findings = tirith_result.get("findings") or []
        rule_id = findings[0].get("rule_id", "unknown") if findings else "unknown"
        tirith_key = f"tirith:{rule_id}"
        tirith_desc = _format_tirith_description(tirith_result)
        if not is_approved(session_key, tirith_key):
            warnings.append((tirith_key, tirith_desc, True))

    if is_dangerous:
        if not is_approved(session_key, pattern_key):
            warnings.append((pattern_key, description, False))

    # Nothing to warn about
    if not warnings:
        return {"approved": True, "message": None}

    # --- Phase 2.5: Smart approval (auxiliary LLM risk assessment) ---
    # When approvals.mode=smart, ask the aux LLM before prompting the user.
    # Inspired by OpenAI Codex's Smart Approvals guardian subagent
    # (openai/codex#13860).
    if approval_mode == "smart":
        combined_desc_for_llm = "; ".join(desc for _, desc, _ in warnings)
        verdict = _smart_approve(command, combined_desc_for_llm)
        if verdict == "approve":
            # Auto-approve and grant session-level approval for these patterns
            for key, _, _ in warnings:
                approve_session(session_key, key)
            logger.debug("Smart approval: auto-approved '%s' (%s)",
                         command[:60], combined_desc_for_llm)
            return {"approved": True, "message": None,
                    "smart_approved": True,
                    "description": combined_desc_for_llm}
        elif verdict == "deny":
            combined_desc_for_llm = "; ".join(desc for _, desc, _ in warnings)
            return {
                "approved": False,
                "message": f"BLOCKED by smart approval: {combined_desc_for_llm}. "
                           "The command was assessed as genuinely dangerous. Do NOT retry.",
                "smart_denied": True,
            }
        # verdict == "escalate" → fall through to manual prompt

    # --- Phase 3: Approval ---

    # Combine descriptions for a single approval prompt
    combined_desc = "; ".join(desc for _, desc, _ in warnings)
    primary_key = warnings[0][0]
    all_keys = [key for key, _, _ in warnings]
    has_tirith = any(is_t for _, _, is_t in warnings)

    # Gateway/async approval — block the agent thread until the user
    # responds with /approve or /deny, mirroring the CLI's synchronous
    # input() flow.  The agent never sees "approval_required"; it either
    # gets the command output (approved) or a definitive "BLOCKED" message.
    if is_gateway or is_ask:
        notify_cb = None
        with _lock:
            notify_cb = _gateway_notify_cbs.get(session_key)

        if notify_cb is not None:
            # --- Blocking gateway approval (queue-based) ---
            # Each call gets its own _ApprovalEntry so parallel subagents
            # and execute_code threads can block concurrently.
            approval_data = {
                "command": command,
                "pattern_key": primary_key,
                "pattern_keys": all_keys,
                "description": combined_desc,
            }
            entry = _ApprovalEntry(approval_data)
            with _lock:
                _gateway_queues.setdefault(session_key, []).append(entry)

            # Notify plugins that an approval is being requested. Fires before
            # the gateway notify callback so observers (e.g. macOS notifier
            # plugins, audit logs, Slack alerts) get the event in real time.
            _fire_approval_hook(
                "pre_approval_request",
                command=command,
                description=combined_desc,
                pattern_key=primary_key,
                pattern_keys=list(all_keys),
                session_key=session_key,
                surface="gateway",
            )

            # Notify the user (bridges sync agent thread → async gateway)
            try:
                notify_cb(approval_data)
            except Exception as exc:
                logger.warning("Gateway approval notify failed: %s", exc)
                with _lock:
                    queue = _gateway_queues.get(session_key, [])
                    if entry in queue:
                        queue.remove(entry)
                    if not queue:
                        _gateway_queues.pop(session_key, None)
                return {
                    "approved": False,
                    "message": "BLOCKED: Failed to send approval request to user. Do NOT retry.",
                    "pattern_key": primary_key,
                    "description": combined_desc,
                }

            # Block until the user responds or timeout (default 5 min).
            # Poll in short slices so we can fire activity heartbeats every
            # ~10s to the agent's inactivity tracker.  Without this, the
            # blocking event.wait() never touches activity, and the
            # gateway's inactivity watchdog (agent.gateway_timeout, default
            # 1800s) kills the agent while the user is still responding to
            # the approval prompt.  Mirrors the _wait_for_process() cadence
            # in tools/environments/base.py.
            timeout = _get_approval_config().get("gateway_timeout", 300)
            try:
                timeout = int(timeout)
            except (ValueError, TypeError):
                timeout = 300

            try:
                from tools.environments.base import touch_activity_if_due
            except Exception:  # pragma: no cover
                touch_activity_if_due = None

            _now = time.monotonic()
            _deadline = _now + max(timeout, 0)
            _activity_state = {"last_touch": _now, "start": _now}
            resolved = False
            while True:
                _remaining = _deadline - time.monotonic()
                if _remaining <= 0:
                    break
                # 1s poll slice — the event is set immediately when the
                # user responds, so slice length only controls heartbeat
                # cadence, not user-visible responsiveness.
                if entry.event.wait(timeout=min(1.0, _remaining)):
                    resolved = True
                    break
                if touch_activity_if_due is not None:
                    touch_activity_if_due(
                        _activity_state, "waiting for user approval"
                    )

            # Clean up this entry from the queue
            with _lock:
                queue = _gateway_queues.get(session_key, [])
                if entry in queue:
                    queue.remove(entry)
                if not queue:
                    _gateway_queues.pop(session_key, None)

            choice = entry.result
            # Normalize outcome for the post hook. Unresolved (timeout) and
            # None both mean the user never responded; report that explicitly
            # so plugins can distinguish timeout from explicit deny.
            _outcome = (
                "timeout" if not resolved
                else (choice if choice else "timeout")
            )
            _fire_approval_hook(
                "post_approval_response",
                command=command,
                description=combined_desc,
                pattern_key=primary_key,
                pattern_keys=list(all_keys),
                session_key=session_key,
                surface="gateway",
                choice=_outcome,
            )

            if not resolved or choice is None or choice == "deny":
                # Consent contract: silence is NOT consent, and an explicit
                # deny is also a hard halt — both produce a BLOCKED outcome
                # that names the agent's most common evasion paths (retry,
                # rephrase, achieve the same outcome via a different command).
                # See issue #24912 for the original incident.
                if not resolved:
                    reason = "timed out without user response"
                    timeout_addendum = " Silence is not consent."
                    outcome = "timeout"
                else:
                    reason = "denied by user"
                    timeout_addendum = ""
                    outcome = "denied"
                return {
                    "approved": False,
                    "message": (
                        f"BLOCKED: Command {reason}. The user has NOT consented "
                        f"to this action. Do NOT retry this command, do NOT "
                        f"rephrase it, and do NOT attempt the same outcome via "
                        f"a different command. Stop the current workflow and "
                        f"wait for the user to respond before taking any "
                        f"further destructive or irreversible action."
                        f"{timeout_addendum}"
                    ),
                    "pattern_key": primary_key,
                    "description": combined_desc,
                    "outcome": outcome,
                    "user_consent": False,
                }

            # User approved — persist based on scope (same logic as CLI)
            for key, _, is_tirith in warnings:
                if choice == "session" or (choice == "always" and is_tirith):
                    approve_session(session_key, key)
                elif choice == "always":
                    approve_session(session_key, key)
                    approve_permanent(key)
                    save_permanent_allowlist(_permanent_approved)
                # choice == "once": no persistence — command allowed this
                # single time only, matching the CLI's behavior.

            return {"approved": True, "message": None,
                    "user_approved": True, "description": combined_desc}

        # Fallback: no gateway callback registered (e.g. cron, batch).
        # Return approval_required for backward compat.
        submit_pending(session_key, {
            "command": command,
            "pattern_key": primary_key,
            "pattern_keys": all_keys,
            "description": combined_desc,
        })
        return {
            "approved": False,
            "pattern_key": primary_key,
            "status": "pending_approval",
            "approval_pending": True,
            "command": command,
            "description": combined_desc,
            "message": (
                f"⚠️ {combined_desc}. Asking the user for approval.\n\n**Command:**\n```\n{command}\n```"
            ),
        }

    # CLI interactive: single combined prompt
    # Hide [a]lways when any tirith warning is present
    _fire_approval_hook(
        "pre_approval_request",
        command=command,
        description=combined_desc,
        pattern_key=primary_key,
        pattern_keys=list(all_keys),
        session_key=session_key,
        surface="cli",
    )
    choice = prompt_dangerous_approval(command, combined_desc,
                                       allow_permanent=not has_tirith,
                                       approval_callback=approval_callback)
    _fire_approval_hook(
        "post_approval_response",
        command=command,
        description=combined_desc,
        pattern_key=primary_key,
        pattern_keys=list(all_keys),
        session_key=session_key,
        surface="cli",
        choice=choice,
    )

    if choice == "deny":
        return {
            "approved": False,
            "message": (
                "BLOCKED: User denied this command. The user has NOT consented "
                "to this action. Do NOT retry this command, do NOT rephrase "
                "it, and do NOT attempt the same outcome via a different "
                "command. Stop the current workflow and wait for the user "
                "to respond before taking any further destructive or "
                "irreversible action."
            ),
            "pattern_key": primary_key,
            "description": combined_desc,
            "outcome": "denied",
            "user_consent": False,
        }

    # Persist approval for each warning individually
    for key, _, is_tirith in warnings:
        if choice == "session" or (choice == "always" and is_tirith):
            # tirith: session only (no permanent broad allowlisting)
            approve_session(session_key, key)
        elif choice == "always":
            # dangerous patterns: permanent allowed
            approve_session(session_key, key)
            approve_permanent(key)
            save_permanent_allowlist(_permanent_approved)

    return {"approved": True, "message": None,
            "user_approved": True, "description": combined_desc}


# Load permanent allowlist from config on module import
load_permanent_allowlist()
