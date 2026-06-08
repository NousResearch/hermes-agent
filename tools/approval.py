"""Dangerous command approval -- detection, prompting, and per-session state.

This module is the single source of truth for the dangerous command system:
- Pattern detection (DANGEROUS_PATTERNS, detect_dangerous_command)
- Per-session approval state (thread-safe, keyed by session_key)
- Approval prompting (CLI interactive + gateway async)
- Smart approval via auxiliary LLM (auto-approve low-risk commands)
- Permanent allowlist persistence (config.yaml)
"""

import contextvars
import logging
import os
import re
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
_approval_turn_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "approval_turn_id",
    default="",
)
_approval_tool_call_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "approval_tool_call_id",
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
        kwargs.setdefault("turn_id", _approval_turn_id.get())
        kwargs.setdefault("tool_call_id", _approval_tool_call_id.get())
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


def set_current_observability_context(
    *,
    turn_id: str = "",
    tool_call_id: str = "",
) -> tuple[contextvars.Token[str], contextvars.Token[str]]:
    """Bind active tool correlation IDs to approval hooks."""
    return (
        _approval_turn_id.set(turn_id or ""),
        _approval_tool_call_id.set(tool_call_id or ""),
    )


def reset_current_observability_context(
    tokens: tuple[contextvars.Token[str], contextvars.Token[str]],
) -> None:
    """Restore prior approval hook correlation IDs."""
    turn_token, tool_token = tokens
    _approval_tool_call_id.reset(tool_token)
    _approval_turn_id.reset(turn_token)


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
# ~/.hermes/config.yaml IS the security policy: approvals.mode, yolo, and the
# permanent-approval allowlist live here, and the config cache is mtime-keyed
# so a write takes effect mid-session (the agent could flip approvals.mode=off
# and immediately bypass the gate). Pair the write_file/patch deny (file_tools
# _check_sensitive_path) with terminal-side coverage so `sed -i`, `tee`, `>`,
# `cp`, etc. targeting it are gated too — otherwise the deny is unpaired
# theater. Mirrors _HERMES_ENV_PATH; matches the HERMES_HOME override form as
# well as ~/.hermes/.
_HERMES_CONFIG_PATH = (
    r'(?:~\/\.hermes/|'
    r'(?:\$home|\$\{home\})/\.hermes/|'
    r'(?:\$hermes_home|\$\{hermes_home\})/)'
    r'config\.yaml\b'
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
    rf'{_HERMES_CONFIG_PATH}|'
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
    r'(?:^|[;&|`\n]|&&|\|\||\$\()\s*sudo\s+-S\b',
    re.IGNORECASE)


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
    normalized = _normalize_command_for_detection(command).lower()
    if _SUDO_STDIN_RE.search(normalized):
        return (True, "sudo password guessing via stdin (sudo -S)")
    return (False, None)


def detect_hardline_command(command: str) -> tuple:
    """Check if a command matches the unconditional hardline blocklist.

    Returns:
        (is_hardline, description) or (False, None)
    """
    normalized = _normalize_command_for_detection(command).lower()
    for pattern_re, description in HARDLINE_PATTERNS_COMPILED:
        if pattern_re.search(normalized):
            return (True, description)
    return (False, None)


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
    # Docker container lifecycle — any user with docker.sock mounted (a common
    # Docker Compose pattern) gives the agent the ability to restart/stop/kill
    # containers without approval.  These are agent-initiated lifecycle operations
    # that should always require user consent, just like `hermes gateway restart`
    # already does for the gateway process.
    (r'\bdocker\s+compose\s+(restart|stop|kill|down)\b', "docker compose restart/stop/kill/down (container lifecycle)"),
    (r'\bdocker\s+(restart|stop|kill)\b', "docker restart/stop/kill (container lifecycle)"),
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
    # In-place edit of a Hermes-managed security file (~/.hermes/config.yaml or
    # .env). sed -i bypasses the redirection/tee patterns above because it
    # mutates the file directly. Pairs the file_tools write_file/patch deny so
    # the terminal side is not an open door. See #14639.
    (rf'\bsed\s+-[^\s]*i.*(?:{_HERMES_CONFIG_PATH}|{_HERMES_ENV_PATH})', "in-place edit of Hermes config/env"),
    (rf'\bsed\s+--in-place\b.*(?:{_HERMES_CONFIG_PATH}|{_HERMES_ENV_PATH})', "in-place edit of Hermes config/env (long flag)"),
    # perl -i and ruby -i perform the same in-place mutation as sed -i but are
    # not caught by the -e/-c script-execution pattern above (which targets code
    # evaluation, not file mutation). Pairs the sed -i coverage from #14639.
    # The -i flag can appear as its own token after other flags
    # (`perl -p -i -e ... config.yaml`), combined (`perl -pi -e`), or with a
    # backup suffix (`perl -i.bak`). Match any flag token containing `i`
    # anywhere in the args, not just the first token — `perl -e '...'` (code
    # eval, no -i) does not trip because it has no `-...i` flag token.
    (rf'\b(?:perl|ruby)\b.*(?:^|\s)-[^\s]*i\b.*(?:{_HERMES_CONFIG_PATH}|{_HERMES_ENV_PATH})', "in-place edit of Hermes config/env (perl/ruby)"),
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
    # Strip shell backslash-escapes: r\m → rm. Prevents \-injection bypass.
    command = re.sub(r'\\([^\n])', r'\1', command)
    # Strip empty-string literals that split tokens: r''m → rm, r""m → rm.
    command = re.sub(r"''|\"\"", '', command)
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
    """One pending dangerous-command approval inside a gateway session.

    The ``event`` + ``result`` pair is the in-process wake-up mechanism
    for the blocked agent thread. It is NOT the source of truth for
    "should this command execute" — that decision is gated by the
    durable approval store (see :mod:`tools.approval_store`). When the
    store is configured, ``approval_id`` ties this entry to a persisted
    row whose ``status`` column is the actual security boundary.
    """
    __slots__ = ("event", "data", "result", "approval_id")

    def __init__(self, data: dict, approval_id: Optional[str] = None):
        self.event = threading.Event()
        self.data = data          # command, description, pattern_keys, …
        self.result: Optional[str] = None  # "once"|"session"|"always"|"deny"
        self.approval_id = approval_id


_gateway_queues: dict[str, list] = {}        # session_key → [_ApprovalEntry, …]
_gateway_notify_cbs: dict[str, object] = {}  # session_key → callable(approval_data)


# ─── Durable approval store (Phase 2/3 wiring) ─────────────────────────────
# Optional: when set, every gateway approval entry also gets persisted to
# the store, and /approve <id> can consume from there atomically. When None,
# the legacy in-memory-only path is used unchanged. This keeps existing
# tests passing while we incrementally wire the store into production
# code paths.
_default_approval_store = None  # type: ignore[assignment]


def set_default_approval_store(store) -> None:  # type: ignore[no-untyped-def]
    """Inject the approval store used by gateway approval flows.

    Tests pass ``None`` (the default) or a fake store. The real gateway
    init (gateway/run.py) wires a SqliteApprovalStore via this hook.
    Set to ``None`` to disable durable persistence (legacy behavior).

    Installing a non-None store also clears the init-failure flag — this
    is the recovery path after a failed boot-time wiring.
    """
    global _default_approval_store
    _default_approval_store = store
    if store is not None:
        clear_approval_store_init_failed()


def get_default_approval_store():  # type: ignore[no-untyped-def]
    """Return the currently-configured store, or None."""
    return _default_approval_store


# ─── Init-failure tracking (concern #1 from Hermes review v2) ──────────────
# Gateway boot calls set_default_approval_store(SqliteApprovalStore()). If
# that raises, the gateway is in a degraded state: store is None, but unlike
# "store is None by design" (tests, scripts) this is a production
# mis-configuration. We must NOT silently fall back to the legacy in-memory
# FIFO under such conditions — dangerous-command approval has to fail closed
# until an operator restores the store.
#
# The flag is process-local and reset on store re-wiring; gateway init's
# except-block sets it via mark_approval_store_init_failed().
_approval_store_init_failed: bool = False
_approval_store_init_failure_reason: str = ""


def mark_approval_store_init_failed(reason: str) -> None:
    """Called by gateway init (or any production wiring code) when
    set_default_approval_store fails. Subsequent gateway-context approval
    requests will fail closed instead of degrading to legacy FIFO."""
    global _approval_store_init_failed, _approval_store_init_failure_reason
    _approval_store_init_failed = True
    _approval_store_init_failure_reason = reason or "unknown"
    logger.error(
        "Approval store initialisation marked as FAILED: %s. Subsequent "
        "gateway approval requests will fail closed.", _approval_store_init_failure_reason,
    )


def clear_approval_store_init_failed() -> None:
    """Reset the init-failure flag. Called by set_default_approval_store
    when a real store is installed (recovery path), and by tests."""
    global _approval_store_init_failed, _approval_store_init_failure_reason
    _approval_store_init_failed = False
    _approval_store_init_failure_reason = ""


def is_approval_store_init_failed() -> bool:
    return _approval_store_init_failed


# ─── Phase 3: stricter-runtime fail-closed guard ───────────────────────────
# Risk ordering for the post-consume re-classification check. Unknown values
# rank ABOVE high so an unrecognised risk_level on the pinned proposal causes
# fail-closed rather than silent acceptance.
_RISK_ORDER = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


def _risk_rank(value) -> int:  # type: ignore[no-untyped-def]
    """Return the order of *value* with unknown ranking above all known."""
    if not value or not isinstance(value, str):
        return 999
    return _RISK_ORDER.get(value.lower(), 999)


def _classify_runtime_risk(command: str) -> str:
    """Re-classify *command* with the live classifier at execution-thread
    wake-up time.

    Returns a coarse risk level keyword: ``low`` / ``medium`` / ``high``.
    This is the same lattice used in :class:`ApprovalProposal`'s
    ``risk_level``. Used by Phase 3 fail-closed guard to detect cases
    where the runtime classifier has become STRICTER than the pinned
    policy between proposal-creation and execution.

    Hardlines (``rm -rf /``, ``mkfs``, …) are already blocked before
    reaching the gateway approval path, but we still report them as
    ``high`` here for defensive parity with the pinning logic.
    """
    is_hardline, _ = detect_hardline_command(command)
    if is_hardline:
        return "high"
    is_dangerous, _, desc = detect_dangerous_command(command)
    if is_dangerous:
        return _classify_pattern_risk(desc or "")
    return "low"


# Keywords in the pattern description that elevate medium → high. These
# are the genuinely-destructive non-hardline patterns: data loss,
# system-file overwrites, raw-device writes, remote-code-execution,
# privilege escalation, ratio-of-blast-radius high enough that "I
# approved without reading" is unrecoverable.
_HIGH_RISK_DESCRIPTION_MARKERS = (
    "recursive delete",      # rm -rf in any path
    "delete in root",
    "recursive world",       # chmod -R 777
    "recursive chown to root",
    "format filesystem",
    "write to block device",
    "raw block device",
    "block device",
    "sql drop",
    "sql delete without",    # DELETE FROM without WHERE
    "sql truncate",
    "overwrite system config",
    "overwrite system file",
    "pipe remote content",   # curl|sh
    "execute remote script",
    "kill all processes",
)


def _render_approval_display_text(
    *,
    approval_id: str,
    risk_level: str,
    command: str,
    risk_reason: str,
    cwd: Optional[str] = None,
    backend: Optional[str] = None,
    diff_text: Optional[str] = None,
    diff_summary: Optional[str] = None,
) -> str:
    """Build the user-visible approval prompt text.

    Phase 4 contract: when ``risk_level == 'high'`` the rendered text MUST
    include a HIGH RISK marker, default-deny statement, the exact command,
    available context (cwd/backend), pinned risk reason, the approval id,
    and EITHER a diff/summary OR an explicit "no diff available" warning.

    Platforms (Telegram/Matrix/Slack/etc) MAY override rendering using the
    structured fields in approval_data; this string is the fallback /
    canonical rendering and is what tests assert against.
    """
    lines: list = []
    is_high = risk_level == "high"

    if is_high:
        lines.append("🚨 HIGH RISK")
        lines.append("Default: DENY")
        lines.append("")

    lines.append(f"Command:")
    lines.append(f"  {command}")
    lines.append("")

    if cwd or backend:
        lines.append("Context:")
        if cwd:
            lines.append(f"  cwd: {cwd}")
        if backend:
            lines.append(f"  backend: {backend}")
        lines.append("")

    lines.append("Pinned policy:")
    lines.append(f"  risk: {risk_level}")
    if risk_reason:
        lines.append(f"  reason: {risk_reason}")
    lines.append("")

    if diff_text or diff_summary:
        lines.append("Diff / change summary:")
        if diff_summary:
            lines.append(f"  {diff_summary}")
        if diff_text:
            # Bound the inline preview so a huge diff doesn't fill the message.
            preview = diff_text if len(diff_text) <= 1500 else (
                diff_text[:1500] + "\n  …[diff truncated; inspect full via command-side tooling]"
            )
            lines.append("  ---")
            for ln in preview.split("\n"):
                lines.append(f"  {ln}")
        lines.append("")
    else:
        # Spec: missing diff MUST be explicit, not silent.
        lines.append(
            "Diff/summary: NOT AVAILABLE — approve only if you have "
            "independently reviewed the command/context."
        )
        lines.append("")

    if is_high:
        lines.append("Approve only after reading the diff/summary:")
        lines.append(f"  /approve {approval_id}")
        lines.append("")
        lines.append("Deny:")
        lines.append(f"  /deny {approval_id}")
    else:
        lines.append(f"Approve: /approve {approval_id}")
        lines.append(f"Deny:    /deny {approval_id}")

    return "\n".join(lines)


def _classify_pattern_risk(description: str) -> str:
    """Map a dangerous-pattern description string to ``high`` or ``medium``.

    Hardlines and missing/None descriptions both map to ``high`` (defensive
    default: an unknown dangerous pattern is treated as high until proven
    safe via explicit categorisation).
    """
    if not description:
        return "high"
    desc_lower = description.lower()
    for marker in _HIGH_RISK_DESCRIPTION_MARKERS:
        if marker in desc_lower:
            return "high"
    return "medium"


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


def resolve_gateway_approval_by_id(session_key: str, approval_id: str,
                                   choice: str) -> int:
    """Atomically consume/deny a specific approval by id and signal the
    matching blocked agent thread.

    This is the *security gate* for the per-id path. Order matters:

      1. Look up the proposal in the durable store. If the id is unknown,
         the requesting session_key does not own it, or the proposal is
         not pending → return 0 with no side effects.
      2. Atomically transition via ``store.consume`` (for approve choices)
         or ``store.deny`` (for deny). If that returns ``None``/``False``
         — already consumed / denied / expired — return 0.
      3. Only after the store transition succeeds do we find the matching
         in-memory ``_ApprovalEntry`` (by approval_id), set its result,
         and signal its event. The blocked agent thread wakes and runs
         the command from its (pinned) entry data.

    A return of 0 means "no execution will happen". 1 means the proposal
    was consumed/denied and the matching waiter (if any) was signalled.

    If the store is not configured, falls back to the legacy FIFO path
    (see :func:`resolve_gateway_approval`).
    """
    store = get_default_approval_store()
    if store is None:
        # No durable store. Per security spec we should not silently
        # fall back to FIFO under a per-id request, but for the rollout
        # window we accept the no-store legacy behaviour to keep the
        # existing test suite running. Production gateway always wires
        # a store, so this branch is operator-misconfiguration only.
        logger.warning(
            "resolve_gateway_approval_by_id called but no store configured; "
            "falling back to FIFO (approval_id=%s ignored)", approval_id,
        )
        return resolve_gateway_approval(session_key, choice)

    # Lookup + session-key authorization check.
    proposal = store.get(approval_id)
    if proposal is None:
        return 0
    if proposal.session_key and proposal.session_key != session_key:
        # Cross-session approval attempt — fail closed silently.
        logger.warning(
            "Gateway approval id %s requested by session %s but owned by %s; "
            "rejecting",
            approval_id, session_key, proposal.session_key,
        )
        return 0
    if proposal.status != "pending":
        return 0

    # Atomic store-level transition. If we lose the race or the proposal
    # is already terminal, this returns None/False and we exit without
    # signalling anything.
    consumed_by = f"session:{session_key}"
    if choice == "deny":
        ok = store.deny(approval_id, denied_by=consumed_by)
        if not ok:
            return 0
    else:
        got = store.consume(approval_id, consumed_by=consumed_by)
        if got is None:
            return 0

    # Now find and signal the matching in-memory waiter. The store gate
    # has already irreversibly committed; the in-memory step is purely
    # the wake mechanism for the blocked agent thread.
    with _lock:
        target = None
        for queued in _gateway_queues.get(session_key, []):
            if getattr(queued, "approval_id", None) == approval_id:
                target = queued
                break
        if target is not None:
            queue = _gateway_queues.get(session_key, [])
            if target in queue:
                queue.remove(target)
                if not queue:
                    _gateway_queues.pop(session_key, None)
    if target is not None:
        target.result = choice
        target.event.set()
        return 1
    # Store accepted but no live waiter — original agent thread is gone
    # (gateway restart, agent run finished). The store row is now
    # consumed/denied; no command will execute. We mark the audit row
    # accordingly and return -1 to signal "orphan: store updated but no
    # execution happened". Handler should surface a DISTINCT message to
    # the user — saying "approved" here would lie about a side effect
    # that never occurred.
    if choice != "deny":
        try:
            store.mark_post_consume(
                approval_id, executed=False,
                reason="orphan_no_waiter",
            )
        except Exception as e:
            logger.warning(
                "Failed to mark orphan-consume execution status "
                "(approval_id=%s): %s", approval_id, e,
            )
    logger.info(
        "Gateway approval %s consumed/denied but no live waiter "
        "(session=%s) — store row updated with execution_status=%s; "
        "no command will execute",
        approval_id, session_key,
        "blocked_after_consume" if choice != "deny" else "not_started",
    )
    return -1


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
                            approval_callback=None) -> dict:
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
    if env_type in {"docker", "singularity", "modal", "daytona"}:
        return {"approved": True, "message": None}

    # Hardline floor: commands with no recovery path (rm -rf /, mkfs, dd
    # to raw device, shutdown/reboot, fork bomb, kill -1) are blocked
    # unconditionally, BEFORE the yolo bypass.  Opting into yolo is
    # trusting the agent with your files and services, not trusting it
    # to wipe the disk or power the box off.
    is_hardline, hardline_desc = detect_hardline_command(command)
    if is_hardline:
        logger.warning("Hardline block: %s (command: %s)", hardline_desc, command[:200])
        return _hardline_block_result(hardline_desc)

    # --yolo: bypass all approval prompts. Gateway /yolo is session-scoped;
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


def _await_gateway_decision(session_key: str, notify_cb, approval_data: dict,
                            *, surface: str = "gateway") -> dict:
    """Enqueue *approval_data*, notify the user, and block the calling agent
    thread until the request is resolved or the gateway approval timeout
    elapses — firing pre/post approval hooks and cleaning up the queue entry.

    Shared by the terminal command guard (``check_all_command_guards``) and
    the execute_code guard (``check_execute_code_guard``) so the fiddly
    heartbeat-polling wait loop lives in one place.

    Returns ``{"resolved": bool, "choice": str|None}`` on completion, or
    ``{"resolved": False, "choice": None, "notify_failed": True}`` if the
    notify callback raised.  Persistence of an approved choice and building
    the final tool-facing result dict remain the caller's responsibility.
    """
    command = approval_data.get("command", "")
    description = approval_data.get("description", "")
    primary_key = approval_data.get("pattern_key", "")
    all_keys = approval_data.get("pattern_keys", [primary_key])

    # ── Durable approval store (Phase 2/3) ─────────────────────────────
    # Generate an approval_id and submit a persisted proposal to the
    # configured store, if any. This is "shadow persistence" in this
    # commit: the in-memory queue still drives execution; the store
    # row is a parallel record that future commits will promote to
    # source-of-truth for /approve <id> consumption.
    #
    # If no store is configured (legacy tests, scripts), approval_id
    # stays None and nothing changes vs. previous behavior.
    store = get_default_approval_store()
    approval_id: Optional[str] = None

    # If the gateway tried to wire a store at boot and failed, we are
    # in a degraded state. Production gateway-context approval requests
    # MUST fail closed here rather than silently degrade to the legacy
    # in-memory FIFO path. This addresses the specific 'silent
    # degradation' concern from review v2.
    if store is None and is_approval_store_init_failed():
        logger.error(
            "FAIL CLOSED: gateway-context approval requested but the "
            "durable store is in init-failed state (reason=%r). Refusing "
            "to fall back to legacy in-memory path. Operator must restore "
            "store via re-wiring before approvals can resume.",
            _approval_store_init_failure_reason,
        )
        _fire_approval_hook(
            "post_approval_response",
            command=command,
            description=description,
            pattern_key=primary_key,
            pattern_keys=list(all_keys),
            session_key=session_key,
            surface=surface,
            choice="store_failed",
        )
        return {
            "resolved": False,
            "choice": None,
            "store_failed": True,
        }

    if store is not None:
        import secrets as _secrets
        approval_id = _secrets.token_urlsafe(8)

        # Phase 4: real risk classification at proposal time. The
        # description coming from check_all_command_guards is the
        # classifier's own match. Map it to high/medium per the keyword
        # table. Hardlines never reach here (blocked earlier) but if a
        # description is empty/unknown, default to high (defensive).
        risk_level = _classify_pattern_risk(description or primary_key or "")
        diff_text = approval_data.get("diff_text")
        diff_summary = approval_data.get("diff_summary")
        cwd = approval_data.get("cwd")
        backend = approval_data.get("backend")

        display_text = _render_approval_display_text(
            approval_id=approval_id,
            risk_level=risk_level,
            command=command,
            risk_reason=description or primary_key or "",
            cwd=cwd,
            backend=backend,
            diff_text=diff_text,
            diff_summary=diff_summary,
        )

        # Carry pinned policy + render data into approval_data so the
        # platform-side notify callback can either render structured or
        # use the pre-built display_text verbatim.
        approval_data = {
            **approval_data,
            "approval_id": approval_id,
            "risk_level": risk_level,
            "default_decision": "deny",
            "display_text": display_text,
        }
        try:
            from tools.approval_store import ApprovalProposal
            timeout_for_proposal = _get_approval_config().get(
                "gateway_timeout", 300
            )
            try:
                timeout_for_proposal = int(timeout_for_proposal)
            except (ValueError, TypeError):
                timeout_for_proposal = 300
            created = time.time()
            proposal = ApprovalProposal(
                approval_id=approval_id,
                created_at=created,
                expires_at=created + max(timeout_for_proposal, 0),
                session_key=session_key,
                command=command,
                cwd=cwd,
                backend=backend,
                risk_level=risk_level,
                risk_reason=description or primary_key or "dangerous-command-pattern",
                policy_decision="needs_approval",
                requires_explicit_approval=True,
                default_decision="deny",
                diff_text=diff_text,
                diff_summary=diff_summary,
                display_text=display_text,
            )
            store.submit(proposal)
        except Exception as e:
            # FAIL CLOSED: when the durable store cannot persist a
            # proposal (DB unavailable, validation refused, disk full,
            # …) we MUST NOT degrade to the legacy in-memory path.
            # An approval flow without persistence is exactly the
            # trust gap this rewrite eliminated; falling back would
            # silently reintroduce it.
            #
            # Per spec: store-failure is no-execution. The agent thread
            # gets {"resolved": False, "choice": None, "store_failed":
            # True} so its caller can surface a clear error to the
            # user/model without ever queuing an _ApprovalEntry that
            # could be approved.
            logger.error(
                "FAIL CLOSED: could not persist gateway approval proposal "
                "(approval_id=%s, command=%r): %s — refusing to fall back "
                "to in-memory path; no command will execute",
                approval_id, command[:100], e, exc_info=True,
            )
            _fire_approval_hook(
                "post_approval_response",
                command=command,
                description=description,
                pattern_key=primary_key,
                pattern_keys=list(all_keys),
                session_key=session_key,
                surface=surface,
                choice="store_failed",
            )
            return {
                "resolved": False,
                "choice": None,
                "store_failed": True,
            }

    entry = _ApprovalEntry(approval_data, approval_id=approval_id)
    with _lock:
        _gateway_queues.setdefault(session_key, []).append(entry)

    def _drop_entry() -> None:
        with _lock:
            queue = _gateway_queues.get(session_key, [])
            if entry in queue:
                queue.remove(entry)
            if not queue:
                _gateway_queues.pop(session_key, None)

    # Notify plugins that an approval is being requested. Fires before the
    # gateway notify callback so observers get the event in real time.
    _fire_approval_hook(
        "pre_approval_request",
        command=command,
        description=description,
        pattern_key=primary_key,
        pattern_keys=list(all_keys),
        session_key=session_key,
        surface=surface,
    )

    # Notify the user (bridges sync agent thread → async gateway)
    try:
        notify_cb(approval_data)
    except Exception as exc:
        logger.warning("Gateway approval notify failed: %s", exc)
        _drop_entry()
        return {"resolved": False, "choice": None, "notify_failed": True}

    # Block until the user responds or timeout (default 5 min). Poll in short
    # slices so we can fire activity heartbeats every ~10s to the agent's
    # inactivity tracker — otherwise the gateway watchdog kills the agent
    # while the user is still responding. Mirrors _wait_for_process() cadence.
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
        if entry.event.wait(timeout=min(1.0, _remaining)):
            resolved = True
            break
        if touch_activity_if_due is not None:
            touch_activity_if_due(_activity_state, "waiting for user approval")

    _drop_entry()

    choice = entry.result

    # ── Phase 3: stricter-runtime fail-closed guard ────────────────────
    # The user just released the consume gate (via /approve <id>). Before
    # the agent thread resumes and runs the command, re-classify the
    # command against the live classifier/policy. If the live classifier
    # has become STRICTER than the policy that was pinned at proposal
    # creation (e.g. a new dangerous pattern landed between proposal and
    # approval, or a regression made a previously-safe command risky),
    # fail closed — do not silently execute under stale weaker risk.
    #
    # Allowed transitions:
    #   pinned high  + runtime low/med  → still high (no downgrade)
    #   pinned low   + runtime high     → FAIL CLOSED
    #   pinned med   + runtime high     → FAIL CLOSED
    #   missing pinned                   → FAIL CLOSED
    #   matching levels                 → proceed
    #
    # The consumed approval is intentionally NOT put back into pending —
    # store row stays consumed/denied. The user must obtain a new approval
    # with fresh pinned policy if they want to retry. That keeps the
    # exactly-once invariant intact.
    if resolved and choice in {"once", "session", "always"} and \
            approval_id is not None and store is not None:
        try:
            pinned_proposal = store.get(approval_id)
        except Exception as e:
            logger.error(
                "Phase 3 guard: could not load pinned proposal %s "
                "(failing closed): %s", approval_id, e,
            )
            choice = "deny"
        else:
            if pinned_proposal is None:
                logger.error(
                    "Phase 3 guard: pinned proposal %s missing post-consume "
                    "(failing closed)", approval_id,
                )
                choice = "deny"
            else:
                pinned_risk = pinned_proposal.risk_level
                try:
                    runtime_risk = _classify_runtime_risk(command)
                except Exception as e:
                    # Classifier crashed — cannot prove safety, fail closed.
                    logger.error(
                        "Phase 3 guard: runtime classifier raised "
                        "(failing closed) for approval %s: %s",
                        approval_id, e, exc_info=True,
                    )
                    choice = "deny"
                else:
                    if _risk_rank(runtime_risk) > _risk_rank(pinned_risk):
                        logger.warning(
                            "Phase 3 fail-closed: runtime risk=%s exceeds "
                            "pinned risk=%s for approval %s (command=%r) — "
                            "overriding choice from %s to deny; user must "
                            "obtain new approval",
                            runtime_risk, pinned_risk, approval_id,
                            command[:120], choice,
                        )
                        choice = "deny"

    # Normalize outcome for the post hook. Unresolved (timeout) and None both
    # mean the user never responded; report that explicitly so plugins can
    # distinguish timeout from explicit deny.
    _outcome = "timeout" if not resolved else (choice if choice else "timeout")

    # ── Mirror outcome to the durable store + record execution audit ───
    # The store row may or may not already be in a terminal state:
    #
    #   - resolve_gateway_approval_by_id (the strict /approve <id> path,
    #     used by the slash-command handler) already transitioned the
    #     row to consumed/denied before signalling the event.
    #
    #   - resolve_gateway_approval (legacy FIFO path, still used by
    #     platform inline-button callbacks in Telegram, Slack, Matrix,
    #     QQBot, Feishu, api_server) signals the event but does NOT
    #     touch the store. We need to transition here.
    #
    # Both consume() and deny() are idempotent-by-WHERE-clause: if the
    # row is already terminal, they return None/False harmlessly.
    #
    # We distinguish ``user_choice`` (what the user actually clicked,
    # captured on entry.result) from ``choice`` (the possibly-Phase-3-
    # overridden final outcome). That distinction is what makes
    # execution_status auditable:
    #
    #   user_choice = approve, final_choice = approve  → executed
    #   user_choice = approve, final_choice = deny     → blocked_after_consume
    #                                                    (Phase 3 overrode)
    #   user_choice = deny                              → not_started (never executes)
    user_choice = entry.result  # raw user input, pre-Phase-3
    if approval_id is not None and store is not None and resolved:
        try:
            if user_choice == "deny":
                store.deny(approval_id, denied_by=f"session:{session_key}")
                # execution_status stays at 'not_started' — denied
                # proposals never reach the post-consume audit path.
            elif user_choice in {"once", "session", "always"}:
                store.consume(approval_id, consumed_by=f"session:{session_key}")
                if choice == "deny":
                    # Phase 3 guard fail-closed the user's approve.
                    # status='consumed' (user clicked /approve), but
                    # execution_status='blocked_after_consume' so the
                    # audit trail doesn't misreport this as executed.
                    store.mark_post_consume(
                        approval_id, executed=False,
                        reason="phase3_runtime_stricter",
                    )
                else:
                    # Approve confirmed by Phase 3 — agent thread about
                    # to release for execution.
                    store.mark_post_consume(approval_id, executed=True)
        except Exception as e:
            logger.warning(
                "Failed to mirror execution outcome to store "
                "(approval_id=%s, user_choice=%s, final_choice=%s): %s",
                approval_id, user_choice, choice, e,
            )

    _fire_approval_hook(
        "post_approval_response",
        command=command,
        description=description,
        pattern_key=primary_key,
        pattern_keys=list(all_keys),
        session_key=session_key,
        surface=surface,
        choice=_outcome,
    )
    return {"resolved": resolved, "choice": choice}


def check_all_command_guards(command: str, env_type: str,
                             approval_callback=None) -> dict:
    """Run all pre-exec security checks and return a single approval decision.

    Gathers findings from tirith and dangerous-command detection, then
    presents them as a single combined approval request. This prevents
    a gateway force=True replay from bypassing one check when only the
    other was shown to the user.
    """
    # Skip containers for both checks
    if env_type in {"docker", "singularity", "modal", "daytona"}:
        return {"approved": True, "message": None}

    # Hardline floor: unconditional block for catastrophic commands
    # (rm -rf /, mkfs, dd to raw device, shutdown/reboot, fork bomb,
    # kill -1). Applies BEFORE yolo / mode=off / cron approve-mode so
    # no session-level setting can bypass it.
    is_hardline, hardline_desc = detect_hardline_command(command)
    if is_hardline:
        logger.warning("Hardline block: %s (command: %s)", hardline_desc, command[:200])
        return _hardline_block_result(hardline_desc)

    # == Sudo stdin guard ==
    # Like the hardline floor above, this is unconditional: there is never a
    # legitimate reason for the agent to pipe passwords to sudo -S when no
    # SUDO_PASSWORD has been configured.  This must fire BEFORE the yolo
    # check so even yolo/smart approval/mode=off cannot bypass it.
    is_sudo_guess, sudo_guess_desc = _check_sudo_stdin_guard(command)
    if is_sudo_guess:
        logger.warning("Sudo stdin guard block: %s (command: %s)",
                       sudo_guess_desc, command[:200])
        return _sudo_stdin_block_result(sudo_guess_desc)

    # --yolo or approvals.mode=off: bypass all approval prompts.
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
            # Block the agent thread until the user responds; the notify +
            # heartbeat wait loop is shared with check_execute_code_guard via
            # _await_gateway_decision().
            approval_data = {
                "command": command,
                "pattern_key": primary_key,
                "pattern_keys": all_keys,
                "description": combined_desc,
            }
            decision = _await_gateway_decision(
                session_key, notify_cb, approval_data, surface="gateway"
            )
            if decision.get("notify_failed"):
                return {
                    "approved": False,
                    "message": "BLOCKED: Failed to send approval request to user. Do NOT retry.",
                    "pattern_key": primary_key,
                    "description": combined_desc,
                }
            if decision.get("store_failed"):
                # FAIL CLOSED: persistent approval store is unavailable.
                # We deliberately do NOT fall back to legacy in-memory
                # approval, because that path lacks the security
                # guarantees the rewrite depends on (durability + atomic
                # consume across processes). Operator action required.
                return {
                    "approved": False,
                    "message": (
                        "BLOCKED: Approval store unavailable — approval "
                        "could not be durably persisted. Do NOT retry "
                        "this command; the operator must investigate "
                        "the gateway approval state database."
                    ),
                    "pattern_key": primary_key,
                    "description": combined_desc,
                }
            resolved = decision["resolved"]
            choice = decision["choice"]

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


def check_execute_code_guard(code: str, env_type: str) -> dict:
    """Approve an execute_code script before its child process is spawned.

    execute_code runs arbitrary local Python — the script can call
    ``subprocess``, ``os.system``, ``ctypes``, or other process/file APIs
    directly, none of which pass through ``terminal()`` /
    ``DANGEROUS_PATTERNS``. In gateway/ask contexts we fail closed by approving
    the script as a whole before it runs (#30882). Returns the same dict
    contract as ``check_all_command_guards``.

    Scope (documented limitation, #30882): in a purely local non-interactive
    non-gateway session (no TTY, not gateway, not cron-deny) this returns
    approved — matching the existing terminal auto-approve contract. The
    hardline floor still blocks catastrophic ``terminal()`` commands the script
    issues; running arbitrary code headlessly without any approval surface is
    trusted-by-config (set a gateway/ask surface or ``approvals.cron_mode`` to
    require approval).
    """
    pattern_key = "execute_code"
    description = (
        "execute_code script execution. The script can spawn subprocesses or "
        "mutate files without passing through terminal command approval; "
        "approval is one-shot for this run."
    )

    # Isolated backends already sandbox the child — matches the container skip
    # in check_all_command_guards / check_dangerous_command.
    if env_type in {"docker", "singularity", "modal", "daytona", "vercel_sandbox"}:
        return {"approved": True, "message": None}

    # --yolo or approvals.mode=off: bypass (session- or process-scoped).
    approval_mode = _get_approval_mode()
    if _YOLO_MODE_FROZEN or is_current_session_yolo_enabled() or approval_mode == "off":
        return {"approved": True, "message": None}

    is_gateway = _is_gateway_approval_context()
    is_ask = env_var_enabled("HERMES_EXEC_ASK")

    # Cron: no user is present to approve arbitrary code.
    if env_var_enabled("HERMES_CRON_SESSION"):
        if _get_cron_approval_mode() == "deny":
            return {
                "approved": False,
                "message": (
                    "BLOCKED: execute_code runs arbitrary local Python "
                    "(including subprocess calls that bypass shell-string "
                    "approval checks). Cron jobs run without a user present "
                    "to approve it. Use normal tools instead, or set "
                    "approvals.cron_mode: approve only if this cron profile "
                    "is intentionally trusted."
                ),
                "pattern_key": pattern_key,
                "description": description,
                "outcome": "blocked",
                "user_consent": False,
            }
        return {"approved": True, "message": None}

    # Only gateway/ask contexts get the one-shot whole-script approval.
    #   * CLI interactive: the script's terminal() calls are guarded per-call
    #     (context now propagates into the RPC thread, #33057); a whole-script
    #     prompt would fire on every execute_code call.
    #   * Local non-interactive non-gateway: documented limitation above.
    if not is_gateway and not is_ask:
        return {"approved": True, "message": None}

    session_key = get_current_session_key()
    # Built only now (past the early-return gates) so the common non-approval
    # paths don't pay to copy a potentially-large script into this string.
    command = f"execute_code <<'PY'\n{code}\nPY"

    # Check session/permanent approval — same gate as check_all_command_guards.
    # Without this, "Approve session" / "Always" choices are stored but never
    # consulted, so every execute_code call re-prompts the user (#39275).
    if is_approved(session_key, pattern_key):
        return {"approved": True, "message": None}

    # Smart mode: ask the aux LLM about the whole script. An APPROVE here only
    # suppresses the redundant whole-script prompt; the per-call terminal()
    # guards (restored by context propagation) still run independently.
    if approval_mode == "smart":
        verdict = _smart_approve(command, description)
        if verdict == "approve":
            logger.debug("Smart approval: auto-approved execute_code for session %s",
                         session_key)
            return {"approved": True, "message": None,
                    "smart_approved": True, "description": description}
        if verdict == "deny":
            return {
                "approved": False,
                "message": ("BLOCKED by smart approval: execute_code script "
                            "execution was assessed as genuinely dangerous. "
                            "Do NOT retry."),
                "smart_denied": True,
                "pattern_key": pattern_key,
                "description": description,
                "outcome": "denied",
                "user_consent": False,
            }
        # verdict == "escalate" → fall through to manual approval

    notify_cb = None
    with _lock:
        notify_cb = _gateway_notify_cbs.get(session_key)

    if notify_cb is None:
        # No gateway callback registered (e.g. ask-mode without a notifier):
        # surface a pending approval for backward compatibility.
        submit_pending(session_key, {
            "command": command,
            "pattern_key": pattern_key,
            "pattern_keys": [pattern_key],
            "description": description,
        })
        return {
            "approved": False,
            "pattern_key": pattern_key,
            "status": "pending_approval",
            "approval_pending": True,
            "command": command,
            "description": description,
            "message": (
                f"⚠️ {description}. Asking the user for approval.\n\n"
                f"**Code:**\n```python\n{code}\n```"
            ),
        }

    approval_data = {
        "command": command,
        "pattern_key": pattern_key,
        "pattern_keys": [pattern_key],
        "description": description,
    }
    decision = _await_gateway_decision(
        session_key, notify_cb, approval_data, surface="gateway"
    )
    if decision.get("notify_failed"):
        return {
            "approved": False,
            "message": ("BLOCKED: Failed to send execute_code approval request "
                        "to user. Do NOT retry."),
            "pattern_key": pattern_key,
            "description": description,
            "outcome": "notify_failed",
            "user_consent": False,
        }
    if decision.get("store_failed"):
        # FAIL CLOSED: see _await_gateway_decision's matching comment.
        return {
            "approved": False,
            "message": ("BLOCKED: Approval store unavailable — execute_code "
                        "approval could not be durably persisted. Do NOT "
                        "retry; operator must investigate gateway state DB."),
            "pattern_key": pattern_key,
            "description": description,
            "outcome": "store_failed",
            "user_consent": False,
        }

    resolved = decision["resolved"]
    choice = decision["choice"]

    if not resolved or choice is None or choice == "deny":
        reason = "timed out without user response" if not resolved else "denied by user"
        addendum = " Silence is not consent." if not resolved else ""
        return {
            "approved": False,
            "message": (
                f"BLOCKED: execute_code script {reason}. The user has NOT "
                f"consented to running this code. Do NOT retry, do NOT rephrase "
                f"the script, and do NOT attempt the same outcome via a "
                f"different tool.{addendum}"
            ),
            "pattern_key": pattern_key,
            "description": description,
            "outcome": "timeout" if not resolved else "denied",
            "user_consent": False,
        }

    # Approved — persist based on scope (same logic as check_all_command_guards).
    if choice == "session":
        approve_session(session_key, pattern_key)
    elif choice == "always":
        approve_session(session_key, pattern_key)
        approve_permanent(pattern_key)
        save_permanent_allowlist(_permanent_approved)
    # choice == "once": no persistence — approval lasts this single call only.

    return {"approved": True, "message": None,
            "user_approved": True, "description": description}


# Load permanent allowlist from config on module import
load_permanent_allowlist()
