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
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

# =========================================================================
# Approval display localization (UI-only; internal keys remain unchanged)
# =========================================================================

_APPROVAL_TITLE_ZH = "⚠️ 这条命令有风险，执行前请你确认"
_APPROVAL_REASON_LABEL_ZH = "原因"
_APPROVAL_REASON_FALLBACK_ZH = "检测到潜在风险命令，请确认是否放行。"
_APPROVAL_ALREADY_RESOLVED_ZH = "这条审批已经处理过了。"
_APPROVAL_NOT_AUTHORIZED_ZH = "你没有权限处理这条审批。"
_APPROVAL_UNKNOWN_DECISION_ZH = "已处理"

_APPROVAL_BUTTON_LABELS_ZH = {
    "approve_once": "✅ 只放行这一次",
    "approve_session": "✅ 本会话内放行",
    "approve_always": "✅ 永久放行",
    "deny": "❌ 拒绝执行",
    "once": "✅ 只放行这一次",
    "session": "✅ 本会话内放行",
    "always": "✅ 永久放行",
}

_APPROVAL_DECISION_LABELS_ZH = {
    "once": "✅ 已放行（仅这一次）",
    "session": "✅ 已放行（本会话内有效）",
    "always": "✅ 已放行（后续默认允许）",
    "deny": "❌ 已拒绝执行",
}

_SEVERITY_LABELS_ZH = {
    "CRITICAL": "高危",
    "HIGH": "高危",
    "MEDIUM": "中风险",
    "LOW": "低风险",
    "INFO": "提示",
}

_DANGEROUS_DESC_ZH = {
    "dangerous command": "这条命令可能有风险，请确认后再执行。",
    "dangerous deletion": "这条命令会删除关键内容，风险很高。",
    "delete in root path": "命令会删除根路径下的内容，风险很高。",
    "recursive delete": "命令包含递归删除，可能一次删掉大量文件。",
    "recursive delete (long flag)": "命令包含递归删除，可能一次删掉大量文件。",
    "world/other-writable permissions": "命令会把权限放得过宽，其他人也可能改文件。",
    "recursive world/other-writable (long flag)": "命令会递归放宽权限，影响范围很大。",
    "recursive chown to root": "命令会递归改为 root 所有者，可能影响系统可用性。",
    "recursive chown to root (long flag)": "命令会递归改为 root 所有者，可能影响系统可用性。",
    "format filesystem": "命令疑似在格式化磁盘分区，数据可能不可恢复。",
    "disk copy": "命令疑似直接读写磁盘设备，风险较高。",
    "write to block device": "命令会直接写块设备，可能破坏磁盘数据。",
    "SQL DROP": "命令包含 DROP，可能直接删库或删表。",
    "SQL DELETE without WHERE": "命令是无条件 DELETE，可能清空整张表。",
    "SQL TRUNCATE": "命令包含 TRUNCATE，可能清空整张表。",
    "overwrite system config": "命令会覆盖系统配置文件，可能影响系统运行。",
    "stop/disable system service": "命令会停用系统服务，可能导致服务中断。",
    "kill all processes": "命令会杀掉大量进程，系统可能立刻失去响应。",
    "force kill processes": "命令会强制终止进程，请确认不会误杀关键服务。",
    "fork bomb": "命令疑似 fork bomb，会迅速耗尽系统资源。",
    "shell command via -c/-lc flag": "命令通过 shell 的 -c/-lc 执行动态脚本，风险较高。",
    "script execution via -e/-c flag": "命令通过解释器参数执行脚本，建议确认脚本内容。",
    "pipe remote content to shell": "命令把远程内容直接喂给 shell 执行，风险很高。",
    "execute remote script via process substitution": "命令会直接执行远程脚本，风险很高。",
    "overwrite system file via tee": "命令会把内容写入系统敏感路径，风险较高。",
    "overwrite system file via redirection": "命令会重定向写入系统敏感路径，风险较高。",
    "xargs with rm": "命令通过 xargs 批量删除文件，请确认范围。",
    "find -exec rm": "命令会用 find 批量删除文件，请确认范围。",
    "find -delete": "命令会用 find -delete 批量删除文件，请确认范围。",
    "start gateway outside systemd (use 'systemctl --user restart hermes-gateway')": "命令会绕过 systemd 直接拉起 gateway，建议用 systemctl 重启。",
    "kill hermes/gateway process (self-termination)": "命令会终止 Hermes/Gateway 进程，当前会话可能中断。",
    "copy/move file into /etc/": "命令会改动 /etc 配置目录，请确认影响范围。",
    "in-place edit of system config": "命令会原地修改系统配置文件，风险较高。",
    "in-place edit of system config (long flag)": "命令会原地修改系统配置文件，风险较高。",
}

_RISK_PHRASE_ZH = {
    "security issue detected": "检测到安全风险",
    "shortened URL detected": "检测到短链接，可能隐藏真实跳转地址",
    "shortened URL": "检测到短链接，可能隐藏真实跳转地址",
    "homograph detected": "检测到疑似域名伪装（同形字符）",
    "homograph URL": "检测到疑似域名伪装（同形字符）",
    "terminal injection": "检测到终端注入风险",
    "pipe to shell": "检测到管道直连执行风险",
    "pipe detected": "检测到可疑管道执行行为",
    "pipe to interpreter": "检测到将内容直接喂给解释器执行",
    "downloaded content executed without inspection": "下载内容未检查就直接执行，风险较高",
    "interpreter hijack": "检测到解释器劫持风险",
}

_RISK_TITLE_ZH = {
    "pipe to interpreter": "管道直接喂给解释器执行",
    "interpreter hijack": "解释器劫持风险",
    "homograph URL": "疑似同形字符域名伪装",
    "shortened URL": "短链接风险",
    "terminal injection": "终端注入风险",
}


def approval_title_text() -> str:
    """Return localized title for dangerous-command approval prompts."""
    return _APPROVAL_TITLE_ZH


def approval_reason_label_text() -> str:
    """Return localized reason label for approval prompts."""
    return _APPROVAL_REASON_LABEL_ZH


def approval_button_label(action: str) -> str:
    """Return localized button text for an approval action key."""
    return _APPROVAL_BUTTON_LABELS_ZH.get(action, _APPROVAL_UNKNOWN_DECISION_ZH)


def approval_decision_label(choice: str) -> str:
    """Return localized resolved-label text for an approval choice."""
    return _APPROVAL_DECISION_LABELS_ZH.get(choice, _APPROVAL_UNKNOWN_DECISION_ZH)


def approval_already_resolved_text() -> str:
    """Return localized text shown when an approval is already resolved."""
    return _APPROVAL_ALREADY_RESOLVED_ZH


def approval_not_authorized_text() -> str:
    """Return localized text shown when user lacks approval permission."""
    return _APPROVAL_NOT_AUTHORIZED_ZH


def _translate_risk_phrase(text: str) -> str:
    """Translate a common risk phrase into plain Chinese when possible."""
    if not text:
        return ""
    stripped = text.strip()
    if not stripped:
        return ""

    if stripped in _DANGEROUS_DESC_ZH:
        return _DANGEROUS_DESC_ZH[stripped]

    lower = stripped.lower()
    if lower in _RISK_PHRASE_ZH:
        return _RISK_PHRASE_ZH[lower]

    translated = stripped
    for en, zh in _RISK_PHRASE_ZH.items():
        translated = re.sub(re.escape(en), zh, translated, flags=re.IGNORECASE)
    return translated


def _translate_tirith_fragment(fragment: str) -> str:
    """Translate a single Tirith finding fragment into Chinese."""
    text = fragment.strip()
    if not text:
        return ""

    m = re.match(r"^\[(?P<severity>[A-Za-z]+)\]\s*(?P<body>.*)$", text)
    severity_text = ""
    body = text
    if m:
        sev = (m.group("severity") or "").upper()
        severity_text = _SEVERITY_LABELS_ZH.get(sev, sev)
        body = (m.group("body") or "").strip()

    title = body
    detail = ""
    if ":" in body:
        title, detail = body.split(":", 1)
        title = title.strip()
        detail = detail.strip()

    title_zh = _RISK_TITLE_ZH.get(title.lower(), _translate_risk_phrase(title))
    detail_zh = _translate_risk_phrase(detail) if detail else ""

    if severity_text and title_zh and detail_zh:
        return f"【{severity_text}】{title_zh}：{detail_zh}"
    if severity_text and title_zh:
        return f"【{severity_text}】{title_zh}"
    if title_zh and detail_zh:
        return f"{title_zh}：{detail_zh}"
    return title_zh or detail_zh


def localize_approval_reason(reason: str) -> str:
    """Translate approval reason text into plain Chinese for user-facing UIs."""
    raw = (reason or "").strip()
    if not raw:
        return _APPROVAL_REASON_FALLBACK_ZH

    # Direct match for known dangerous-command descriptors
    if raw in _DANGEROUS_DESC_ZH:
        return _DANGEROUS_DESC_ZH[raw]

    lower = raw.lower()
    if lower.startswith("security scan"):
        payload = raw
        for sep in ("—", ":"):
            if sep in raw:
                payload = raw.split(sep, 1)[1].strip()
                break

        if not payload:
            return "安全检查：检测到安全风险，请你确认后再执行。"

        parts = [p.strip() for p in payload.split(";") if p.strip()]
        translated_parts = [_translate_tirith_fragment(p) for p in parts]
        translated_parts = [p for p in translated_parts if p]
        if not translated_parts:
            summary = _translate_risk_phrase(payload)
            return f"安全检查：{summary}"
        return "安全检查：" + "；".join(translated_parts)

    translated = _translate_risk_phrase(raw).strip()
    if translated and translated != raw:
        return translated

    if re.search(r"[\u4e00-\u9fff]", raw):
        return raw

    return f"检测到潜在风险：{raw}"

# Per-thread/per-task gateway session identity.
# Gateway runs agent turns concurrently in executor threads, so reading a
# process-global env var for session identity is racy. Keep env fallback for
# legacy single-threaded callers, but prefer the context-local value when set.
_approval_session_key: contextvars.ContextVar[str] = contextvars.ContextVar(
    "approval_session_key",
    default="",
)


def set_current_session_key(session_key: str) -> contextvars.Token[str]:
    """Bind the active approval session key to the current context."""
    return _approval_session_key.set(session_key or "")


def reset_current_session_key(token: contextvars.Token[str]) -> None:
    """Restore the prior approval session key context."""
    _approval_session_key.reset(token)


def get_current_session_key(default: str = "default") -> str:
    """Return the active session key, preferring context-local state."""
    session_key = _approval_session_key.get()
    if session_key:
        return session_key
    return os.getenv("HERMES_SESSION_KEY", default)

# Sensitive write targets that should trigger approval even when referenced
# via shell expansions like $HOME or $HERMES_HOME.
_SSH_SENSITIVE_PATH = r'(?:~|\$home|\$\{home\})/\.ssh(?:/|$)'
_HERMES_ENV_PATH = (
    r'(?:~\/\.hermes/|'
    r'(?:\$home|\$\{home\})/\.hermes/|'
    r'(?:\$hermes_home|\$\{hermes_home\})/)'
    r'\.env\b'
)
_SENSITIVE_WRITE_TARGET = (
    r'(?:/etc/|/dev/sd|'
    rf'{_SSH_SENSITIVE_PATH}|'
    rf'{_HERMES_ENV_PATH})'
)

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
    (r'\bDELETE\s+FROM\b(?!.*\bWHERE\b)', "SQL DELETE without WHERE"),
    (r'\bTRUNCATE\s+(TABLE)?\s*\w', "SQL TRUNCATE"),
    (r'>\s*/etc/', "overwrite system config"),
    (r'\bsystemctl\s+(stop|disable|mask)\b', "stop/disable system service"),
    (r'\bkill\s+-9\s+-1\b', "kill all processes"),
    (r'\bpkill\s+-9\b', "force kill processes"),
    (r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:', "fork bomb"),
    # Any shell invocation via -c or combined flags like -lc, -ic, etc.
    (r'\b(bash|sh|zsh|ksh)\s+-[^\s]*c(\s+|$)', "shell command via -c/-lc flag"),
    (r'\b(python[23]?|perl|ruby|node)\s+-[ec]\s+', "script execution via -e/-c flag"),
    (r'\b(curl|wget)\b.*\|\s*(ba)?sh\b', "pipe remote content to shell"),
    (r'\b(bash|sh|zsh|ksh)\s+<\s*<?\s*\(\s*(curl|wget)\b', "execute remote script via process substitution"),
    (rf'\btee\b.*["\']?{_SENSITIVE_WRITE_TARGET}', "overwrite system file via tee"),
    (rf'>>?\s*["\']?{_SENSITIVE_WRITE_TARGET}', "overwrite system file via redirection"),
    (r'\bxargs\s+.*\brm\b', "xargs with rm"),
    (r'\bfind\b.*-exec\s+(/\S*/)?rm\b', "find -exec rm"),
    (r'\bfind\b.*-delete\b', "find -delete"),
    # Gateway protection: never start gateway outside systemd management
    (r'gateway\s+run\b.*(&\s*$|&\s*;|\bdisown\b|\bsetsid\b)', "start gateway outside systemd (use 'systemctl --user restart hermes-gateway')"),
    (r'\bnohup\b.*gateway\s+run\b', "start gateway outside systemd (use 'systemctl --user restart hermes-gateway')"),
    # Self-termination protection: prevent agent from killing its own process
    (r'\b(pkill|killall)\b.*\b(hermes|gateway|cli\.py)\b', "kill hermes/gateway process (self-termination)"),
    # File copy/move/edit into sensitive system paths
    (r'\b(cp|mv|install)\b.*\s/etc/', "copy/move file into /etc/"),
    (r'\bsed\s+-[^\s]*i.*\s/etc/', "in-place edit of system config"),
    (r'\bsed\s+--in-place\b.*\s/etc/', "in-place edit of system config (long flag)"),
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
    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, command_lower, re.IGNORECASE | re.DOTALL):
            pattern_key = description
            return (True, pattern_key, description)
    return (False, None, None)


# =========================================================================
# Per-session approval state (thread-safe)
# =========================================================================

_lock = threading.Lock()
_pending: dict[str, dict] = {}
_session_approved: dict[str, set] = {}
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


def pending_approval_count(session_key: str) -> int:
    """Return the number of pending blocking approvals for a session."""
    with _lock:
        return len(_gateway_queues.get(session_key, []))


def submit_pending(session_key: str, approval: dict):
    """Store a pending approval request for a session."""
    with _lock:
        _pending[session_key] = approval


def pop_pending(session_key: str) -> Optional[dict]:
    """Retrieve and remove a pending approval for a session."""
    with _lock:
        return _pending.pop(session_key, None)


def has_pending(session_key: str) -> bool:
    """Check if a session has a pending approval request."""
    with _lock:
        return session_key in _pending


def approve_session(session_key: str, pattern_key: str):
    """Approve a pattern for this session only."""
    with _lock:
        _session_approved.setdefault(session_key, set()).add(pattern_key)


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


def clear_session(session_key: str):
    """Clear all approvals and pending requests for a session."""
    with _lock:
        _session_approved.pop(session_key, None)
        _pending.pop(session_key, None)
        _gateway_notify_cbs.pop(session_key, None)
        # Signal ALL blocked threads so they don't hang forever
        entries = _gateway_queues.pop(session_key, [])
        for entry in entries:
            entry.event.set()


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
    except Exception:
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
        except Exception:
            return "deny"

    os.environ["HERMES_SPINNER_PAUSE"] = "1"
    try:
        while True:
            print()
            print(f"  ⚠️  DANGEROUS COMMAND: {description}")
            print(f"      {command}")
            print()
            if allow_permanent:
                print("      [o]nce  |  [s]ession  |  [a]lways  |  [d]eny")
            else:
                print("      [o]nce  |  [s]ession  |  [d]eny")
            print()
            sys.stdout.flush()

            result = {"choice": ""}

            def get_input():
                try:
                    prompt = "      Choice [o/s/a/D]: " if allow_permanent else "      Choice [o/s/D]: "
                    result["choice"] = input(prompt).strip().lower()
                except (EOFError, OSError):
                    result["choice"] = ""

            thread = threading.Thread(target=get_input, daemon=True)
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                print("\n      ⏱ Timeout - denying command")
                return "deny"

            choice = result["choice"]
            if choice in ('o', 'once'):
                print("      ✓ Allowed once")
                return "once"
            elif choice in ('s', 'session'):
                print("      ✓ Allowed for this session")
                return "session"
            elif choice in ('a', 'always'):
                if not allow_permanent:
                    print("      ✓ Allowed for this session")
                    return "session"
                print("      ✓ Added to permanent allowlist")
                return "always"
            else:
                print("      ✗ Denied")
                return "deny"

    except (EOFError, KeyboardInterrupt):
        print("\n      ✗ Cancelled")
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
    except Exception:
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


def _smart_approve(command: str, description: str) -> str:
    """Use the auxiliary LLM to assess risk and decide approval.

    Returns 'approve' if the LLM determines the command is safe,
    'deny' if genuinely dangerous, or 'escalate' if uncertain.

    Inspired by OpenAI Codex's Smart Approvals guardian subagent
    (openai/codex#13860).
    """
    try:
        from agent.auxiliary_client import get_text_auxiliary_client, auxiliary_max_tokens_param

        client, model = get_text_auxiliary_client(task="approval")
        if not client or not model:
            logger.debug("Smart approvals: no aux client available, escalating")
            return "escalate"

        prompt = f"""You are a security reviewer for an AI coding agent. A terminal command was flagged by pattern matching as potentially dangerous.

Command: {command}
Flagged reason: {description}

Assess the ACTUAL risk of this command. Many flagged commands are false positives — for example, `python -c "print('hello')"` is flagged as "script execution via -c flag" but is completely harmless.

Rules:
- APPROVE if the command is clearly safe (benign script execution, safe file operations, development tools, package installs, git operations, etc.)
- DENY if the command could genuinely damage the system (recursive delete of important paths, overwriting system files, fork bombs, wiping disks, dropping databases, etc.)
- ESCALATE if you're uncertain

Respond with exactly one word: APPROVE, DENY, or ESCALATE"""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **auxiliary_max_tokens_param(16),
            temperature=0,
        )

        answer = (response.choices[0].message.content or "").strip().upper()

        if "APPROVE" in answer:
            return "approve"
        elif "DENY" in answer:
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
    if env_type in ("docker", "singularity", "modal", "daytona"):
        return {"approved": True, "message": None}

    # --yolo: bypass all approval prompts
    if os.getenv("HERMES_YOLO_MODE"):
        return {"approved": True, "message": None}

    is_dangerous, pattern_key, description = detect_dangerous_command(command)
    if not is_dangerous:
        return {"approved": True, "message": None}

    session_key = get_current_session_key()
    if is_approved(session_key, pattern_key):
        return {"approved": True, "message": None}

    is_cli = os.getenv("HERMES_INTERACTIVE")
    is_gateway = os.getenv("HERMES_GATEWAY_SESSION")

    if not is_cli and not is_gateway:
        return {"approved": True, "message": None}

    if is_gateway or os.getenv("HERMES_EXEC_ASK"):
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
        return f"安全检查：{_translate_risk_phrase(summary)}"

    parts = []
    for f in findings:
        severity = f.get("severity", "")
        title = f.get("title", "")
        desc = f.get("description", "")
        if title and desc:
            combined = f"[{severity}] {title}: {desc}" if severity else f"{title}: {desc}"
            parts.append(_translate_tirith_fragment(combined))
        elif title:
            combined = f"[{severity}] {title}" if severity else title
            parts.append(_translate_tirith_fragment(combined))
    if not parts:
        summary = tirith_result.get("summary") or "security issue detected"
        return f"安全检查：{_translate_risk_phrase(summary)}"

    return "安全检查：" + "；".join(p for p in parts if p)


def check_all_command_guards(command: str, env_type: str,
                             approval_callback=None) -> dict:
    """Run all pre-exec security checks and return a single approval decision.

    Gathers findings from tirith and dangerous-command detection, then
    presents them as a single combined approval request. This prevents
    a gateway force=True replay from bypassing one check when only the
    other was shown to the user.
    """
    # Skip containers for both checks
    if env_type in ("docker", "singularity", "modal", "daytona"):
        return {"approved": True, "message": None}

    # --yolo or approvals.mode=off: bypass all approval prompts
    approval_mode = _get_approval_mode()
    if os.getenv("HERMES_YOLO_MODE") or approval_mode == "off":
        return {"approved": True, "message": None}

    is_cli = os.getenv("HERMES_INTERACTIVE")
    is_gateway = os.getenv("HERMES_GATEWAY_SESSION")
    is_ask = os.getenv("HERMES_EXEC_ASK")

    # Preserve the existing non-interactive behavior: outside CLI/gateway/ask
    # flows, we do not block on approvals and we skip external guard work.
    if not is_cli and not is_gateway and not is_ask:
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
    if tirith_result["action"] in ("block", "warn"):
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

            # Block until the user responds or timeout (default 5 min)
            timeout = _get_approval_config().get("gateway_timeout", 300)
            try:
                timeout = int(timeout)
            except (ValueError, TypeError):
                timeout = 300
            resolved = entry.event.wait(timeout=timeout)

            # Clean up this entry from the queue
            with _lock:
                queue = _gateway_queues.get(session_key, [])
                if entry in queue:
                    queue.remove(entry)
                if not queue:
                    _gateway_queues.pop(session_key, None)

            choice = entry.result
            if not resolved or choice is None or choice == "deny":
                reason = "timed out" if not resolved else "denied by user"
                return {
                    "approved": False,
                    "message": f"BLOCKED: Command {reason}. Do NOT retry this command.",
                    "pattern_key": primary_key,
                    "description": combined_desc,
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
            "status": "approval_required",
            "command": command,
            "description": combined_desc,
            "message": (
                f"⚠️ {combined_desc}. Asking the user for approval.\n\n**Command:**\n```\n{command}\n```"
            ),
        }

    # CLI interactive: single combined prompt
    # Hide [a]lways when any tirith warning is present
    choice = prompt_dangerous_approval(command, combined_desc,
                                       allow_permanent=not has_tirith,
                                       approval_callback=approval_callback)

    if choice == "deny":
        return {
            "approved": False,
            "message": "BLOCKED: User denied. Do NOT retry.",
            "pattern_key": primary_key,
            "description": combined_desc,
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
