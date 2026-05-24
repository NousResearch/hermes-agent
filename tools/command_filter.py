"""Command allowlist/denylist enforcement for the terminal tool.

Provides pattern-based command filtering that sits between the hardline
block (unconditional) and the yolo/approval flow.  Config-driven via
``terminal.command_denylist`` and ``terminal.command_allowlist`` in
config.yaml.

Design
------
- **Denylist** (default): regex patterns that are **always blocked**, even
  when yolo mode is on.  Ships with sensible defaults for catastrophic
  commands.
- **Allowlist** (opt-in): when non-empty, ONLY commands matching at least
  one pattern are permitted.  Everything else is rejected.
- Denylist is checked **before** allowlist so a deny pattern can override
  an allow pattern (safety-first).

Config example
--------------
```yaml
terminal:
  command_denylist:
    - "rm\\s+-rf\\s+/"       # block rm -rf /
    - "mkfs\\.?"             # block filesystem format
  command_allowlist:
    - "^ls\\b"               # only ls commands allowed
    - "^cat\\b"
```
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Default denylist patterns (applied even when yolo/allowlist is on) ──

_DEFAULT_DENY_PATTERNS: list[tuple[str, str]] = [
    (r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*|-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*)\s+/",
     "rm -rf / — recursive force-delete from root"),
    (r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*|-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*)\s+\*/",
     "rm -rf */ — recursive force-delete of all top-level directories"),
    (r"mkfs\.?[a-z]*",
     "mkfs — filesystem format (destroys data)"),
    (r"dd\s+if=\S*\s+of=/dev/",
     "dd to raw device — can overwrite disk/MBR"),
    (r"dd\s+of=/dev/",
     "dd to raw device — can overwrite disk/MBR"),
    (r">\s*/dev/sd",
     "truncate to raw disk device"),
    (r":\s*>\s*/dev/sd",
     "truncate to raw disk device"),
    (r"(shutdown|reboot|halt|poweroff)\b",
     "system shutdown/reboot/halt/poweroff"),
    (r"kill\s+-9\s+-?\d+",
     "kill -9 of all processes"),
    (r"killall\s+-9",
     "killall -9 — force-kill all matching processes"),
    (r"\bchmod\s+-R\s+777\s+/",
     "chmod -R 777 / — make entire filesystem world-writable"),
    (r">\s*/etc/(passwd|shadow|sudoers)",
     "truncate critical system config files"),
    (r":\s*>\s*/etc/(passwd|shadow|sudoers)",
     "truncate critical system config files"),
]

_COMPILED_DENY = [
    (re.compile(pat, re.IGNORECASE), desc)
    for pat, desc in _DEFAULT_DENY_PATTERNS
]


def _load_config_patterns(config_path: Optional[str] = None) -> tuple[list[tuple[str, str]], list[str]]:
    """Load denylist/allowlist patterns from config.yaml.

    Returns (denylist, allowlist) where each item is a regex string.
    Denylist items include descriptions; allowlist items are bare patterns.
    """
    deny: list[tuple[str, str]] = []
    allow: list[str] = []

    # Lazy import to avoid circular deps
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        return deny, allow

    terminal_cfg = cfg.get("terminal", {}) if isinstance(cfg, dict) else {}

    # Load denylist
    denylist = terminal_cfg.get("command_denylist", [])
    if isinstance(denylist, list):
        for entry in denylist:
            if isinstance(entry, str) and entry.strip():
                deny.append((entry.strip(), "user-configured deny pattern"))
            elif isinstance(entry, dict):
                pat = entry.get("pattern", "")
                desc = entry.get("description", "user-configured deny pattern")
                if pat:
                    deny.append((pat.strip(), desc))

    # Load allowlist
    allowlist = terminal_cfg.get("command_allowlist", [])
    if isinstance(allowlist, list):
        for entry in allowlist:
            if isinstance(entry, str) and entry.strip():
                allow.append(entry.strip())

    return deny, allow


def _compile_patterns(
    deny_patterns: list[tuple[str, str]],
    allow_patterns: list[str],
) -> tuple[list[tuple[re.Pattern, str]], list[re.Pattern]]:
    """Compile regex patterns, logging any that fail to compile."""
    compiled_deny = []
    for pat, desc in deny_patterns:
        try:
            compiled_deny.append((re.compile(pat, re.IGNORECASE), desc))
        except re.error as e:
            logger.warning("Invalid deny pattern %r: %s", pat, e)

    compiled_allow = []
    for pat in allow_patterns:
        try:
            compiled_allow.append(re.compile(pat, re.IGNORECASE))
        except re.error as e:
            logger.warning("Invalid allow pattern %r: %s", pat, e)

    return compiled_deny, compiled_allow


def check_command_filter(
    command: str,
    *,
    config_path: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """Check whether a command passes the allowlist/denylist filter.

    Parameters
    ----------
    command:
        The shell command to check.
    config_path:
        Optional path to config.yaml (uses default loader if None).

    Returns
    -------
    (allowed, reason)
        ``allowed`` is ``True`` when the command passes all checks.
        ``reason`` is a human-readable explanation when blocked.
    """
    # Load patterns (cached at module level after first load)
    user_deny, user_allow = _load_config_patterns(config_path)
    compiled_deny, compiled_allow = _compile_patterns(
        user_deny + _DEFAULT_DENY_PATTERNS,  # defaults always apply
        user_allow,
    )

    # Step 1: Check denylist — always blocks, even with allowlist
    for pat_re, desc in compiled_deny:
        if pat_re.search(command):
            logger.info(
                "Command denied by denylist pattern: %s (command: %s)",
                desc, command[:200],
            )
            return False, f"Command denied: {desc}"

    # Step 2: If allowlist is configured, command MUST match at least one pattern
    if compiled_allow:
        for pat_re in compiled_allow:
            if pat_re.search(command):
                return True, None
        # No allow pattern matched
        return False, (
            "Command denied: not in the command allowlist. "
            "Add a matching pattern to terminal.command_allowlist in config.yaml "
            "or remove the allowlist to allow all non-denied commands."
        )

    # No allowlist configured, no deny match → allowed
    return True, None


# ── Module-level cache (populated on first check) ──
_compiled_deny_cache: Optional[list[tuple[re.Pattern, str]]] = None
_compiled_allow_cache: Optional[list[re.Pattern]] = None
_cache_initialized = False


def _ensure_cache() -> None:
    """Compile and cache patterns once at startup."""
    global _compiled_deny_cache, _compiled_allow_cache, _cache_initialized
    if _cache_initialized:
        return
    user_deny, user_allow = _load_config_patterns()
    _compiled_deny_cache, _compiled_allow_cache = _compile_patterns(
        user_deny + _DEFAULT_DENY_PATTERNS,
        user_allow,
    )
    _cache_initialized = True


def check_command_filter_fast(command: str) -> tuple[bool, Optional[str]]:
    """Fast path using pre-compiled cached patterns.

    Same semantics as ``check_command_filter`` but skips config reload.
    Use this in hot paths (terminal tool per-execution).
    """
    _ensure_cache()

    # Check denylist
    if _compiled_deny_cache:
        for pat_re, desc in _compiled_deny_cache:
            if pat_re.search(command):
                return False, f"Command denied: {desc}"

    # Check allowlist (if configured)
    if _compiled_allow_cache:
        for pat_re in _compiled_allow_cache:
            if pat_re.search(command):
                return True, None
        return False, (
            "Command denied: not in the command allowlist. "
            "Add a matching pattern to terminal.command_allowlist in config.yaml "
            "or remove the allowlist to allow all non-denied commands."
        )

    return True, None


def get_denylist_patterns() -> list[tuple[str, str]]:
    """Return the compiled denylist patterns (for diagnostics/logging)."""
    _ensure_cache()
    return [
        (pat.pattern, desc)
        for pat, desc in (_compiled_deny_cache or [])
    ]


def get_allowlist_patterns() -> list[str]:
    """Return the compiled allowlist patterns (for diagnostics/logging)."""
    _ensure_cache()
    return [
        pat.pattern
        for pat in (_compiled_allow_cache or [])
    ]
