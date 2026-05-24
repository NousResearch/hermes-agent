"""Structured audit logging for terminal command execution.

Writes JSON-formatted audit entries for every command executed through the
terminal tool, with log rotation (max size + max age).

Config
------
```yaml
logging:
  audit_log: "~/.hermes/logs/audit.log"       # path (HERMES_HOME-aware)
  audit_log_max_bytes: 104857600              # 100 MB default
  audit_log_max_days: 7                       # keep 7 days
  audit_enabled: true                         # on/off toggle
```

CLI
---
``hermes logs --audit`` — tails the audit log.
``hermes logs --audit --lines 50`` — show last 50 entries.
``hermes logs --audit --session <key>`` — filter by session.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────
_DEFAULT_MAX_BYTES = 100 * 1024 * 1024  # 100 MB
_DEFAULT_MAX_DAYS = 7
_DEFAULT_AUDIT_LOG = None  # resolved at runtime


def _resolve_audit_log_path() -> Path:
    """Resolve the audit log path, respecting HERMES_HOME."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        if isinstance(cfg, dict):
            log_cfg = cfg.get("logging", {})
            if isinstance(log_cfg, dict):
                audit_path = log_cfg.get("audit_log")
                if audit_path:
                    path = Path(audit_path).expanduser()
                    # If relative, resolve against HERMES_HOME/logs/
                    if not path.is_absolute():
                        return get_hermes_home() / "logs" / audit_path
                    return path
    except Exception:
        pass
    # Fallback: HERMES_HOME/logs/audit.log
    return get_hermes_home() / "logs" / "audit.log"


def _resolve_config_value(key: str, default: Any) -> Any:
    """Read a config value with a safe fallback."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        if isinstance(cfg, dict):
            log_cfg = cfg.get("logging", {})
            if isinstance(log_cfg, dict):
                val = log_cfg.get(key)
                if val is not None:
                    return val
    except Exception:
        pass
    return default


def is_audit_enabled() -> bool:
    """Check whether audit logging is enabled."""
    return bool(_resolve_config_value("audit_enabled", True))


def get_audit_log_path() -> Path:
    """Get the resolved audit log path."""
    return _resolve_audit_log_path()


def get_audit_config() -> tuple[int, int]:
    """Return (max_bytes, max_days) from config."""
    max_bytes = _resolve_config_value("audit_log_max_bytes", _DEFAULT_MAX_BYTES)
    max_days = _resolve_config_value("audit_log_max_days", _DEFAULT_MAX_DAYS)
    return int(max_bytes), int(max_days)


def _rotate_if_needed(log_path: Path) -> None:
    """Rotate the audit log if it exceeds size or age limits."""
    if not log_path.exists():
        return

    max_bytes, max_days = get_audit_config()

    # Size-based rotation
    try:
        file_size = log_path.stat().st_size
        if file_size >= max_bytes:
            _rotate_file(log_path, "size")
            return
    except OSError:
        pass

    # Age-based rotation
    try:
        mtime = log_path.stat().st_mtime
        age_days = (time.time() - mtime) / 86400
        if age_days >= max_days:
            _rotate_file(log_path, "age")
    except OSError:
        pass


def _rotate_file(log_path: Path, reason: str) -> None:
    """Rotate log_path → log_path.1, shifting existing rotations."""
    try:
        # Shift existing rotations (.2 → .3, .1 → .2, etc.)
        max_rotations = 3  # keep at most 3 rotated files
        for i in range(max_rotations, 0, -1):
            old = log_path.with_suffix(f"{log_path.suffix}.{i}")
            new = log_path.with_suffix(f"{log_path.suffix}.{i + 1}")
            if old.exists():
                if i == max_rotations:
                    old.unlink(missing_ok=True)  # drop oldest
                else:
                    old.rename(new)

        # Current → .1
        log_path.rename(log_path.with_suffix(f"{log_path.suffix}.1"))
        logger.info("Audit log rotated (%s): %s", reason, log_path)
    except OSError as e:
        logger.warning("Audit log rotation failed: %s", e)


def write_audit_entry(
    command: str,
    session_key: str = "",
    task_id: str = "",
    workdir: str = "",
    exit_code: Optional[int] = None,
    blocked: bool = False,
    block_reason: str = "",
    user_approved: bool = False,
    env_type: str = "",
    duration_ms: float = 0,
    **extra: Any,
) -> None:
    """Write a structured audit log entry.

    Parameters
    ----------
    command:
        The command that was executed (or attempted).
    session_key:
        Gateway session key (for filtering by session).
    task_id:
        Terminal task/agent identifier.
    workdir:
        Working directory for the command.
    exit_code:
        Exit code of the command (None if blocked or not yet finished).
    blocked:
        Whether the command was blocked by a guard.
    block_reason:
        Reason for blocking (if blocked=True).
    user_approved:
        Whether the user explicitly approved a flagged command.
    env_type:
        Terminal environment type (local, docker, modal, ...).
    duration_ms:
        Execution duration in milliseconds.
    extra:
        Additional fields to include in the entry.
    """
    if not is_audit_enabled():
        return

    log_path = get_audit_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Rotate before writing
    _rotate_if_needed(log_path)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "session_key": session_key,
        "task_id": task_id,
        "workdir": workdir,
        "exit_code": exit_code,
        "blocked": blocked,
        "block_reason": block_reason,
        "user_approved": user_approved,
        "env_type": env_type,
        "duration_ms": duration_ms,
        **{k: v for k, v in extra.items() if v is not None},
    }

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            f.flush()
    except OSError as e:
        logger.warning("Failed to write audit entry: %s", e)


def read_audit_entries(
    log_path: Optional[Path] = None,
    max_lines: int = 100,
    session_filter: str = "",
    reverse: bool = True,
) -> list[dict[str, Any]]:
    """Read audit log entries, optionally filtered.

    Parameters
    ----------
    log_path:
        Path to the audit log (resolved automatically if None).
    max_lines:
        Maximum number of entries to return.
    session_filter:
        Only return entries matching this session key.
    reverse:
        Return newest entries first.

    Returns
    -------
    list[dict]
        Parsed audit entries.
    """
    path = log_path or get_audit_log_path()
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if session_filter and entry.get("session_key") != session_filter:
                        continue
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        logger.warning("Failed to read audit log: %s", e)
        return []

    if reverse:
        entries.reverse()

    return entries[:max_lines]


def tail_audit_log(
    log_path: Optional[Path] = None,
    lines: int = 20,
    follow: bool = False,
    session_filter: str = "",
) -> None:
    """Tail the audit log (CLI helper).

    Parameters
    ----------
    log_path:
        Path to the audit log.
    lines:
        Number of recent lines to show.
    follow:
        If True, continue tailing new entries (like ``tail -f``).
    session_filter:
        Only show entries matching this session key.
    """
    import sys

    path = log_path or get_audit_log_path()
    if not path.exists():
        print(f"Audit log not found: {path}", file=sys.stderr)
        return

    # Read last N lines
    try:
        with open(path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            recent = all_lines[-lines:] if len(all_lines) > lines else all_lines

        for line in recent:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if session_filter and entry.get("session_key") != session_filter:
                    continue
                _format_audit_entry(entry)
            except json.JSONDecodeError:
                print(line)
    except OSError as e:
        print(f"Error reading audit log: {e}", file=sys.stderr)

    if follow:
        _follow_audit_log(path, session_filter)


def _follow_audit_log(log_path: Path, session_filter: str = "") -> None:
    """Follow new audit log entries (like tail -f)."""
    import sys
    import time

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            f.seek(0, 2)  # Seek to end
            while True:
                line = f.readline()
                if line:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            if session_filter and entry.get("session_key") != session_filter:
                                continue
                            _format_audit_entry(entry)
                        except json.JSONDecodeError:
                            print(line)
                        sys.stdout.flush()
                else:
                    time.sleep(0.2)
    except (OSError, KeyboardInterrupt):
        pass


def _format_audit_entry(entry: dict[str, Any]) -> None:
    """Format a single audit entry for display."""
    import sys

    ts = entry.get("timestamp", "?")
    cmd = entry.get("command", "?")[:100]
    blocked = entry.get("blocked", False)
    exit_code = entry.get("exit_code")
    session = entry.get("session_key", "")[:20]

    status = "BLOCKED" if blocked else f"exit={exit_code}"
    prefix = "❌" if blocked else "✅"

    print(f"{prefix} [{ts}] {status:10s} {session:20s} {cmd}", file=sys.stdout)


def format_audit_summary(entries: list[dict[str, Any]]) -> str:
    """Format a summary of audit entries."""
    total = len(entries)
    blocked = sum(1 for e in entries if e.get("blocked"))
    approved = sum(1 for e in entries if e.get("user_approved"))

    if total == 0:
        return "No audit entries."

    return (
        f"Audit summary: {total} total, {blocked} blocked, "
        f"{approved} user-approved"
    )
