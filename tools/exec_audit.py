"""Append-only command-execution audit log.

``terminal`` and ``execute_code`` previously left no durable record of what commands
or code actually ran on the host — only cron-fire ``usage_audit.jsonl`` metadata
survived. This module records every invocation as one JSONL line, append-only with
0600 permissions, and NEVER raises: a logger bug must not break the tool it observes.

The recorded body is scrubbed through the shared ``agent.redact`` path before it is
persisted, so a raw command/script carrying inline credentials or PII is not written to
the log in plaintext. The write itself fails closed (a record may be dropped with a
warning); the observed tool is never failed by the logger.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

EXEC_AUDIT_FILENAME = "exec_audit.jsonl"


def exec_audit_path() -> Path:
    """Per-profile audit log, resolved the same way as ``cron/usage_audit.jsonl``."""
    return get_hermes_home() / "security" / EXEC_AUDIT_FILENAME


def _utcnow_iso_ms() -> str:
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _redact_body(body: str) -> str:
    """Best-effort secret/PII redaction of the command/code body; raw on failure.

    ``force=True`` (never persist raw secrets regardless of the ``redact_secrets``
    preference) and ``redact_url_credentials=True`` (scrub URL userinfo too) — this is
    a forensics log that is never re-executed, so over-redaction is harmless while
    under-redaction leaks credentials out of the session transcript.
    """
    try:
        from agent.redact import redact_sensitive_text

        scrubbed = redact_sensitive_text(body, force=True, redact_url_credentials=True)
        return scrubbed if scrubbed is not None else (body or "")
    except Exception as exc:  # redaction must never break the record
        logger.debug("exec_audit redaction failed: %s", exc)
        return body or ""


def _enforce_0600(path: Path) -> None:
    try:
        path.chmod(0o600)
    except OSError:
        pass


def record_exec(
    *,
    tool: str,
    body: str,
    profile: str = "",
    session_key: Optional[str] = "",
    task_id: Optional[str] = "",
    background: bool = False,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Append one invocation record. NEVER raises (mirrors cron ``_write_usage_audit``)."""
    try:
        path = exec_audit_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        record: dict[str, Any] = {
            "ts": _utcnow_iso_ms(),
            "tool": tool,
            "profile": profile,
            "session_key": session_key,
            "task_id": task_id or "",
            "background": bool(background),
            "body": _redact_body(body),
        }
        if extra:
            record.update(extra)
        line = json.dumps(record, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        _enforce_0600(path)
    except Exception as exc:
        logger.warning("exec_audit write failed: %s", exc)
