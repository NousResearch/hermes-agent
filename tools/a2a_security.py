"""Shared A2A security utilities — used by both the gateway adapter and client tools.

Single source of truth for:
- Prompt injection detection and filtering (inbound)
- Sensitive data redaction (outbound)
- Audit logging
- Rate limiting
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt injection patterns (inbound filtering)
# ---------------------------------------------------------------------------

INJECTION_PATTERNS = [
    re.compile(r"(?i)<\s*system\s*>.*?<\s*/\s*system\s*>", re.DOTALL),
    re.compile(r"(?i)\[INST\].*?\[/INST\]", re.DOTALL),
    re.compile(r"(?i)ignore\s+(all\s+)?previous\s+instructions?"),
    re.compile(r"(?i)you\s+are\s+now\s+"),
    re.compile(r"(?i)new\s+system\s+prompt"),
    re.compile(r"(?i)disregard\s+(all\s+)?(prior|earlier|above)"),
    re.compile(r"(?i)override\s+(your\s+)?(instructions?|rules?|guidelines?)"),
]


def sanitize_inbound(text: str, max_length: int = 50_000) -> str:
    """Strip prompt injection patterns from inbound A2A messages."""
    if len(text) > max_length:
        text = text[:max_length] + "\n[... message truncated for safety]"
        logger.warning("Inbound A2A message truncated: %d chars", max_length)

    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning("Prompt injection pattern detected in A2A message")
            text = pattern.sub("[FILTERED]", text)

    return text


# ---------------------------------------------------------------------------
# Sensitive data patterns (outbound filtering)
# ---------------------------------------------------------------------------

SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|password|token|credential)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(sk-[a-zA-Z0-9]{20,})"),
    re.compile(r"(?i)(ghp_[a-zA-Z0-9]{20,})"),
    re.compile(r"(?i)(xoxb-[a-zA-Z0-9-]+)"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
]


def filter_outbound(text: str) -> str:
    """Redact sensitive patterns from outbound A2A responses."""
    for pattern in SENSITIVE_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Thread-safe sliding-window rate limiter."""

    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._buckets: Dict[str, list] = defaultdict(list)
        self._lock = Lock()

    def allow(self, client_id: str) -> bool:
        now = time.time()
        with self._lock:
            bucket = self._buckets[client_id]
            self._buckets[client_id] = [ts for ts in bucket if ts > now - self.window]
            if len(self._buckets[client_id]) >= self.max_requests:
                return False
            self._buckets[client_id].append(now)
            return True

    def remaining(self, client_id: str) -> int:
        now = time.time()
        with self._lock:
            bucket = self._buckets.get(client_id, [])
            active = [ts for ts in bucket if ts > now - self.window]
            return max(0, self.max_requests - len(active))


# ---------------------------------------------------------------------------
# Audit logger
# ---------------------------------------------------------------------------

class AuditLogger:
    """Thread-safe append-only audit log for A2A interactions."""

    def __init__(self, log_path: Optional[Path] = None):
        if log_path is None:
            try:
                from hermes_constants import get_hermes_home
                log_path = get_hermes_home() / "a2a_audit.jsonl"
            except ImportError:
                log_path = Path.home() / ".hermes" / "a2a_audit.jsonl"
        self.log_path = log_path
        self._lock = Lock()

    def log(self, event_type: str, data: dict) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **data,
        }
        try:
            with self._lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("Failed to write A2A audit log", exc_info=True)


# Singleton instance
audit = AuditLogger()
