from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

EXECUTION_STATES = {"generated", "persisted", "delivered", "failed", "retried"}
DATA_PROVENANCE = {"live", "cached", "synthetic", "pending", "unavailable"}
FAILURE_CLASSIFICATIONS = {
    "configuration",
    "authentication",
    "network",
    "telegram_api",
    "storage",
    "validation",
    "unknown",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _env_or(default: str) -> str:
    return os.environ.get(default, default)


@dataclass(frozen=True)
class BriefRecord:
    id: str = field(default_factory=lambda: "BR-" + uuid.uuid4().hex[:12])
    execution_id: str = field(default_factory=lambda: "EX-" + uuid.uuid4().hex[:12])
    scheduled_date: str = ""
    environment: str = field(default_factory=lambda: os.environ.get("HERMES_ENV", "unknown"))
    version: str = field(default_factory=lambda: os.environ.get("HERMES_VERSION", "0.18.2"))
    git_sha: str = field(default_factory=lambda: os.environ.get("HERMES_GIT_SHA", "unknown"))


@dataclass(frozen=True)
class ExecutionMetrics:
    started_at: str = field(default_factory=_now)
    ended_at: Optional[str] = None
    duration_ms: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    retry_count: int = 0
    exit_code: Optional[int] = None
    failure_classification: Optional[str] = None
    error_message: Optional[str] = None

    def finish(self, exit_code: int = 0) -> None:
        object.__setattr__(self, "ended_at", _now())
        object.__setattr__(self, "exit_code", exit_code)
        start = datetime.fromisoformat(self.started_at)
        end = datetime.fromisoformat(self.ended_at)
        object.__setattr__(self, "duration_ms", int((end - start).total_seconds() * 1000))
        try:
            import resource
            object.__setattr__(self, "memory_usage_mb", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        except Exception:
            pass


class OperationalMetrics:
    def __init__(self) -> None:
        self.successful_executions: int = 0
        self.failed_executions: int = 0
        self.total_duration_ms: int = 0
        self.total_delivery_latency_ms: int = 0
        self.consecutive_failures: int = 0
        self.last_successful_execution: Optional[str] = None
        self.last_successful_delivery: Optional[str] = None

    def record_success(self, duration_ms: int, delivery_latency_ms: Optional[int] = None) -> None:
        self.successful_executions += 1
        self.total_duration_ms += duration_ms
        if delivery_latency_ms is not None:
            self.total_delivery_latency_ms += delivery_latency_ms
        self.consecutive_failures = 0
        self.last_successful_execution = _now()

    def record_failure(self) -> None:
        self.failed_executions += 1
        self.consecutive_failures += 1

    def record_delivery(self) -> None:
        self.last_successful_delivery = _now()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "average_runtime_ms": (self.total_duration_ms / self.successful_executions) if self.successful_executions else 0,
            "average_delivery_latency_ms": (self.total_delivery_latency_ms / self.successful_executions) if self.successful_executions else 0,
            "consecutive_failures": self.consecutive_failures,
            "last_successful_execution": self.last_successful_execution,
            "last_successful_delivery": self.last_successful_delivery,
        }


_metrics = OperationalMetrics()


def get_metrics() -> OperationalMetrics:
    return _metrics


def make_executive_brief(
    *,
    calendar: Optional[Dict[str, Any]] = None,
    blackgold: Optional[Dict[str, Any]] = None,
    pipeline: Optional[Dict[str, Any]] = None,
    approvals: Optional[Dict[str, Any]] = None,
    banking: Optional[Dict[str, Any]] = None,
    portfolio: Optional[Dict[str, Any]] = None,
    market: Optional[Dict[str, Any]] = None,
    workforce: Optional[Dict[str, Any]] = None,
    infra: Optional[Dict[str, Any]] = None,
    failed_jobs: Optional[Dict[str, Any]] = None,
    financial: Optional[Dict[str, Any]] = None,
    weather: Optional[Dict[str, Any]] = None,
    scripture: Optional[str] = None,
    priorities: Optional[Dict[str, Any]] = None,
    execution: Optional[BriefRecord] = None,
) -> Dict[str, Any]:
    now = _now()
    record = execution or BriefRecord(scheduled_date=datetime.now(timezone.utc).date().isoformat())

    brief = {
        "id": record.id,
        "execution_id": record.execution_id,
        "scheduled_date": record.scheduled_date,
        "environment": record.environment,
        "version": record.version,
        "git_sha": record.git_sha,
        "generated_at": now,
        "generated_timestamp": now,
        "currency": "USD",
        "timezone": "UTC",
        "execution_state": "generated",
        "delivery": {
            "status": "pending",
            "message_id": None,
            "telegram_delivery_timestamp": None,
            "retry_count": 0,
            "error_classification": None,
        },
        "execution_log": [
            {"timestamp": now, "state": "generated", "message": "Brief generated"}
        ],
        "sections": [
            "calendar",
            "blackgold_priority_items",
            "deal_pipeline",
            "approval_queue",
            "banking_and_gci",
            "portfolio_alerts",
            "market_intelligence",
            "ai_workforce_status",
            "infrastructure_health",
            "failed_jobs",
            "financial_alerts",
            "weather",
            "scripture",
            "top_executive_priorities",
        ],
        "calendar": _section(calendar, now),
        "blackgold_priority_items": _section(blackgold, now),
        "deal_pipeline": _section(pipeline, now),
        "approval_queue": _section(approvals, now),
        "banking_and_gci": _section(banking, now),
        "portfolio_alerts": _section(portfolio, now),
        "market_intelligence": _section(market, now),
        "ai_workforce_status": _section(workforce, now),
        "infrastructure_health": _section(infra, now),
        "failed_jobs": _section(failed_jobs, now),
        "financial_alerts": _section(financial, now),
        "weather": _section(weather, now),
        "scripture": _scripture_section(scripture, now),
        "top_executive_priorities": _section(priorities, now),
        "provenance_legend": {
            "live": "Real operational data",
            "cached": "Recent data returned from storage",
            "synthetic": "Generated placeholder pending integration",
            "pending": "Awaiting upstream source",
            "unavailable": "Integration not yet connected",
        },
        "data_source_status": {
            section: _source_status(section)
            for section in [
                "calendar",
                "blackgold_priority_items",
                "deal_pipeline",
                "approval_queue",
                "banking_and_gci",
                "portfolio_alerts",
                "market_intelligence",
                "ai_workforce_status",
                "infrastructure_health",
                "failed_jobs",
                "financial_alerts",
                "weather",
                "scripture",
                "top_executive_priorities",
            ]
        },
        "metrics": {
            "execution_id": record.execution_id,
            "environment": record.environment,
            "git_sha": record.git_sha,
        },
    }
    return brief


def _source_status(section: str) -> str:
    return "available"


def _section(value: Optional[Dict[str, Any]], updated_at: str) -> Dict[str, Any]:
    if value is None:
        return {"data": {}, "provenance": "unavailable", "updated_at": updated_at, "error": None}
    return {
        "data": value.get("data", value),
        "provenance": value.get("provenance", "synthetic"),
        "updated_at": value.get("updated_at", updated_at),
        "error": value.get("error"),
    }


def _scripture_section(value: Optional[str], updated_at: str) -> Dict[str, Any]:
    if not value:
        return {"data": "", "provenance": "unavailable", "updated_at": updated_at, "error": None}
    return {"data": value, "provenance": "cached", "updated_at": updated_at, "error": None}


def redact_secrets(pack: Dict[str, Any]) -> Dict[str, Any]:
    redacted = json.loads(json.dumps(pack))
    keys = ["telegram_bot_token", "openrouter_api_key", "google_client_secret", "onepassword_token", "secret"]
    for key in list(redacted.keys()):
        if any(k in key.lower() for k in keys):
            redacted[key] = "***REDACTED***"
    if "delivery" in redacted and isinstance(redacted["delivery"], dict):
        redacted["delivery"] = {k: v for k, v in redacted["delivery"].items() if k != "error"}
    if "execution_log" in redacted and isinstance(redacted["execution_log"], list):
        redacted["execution_log"] = [
            {k: (_sanitize_value(v) if isinstance(v, str) else v) for k, v in entry.items()}
            for entry in redacted["execution_log"]
        ]
    return redacted


def _sanitize_value(value: str) -> str:
    patterns = [
        r"telegram_bot_token",
        r"openrouter_api_key",
        r"google_client_secret",
        r"onepassword_token",
        r"Authorization",
        r"Bearer [A-Za-z0-9\-_\.]+",
        r"api[_-]?key",
        r"secret",
    ]
    text = value
    for pat in patterns:
        text = re.sub(pat, "***REDACTED***", text, flags=re.IGNORECASE)
    return text


import re


def validate_brief_schema(brief: Dict[str, Any]) -> Dict[str, Any]:
    required = ["id", "execution_id", "scheduled_date", "environment", "generated_at", "execution_state", "delivery", "execution_log"]
    missing = [k for k in required if k not in brief]
    if missing:
        return {"success": False, "error": {"code": "invalid_schema", "missing": missing}}
    if brief["execution_state"] not in EXECUTION_STATES:
        return {"success": False, "error": {"code": "invalid_state", "state": brief["execution_state"]}}
    return {"success": True}


def append_execution_log(brief: Dict[str, Any], state: str, message: str) -> Dict[str, Any]:
    brief["execution_state"] = state
    brief.setdefault("execution_log", []).append({"timestamp": _now(), "state": state, "message": message})
    return brief


def mark_delivered(brief: Dict[str, Any], message_id: Any, timestamp: Optional[str] = None) -> Dict[str, Any]:
    brief["delivery"] = {
        "status": "delivered",
        "message_id": message_id,
        "telegram_delivery_timestamp": timestamp or _now(),
        "retry_count": brief.get("delivery", {}).get("retry_count", 0),
        "error_classification": None,
    }
    return append_execution_log(brief, "delivered", f"Delivered message_id={message_id}")


def mark_failed(brief: Dict[str, Any], error: str, classification: str, retry: bool = False) -> Dict[str, Any]:
    if classification not in FAILURE_CLASSIFICATIONS:
        classification = "unknown"
    delivery = brief.setdefault("delivery", {})
    delivery["status"] = "failed"
    delivery["error_classification"] = classification
    if retry:
        delivery["retry_count"] = delivery.get("retry_count", 0) + 1
    return append_execution_log(brief, "failed", f"Delivery failed: {_sanitize_value(error)}")


def classify_failure(error: Exception) -> str:
    text = str(error).lower()
    if "auth" in text or "token" in text or "unauthorized" in text:
        return "authentication"
    if "telegram" in text or "bot" in text:
        return "telegram_api"
    if "network" in text or "timeout" in text or "dns" in text or "connect" in text:
        return "network"
    if "config" in text or "env" in text or "missing" in text:
        return "configuration"
    if "storage" in text or "database" in text or "sqlite" in text or "volume" in text:
        return "storage"
    if "validation" in text or "schema" in text:
        return "validation"
    return "unknown"


def duplicate_delivery_key(brief: Dict[str, Any]) -> str:
    # Execution ID is intentionally excluded so repeated reruns of the same
    # scheduled brief resolve to one stable key and are blocked by the
    # delivery_blocklist instead of always being treated as new sends.
    scheduled = brief.get("scheduled_date", "")
    delivery = brief.get("delivery", {})
    ts = delivery.get("telegram_delivery_timestamp", "")
    msg = delivery.get("message_id", "")
    return f"brief:{scheduled}:{msg}:{ts}"


def store_delivery_blocklist(idempotency_key: str, db_path: str = "/app/.hermes/executions.db") -> Dict[str, Any]:
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS delivery_blocklist(idempotency_key TEXT PRIMARY KEY, created_at TEXT)"
        )
        cur.execute(
            "INSERT OR IGNORE INTO delivery_blocklist(idempotency_key, created_at) VALUES(?, ?)",
            (idempotency_key, _now()),
        )
        conn.commit()
        inserted = cur.rowcount == 1
        conn.close()
        return {"success": True, "inserted": inserted}
    except Exception as exc:
        logger.exception("Failed to write delivery_blocklist")
        return {"success": False, "error": str(exc)}


def is_delivery_blocked(idempotency_key: str, db_path: str = "/app/.hermes/executions.db") -> bool:
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS delivery_blocklist(idempotency_key TEXT PRIMARY KEY, created_at TEXT)"
        )
        cur.execute("SELECT 1 FROM delivery_blocklist WHERE idempotency_key = ?", (idempotency_key,))
        row = cur.fetchone()
        conn.close()
        return row is not None
    except Exception as exc:
        logger.exception("Failed to read delivery_blocklist")
        return False


def store_brief(pack: Dict[str, Any], db_path: str = "/app/.hermes/executions.db") -> Dict[str, Any]:
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS briefs (id TEXT, execution_id TEXT, scheduled_date TEXT, environment TEXT, payload TEXT, created_at TEXT)"
        )
        cur.execute(
            "INSERT INTO briefs(id, execution_id, scheduled_date, environment, payload, created_at) VALUES(?, ?, ?, ?, ?, ?)",
            (pack.get("id"), pack.get("execution_id"), pack.get("scheduled_date"), pack.get("environment"), json.dumps(pack), _now()),
        )
        conn.commit()
        conn.close()
        return {"success": True, "id": pack.get("id"), "execution_id": pack.get("execution_id")}
    except Exception as exc:
        logger.exception("Failed to store brief")
        return {"success": False, "error": str(exc)}


def is_dry_run() -> bool:
    return os.environ.get("BRIEF_DRY_RUN", "false").lower() in {"1", "true", "yes"}
