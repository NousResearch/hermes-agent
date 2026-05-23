"""Reliability gate for Hermes self-improvement evidence.

This is the small repo-local reliability floor retained after the Hadto-specific
orchestration stack moved to plugins. It deliberately avoids Linear writeback
and cross-repo ontology orchestration; callers provide paths to evidence files
and get a deterministic scorecard back.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_constants import display_hermes_home, get_hermes_home
from tools.registry import registry
from utils import atomic_json_write

logger = logging.getLogger(__name__)


DEFAULT_JOURNAL_PATH = Path("/home/david/stacks/hermes-journal/src/data/journal.json")
DEFAULT_CODEX_RUNS_PATH = get_hermes_home() / "codex" / "runs.json"
DEFAULT_CTX_BINDINGS_PATH = get_hermes_home() / "ctx" / "session_bindings.json"
DEFAULT_ONTOLOGY_ROOT = Path("/home/david/stacks/smb-ontology-platform")
DEFAULT_BENCHMARK_HISTORY_PATH = get_hermes_home() / "self_improvement" / "benchmark_history.json"
DEFAULT_FRESHNESS_HOURS = 72
DEFAULT_ACTIVE_STALE_HOURS = 12
PROVENANCE_CONTRACT_VERSION = "v1"
BENCHMARK_CONTRACT_VERSION = "v1"
_BENCHMARK_HISTORY_LIMIT = 200
_LEADING_INDICATOR_CHECK_IDS = (
    "reliability_gate",
    "anti_make_work_check",
    "operator_value_alignment",
)
_LEADING_INDICATOR_HARBINGERS = (
    "critical_slowing_down",
    "variance_explosion",
    "flickering",
    "correlation_explosion",
)
_ONTOLOGY_SCAN_SUFFIXES = {".json", ".yaml", ".yml", ".md"}
_ONTOLOGY_SCAN_PRUNED_DIRS = {".git", "__pycache__", ".pytest_cache", "tests"}
_ONTOLOGY_REQUIRED_ARTIFACTS = (
    ("ontology_metrics", Path("evolution/metrics.json")),
    ("ontology_delta_report", Path("evolution/delta_report.json")),
    ("ontology_daily_report", Path("evolution/daily_report.md")),
)
_FUTURE_TIMESTAMP_TOLERANCE_SECONDS = 300
_TEXT_EVIDENCE_EXCLUDED_KEYS = {
    "command",
    "prompt",
    "command_args",
    "ctx_worktree_path",
    "latest_path",
    "last_message_path",
    "record_path",
    "workdir",
    "worktree_path",
}
_CLAIM_TEXT_KEYS = {
    "detail",
    "final_message",
    "last_agent_message",
    "notes",
    "outcome_note",
    "reason",
    "result",
    "summary",
    "title",
}
_CLAIM_CONTAINER_KEYS = {
    "active_agenda",
    "current_strategy",
    "lane_links",
    "self_improvement_focus",
}
_DURABLE_EVIDENCE_KEYS = {
    "artifact_path",
    "artifact_paths",
    "artifacts",
    "changed_files",
    "changed_paths",
    "checks",
    "ci",
    "commit",
    "commit_sha",
    "commit_shas",
    "commits",
    "decision",
    "decisions",
    "durable_artifacts",
    "evidence",
    "files_changed",
    "operator_decision_support",
    "pr_url",
    "pr_urls",
    "proof_artifact",
    "proof_artifacts",
    "pull_request",
    "pull_request_url",
    "pull_requests",
    "risk_reduction",
    "test_results",
    "tests",
    "verification",
}
_STATUS_ONLY_PATTERNS = (
    re.compile(r"\bactionable\b", re.IGNORECASE),
    re.compile(r"\bactive work\b", re.IGNORECASE),
    re.compile(r"\bin[- ]progress\b", re.IGNORECASE),
    re.compile(r"\bnext steps?\b", re.IGNORECASE),
    re.compile(r"\bqueued\b", re.IGNORECASE),
    re.compile(r"\bselected\b", re.IGNORECASE),
    re.compile(r"\bstatus(?:\s+update)?\b", re.IGNORECASE),
    re.compile(r"\bsummary\b", re.IGNORECASE),
    re.compile(r"\btriage(?:d|s|)\b", re.IGNORECASE),
    re.compile(r"\bworking on\b", re.IGNORECASE),
)
_DURABLE_TEXT_PATTERNS = (
    ("commit", re.compile(r"\b(?:commit|committed|sha)\b[^.\n]{0,120}\b[0-9a-f]{7,40}\b", re.IGNORECASE)),
    ("commit", re.compile(r"\b[0-9a-f]{7,40}\b[^.\n]{0,120}\b(?:commit|sha)\b", re.IGNORECASE)),
    ("pull_request", re.compile(r"https://github\.com/[^\s)]+/[^\s)]+/pull/\d+", re.IGNORECASE)),
    ("pull_request", re.compile(r"\b(?:PR|pull request)\s*#?\d+\b", re.IGNORECASE)),
    (
        "verification",
        re.compile(
            r"\b(?:pytest|npm test|uv run pytest|ruff|mypy|git diff --check|GitHub Actions|CI)\b"
            r"[^.\n]{0,160}\b(?:passed|pass|success|succeeded|green|\d+\s+passed|0\s+failed)\b",
            re.IGNORECASE,
        ),
    ),
    ("changed_files", re.compile(r"\bCHANGED_FILES\b|\bchanged files?\b", re.IGNORECASE)),
    (
        "artifact",
        re.compile(
            r"\b(?:durable|checked-in|repo-visible)\b[^.\n]{0,120}\b(?:artifact|evidence|record)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "state_transition",
        re.compile(
            r"\b(?:merged|pushed|opened|created|closed|resolved|completed)\b"
            r"[^.\n]{0,120}\b(?:PR|pull request|branch|issue|commit|state|artifact|file|test)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "operator_decision_support",
        re.compile(
            r"\b(?:blocked|blocker|missing|unavailable|permission|403|401|unable to)\b"
            r"[^.\n]{0,160}\b(?:operator|token|scope|credential|auth|permission|artifact|secret|manual)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "operator_decision_support",
        re.compile(
            r"\b(?:operator|human|user)\b[^.\n]{0,160}"
            r"\b(?:decision|decide|choose|approval|blocker|risk|trade[- ]off|manual step|recommended)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "operator_decision_support",
        re.compile(
            r"\b(?:decision|decide|choose|approval|blocker|risk|trade[- ]off|manual step|recommended)\b"
            r"[^.\n]{0,160}\b(?:operator|human|user)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "capability_change",
        re.compile(
            r"\b(?:added|implemented|fixed|hardened|repaired|wired|enabled)\b"
            r"[^.\n]{0,160}\b(?:tool|runtime|service|gateway|config|workflow|benchmark|check|test|schema)\b",
            re.IGNORECASE,
        ),
    ),
)
_OPERATOR_DECISION_SUPPORT_SIGNALS = {
    "decision",
    "decisions",
    "operator_decision_support",
    "risk_reduction",
}
_VERIFIED_SYSTEM_CHANGE_SIGNALS = {
    "artifact",
    "artifact_path",
    "artifact_paths",
    "artifacts",
    "capability_change",
    "changed_files",
    "changed_paths",
    "checks",
    "ci",
    "commit",
    "commit_sha",
    "commit_shas",
    "commits",
    "durable_artifacts",
    "evidence",
    "files_changed",
    "pr_url",
    "pr_urls",
    "proof_artifact",
    "proof_artifacts",
    "pull_request",
    "pull_request_url",
    "pull_requests",
    "state_transition",
    "test_results",
    "tests",
    "verification",
}


SELF_IMPROVEMENT_EVIDENCE_SCHEMA = {
    "name": "self_improvement_evidence_gate",
    "description": (
        "Evaluate freshness and consistency of Hermes self-improvement evidence "
        "without creating or updating backlog work."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "journal_path": {"type": "string"},
            "codex_runs_path": {"type": "string"},
            "ctx_bindings_path": {"type": "string"},
            "ontology_root": {"type": "string"},
            "now": {"type": "string"},
            "freshness_hours": {"type": "integer", "minimum": 1},
            "active_stale_hours": {"type": "integer", "minimum": 1},
        },
        "required": [],
    },
}


SELF_IMPROVEMENT_BENCHMARK_SCHEMA = {
    "name": "self_improvement_benchmark",
    "description": (
        "Score the Hermes self-improvement reliability floor and optionally "
        f"persist benchmark history under {display_hermes_home()}/self_improvement."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "journal_path": {"type": "string"},
            "codex_runs_path": {"type": "string"},
            "ctx_bindings_path": {"type": "string"},
            "ontology_root": {"type": "string"},
            "history_path": {"type": "string"},
            "now": {"type": "string"},
            "freshness_hours": {"type": "integer", "minimum": 1},
            "active_stale_hours": {"type": "integer", "minimum": 1},
            "persist": {"type": "boolean"},
        },
        "required": [],
    },
}


def _parse_time(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        try:
            parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read JSON evidence file %s", path, exc_info=True)
        return None


def _iter_records(payload: Any, key: str) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        value = payload.get(key)
        if isinstance(value, dict):
            for item in value.values():
                if isinstance(item, dict):
                    yield item
            return
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item
            return
        if all(not isinstance(item, (dict, list)) for item in payload.values()):
            yield payload
            return
        for item in payload.values():
            if isinstance(item, dict):
                yield item
            elif isinstance(item, list):
                for child in item:
                    if isinstance(child, dict):
                        yield child
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def _record_timestamp(record: dict[str, Any], *keys: str) -> Optional[datetime]:
    for key in keys:
        parsed = _parse_time(record.get(key))
        if parsed is not None:
            return parsed
    return None


def _iter_journal_timestamps(payload: Any) -> Iterable[datetime]:
    for record in _iter_records(payload, "entries"):
        parsed = _record_timestamp(
            record,
            "occurredAt",
            "occurred_at",
            "updatedAt",
            "updated_at",
            "createdAt",
            "created_at",
            "timestamp",
            "date",
        )
        if parsed is not None:
            yield parsed


def _iter_codex_records(payload: Any) -> Iterable[dict[str, Any]]:
    yield from _iter_records(payload, "runs")


def _iter_codex_timestamps(payload: Any) -> Iterable[datetime]:
    for record in _iter_codex_records(payload):
        parsed = _record_timestamp(
            record,
            "completed_at",
            "updated_at",
            "started_at",
            "process_started_at",
            "created_at",
            "timestamp",
        )
        if parsed is not None:
            yield parsed


def _iter_ctx_records(payload: Any) -> Iterable[dict[str, Any]]:
    yield from _iter_records(payload, "sessions")


def _iter_ctx_timestamps(payload: Any) -> Iterable[datetime]:
    for record in _iter_ctx_records(payload):
        parsed = _record_timestamp(
            record,
            "updated_at",
            "updatedAt",
            "created_at",
            "createdAt",
            "timestamp",
        )
        if parsed is not None:
            yield parsed


def _ctx_record_is_active(record: dict[str, Any]) -> bool:
    return record.get("active") is True


def _latest_timestamp(values: Iterable[datetime]) -> Optional[datetime]:
    return max(values, default=None)


def _summarize_source(
    name: str,
    latest: Optional[datetime],
    freshness_hours: int,
    now: datetime,
) -> dict[str, Any]:
    if latest is None:
        return {
            "source": name,
            "status": "missing",
            "age_hours": None,
            "latest_timestamp": None,
        }
    age_hours = max(0.0, (now - latest).total_seconds() / 3600)
    return {
        "source": name,
        "status": "fresh" if age_hours <= freshness_hours else "stale",
        "age_hours": round(age_hours, 2),
        "latest_timestamp": latest.isoformat(),
    }


def _summarize_ctx_bindings(
    payload: Any,
    freshness_hours: int,
    now: datetime,
) -> dict[str, Any]:
    if payload is None:
        summary = _summarize_source("ctx_bindings", None, freshness_hours, now)
        summary.update(
            {
                "record_count": None,
                "active_count": None,
                "freshness_required": True,
                "detail": "ctx bindings evidence unavailable.",
            }
        )
        return summary

    records = list(_iter_ctx_records(payload))
    latest = _latest_timestamp(_iter_ctx_timestamps(payload))
    active_count = sum(1 for record in records if _ctx_record_is_active(record))
    summary = _summarize_source("ctx_bindings", latest, freshness_hours, now)
    summary.update(
        {
            "record_count": len(records),
            "active_count": active_count,
            "freshness_required": bool(active_count),
        }
    )

    if active_count == 0 and summary["status"] in {"missing", "stale"}:
        summary["status"] = "inactive"
        summary["detail"] = (
            "No active ctx bindings; retired binding timestamps are informational."
            if latest is not None
            else "No ctx bindings recorded; no active ctx sessions require freshness."
        )

    return summary


def _extract_timestamps_from_text(text: str) -> Iterable[datetime]:
    for match in re.finditer(
        r"\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+(?:Z|[+-]\d{2}:?\d{2})?",
        text,
    ):
        parsed = _parse_time(match.group(0).replace(" ", "T", 1))
        if parsed is not None:
            yield parsed


def _scan_ontology_file(path: Path) -> tuple[list[datetime], list[str]]:
    timestamps: list[datetime] = []
    alerts: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return timestamps, [f"{path.name} unreadable"]

    if path.suffix == ".json":
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            for key in (
                "generated_at",
                "updated_at",
                "prepared_at",
                "last_evolved",
                "timestamp",
                "created_at",
            ):
                parsed = _parse_time(payload.get(key))
                if parsed is not None:
                    timestamps.append(parsed)
            reliability = payload.get("reliability")
            status = ""
            if isinstance(reliability, dict):
                status = str(reliability.get("status") or "")
            status = status or str(payload.get("status") or "")
            if status.strip().lower() in {"degraded", "error", "failed", "missing", "stale"}:
                alerts.append(f"{path.name} status={status.strip().lower()}")
    else:
        timestamps.extend(_extract_timestamps_from_text(text))

    return timestamps, alerts


def _iter_ontology_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in _ONTOLOGY_SCAN_PRUNED_DIRS
        ]
        current_dir = Path(dirpath)
        for filename in filenames:
            path = current_dir / filename
            if path.suffix.lower() in _ONTOLOGY_SCAN_SUFFIXES:
                yield path


def _split_future_timestamps(
    timestamps: Iterable[datetime],
    now: datetime,
) -> tuple[list[datetime], list[datetime]]:
    valid: list[datetime] = []
    future: list[datetime] = []
    for timestamp in timestamps:
        if (timestamp - now).total_seconds() > _FUTURE_TIMESTAMP_TOLERANCE_SECONDS:
            future.append(timestamp)
        else:
            valid.append(timestamp)
    return valid, future


def _artifact_summary_from_timestamps(
    *,
    name: str,
    path: Path,
    timestamps: Iterable[datetime],
    alerts: Iterable[str],
    freshness_hours: int,
    now: datetime,
) -> dict[str, Any]:
    valid_timestamps, future_timestamps = _split_future_timestamps(timestamps, now)
    latest = _latest_timestamp(valid_timestamps)
    alert_reasons = [str(item).strip() for item in alerts if str(item).strip()]
    reasons = list(alert_reasons)

    if latest is None:
        status = "missing"
        age_hours = None
        latest_timestamp = None
        if future_timestamps:
            reasons.append(f"{name} only has future timestamps")
    else:
        age_hours = round(max(0.0, (now - latest).total_seconds() / 3600), 2)
        latest_timestamp = latest.isoformat()
        status = "fresh" if age_hours <= freshness_hours else "stale"
        if status == "stale":
            reasons.append(f"{name} stale ({age_hours}h)")

    if alert_reasons and status in {"fresh", "stale"}:
        status = "degraded"

    return {
        "source": name,
        "path": str(path),
        "status": status,
        "age_hours": age_hours,
        "latest_timestamp": latest_timestamp,
        "future_timestamp_count": len(future_timestamps),
        "ignored_future_timestamps": [
            timestamp.isoformat() for timestamp in sorted(future_timestamps)[-5:]
        ],
        "reasons": reasons,
    }


def _summarize_required_ontology_artifacts(
    root: Path,
    freshness_hours: int,
    now: datetime,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for name, relative_path in _ONTOLOGY_REQUIRED_ARTIFACTS:
        path = root / relative_path
        if not path.exists():
            summaries[name] = {
                "source": name,
                "path": str(path),
                "status": "missing",
                "age_hours": None,
                "latest_timestamp": None,
                "future_timestamp_count": 0,
                "ignored_future_timestamps": [],
                "reasons": [f"{name} missing"],
            }
            continue
        timestamps, alerts = _scan_ontology_file(path)
        summaries[name] = _artifact_summary_from_timestamps(
            name=name,
            path=path,
            timestamps=timestamps,
            alerts=alerts,
            freshness_hours=freshness_hours,
            now=now,
        )
    return summaries


def _ontology_external_repair(root: Path, required_artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    invalid_artifacts = [
        summary
        for summary in required_artifacts.values()
        if str(summary.get("status") or "") in {"stale", "missing", "degraded"}
    ]
    return {
        "required": bool(invalid_artifacts),
        "repository": str(root),
        "action": "refresh ontology evolution reporting artifacts",
        "artifacts": invalid_artifacts,
    }


def _summarize_ontology(root: Path, freshness_hours: int, now: datetime) -> dict[str, Any]:
    if not root.exists():
        return {
            "status": "missing",
            "latest_timestamp": None,
            "age_hours": None,
            "reasons": ["ontology root missing"],
            "alerts": [],
            "required_artifacts": {},
        }

    timestamps: list[datetime] = []
    alerts: list[str] = []
    for path in _iter_ontology_files(root):
        file_timestamps, file_alerts = _scan_ontology_file(path)
        timestamps.extend(file_timestamps)
        alerts.extend(file_alerts)

    required_artifacts = _summarize_required_ontology_artifacts(root, freshness_hours, now)
    required_latest = [
        parsed
        for summary in required_artifacts.values()
        if (parsed := _parse_time(summary.get("latest_timestamp"))) is not None
    ]
    invalid_required = [
        summary
        for summary in required_artifacts.values()
        if str(summary.get("status") or "") in {"stale", "missing", "degraded"}
    ]
    external_repair = _ontology_external_repair(root, required_artifacts)

    valid_timestamps, future_timestamps = _split_future_timestamps(timestamps, now)
    latest = _latest_timestamp(required_latest) or _latest_timestamp(valid_timestamps)
    if latest is None:
        reasons = ["ontology intelligence timestamp missing"]
        if future_timestamps:
            reasons.append("ontology intelligence only has future timestamps")
        return {
            "status": "missing",
            "latest_timestamp": None,
            "age_hours": None,
            "reasons": reasons,
            "alerts": alerts,
            "ignored_future_timestamp_count": len(future_timestamps),
            "required_artifacts": required_artifacts,
            "external_repair": external_repair,
        }

    age_hours = max(0.0, (now - latest).total_seconds() / 3600)
    status = "fresh" if age_hours <= freshness_hours else "stale"
    reasons: list[str] = []
    if status == "stale":
        reasons.append(f"ontology_intelligence stale ({round(age_hours, 2)}h)")
    for summary in invalid_required:
        for reason in summary.get("reasons") or [f"{summary.get('source')} {summary.get('status')}"]:
            text = str(reason).strip()
            if text and text not in reasons:
                reasons.append(text)
    if invalid_required:
        required_statuses = {str(summary.get("status") or "") for summary in invalid_required}
        status = "degraded" if required_statuses.intersection({"missing", "degraded"}) else "stale"
    if alerts:
        status = "degraded"
        reasons.extend(alerts)
    return {
        "status": status,
        "latest_timestamp": latest.isoformat(),
        "age_hours": round(age_hours, 2),
        "reasons": reasons,
        "alerts": alerts,
        "ignored_future_timestamp_count": len(future_timestamps),
        "required_artifacts": required_artifacts,
        "external_repair": external_repair,
    }


def _codex_record_status(record: dict[str, Any]) -> str:
    return str(record.get("status") or "").strip().lower()


def _codex_record_is_active(record: dict[str, Any]) -> bool:
    status = _codex_record_status(record)
    if status in {"running", "queued", "in_progress", "active", "unknown"}:
        return True
    if record.get("active") is True:
        return True
    return record.get("completed_at") is None and record.get("exit_code") is None and bool(status)


def _find_stale_active_codex(
    payload: Any,
    now: datetime,
    active_stale_hours: int,
) -> list[dict[str, Any]]:
    stale: list[dict[str, Any]] = []
    for record in _iter_codex_records(payload):
        if not _codex_record_is_active(record):
            continue
        started = _record_timestamp(
            record,
            "updated_at",
            "started_at",
            "process_started_at",
            "created_at",
            "timestamp",
        )
        if started is None:
            continue
        age_hours = max(0.0, (now - started).total_seconds() / 3600)
        if age_hours > active_stale_hours:
            stale.append(
                {
                    "run_id": record.get("run_id") or record.get("id"),
                    "status": record.get("status"),
                    "age_hours": round(age_hours, 2),
                    "latest_timestamp": started.isoformat(),
                }
            )
    return stale


def _find_stale_active_ctx(
    payload: Any,
    now: datetime,
    active_stale_hours: int,
) -> list[dict[str, Any]]:
    stale: list[dict[str, Any]] = []
    for record in _iter_ctx_records(payload):
        if not _ctx_record_is_active(record):
            continue
        updated = _record_timestamp(record, "updated_at", "updatedAt", "created_at", "createdAt")
        if updated is None:
            continue
        age_hours = max(0.0, (now - updated).total_seconds() / 3600)
        if age_hours > active_stale_hours:
            stale.append(
                {
                    "session_id": record.get("session_id") or record.get("id"),
                    "task_id": record.get("task_id"),
                    "age_hours": round(age_hours, 2),
                    "latest_timestamp": updated.isoformat(),
                }
            )
    return stale


def _find_planning_contradictions(codex_payload: Any, ctx_payload: Any) -> list[dict[str, Any]]:
    contradictions: list[dict[str, Any]] = []
    for record in _iter_ctx_records(ctx_payload):
        if not _ctx_record_is_active(record):
            continue
        worktree_path = str(record.get("worktree_path") or "").strip()
        reason = str(record.get("reason") or "").strip().lower()
        if "retired" in reason:
            contradictions.append(
                {
                    "type": "ctx_binding_retired_but_active",
                    "session_id": record.get("session_id"),
                    "task_id": record.get("task_id"),
                }
            )
        if not worktree_path:
            contradictions.append(
                {
                    "type": "ctx_binding_missing_worktree_path",
                    "session_id": record.get("session_id"),
                    "task_id": record.get("task_id"),
                }
            )
        elif not Path(worktree_path).exists():
            contradictions.append(
                {
                    "type": "ctx_binding_worktree_missing",
                    "session_id": record.get("session_id"),
                    "task_id": record.get("task_id"),
                    "worktree_path": worktree_path,
                }
            )

    for record in _iter_codex_records(codex_payload):
        if not _codex_record_is_active(record):
            continue
        if record.get("completed_at") is not None or record.get("exit_code") is not None:
            contradictions.append(
                {
                    "type": "codex_running_but_completed",
                    "run_id": record.get("run_id") or record.get("id"),
                }
            )
        worktree_path = str(record.get("ctx_worktree_path") or "").strip()
        if worktree_path and not Path(worktree_path).exists():
            contradictions.append(
                {
                    "type": "codex_worktree_missing",
                    "run_id": record.get("run_id") or record.get("id"),
                    "ctx_worktree_path": worktree_path,
                }
            )
    return contradictions


def _build_ctx_remediation(
    ctx_bindings_path: Path,
    ctx_summary: dict[str, Any],
    stale_active_ctx: list[dict[str, Any]],
    planning_contradictions: list[dict[str, Any]],
) -> dict[str, Any]:
    status = str(ctx_summary.get("status") or "")
    ctx_contradictions = [
        item
        for item in planning_contradictions
        if str(item.get("type") or "").startswith("ctx_")
    ]
    required = status in {"missing", "stale", "degraded"} or bool(stale_active_ctx or ctx_contradictions)

    if not required:
        return {
            "required": False,
            "path": str(ctx_bindings_path),
            "action": "none",
            "reason": str(
                ctx_summary.get("detail")
                or "ctx session-binding evidence is current."
            ),
            "active_count": ctx_summary.get("active_count"),
            "stale_active_count": 0,
            "contradiction_count": 0,
        }

    actions: list[str] = []
    reasons: list[str] = []
    if status == "missing":
        actions.append("restore or regenerate ctx session-binding evidence")
        reasons.append("ctx session-binding evidence is unavailable")
    elif status == "stale":
        actions.append("refresh active ctx session bindings or retire sessions that are no longer live")
        reasons.append(f"ctx session-binding evidence is stale ({ctx_summary.get('age_hours')}h)")
    elif status == "degraded":
        actions.append("repair degraded ctx session-binding evidence before selecting new work")
        reasons.append("ctx session-binding evidence is degraded")

    if stale_active_ctx:
        actions.append("retire stale active ctx bindings or refresh them from the live ctx runtime")
        reasons.append(f"{len(stale_active_ctx)} active ctx binding(s) exceed freshness limits")
    if ctx_contradictions:
        actions.append("repair ctx binding store contradictions")
        reasons.append(f"{len(ctx_contradictions)} ctx binding contradiction(s) detected")

    return {
        "required": True,
        "path": str(ctx_bindings_path),
        "action": "; ".join(dict.fromkeys(actions)),
        "reasons": list(dict.fromkeys(reasons)),
        "active_count": ctx_summary.get("active_count"),
        "stale_active_count": len(stale_active_ctx),
        "stale_active_sessions": stale_active_ctx[:5],
        "contradiction_count": len(ctx_contradictions),
        "contradictions": ctx_contradictions[:5],
    }


def _build_provenance_item(tag: str, path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "tag": tag,
        "path": str(path),
        "status": summary.get("status"),
        "latest_timestamp": summary.get("latest_timestamp"),
        "age_hours": summary.get("age_hours"),
    }


def format_evidence_provenance(items: Iterable[dict[str, Any]]) -> str:
    lines = ["Evidence provenance:"]
    for item in items:
        details: list[str] = []
        for key in ("status", "age_hours", "latest_timestamp", "path", "notes"):
            value = item.get(key)
            if value is not None and value != "":
                label = "latest" if key == "latest_timestamp" else key
                details.append(f"{label}={value}")
        lines.append(f"- [{item.get('tag')}] " + "; ".join(details))
    return "\n".join(lines)


def evaluate_self_improvement_evidence(
    *,
    journal_path: Path = DEFAULT_JOURNAL_PATH,
    codex_runs_path: Path = DEFAULT_CODEX_RUNS_PATH,
    ctx_bindings_path: Path = DEFAULT_CTX_BINDINGS_PATH,
    ontology_root: Path = DEFAULT_ONTOLOGY_ROOT,
    now: Optional[datetime] = None,
    freshness_hours: int = DEFAULT_FRESHNESS_HOURS,
    active_stale_hours: int = DEFAULT_ACTIVE_STALE_HOURS,
) -> dict[str, Any]:
    current = now or datetime.now(tz=timezone.utc)
    journal_payload = _load_json(journal_path)
    codex_payload = _load_json(codex_runs_path)
    ctx_payload = _load_json(ctx_bindings_path)
    ontology_summary = _summarize_ontology(ontology_root, freshness_hours, current)

    journal_latest = _latest_timestamp(_iter_journal_timestamps(journal_payload))
    codex_latest = _latest_timestamp(_iter_codex_timestamps(codex_payload))
    ctx_latest = _latest_timestamp(_iter_ctx_timestamps(ctx_payload))
    ctx_summary = _summarize_ctx_bindings(ctx_payload, freshness_hours, current)
    ontology_latest = _parse_time(ontology_summary.get("latest_timestamp"))

    sources = {
        "journal_entries": _summarize_source("journal_entries", journal_latest, freshness_hours, current),
        "codex_runs": _summarize_source("codex_runs", codex_latest, freshness_hours, current),
        "ctx_bindings": ctx_summary,
        "ontology_intelligence": {
            "source": "ontology_intelligence",
            "status": ontology_summary.get("status"),
            "age_hours": ontology_summary.get("age_hours"),
            "latest_timestamp": ontology_summary.get("latest_timestamp"),
        },
    }

    stale_active_codex = _find_stale_active_codex(codex_payload, current, active_stale_hours)
    stale_active_ctx = _find_stale_active_ctx(ctx_payload, current, active_stale_hours)
    planning_contradictions = _find_planning_contradictions(codex_payload, ctx_payload)
    ctx_remediation = _build_ctx_remediation(
        ctx_bindings_path,
        ctx_summary,
        stale_active_ctx,
        planning_contradictions,
    )

    latest_timestamps = [
        item
        for item in (
            journal_latest,
            codex_latest,
            None if ctx_summary.get("status") == "inactive" else ctx_latest,
            ontology_latest,
        )
        if item is not None
    ]
    freshness_spread_hours = None
    if len(latest_timestamps) >= 2:
        spread_seconds = (max(latest_timestamps) - min(latest_timestamps)).total_seconds()
        freshness_spread_hours = round(spread_seconds / 3600, 2)

    reasons: list[str] = []
    contradictions: list[str] = []
    warnings: list[str] = []
    for name, entry in sources.items():
        status = str(entry.get("status") or "")
        if status == "missing":
            reasons.append(f"{name} evidence missing")
            warnings.append(f"{name} evidence missing")
        elif status == "stale":
            reasons.append(f"{name} evidence stale ({entry.get('age_hours')}h)")
            warnings.append(f"{name} evidence stale")
        elif status == "degraded":
            reasons.append(f"{name} evidence degraded")
            warnings.append(f"{name} evidence degraded")

    statuses = {str(entry.get("status") or "") for entry in sources.values()}
    if "fresh" in statuses and statuses.intersection({"stale", "missing", "degraded"}):
        contradictions.append("evidence freshness mismatch across sources")
    if stale_active_codex:
        contradictions.append("stale active Codex runs detected")
        warnings.append("stale active Codex runs detected")
        reasons.append(f"{len(stale_active_codex)} active Codex run(s) exceed {active_stale_hours}h")
    if stale_active_ctx:
        contradictions.append("stale active ctx bindings detected")
        warnings.append("stale active ctx bindings detected")
        reasons.append(f"{len(stale_active_ctx)} active ctx binding(s) exceed {active_stale_hours}h")
    if planning_contradictions:
        contradictions.append("planning contradictions detected")
        warnings.append("planning contradictions detected")
        reasons.append(f"{len(planning_contradictions)} planning contradiction(s) detected")

    ontology_alerts = [
        str(item).strip()
        for item in ontology_summary.get("alerts", [])
        if str(item).strip()
    ]
    for item in ontology_summary.get("reasons", []):
        text = str(item).strip()
        if text and text not in reasons:
            reasons.append(text)
    if str(ontology_summary.get("status") or "") in {"missing", "stale", "degraded"}:
        message = "ontology intelligence artifacts are stale, missing, or degraded"
        if message not in contradictions:
            contradictions.append(message)

    degraded = bool(reasons or contradictions)
    gate_status = "degraded" if degraded else "healthy"

    provenance_items = [
        _build_provenance_item("journal", journal_path, sources["journal_entries"]),
        _build_provenance_item("codex", codex_runs_path, sources["codex_runs"]),
        _build_provenance_item("ctx", ctx_bindings_path, sources["ctx_bindings"]),
        _build_provenance_item("ontology", ontology_root, sources["ontology_intelligence"]),
    ]
    if stale_active_codex:
        provenance_items[1]["notes"] = f"{len(stale_active_codex)} active run(s) exceed {active_stale_hours}h"
    if stale_active_ctx:
        provenance_items[2]["notes"] = f"{len(stale_active_ctx)} active session(s) exceed {active_stale_hours}h"
    elif ctx_summary.get("status") == "inactive":
        provenance_items[2]["notes"] = str(ctx_summary.get("detail") or "ctx inactive")
    if ontology_alerts:
        provenance_items[3]["notes"] = " | ".join(ontology_alerts)

    return {
        "status": gate_status,
        "freshness_hours": freshness_hours,
        "active_stale_hours": active_stale_hours,
        "sources": sources,
        "freshness_spread_hours": freshness_spread_hours,
        "stale_active_codex": stale_active_codex,
        "stale_active_ctx": stale_active_ctx,
        "planning_contradictions": planning_contradictions,
        "warnings": warnings,
        "ontology": ontology_summary,
        "ontology_alerts": ontology_alerts,
        "ctx_remediation": ctx_remediation,
        "contradictions": contradictions,
        "reasons": reasons,
        "suppression": {
            "suppress_non_maintenance": degraded,
            "message": (
                "Reliability floor degraded: non-maintenance work suppressed."
                if degraded
                else "Reliability floor healthy: normal lane selection permitted."
            ),
        },
        "provenance": {
            "contract_version": PROVENANCE_CONTRACT_VERSION,
            "items": provenance_items,
            "summary_markdown": format_evidence_provenance(provenance_items),
        },
    }


def _check_status(score: float) -> str:
    if score >= 0.85:
        return "pass"
    if score >= 0.6:
        return "warn"
    return "fail"


def _build_benchmark_item(
    benchmark_id: str,
    label: str,
    *,
    score: float,
    weight: int,
    detail: str,
    critical: bool,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": benchmark_id,
        "label": label,
        "score": round(max(0.0, min(1.0, score)), 4),
        "weight": weight,
        "status": _check_status(score),
        "detail": detail,
        "critical": critical,
        "metrics": metrics,
    }


def _load_benchmark_history(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if isinstance(payload, dict):
        history = payload
    elif isinstance(payload, list):
        history = {"runs": payload}
    else:
        history = {"runs": []}
    if not isinstance(history.get("runs"), list):
        history["runs"] = []
    return history


def _save_benchmark_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, payload)


def _normalize_evidence_key(key: Any) -> str:
    text = str(key or "").strip()
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _value_has_content(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_value_has_content(item) for item in value.values())
    if isinstance(value, list):
        return any(_value_has_content(item) for item in value)
    return True


def _collect_record_text(value: Any, key: str = "") -> list[str]:
    normalized_key = _normalize_evidence_key(key)
    if normalized_key in _TEXT_EVIDENCE_EXCLUDED_KEYS:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        parts: list[str] = []
        for child_key, child_value in value.items():
            parts.extend(_collect_record_text(child_value, str(child_key)))
        return parts
    if isinstance(value, list):
        parts = []
        for child_value in value:
            parts.extend(_collect_record_text(child_value, key))
        return parts
    return []


def _record_has_claim_field(value: Any, key: str = "") -> bool:
    normalized_key = _normalize_evidence_key(key)
    if normalized_key in _CLAIM_TEXT_KEYS and _value_has_content(value):
        return True
    if normalized_key in _CLAIM_CONTAINER_KEYS and _value_has_content(value):
        return True
    if normalized_key in _TEXT_EVIDENCE_EXCLUDED_KEYS:
        return False
    if isinstance(value, dict):
        return any(_record_has_claim_field(child_value, str(child_key)) for child_key, child_value in value.items())
    if isinstance(value, list):
        return any(_record_has_claim_field(child_value, key) for child_value in value)
    return False


def _structured_durable_signals(value: Any, key: str = "") -> set[str]:
    normalized_key = _normalize_evidence_key(key)
    signals: set[str] = set()
    if normalized_key in _DURABLE_EVIDENCE_KEYS and _value_has_content(value):
        signals.add(normalized_key)
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            signals.update(_structured_durable_signals(child_value, str(child_key)))
    elif isinstance(value, list):
        for child_value in value:
            signals.update(_structured_durable_signals(child_value, key))
    return signals


def _text_durable_signals(text: str) -> set[str]:
    return {label for label, pattern in _DURABLE_TEXT_PATTERNS if pattern.search(text)}


def _status_only_markers(text: str) -> set[str]:
    return {pattern.pattern for pattern in _STATUS_ONLY_PATTERNS if pattern.search(text)}


def _record_claimed_timestamp(record: dict[str, Any]) -> Optional[datetime]:
    return _record_timestamp(
        record,
        "completed_at",
        "updated_at",
        "updatedAt",
        "occurredAt",
        "occurred_at",
        "created_at",
        "createdAt",
        "started_at",
        "process_started_at",
        "timestamp",
        "date",
    )


def _record_claims_work(record: dict[str, Any], text: str) -> bool:
    if _record_has_claim_field(record):
        return True
    if record.get("active") is True and _status_only_markers(text):
        return True
    status = str(record.get("status") or "").strip().lower()
    if status in {"active", "in_progress", "running", "queued"} and _status_only_markers(text):
        return True
    return False


def _iter_recent_claimed_work_items(
    *,
    journal_payload: Any,
    codex_payload: Any,
    ctx_payload: Any,
    now: datetime,
    freshness_hours: int,
) -> Iterable[dict[str, Any]]:
    source_records = (
        ("journal_entries", _iter_records(journal_payload, "entries")),
        ("codex_runs", _iter_codex_records(codex_payload)),
        ("ctx_bindings", _iter_ctx_records(ctx_payload)),
    )
    for source, records in source_records:
        for record in records:
            timestamp = _record_claimed_timestamp(record)
            if timestamp is not None:
                age_hours = max(0.0, (now - timestamp).total_seconds() / 3600)
                if age_hours > freshness_hours:
                    continue
            text = "\n".join(_collect_record_text(record))
            if not _record_claims_work(record, text):
                continue
            yield {
                "source": source,
                "id": (
                    record.get("id")
                    or record.get("run_id")
                    or record.get("session_id")
                    or record.get("external_key")
                ),
                "timestamp": timestamp.isoformat() if timestamp is not None else None,
                "record": record,
                "text": text,
            }


def _assess_make_work_item(item: dict[str, Any]) -> dict[str, Any]:
    record = item.get("record") or {}
    text = str(item.get("text") or "")
    durable_signals = _structured_durable_signals(record)
    durable_signals.update(_text_durable_signals(text))
    status_markers = _status_only_markers(text)
    durable = bool(durable_signals)
    issue = None
    if not durable and status_markers:
        issue = "status_language_without_durable_evidence"
    elif not durable:
        issue = "claimed_work_without_durable_evidence"

    return {
        "source": item.get("source"),
        "id": item.get("id"),
        "timestamp": item.get("timestamp"),
        "durable": durable,
        "signals": sorted(durable_signals),
        "status_language": bool(status_markers),
        "issue": issue,
    }


def _assess_operator_value_item(item: dict[str, Any]) -> dict[str, Any]:
    make_work = _assess_make_work_item(item)
    durable_signals = set(make_work.get("signals") or [])
    decision_support_signals = durable_signals.intersection(_OPERATOR_DECISION_SUPPORT_SIGNALS)
    verified_change_signals = durable_signals.intersection(_VERIFIED_SYSTEM_CHANGE_SIGNALS)

    item_score = 0.0
    issue = make_work.get("issue")
    if make_work["durable"]:
        if decision_support_signals and verified_change_signals:
            item_score = 1.0
        elif decision_support_signals:
            item_score = 0.65
            issue = "decision_support_without_verified_system_change"
        elif verified_change_signals:
            item_score = 0.45
            issue = "verified_change_without_operator_decision_support"
        else:
            item_score = 0.25
            issue = "durable_evidence_without_operator_value_signal"

    return {
        "source": make_work.get("source"),
        "id": make_work.get("id"),
        "timestamp": make_work.get("timestamp"),
        "score": item_score,
        "durable": make_work["durable"],
        "signals": sorted(durable_signals),
        "operator_decision_support": bool(decision_support_signals),
        "operator_decision_support_signals": sorted(decision_support_signals),
        "verified_system_change": bool(verified_change_signals),
        "verified_system_change_signals": sorted(verified_change_signals),
        "aligned": bool(decision_support_signals and verified_change_signals),
        "issue": issue,
    }


def _evaluate_anti_make_work_check(
    *,
    journal_path: Path,
    codex_runs_path: Path,
    ctx_bindings_path: Path,
    now: datetime,
    freshness_hours: int,
) -> dict[str, Any]:
    journal_payload = _load_json(journal_path)
    codex_payload = _load_json(codex_runs_path)
    ctx_payload = _load_json(ctx_bindings_path)
    assessments = [
        _assess_make_work_item(item)
        for item in _iter_recent_claimed_work_items(
            journal_payload=journal_payload,
            codex_payload=codex_payload,
            ctx_payload=ctx_payload,
            now=now,
            freshness_hours=freshness_hours,
        )
    ]
    assessed_count = len(assessments)
    durable_count = sum(1 for item in assessments if item["durable"])
    shallow_items = [item for item in assessments if not item["durable"]]
    status_only_count = sum(1 for item in shallow_items if item["status_language"])

    if assessed_count == 0:
        score = 1.0
        detail = "No claimed work items required anti-make-work evidence."
    elif not shallow_items:
        score = 1.0
        detail = "Claimed work includes durable evidence."
    else:
        score = durable_count / assessed_count
        if status_only_count:
            score = min(score, 0.55 if durable_count else 0.0)
        else:
            score = min(score, 0.4)
        examples = [
            f"{item.get('source')}:{item.get('id') or 'unknown'}"
            for item in shallow_items[:3]
        ]
        detail = (
            "Claimed work lacks durable state-change evidence: "
            + ", ".join(examples)
        )

    return _build_benchmark_item(
        "anti_make_work_check",
        "Anti make-work check",
        score=score,
        weight=25,
        detail=detail,
        critical=True,
        metrics={
            "assessed_work_item_count": assessed_count,
            "durable_evidence_count": durable_count,
            "shallow_work_item_count": len(shallow_items),
            "status_language_only_count": status_only_count,
            "durable_examples": [item for item in assessments if item["durable"]][:5],
            "shallow_examples": shallow_items[:5],
        },
    )


def _evaluate_operator_value_alignment_check(
    *,
    journal_path: Path,
    codex_runs_path: Path,
    ctx_bindings_path: Path,
    now: datetime,
    freshness_hours: int,
) -> dict[str, Any]:
    journal_payload = _load_json(journal_path)
    codex_payload = _load_json(codex_runs_path)
    ctx_payload = _load_json(ctx_bindings_path)
    assessments = [
        _assess_operator_value_item(item)
        for item in _iter_recent_claimed_work_items(
            journal_payload=journal_payload,
            codex_payload=codex_payload,
            ctx_payload=ctx_payload,
            now=now,
            freshness_hours=freshness_hours,
        )
    ]

    assessed_count = len(assessments)
    durable_count = sum(1 for item in assessments if item["durable"])
    decision_support_count = sum(1 for item in assessments if item["operator_decision_support"])
    verified_change_count = sum(1 for item in assessments if item["verified_system_change"])
    aligned_count = sum(1 for item in assessments if item["aligned"])
    issue_items = [item for item in assessments if item.get("issue")]

    if assessed_count == 0:
        score = 1.0
        detail = "No claimed work items required operator-value assessment."
    else:
        score = sum(float(item["score"]) for item in assessments) / assessed_count
        if verified_change_count and not decision_support_count:
            score = min(score, 0.55)
        if assessed_count >= 3 and not aligned_count:
            score = min(score, 0.55)

        if aligned_count == assessed_count:
            detail = "Claimed work pairs operator decision support with verified system change."
        elif not decision_support_count:
            detail = "Claimed work shows throughput, but lacks operator decision support."
        elif not verified_change_count:
            detail = "Claimed work supports operator decisions, but lacks verified system change."
        else:
            detail = "Operator-value evidence is incomplete across claimed work."

    return _build_benchmark_item(
        "operator_value_alignment",
        "Operator-value alignment",
        score=score,
        weight=30,
        detail=detail,
        critical=True,
        metrics={
            "assessed_work_item_count": assessed_count,
            "durable_evidence_count": durable_count,
            "operator_decision_support_count": decision_support_count,
            "verified_system_change_count": verified_change_count,
            "aligned_work_item_count": aligned_count,
            "operator_decision_support_rate": (
                round(decision_support_count / assessed_count, 4)
                if assessed_count
                else 1.0
            ),
            "verified_system_change_rate": (
                round(verified_change_count / assessed_count, 4)
                if assessed_count
                else 1.0
            ),
            "aligned_work_rate": (
                round(aligned_count / assessed_count, 4)
                if assessed_count
                else 1.0
            ),
            "quantity_guardrail_basis": "average_evidence_quality_not_item_count",
            "issue_examples": issue_items[:5],
            "aligned_examples": [item for item in assessments if item["aligned"]][:5],
        },
    )


def _weighted_project_score(checks: dict[str, dict[str, Any]]) -> float:
    total_weight = sum(max(0, int(check.get("weight") or 0)) for check in checks.values())
    if total_weight <= 0:
        return 0.0
    weighted_score = sum(
        float(check.get("score") or 0.0) * max(0, int(check.get("weight") or 0))
        for check in checks.values()
    )
    return round((weighted_score / total_weight) * 100, 2)


def _coerce_score(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        value = value.get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iter_benchmark_history_entries(history: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ("evaluations", "runs"):
        entries = history.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                yield entry


def _history_check_scores(history: dict[str, Any], check_id: str) -> list[float]:
    scores: list[float] = []
    for entry in _iter_benchmark_history_entries(history):
        checks = entry.get("checks")
        if not isinstance(checks, dict):
            continue
        score = _coerce_score(checks.get(check_id))
        if score is not None:
            scores.append(score)
    return scores


def _latest_history_project_score(history: dict[str, Any]) -> Optional[float]:
    for entry in reversed(list(_iter_benchmark_history_entries(history))):
        score = _coerce_score(entry.get("project_score"))
        if score is None:
            score = _coerce_score(entry.get("score"))
        if score is not None:
            return score
    return None


def _score_direction(current: float, previous: Optional[float], *, threshold: float = 0.01) -> str:
    if previous is None:
        return "stable"
    delta = current - previous
    if delta > threshold:
        return "positive"
    if delta < -threshold:
        return "negative"
    return "stable"


def _rounded_float(value: Any, digits: int = 4) -> Optional[float]:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _population_stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _coerce_check_status(value: Any) -> str:
    score = _coerce_score(value)
    if isinstance(value, dict):
        status = str(value.get("status") or "").strip().lower()
        if status:
            return status
    if score is not None:
        return _check_status(score)
    return "unknown"


def _normalize_benchmark_check(value: Any) -> Optional[dict[str, Any]]:
    score = _coerce_score(value)
    if score is None:
        return None
    return {"score": round(score, 4), "status": _coerce_check_status(value)}


def _benchmark_indicator_series(
    history: dict[str, Any],
    current_checks: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for entry in _iter_benchmark_history_entries(history):
        checks = entry.get("checks")
        if not isinstance(checks, dict):
            continue
        normalized_checks = {
            check_id: normalized
            for check_id in _LEADING_INDICATOR_CHECK_IDS
            if (normalized := _normalize_benchmark_check(checks.get(check_id))) is not None
        }
        if not normalized_checks:
            continue
        series.append(
            {
                "source": "history",
                "generated_at": entry.get("generated_at") or entry.get("evaluated_at"),
                "checks": normalized_checks,
            }
        )

    normalized_current = {
        check_id: normalized
        for check_id in _LEADING_INDICATOR_CHECK_IDS
        if (normalized := _normalize_benchmark_check(current_checks.get(check_id))) is not None
    }
    if normalized_current:
        series.append({"source": "current", "generated_at": None, "checks": normalized_current})
    return series


def _check_score_series(series: list[dict[str, Any]], check_id: str) -> list[float]:
    scores: list[float] = []
    for entry in series:
        check = (entry.get("checks") or {}).get(check_id)
        score = _coerce_score(check)
        if score is not None:
            scores.append(score)
    return scores


def _check_status_series(series: list[dict[str, Any]], check_id: str) -> list[str]:
    statuses: list[str] = []
    for entry in series:
        check = (entry.get("checks") or {}).get(check_id)
        if check is None:
            continue
        status = _coerce_check_status(check)
        if status != "unknown":
            statuses.append(status)
    return statuses


def _harbinger_payload(
    *,
    triggered: bool,
    evidence: dict[str, Any],
    mitigation: str,
    next_action: str,
) -> dict[str, Any]:
    return {
        "triggered": triggered,
        "severity": "fail" if triggered else "none",
        "evidence": evidence,
        "mitigation": mitigation,
        "next_action": next_action,
    }


def _detect_critical_slowing_down(scores: list[float]) -> dict[str, Any]:
    recent_scores = [round(score, 4) for score in scores[-5:]]
    prior_peak = max(scores[:-1]) if len(scores) > 1 else (scores[-1] if scores else None)
    current_score = scores[-1] if scores else None
    recovery_gap = (prior_peak - current_score) if prior_peak is not None and current_score is not None else 0.0
    deltas = [scores[idx] - scores[idx - 1] for idx in range(1, len(scores))]
    recent_deltas = deltas[-3:]
    flat_or_negative_count = sum(1 for delta in recent_deltas if delta <= 0.01)
    average_recent_delta = sum(recent_deltas) / len(recent_deltas) if recent_deltas else None
    triggered = (
        len(scores) >= 5
        and recovery_gap >= 0.05
        and len(recent_deltas) >= 3
        and flat_or_negative_count == len(recent_deltas)
    )

    return _harbinger_payload(
        triggered=triggered,
        evidence={
            "sample_count": len(scores),
            "recent_scores": recent_scores,
            "prior_peak": _rounded_float(prior_peak),
            "current_score": _rounded_float(current_score),
            "recovery_gap": round(recovery_gap, 4),
            "recent_deltas": [round(delta, 4) for delta in recent_deltas],
            "average_recent_delta": _rounded_float(average_recent_delta),
            "flat_or_negative_delta_count": flat_or_negative_count,
        },
        mitigation="Stop expanding self-improvement scope until the lagging operator-value signal recovers.",
        next_action="Select the next maintenance item only if it restores operator decision support plus verified change evidence.",
    )


def _detect_variance_explosion(scores: list[float]) -> dict[str, Any]:
    recent_scores = scores[-4:]
    baseline_scores = scores[:-4]
    baseline_stddev = _population_stddev(baseline_scores)
    recent_stddev = _population_stddev(recent_scores)
    recent_range = max(recent_scores) - min(recent_scores) if recent_scores else 0.0
    triggered = (
        len(scores) >= 6
        and len(recent_scores) >= 4
        and recent_range >= 0.2
        and recent_stddev >= max(0.08, baseline_stddev * 3)
    )

    return _harbinger_payload(
        triggered=triggered,
        evidence={
            "sample_count": len(scores),
            "baseline_scores": [round(score, 4) for score in baseline_scores[-4:]],
            "recent_scores": [round(score, 4) for score in recent_scores],
            "baseline_stddev": round(baseline_stddev, 4),
            "recent_stddev": round(recent_stddev, 4),
            "recent_range": round(recent_range, 4),
        },
        mitigation="Treat the benchmark as unstable and stop optimizing for throughput until score variance narrows.",
        next_action="Run one focused stabilization pass and require the next run to include low-variance evidence before broadening lane selection.",
    )


def _detect_flickering(series: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = _check_status_series(series, "operator_value_alignment")[-6:]
    transition_count = sum(
        1
        for idx in range(1, len(statuses))
        if statuses[idx] != statuses[idx - 1]
    )
    pass_boundary_crossings = sum(
        1
        for idx in range(1, len(statuses))
        if (statuses[idx] == "pass") != (statuses[idx - 1] == "pass")
    )
    triggered = len(statuses) >= 5 and transition_count >= 3 and pass_boundary_crossings >= 2

    return _harbinger_payload(
        triggered=triggered,
        evidence={
            "sample_count": len(statuses),
            "recent_statuses": statuses,
            "transition_count": transition_count,
            "pass_boundary_crossings": pass_boundary_crossings,
        },
        mitigation="Do not treat a single passing run as stable while the signal flickers across pass/fail boundaries.",
        next_action="Require two consecutive stable passing runs before raw issue-selection volume is allowed again.",
    )


def _detect_correlation_explosion(series: list[dict[str, Any]]) -> dict[str, Any]:
    current = series[-1] if series else {}
    current_checks = current.get("checks") if current.get("source") == "current" else {}
    check_deltas: dict[str, float] = {}
    for check_id in _LEADING_INDICATOR_CHECK_IDS:
        current_score = _coerce_score((current_checks or {}).get(check_id))
        if current_score is None:
            continue
        previous_score = None
        for entry in reversed(series[:-1]):
            previous_score = _coerce_score((entry.get("checks") or {}).get(check_id))
            if previous_score is not None:
                break
        if previous_score is not None:
            check_deltas[check_id] = round(current_score - previous_score, 4)

    dropped_checks = {
        check_id: delta
        for check_id, delta in check_deltas.items()
        if delta <= -0.05
    }
    triggered = len(dropped_checks) >= 3

    return _harbinger_payload(
        triggered=triggered,
        evidence={
            "check_deltas": check_deltas,
            "dropped_check_count": len(dropped_checks),
            "dropped_checks": sorted(dropped_checks),
            "correlated_drop_threshold": -0.05,
        },
        mitigation="Treat simultaneous benchmark-check degradation as coupled risk, not isolated check noise.",
        next_action="Pick a maintenance item that improves the shared evidence path before selecting feature or volume work.",
    )


def _build_leading_indicator_scorecard(
    history: dict[str, Any],
    current_checks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    series = _benchmark_indicator_series(history, current_checks)
    operator_scores = _check_score_series(series, "operator_value_alignment")
    scorecard = {
        "critical_slowing_down": _detect_critical_slowing_down(operator_scores),
        "variance_explosion": _detect_variance_explosion(operator_scores),
        "flickering": _detect_flickering(series),
        "correlation_explosion": _detect_correlation_explosion(series),
    }
    triggered_harbingers = [
        harbinger
        for harbinger in _LEADING_INDICATOR_HARBINGERS
        if scorecard[harbinger]["triggered"]
    ]
    return {
        "series_sample_count": len(series),
        "operator_value_score_series": [round(score, 4) for score in operator_scores[-8:]],
        "scorecard": scorecard,
        "triggered_harbingers": triggered_harbingers,
        "recommended_mitigations": [
            {
                "harbinger": harbinger,
                "mitigation": scorecard[harbinger]["mitigation"],
                "next_action": scorecard[harbinger]["next_action"],
                "evidence": scorecard[harbinger]["evidence"],
            }
            for harbinger in triggered_harbingers
        ],
    }


def _evaluate_leading_indicator_drift_check(
    operator_value_check: dict[str, Any],
    history: dict[str, Any],
    current_checks: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    current_score = float(operator_value_check.get("score") or 0.0)
    prior_scores = _history_check_scores(history, "operator_value_alignment")
    previous_score = prior_scores[-1] if prior_scores else None
    delta = round(current_score - previous_score, 4) if previous_score is not None else None
    regressing = delta is not None and delta < -0.01
    indicator_payload = _build_leading_indicator_scorecard(
        history,
        current_checks or {"operator_value_alignment": operator_value_check},
    )
    triggered_harbingers = indicator_payload["triggered_harbingers"]

    if triggered_harbingers:
        score = max(0.2, 0.6 - (0.15 * len(triggered_harbingers)))
        if regressing:
            score = min(score, 0.5)
        detail = (
            "Leading indicators triggered: "
            + ", ".join(triggered_harbingers)
            + "; run mitigation before expanding self-improvement scope."
        )
    elif previous_score is None:
        score = 1.0
        detail = "No prior operator-value score; drift not assessed."
    elif regressing:
        score = 0.5
        detail = "Operator-value alignment is regressing; keep quantity guardrail active."
    else:
        score = 1.0
        detail = "Operator-value leading indicator is stable or improving."

    metrics = dict(operator_value_check.get("metrics") or {})
    metrics.update(
        {
            "previous_operator_value_score": (
                round(previous_score, 4) if previous_score is not None else None
            ),
            "current_operator_value_score": round(current_score, 4),
            "operator_value_delta": delta,
            "prior_operator_value_sample_count": len(prior_scores),
            "leading_indicator_contract_version": "harbinger_scorecard.v1",
            "series_sample_count": indicator_payload["series_sample_count"],
            "operator_value_score_series": indicator_payload["operator_value_score_series"],
            "triggered_harbingers": triggered_harbingers,
            "harbinger_scorecard": indicator_payload["scorecard"],
            "recommended_mitigations": indicator_payload["recommended_mitigations"],
        }
    )
    return _build_benchmark_item(
        "leading_indicator_drift",
        "Leading-indicator drift",
        score=score,
        weight=20,
        detail=detail,
        critical=True,
        metrics=metrics,
    )


def _build_issue_selection_summary(
    checks: dict[str, dict[str, Any]],
    gate: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    guardrail_checks = {
        name: check
        for name, check in checks.items()
        if name in {
            "reliability_gate",
            "anti_make_work_check",
            "operator_value_alignment",
            "leading_indicator_drift",
        }
        and check.get("status") != "pass"
    }
    quantity_guardrail_active = bool(guardrail_checks)
    reliability_blocked = "reliability_gate" in guardrail_checks
    gate = gate or {}
    ctx_remediation = gate.get("ctx_remediation") or {}
    ontology_repair = (gate.get("ontology") or {}).get("external_repair") or {}
    remediation_actions = [
        str(item.get("action") or "").strip()
        for item in (ctx_remediation, ontology_repair)
        if item.get("required") and str(item.get("action") or "").strip()
    ]
    if reliability_blocked:
        recommended_focus = "self-improvement evidence freshness repair"
        detail = (
            "Repair self-improvement evidence freshness before selecting throughput or operator-value work: "
            + "; ".join(remediation_actions or ["inspect reliability gate provenance"])
        )
    elif quantity_guardrail_active:
        recommended_focus = "operator decision support plus verified system change"
        detail = (
            "Do not select issues because they increase task count; "
            "prefer work with operator decision support and verified change evidence."
        )
    else:
        recommended_focus = "normal lane selection"
        detail = "Benchmark guardrails permit normal lane selection."

    return {
        "quantity_guardrail_active": quantity_guardrail_active,
        "suppress_raw_throughput_selection": quantity_guardrail_active,
        "blocked_checks": sorted(guardrail_checks),
        "recommended_focus": recommended_focus,
        "detail": detail,
        "remediation_actions": remediation_actions,
    }


def _build_operator_summary(
    checks: dict[str, dict[str, Any]],
    issue_selection: dict[str, Any],
) -> dict[str, str]:
    operator_value = checks["operator_value_alignment"]
    drift = checks["leading_indicator_drift"]
    return {
        "operator_value_alignment": str(operator_value.get("detail") or ""),
        "leading_indicator_drift": str(drift.get("detail") or ""),
        "issue_selection": str(issue_selection.get("detail") or ""),
    }


def evaluate_self_improvement_benchmark(
    *,
    journal_path: Path = DEFAULT_JOURNAL_PATH,
    codex_runs_path: Path = DEFAULT_CODEX_RUNS_PATH,
    ctx_bindings_path: Path = DEFAULT_CTX_BINDINGS_PATH,
    ontology_root: Path = DEFAULT_ONTOLOGY_ROOT,
    history_path: Path = DEFAULT_BENCHMARK_HISTORY_PATH,
    now: Optional[datetime] = None,
    freshness_hours: int = DEFAULT_FRESHNESS_HOURS,
    active_stale_hours: int = DEFAULT_ACTIVE_STALE_HOURS,
    persist: bool = True,
) -> dict[str, Any]:
    current = now or datetime.now(tz=timezone.utc)
    gate = evaluate_self_improvement_evidence(
        journal_path=journal_path,
        codex_runs_path=codex_runs_path,
        ctx_bindings_path=ctx_bindings_path,
        ontology_root=ontology_root,
        now=current,
        freshness_hours=freshness_hours,
        active_stale_hours=active_stale_hours,
    )
    source_statuses = [
        str((entry or {}).get("status") or "")
        for entry in (gate.get("sources") or {}).values()
        if isinstance(entry, dict)
    ]
    stale_source_count = sum(status in {"stale", "missing", "degraded"} for status in source_statuses)

    reliability_score = 1.0
    reliability_score -= 0.2 * stale_source_count
    reliability_score -= 0.15 if gate.get("stale_active_codex") else 0.0
    reliability_score -= 0.15 if gate.get("stale_active_ctx") else 0.0
    reliability_score -= 0.15 if gate.get("planning_contradictions") else 0.0
    reliability_score -= (
        0.15
        if str((gate.get("ontology") or {}).get("status") or "") in {"stale", "missing", "degraded"}
        else 0.0
    )
    if gate.get("status") == "degraded":
        reliability_score = min(reliability_score, 0.45)

    detail = (
        "Reliability floor is healthy."
        if gate.get("status") == "healthy"
        else "; ".join(
            gate.get("reasons")
            or gate.get("warnings")
            or gate.get("contradictions")
            or ["Reliability floor degraded."]
        )
    )
    reliability_gate = _build_benchmark_item(
        "reliability_gate",
        "Reliability gate",
        score=reliability_score,
        weight=25,
        detail=detail,
        critical=True,
        metrics={
            "gate_status": gate.get("status"),
            "stale_source_count": stale_source_count,
            "warning_count": len(gate.get("warnings") or []),
            "contradiction_count": len(gate.get("contradictions") or []),
            "freshness_spread_hours": gate.get("freshness_spread_hours"),
            "ctx_remediation_required": bool(
                (gate.get("ctx_remediation") or {}).get("required")
            ),
        },
    )
    anti_make_work_check = _evaluate_anti_make_work_check(
        journal_path=journal_path,
        codex_runs_path=codex_runs_path,
        ctx_bindings_path=ctx_bindings_path,
        now=current,
        freshness_hours=freshness_hours,
    )
    operator_value_alignment = _evaluate_operator_value_alignment_check(
        journal_path=journal_path,
        codex_runs_path=codex_runs_path,
        ctx_bindings_path=ctx_bindings_path,
        now=current,
        freshness_hours=freshness_hours,
    )
    history = _load_benchmark_history(history_path)
    leading_indicator_drift = _evaluate_leading_indicator_drift_check(
        operator_value_alignment,
        history,
        {
            "reliability_gate": reliability_gate,
            "anti_make_work_check": anti_make_work_check,
            "operator_value_alignment": operator_value_alignment,
        },
    )
    checks = {
        "reliability_gate": reliability_gate,
        "anti_make_work_check": anti_make_work_check,
        "operator_value_alignment": operator_value_alignment,
        "leading_indicator_drift": leading_indicator_drift,
    }
    project_score = _weighted_project_score(checks)
    critical_failures = [
        name
        for name, check in checks.items()
        if check.get("critical") and check.get("status") == "fail"
    ]
    issue_selection = _build_issue_selection_summary(checks, gate)
    operator_summary = _build_operator_summary(checks, issue_selection)
    previous_project_score = _latest_history_project_score(history)
    direction = _score_direction(project_score, previous_project_score, threshold=0.1)
    trend = "single_run" if previous_project_score is None else direction
    if leading_indicator_drift.get("status") == "fail":
        direction = "negative"
        trend = "regressing"

    benchmark = {
        "contract_version": BENCHMARK_CONTRACT_VERSION,
        "generated_at": current.isoformat(),
        "project_score": project_score,
        "score": project_score,
        "direction": direction,
        "trend": trend,
        "gate": gate,
        "checks": checks,
        "critical_failures": critical_failures,
        "operator_value_score": operator_value_alignment.get("score"),
        "operator_value_checks": {
            "operator_decision_support_rate": (
                operator_value_alignment.get("metrics", {}).get("operator_decision_support_rate")
            ),
            "verified_system_change_rate": (
                operator_value_alignment.get("metrics", {}).get("verified_system_change_rate")
            ),
            "aligned_work_rate": (
                operator_value_alignment.get("metrics", {}).get("aligned_work_rate")
            ),
            "operator_value_score": operator_value_alignment.get("score"),
        },
        "anti_make_work": {
            "status": anti_make_work_check.get("status"),
            "score": anti_make_work_check.get("score"),
            "flags": [
                item.get("issue")
                for item in anti_make_work_check.get("metrics", {}).get("shallow_examples", [])
                if item.get("issue")
            ],
        },
        "issue_selection": issue_selection,
        "summary": operator_summary,
        "history_path": str(history_path),
    }

    if persist:
        history["runs"].append(
            {
                "generated_at": benchmark["generated_at"],
                "project_score": benchmark["project_score"],
                "direction": benchmark["direction"],
                "critical_failures": benchmark["critical_failures"],
                "operator_value_score": benchmark["operator_value_score"],
                "checks": {
                    name: {
                        "score": check.get("score"),
                        "status": check.get("status"),
                    }
                    for name, check in checks.items()
                },
            }
        )
        history["runs"] = history["runs"][-_BENCHMARK_HISTORY_LIMIT:]
        _save_benchmark_history(history_path, history)

    return benchmark


def self_improvement_evidence_gate(
    journal_path: Optional[str] = None,
    codex_runs_path: Optional[str] = None,
    ctx_bindings_path: Optional[str] = None,
    ontology_root: Optional[str] = None,
    now: Optional[str] = None,
    freshness_hours: Optional[int] = None,
    active_stale_hours: Optional[int] = None,
    task_id: Optional[str] = None,
) -> str:
    gate = evaluate_self_improvement_evidence(
        journal_path=Path(journal_path).expanduser() if journal_path else DEFAULT_JOURNAL_PATH,
        codex_runs_path=Path(codex_runs_path).expanduser() if codex_runs_path else DEFAULT_CODEX_RUNS_PATH,
        ctx_bindings_path=Path(ctx_bindings_path).expanduser() if ctx_bindings_path else DEFAULT_CTX_BINDINGS_PATH,
        ontology_root=Path(ontology_root).expanduser() if ontology_root else DEFAULT_ONTOLOGY_ROOT,
        now=_parse_time(now) if now else None,
        freshness_hours=int(freshness_hours) if freshness_hours else DEFAULT_FRESHNESS_HOURS,
        active_stale_hours=int(active_stale_hours) if active_stale_hours else DEFAULT_ACTIVE_STALE_HOURS,
    )
    return json.dumps({"success": True, "gate": gate, "task_id": task_id})


def self_improvement_benchmark(
    journal_path: Optional[str] = None,
    codex_runs_path: Optional[str] = None,
    ctx_bindings_path: Optional[str] = None,
    ontology_root: Optional[str] = None,
    history_path: Optional[str] = None,
    now: Optional[str] = None,
    freshness_hours: Optional[int] = None,
    active_stale_hours: Optional[int] = None,
    persist: Optional[bool] = None,
    task_id: Optional[str] = None,
) -> str:
    benchmark = evaluate_self_improvement_benchmark(
        journal_path=Path(journal_path).expanduser() if journal_path else DEFAULT_JOURNAL_PATH,
        codex_runs_path=Path(codex_runs_path).expanduser() if codex_runs_path else DEFAULT_CODEX_RUNS_PATH,
        ctx_bindings_path=Path(ctx_bindings_path).expanduser() if ctx_bindings_path else DEFAULT_CTX_BINDINGS_PATH,
        ontology_root=Path(ontology_root).expanduser() if ontology_root else DEFAULT_ONTOLOGY_ROOT,
        history_path=Path(history_path).expanduser() if history_path else DEFAULT_BENCHMARK_HISTORY_PATH,
        now=_parse_time(now) if now else None,
        freshness_hours=int(freshness_hours) if freshness_hours else DEFAULT_FRESHNESS_HOURS,
        active_stale_hours=int(active_stale_hours) if active_stale_hours else DEFAULT_ACTIVE_STALE_HOURS,
        persist=True if persist is None else bool(persist),
    )
    return json.dumps({"success": True, "benchmark": benchmark, "task_id": task_id})


registry.register(
    name="self_improvement_evidence_gate",
    toolset="self_improvement",
    schema=SELF_IMPROVEMENT_EVIDENCE_SCHEMA,
    handler=lambda args, **kw: self_improvement_evidence_gate(
        journal_path=args.get("journal_path"),
        codex_runs_path=args.get("codex_runs_path"),
        ctx_bindings_path=args.get("ctx_bindings_path"),
        ontology_root=args.get("ontology_root"),
        now=args.get("now"),
        freshness_hours=args.get("freshness_hours"),
        active_stale_hours=args.get("active_stale_hours"),
        task_id=kw.get("task_id"),
    ),
)

registry.register(
    name="self_improvement_benchmark",
    toolset="self_improvement",
    schema=SELF_IMPROVEMENT_BENCHMARK_SCHEMA,
    handler=lambda args, **kw: self_improvement_benchmark(
        journal_path=args.get("journal_path"),
        codex_runs_path=args.get("codex_runs_path"),
        ctx_bindings_path=args.get("ctx_bindings_path"),
        ontology_root=args.get("ontology_root"),
        history_path=args.get("history_path"),
        now=args.get("now"),
        freshness_hours=args.get("freshness_hours"),
        active_stale_hours=args.get("active_stale_hours"),
        persist=args.get("persist"),
        task_id=kw.get("task_id"),
    ),
)
