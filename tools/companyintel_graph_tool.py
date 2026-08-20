"""Deterministic, profile-gated graph runtime for companyintel.

This is intentionally a narrow Hermes tool boundary: the tool is exposed only
when the active profile is ``companyintel``.  SQLite owns the authoritative
state; JSON/JSONL files are projections for validators and operators.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from hermes_constants import get_hermes_home
from tools.registry import registry

SCHEMA_VERSION = "corporate-intelligence-graph/v1"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ALLOWED_NODE_TYPES = {
    "domain", "url", "phone", "email", "address", "place", "brand", "legal_entity",
    "person", "document", "image", "favicon", "marketplace_profile",
    "social_profile", "analytics_id", "merchant_id", "iban", "legal_id",
    "product", "organization_candidate",
}


def _home() -> Path:
    raw = os.environ.get("HERMES_HOME")
    return Path(raw).expanduser() if raw else Path(get_hermes_home())


def check_companyintel_requirements() -> bool:
    profile = os.environ.get("HERMES_PROFILE", "")
    if profile:
        return profile == "companyintel"
    return _home().name == "companyintel"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_id(prefix: str, *parts: str) -> str:
    payload = "|".join(str(part).strip().lower() for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _normalize_url(value: str) -> str:
    value = value.strip()
    if "://" not in value:
        value = "https://" + value
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("target_url must be a public HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("target_url must not contain credentials")
    parsed = parsed._replace(fragment="")
    return parsed.geturl().rstrip("/")


def _domain(value: str) -> str:
    return (urlparse(value).hostname or "").lower().rstrip(".")


def _normalize_value(node_type: str, value: str) -> str:
    value = " ".join(str(value).strip().split())
    if node_type in {"domain", "email"}:
        value = value.lower()
    if node_type == "domain":
        value = value.rstrip(".")
    if node_type == "email" and "@" not in value:
        raise ValueError("email node requires an email address")
    if node_type == "phone":
        value = re.sub(r"[^0-9+]", "", value)
    if not value:
        raise ValueError("node value must not be empty")
    return value


def _run_dir(run_id: str) -> Path:
    return _home() / "companyintel" / "runs" / run_id


def _connect(run_id: str) -> sqlite3.Connection:
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(run_dir / "graph.sqlite3", timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS run_state (
            run_id TEXT PRIMARY KEY,
            target_url TEXT NOT NULL,
            status TEXT NOT NULL,
            round INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            max_tasks INTEGER NOT NULL DEFAULT 50,
            max_searches INTEGER NOT NULL DEFAULT 200,
            max_nodes INTEGER NOT NULL DEFAULT 250,
            max_edges INTEGER NOT NULL DEFAULT 500,
            deadline_at REAL,
            tasks_executed INTEGER NOT NULL DEFAULT 0,
            source_calls INTEGER NOT NULL DEFAULT 0,
            saturation_reason TEXT,
            legal_identity_status TEXT NOT NULL DEFAULT 'UNRESOLVED',
            legal_identity_reason TEXT,
            legal_identity_checked_at TEXT,
            legal_identity_coverage TEXT NOT NULL DEFAULT '{}',
            retry_at REAL,
            retry_reason TEXT,
            retry_count INTEGER NOT NULL DEFAULT 0,
            resume_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            value TEXT NOT NULL,
            normalized_value TEXT NOT NULL,
            status TEXT NOT NULL,
            confidence REAL NOT NULL,
            evidence_ids TEXT NOT NULL DEFAULT '[]',
            discovered_from TEXT NOT NULL DEFAULT '[]',
            first_seen_round INTEGER NOT NULL,
            last_expanded_round INTEGER,
            frontier_state TEXT NOT NULL DEFAULT 'OPEN'
        );
        CREATE TABLE IF NOT EXISTS evidence (
            evidence_id TEXT PRIMARY KEY,
            source_url TEXT NOT NULL,
            source_title TEXT NOT NULL DEFAULT '',
            source_tier TEXT NOT NULL DEFAULT 'C',
            retrieved_at TEXT NOT NULL,
            excerpt TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            retrieval_method TEXT NOT NULL,
            access_status TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS edges (
            edge_id TEXT PRIMARY KEY,
            from_node TEXT NOT NULL,
            to_node TEXT NOT NULL,
            relation TEXT NOT NULL,
            confidence REAL NOT NULL,
            evidence_ids TEXT NOT NULL DEFAULT '[]',
            first_seen_round INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS frontier (
            node_id TEXT PRIMARY KEY,
            priority INTEGER NOT NULL,
            reason TEXT NOT NULL,
            planned_pivots TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'OPEN'
        );
        CREATE TABLE IF NOT EXISTS frontier_tasks (
            task_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            pivot_type TEXT NOT NULL,
            priority INTEGER NOT NULL,
            state TEXT NOT NULL DEFAULT 'OPEN',
            attempt INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 3,
            available_at REAL NOT NULL,
            worker_id TEXT,
            lease_token TEXT,
            lease_expires_at REAL,
            last_error TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(run_id, node_id, pivot_type)
        );
        CREATE INDEX IF NOT EXISTS frontier_tasks_claim_idx
            ON frontier_tasks(state, available_at, priority DESC, created_at);
        CREATE TABLE IF NOT EXISTS search_log (
            search_id TEXT PRIMARY KEY,
            round INTEGER NOT NULL,
            pivot_node_id TEXT NOT NULL,
            pivot_type TEXT NOT NULL,
            query TEXT NOT NULL,
            source TEXT NOT NULL,
            status TEXT NOT NULL,
            new_node_ids TEXT NOT NULL DEFAULT '[]',
            limitation TEXT
        );
        CREATE TABLE IF NOT EXISTS identity_candidates (
            candidate_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            legal_name TEXT NOT NULL DEFAULT '',
            legal_id TEXT NOT NULL DEFAULT '',
            country TEXT NOT NULL DEFAULT '',
            score INTEGER NOT NULL,
            status TEXT NOT NULL,
            matching_dimensions TEXT NOT NULL DEFAULT '[]',
            conflicting_dimensions TEXT NOT NULL DEFAULT '[]',
            missing_dimensions TEXT NOT NULL DEFAULT '[]',
            evidence_ids TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(run_id, candidate_id)
        );
        CREATE INDEX IF NOT EXISTS identity_candidates_run_idx
            ON identity_candidates(run_id, score DESC, updated_at DESC);
        CREATE TABLE IF NOT EXISTS worker_checkpoints (
            checkpoint_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            worker_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            phase TEXT NOT NULL,
            status TEXT NOT NULL,
            cursor_json TEXT NOT NULL DEFAULT '{}',
            graph_digest TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(run_id, task_id, attempt)
        );
        CREATE INDEX IF NOT EXISTS worker_checkpoints_latest_idx
            ON worker_checkpoints(run_id, task_id, updated_at DESC);
        CREATE TABLE IF NOT EXISTS checkpoints (
            checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            reason TEXT NOT NULL,
            graph_digest TEXT NOT NULL
        );
        """
    )
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(run_state)").fetchall()}
    for column, definition in {
        "max_tasks": "INTEGER NOT NULL DEFAULT 50",
        "max_searches": "INTEGER NOT NULL DEFAULT 200",
        "max_nodes": "INTEGER NOT NULL DEFAULT 250",
        "max_edges": "INTEGER NOT NULL DEFAULT 500",
        "deadline_at": "REAL",
        "tasks_executed": "INTEGER NOT NULL DEFAULT 0",
        "source_calls": "INTEGER NOT NULL DEFAULT 0",
        "saturation_reason": "TEXT",
        "legal_identity_status": "TEXT NOT NULL DEFAULT 'UNRESOLVED'",
        "legal_identity_reason": "TEXT",
        "legal_identity_checked_at": "TEXT",
        "legal_identity_coverage": "TEXT NOT NULL DEFAULT '{}'",
        "retry_at": "REAL",
        "retry_reason": "TEXT",
        "retry_count": "INTEGER NOT NULL DEFAULT 0",
        "resume_count": "INTEGER NOT NULL DEFAULT 0",
    }.items():
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE run_state ADD COLUMN {column} {definition}")
    return conn


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _materialize(conn: sqlite3.Connection, run_id: str) -> None:
    run = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    nodes = [dict(row) for row in conn.execute("SELECT * FROM nodes ORDER BY node_id")]
    edges = [dict(row) for row in conn.execute("SELECT * FROM edges ORDER BY edge_id")]
    evidence = [dict(row) for row in conn.execute("SELECT * FROM evidence ORDER BY evidence_id")]
    frontier = [dict(row) for row in conn.execute("SELECT * FROM frontier ORDER BY priority DESC,node_id")]
    frontier_tasks = [dict(row) for row in conn.execute("SELECT * FROM frontier_tasks ORDER BY priority DESC,created_at ASC,task_id")]
    search_log = [dict(row) for row in conn.execute("SELECT * FROM search_log ORDER BY search_id")]
    identity_candidates = [dict(row) for row in conn.execute("SELECT * FROM identity_candidates WHERE run_id=? ORDER BY score DESC,updated_at DESC", (run_id,))]
    worker_checkpoints = [dict(row) for row in conn.execute("SELECT * FROM worker_checkpoints WHERE run_id=? ORDER BY updated_at DESC,checkpoint_id DESC", (run_id,))]
    for row in frontier_tasks:
        row.pop("lease_token", None)
    for row in nodes + edges + evidence + frontier + search_log + identity_candidates:
        for key in ("evidence_ids", "discovered_from", "planned_pivots", "new_node_ids", "matching_dimensions", "conflicting_dimensions", "missing_dimensions"):
            if key in row and isinstance(row[key], str):
                row[key] = json.loads(row[key])
    for row in worker_checkpoints:
        row.pop("graph_digest", None)
        if isinstance(row.get("cursor_json"), str):
            row["cursor"] = json.loads(row.pop("cursor_json"))
    open_frontier = sum(row["state"] == "OPEN" for row in frontier)
    graph = {
        "schema_version": SCHEMA_VERSION,
        "target": {"url": run["target_url"], "domain": _domain(run["target_url"])},
        "research_state": {
            "status": run["status"], "round": run["round"], "frontier_open": open_frontier,
            "stale_rounds": 0,
            "budgets": {"max_rounds": 8, "max_tasks": run["max_tasks"], "max_searches": run["max_searches"], "max_nodes": run["max_nodes"], "max_edges": run["max_edges"], "deadline_at": run["deadline_at"]},
            "usage": {"tasks_executed": run["tasks_executed"], "source_calls": run["source_calls"]},
            "saturation_reason": run["saturation_reason"],
            "retry_at": run["retry_at"], "retry_reason": run["retry_reason"], "retry_count": run["retry_count"], "resume_count": run["resume_count"],
        },
        "identity_resolution": {"status": "AMBIGUOUS", "candidate_ids": [row["candidate_id"] for row in identity_candidates]},
        "identity_candidates": identity_candidates,
        "worker_checkpoints": worker_checkpoints,
        "nodes": nodes, "edges": edges, "evidence": evidence,
        "frontier": frontier, "frontier_tasks": frontier_tasks, "search_log": search_log,
        "pivot_coverage": {},
        "legal_identity_exhaustion": {"status": run["legal_identity_status"], "reason": run["legal_identity_reason"], "checked_at": run["legal_identity_checked_at"], "coverage": json.loads(run["legal_identity_coverage"] or "{}")},
    }
    inventory_path = _run_dir(run_id) / "inventory.json"
    if inventory_path.exists():
        try:
            graph["inventory"] = json.loads(inventory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            graph["inventory"] = {"error": "inventory_projection_unavailable"}
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(run_dir / "graph.json", graph)
    for name, rows in (("nodes.jsonl", nodes), ("edges.jsonl", edges), ("evidence.jsonl", evidence), ("frontier_tasks.jsonl", frontier_tasks), ("search_log.jsonl", search_log), ("identity_candidates.jsonl", identity_candidates), ("worker_checkpoints.jsonl", worker_checkpoints)):
        _atomic_text(run_dir / name, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _checkpoint(conn: sqlite3.Connection, run_id: str, reason: str) -> None:
    graph_path = _run_dir(run_id) / "graph.json"
    graph_json = graph_path.read_bytes() if graph_path.exists() else b""
    digest = hashlib.sha256(graph_json).hexdigest()
    created_at = _now()
    conn.execute("INSERT INTO checkpoints(created_at,reason,graph_digest) VALUES(?,?,?)", (created_at, reason, digest))
    conn.commit()
    _atomic_json(_run_dir(run_id) / "checkpoints" / "latest.json", {
        "run_id": run_id,
        "created_at": created_at,
        "reason": reason,
        "graph_sha256": digest,
    })


def _validate_worker_cursor(cursor) -> tuple[dict, str]:
    if cursor is None:
        cursor = {}
    if not isinstance(cursor, dict):
        raise ValueError("worker checkpoint cursor must be an object")
    encoded = _json(cursor)
    if len(encoded.encode("utf-8")) > 8192:
        raise ValueError("worker checkpoint cursor exceeds 8192 bytes")
    return cursor, encoded


def _upsert_worker_checkpoint(conn: sqlite3.Connection, run_id: str, worker_id: str, task: sqlite3.Row | dict, *, phase: str, status: str, cursor=None) -> dict:
    cursor, cursor_json = _validate_worker_cursor(cursor)
    phase = " ".join(str(phase).split())[:128] or "worker"
    status = " ".join(str(status).split())[:32]
    now = time.time()
    checkpoint_id = _stable_id("worker-checkpoint", run_id, task["task_id"], task["attempt"])
    graph_path = _run_dir(run_id) / "graph.json"
    digest = hashlib.sha256(graph_path.read_bytes()).hexdigest() if graph_path.exists() else ""
    conn.execute(
        """INSERT OR REPLACE INTO worker_checkpoints(checkpoint_id,run_id,worker_id,task_id,attempt,phase,status,cursor_json,graph_digest,created_at,updated_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (checkpoint_id, run_id, worker_id[:128], task["task_id"], int(task["attempt"]), phase, status, cursor_json, digest, now, now),
    )
    return {"checkpoint_id": checkpoint_id, "run_id": run_id, "worker_id": worker_id[:128], "task_id": task["task_id"], "attempt": int(task["attempt"]), "phase": phase, "status": status, "cursor": cursor, "created_at": now, "updated_at": now}


def _latest_worker_checkpoint(conn: sqlite3.Connection, run_id: str, task_id: str = "") -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM worker_checkpoints WHERE run_id=? AND (?='' OR task_id=?) ORDER BY updated_at DESC,checkpoint_id DESC LIMIT 1",
        (run_id, task_id, task_id),
    ).fetchone()


def _enqueue_frontier_tasks(conn: sqlite3.Connection, run_id: str, node_id: str, priority: int, planned_pivots: list[str]) -> int:
    now = time.time()
    inserted = 0
    for pivot_type in sorted({str(value).strip() for value in planned_pivots if str(value).strip()}):
        task_id = _stable_id("frontier-task", run_id, node_id, pivot_type)
        cursor = conn.execute(
            """INSERT OR IGNORE INTO frontier_tasks(
                task_id,run_id,node_id,pivot_type,priority,state,attempt,max_attempts,
                available_at,created_at,updated_at
            ) VALUES(?,?,?,?,?,'OPEN',0,3,?,?,?)""",
            (task_id, run_id, node_id, pivot_type, int(priority), now, now, now),
        )
        inserted += cursor.rowcount
    return inserted


def _task_dict(row: sqlite3.Row | None, *, include_lease: bool = True) -> dict | None:
    if row is None:
        return None
    result = dict(row)
    if not include_lease:
        result.pop("lease_token", None)
    return result


def _recover_expired_leases(conn: sqlite3.Connection, now: float) -> int:
    cursor = conn.execute(
        """UPDATE frontier_tasks
           SET state=CASE WHEN attempt < max_attempts THEN 'RETRY_WAIT' ELSE 'FAILED' END,
               available_at=CASE WHEN attempt < max_attempts THEN ? ELSE available_at END,
               worker_id=NULL, lease_token=NULL, lease_expires_at=NULL,
               last_error=CASE WHEN attempt < max_attempts THEN 'lease_expired' ELSE COALESCE(last_error, 'lease_expired') END,
               updated_at=?
         WHERE state='CLAIMED' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?""",
        (now, now, now),
    )
    return cursor.rowcount


def _refresh_run_retry_state(conn: sqlite3.Connection, run_id: str, now: float, *, promote_ready: bool = True) -> sqlite3.Row | None:
    run = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    if run is None or run["status"] in {"COMPLETED", "FAILED", "CANCELLED", "PARTIAL"}:
        return run
    retry = conn.execute("SELECT MIN(available_at) AS retry_at, COUNT(*) AS count, MAX(last_error) AS reason FROM frontier_tasks WHERE run_id=? AND state='RETRY_WAIT'", (run_id,)).fetchone()
    if retry["count"]:
        retry_at = float(retry["retry_at"])
        if run["status"] == "RUNNING":
            status = "RUNNING"
        else:
            status = "RESUMABLE" if promote_ready and retry_at <= now else "RETRY_WAIT"
        conn.execute("UPDATE run_state SET status=?,retry_at=?,retry_reason=?,updated_at=? WHERE run_id=?", (status, retry_at, retry["reason"], _now(), run_id))
    elif run["status"] in {"RETRY_WAIT", "RESUMABLE"}:
        conn.execute("UPDATE run_state SET status='RUNNING',retry_at=NULL,retry_reason=NULL,updated_at=? WHERE run_id=?", (_now(), run_id))
    return conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()


def _schedule_frontier(args: dict) -> dict:
    run_id = args.get("run_id", "")
    conn = _connect(run_id)
    conn.execute("BEGIN IMMEDIATE")
    try:
        if conn.execute("SELECT 1 FROM run_state WHERE run_id=?", (run_id,)).fetchone() is None:
            raise ValueError("run is not initialized")
        node_id = str(args.get("node_id", "")).strip()
        rows = conn.execute(
            "SELECT node_id,priority,planned_pivots FROM frontier WHERE state='OPEN' AND (?='' OR node_id=?) ORDER BY priority DESC,node_id ASC",
            (node_id, node_id),
        ).fetchall()
        scheduled = 0
        for row in rows:
            try:
                pivots = json.loads(row["planned_pivots"])
            except json.JSONDecodeError:
                pivots = []
            if isinstance(pivots, list):
                scheduled += _enqueue_frontier_tasks(conn, run_id, row["node_id"], row["priority"], pivots)
        conn.commit()
        return {"ok": True, "run_id": run_id, "scheduled": scheduled, "frontier_rows": len(rows)}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _claim_frontier(args: dict) -> dict:
    run_id = args.get("run_id", "")
    worker_id = str(args.get("worker_id", "")).strip()[:128]
    if not worker_id:
        raise ValueError("worker_id is required")
    lease_seconds = min(3600.0, max(0.0, float(args.get("lease_seconds", 300))))
    conn = _connect(run_id)
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        if conn.execute("SELECT 1 FROM run_state WHERE run_id=?", (run_id,)).fetchone() is None:
            raise ValueError("run is not initialized")
        recovered = _recover_expired_leases(conn, now)
        run = _refresh_run_retry_state(conn, run_id, now, promote_ready=True)
        if run["status"] in {"COMPLETED", "FAILED", "CANCELLED", "PARTIAL"}:
            conn.commit()
            return {"ok": True, "task": None, "recovered": recovered, "run_status": run["status"], "stop_reason": run["saturation_reason"] or run["status"]}
        if run["status"] == "RETRY_WAIT":
            conn.commit()
            return {"ok": True, "task": None, "recovered": recovered, "run_status": "RETRY_WAIT", "retry_at": run["retry_at"], "stop_reason": "RETRY_WAIT"}
        if run["status"] == "RESUMABLE":
            conn.commit()
            return {"ok": True, "task": None, "recovered": recovered, "run_status": "RESUMABLE", "retry_at": run["retry_at"], "stop_reason": "RESUMABLE"}
        node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        budget_reason = None
        if run["deadline_at"] is not None and now >= run["deadline_at"]:
            budget_reason = "BUDGET_EXHAUSTED:deadline"
        elif run["tasks_executed"] >= run["max_tasks"]:
            budget_reason = "BUDGET_EXHAUSTED:max_tasks"
        elif run["source_calls"] >= run["max_searches"]:
            budget_reason = "BUDGET_EXHAUSTED:max_searches"
        elif node_count >= run["max_nodes"]:
            budget_reason = "BUDGET_EXHAUSTED:max_nodes"
        elif edge_count >= run["max_edges"]:
            budget_reason = "BUDGET_EXHAUSTED:max_edges"
        if budget_reason:
            conn.execute("UPDATE run_state SET status='PARTIAL',saturation_reason=?,updated_at=? WHERE run_id=?", (budget_reason, _now(), run_id))
            conn.commit()
            return {"ok": True, "task": None, "recovered": recovered, "stop_reason": budget_reason}
        row = conn.execute(
            """SELECT * FROM frontier_tasks
               WHERE (state='OPEN' OR (state='RETRY_WAIT' AND available_at <= ?))
                 AND (?='' OR pivot_type=?)
                 AND (?='' OR task_id=?)
               ORDER BY priority DESC, created_at ASC, task_id ASC LIMIT 1""",
            (now, str(args.get("pivot_type", "")).strip(), str(args.get("pivot_type", "")).strip(), str(args.get("task_id", "")).strip(), str(args.get("task_id", "")).strip()),
        ).fetchone()
        if row is None:
            conn.commit()
            return {"ok": True, "task": None, "recovered": recovered}
        token = secrets.token_hex(16)
        expires = now + lease_seconds
        requested_max_attempts = min(10, max(1, int(args.get("max_attempts", 3))))
        if row["attempt"] == 0:
            conn.execute("UPDATE frontier_tasks SET max_attempts=? WHERE task_id=?", (requested_max_attempts, row["task_id"]))
        conn.execute(
            """UPDATE frontier_tasks
               SET state='CLAIMED', attempt=attempt+1, worker_id=?, lease_token=?,
                   lease_expires_at=?, updated_at=?
             WHERE task_id=?""",
            (worker_id, token, expires, now, row["task_id"]),
        )
        conn.execute(
            "UPDATE run_state SET tasks_executed=tasks_executed+1,source_calls=source_calls+1,updated_at=? WHERE run_id=?",
            (_now(), run_id),
        )
        claimed = conn.execute("SELECT * FROM frontier_tasks WHERE task_id=?", (row["task_id"],)).fetchone()
        checkpoint = _upsert_worker_checkpoint(conn, run_id, worker_id, claimed, phase="claimed", status="CLAIMED")
        conn.commit()
        return {"ok": True, "task": _task_dict(claimed), "checkpoint": checkpoint, "recovered": 0}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _lease_mutation(args: dict, action: str) -> dict:
    run_id = args.get("run_id", "")
    task_id = str(args.get("task_id", "")).strip()
    worker_id = str(args.get("worker_id", "")).strip()
    lease_token = str(args.get("lease_token", "")).strip()
    if not task_id or not worker_id or not lease_token:
        raise ValueError("task_id, worker_id and lease_token are required")
    conn = _connect(run_id)
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT * FROM frontier_tasks WHERE task_id=? AND run_id=?", (task_id, run_id)).fetchone()
        if row is None:
            raise ValueError("frontier task not found")
        if row["state"] != "CLAIMED" or row["worker_id"] != worker_id or row["lease_token"] != lease_token:
            raise ValueError("frontier lease is not owned by worker")
        if row["lease_expires_at"] is None or row["lease_expires_at"] <= now:
            raise ValueError("frontier lease expired")
        if action == "complete":
            conn.execute("UPDATE frontier_tasks SET state='COMPLETED',worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,updated_at=? WHERE task_id=?", (now, task_id))
            state = "COMPLETED"
        elif action == "renew":
            lease_seconds = min(3600.0, max(1.0, float(args.get("lease_seconds", 300))))
            conn.execute("UPDATE frontier_tasks SET lease_expires_at=?,updated_at=? WHERE task_id=?", (now + lease_seconds, now, task_id))
            state = "CLAIMED"
        else:
            error = " ".join(str(args.get("error", "frontier task failed")).split())[:500]
            retry_after = min(3600.0, max(0.0, float(args.get("retry_after_seconds", 30))))
            if row["attempt"] < row["max_attempts"]:
                state = "RETRY_WAIT"
                conn.execute("UPDATE frontier_tasks SET state=?,available_at=?,worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,last_error=?,updated_at=? WHERE task_id=?", (state, now + retry_after, error, now, task_id))
            else:
                state = "FAILED"
                conn.execute("UPDATE frontier_tasks SET state=?,worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,last_error=?,updated_at=? WHERE task_id=?", (state, error, now, task_id))
            if state == "RETRY_WAIT":
                conn.execute("UPDATE run_state SET status='RETRY_WAIT',retry_at=?,retry_reason=?,retry_count=retry_count+1,updated_at=? WHERE run_id=? AND status NOT IN ('COMPLETED','FAILED','CANCELLED','PARTIAL')", (now + retry_after, error, _now(), run_id))
        updated = conn.execute("SELECT * FROM frontier_tasks WHERE task_id=?", (task_id,)).fetchone()
        prior = _latest_worker_checkpoint(conn, run_id, task_id)
        cursor = json.loads(prior["cursor_json"] or "{}") if prior else {}
        checkpoint = _upsert_worker_checkpoint(conn, run_id, worker_id, updated, phase=prior["phase"] if prior else action, status=state, cursor=cursor)
        conn.commit()
        return {"ok": True, "task_id": task_id, "state": state, "attempt": updated["attempt"], "task": _task_dict(updated), "checkpoint": checkpoint}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _checkpoint_worker(args: dict) -> dict:
    run_id = args.get("run_id", "")
    task_id = str(args.get("task_id", "")).strip()
    worker_id = str(args.get("worker_id", "")).strip()
    lease_token = str(args.get("lease_token", "")).strip()
    if not task_id or not worker_id or not lease_token:
        raise ValueError("task_id, worker_id and lease_token are required")
    cursor, _ = _validate_worker_cursor(args.get("cursor", {}))
    conn = _connect(run_id)
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        task = conn.execute("SELECT * FROM frontier_tasks WHERE task_id=? AND run_id=?", (task_id, run_id)).fetchone()
        if task is None or task["state"] != "CLAIMED" or task["worker_id"] != worker_id or task["lease_token"] != lease_token:
            raise ValueError("frontier lease is not owned by worker")
        if task["lease_expires_at"] is None or task["lease_expires_at"] <= now:
            raise ValueError("frontier lease expired")
        checkpoint = _upsert_worker_checkpoint(conn, run_id, worker_id, task, phase=args.get("phase", "worker"), status="CHECKPOINTED", cursor=cursor)
        conn.execute("UPDATE run_state SET updated_at=? WHERE run_id=?", (_now(), run_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    conn = _connect(run_id)
    _materialize(conn, run_id)
    _checkpoint(conn, run_id, "worker_checkpoint")
    conn.close()
    return {"ok": True, "run_id": run_id, "task_id": task_id, "checkpoint": checkpoint, "cursor": cursor}


def _resume_worker(args: dict) -> dict:
    run_id = args.get("run_id", "")
    worker_id = str(args.get("worker_id", "")).strip()[:128]
    requested_task_id = str(args.get("task_id", "")).strip()
    if not worker_id:
        raise ValueError("worker_id is required")
    conn = _connect(run_id)
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        if conn.execute("SELECT 1 FROM run_state WHERE run_id=?", (run_id,)).fetchone() is None:
            raise ValueError("run is not initialized")
        _recover_expired_leases(conn, now)
        checkpoint = _latest_worker_checkpoint(conn, run_id, requested_task_id)
        if checkpoint is None:
            conn.commit()
            return {"ok": False, "run_id": run_id, "resumed_from_checkpoint": False, "reason": "checkpoint_not_found"}
        task = conn.execute("SELECT * FROM frontier_tasks WHERE task_id=? AND run_id=?", (checkpoint["task_id"], run_id)).fetchone()
        if task is None:
            raise ValueError("checkpoint task not found")
        if task["state"] == "CLAIMED" and task["lease_expires_at"] and task["lease_expires_at"] > now:
            conn.commit()
            return {"ok": False, "run_id": run_id, "resumed_from_checkpoint": False, "reason": "worker_lease_active"}
        cursor = json.loads(checkpoint["cursor_json"] or "{}")
        pivot_type = task["pivot_type"]
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    claimed = _claim_frontier({**args, "task_id": task["task_id"], "pivot_type": pivot_type})
    if claimed.get("task") is None:
        return {"ok": False, "run_id": run_id, "resumed_from_checkpoint": False, "reason": claimed.get("stop_reason", "task_not_ready"), "checkpoint": {"checkpoint_id": checkpoint["checkpoint_id"], "task_id": checkpoint["task_id"], "phase": checkpoint["phase"], "status": checkpoint["status"], "cursor": cursor}}
    return {"ok": True, "run_id": run_id, "worker_id": worker_id, "resumed_from_checkpoint": True, "task": claimed["task"], "checkpoint": {"checkpoint_id": checkpoint["checkpoint_id"], "task_id": checkpoint["task_id"], "phase": checkpoint["phase"], "status": checkpoint["status"], "cursor": cursor}, "new_checkpoint": claimed.get("checkpoint")}


def _frontier_status(args: dict) -> dict:
    run_id = args.get("run_id", "")
    conn = _connect(run_id)
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        recovered = _recover_expired_leases(conn, now)
        run = _refresh_run_retry_state(conn, run_id, now, promote_ready=True)
        rows = conn.execute("SELECT state,COUNT(*) AS count FROM frontier_tasks WHERE run_id=? GROUP BY state", (run_id,)).fetchall()
        tasks = conn.execute("SELECT * FROM frontier_tasks WHERE run_id=? ORDER BY priority DESC,created_at ASC,task_id ASC", (run_id,)).fetchall()
        conn.commit()
        counts = {row["state"]: row["count"] for row in rows}
        return {"ok": True, "run_id": run_id, "recovered": recovered, "run_status": run["status"], "retry_at": run["retry_at"], "retry_reason": run["retry_reason"], "counts": counts, "tasks": [_task_dict(row, include_lease=False) for row in tasks]}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _init_run(args: dict) -> dict:
    run_id = args.get("run_id", "")
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id is required and must be safe")
    target_url = _normalize_url(args.get("target_url", ""))
    max_tasks = min(500, max(1, int(args.get("max_tasks", 50))))
    max_searches = min(2000, max(1, int(args.get("max_searches", 200))))
    max_nodes = min(5000, max(1, int(args.get("max_nodes", 250))))
    max_edges = min(10000, max(1, int(args.get("max_edges", 500))))
    deadline_seconds = min(86400.0, max(60.0, float(args.get("deadline_seconds", 3600))))
    deadline_at = time.time() + deadline_seconds
    conn = _connect(run_id)
    now = _now()
    with conn:
        conn.execute("INSERT OR REPLACE INTO run_state(run_id,target_url,status,round,created_at,updated_at,max_tasks,max_searches,max_nodes,max_edges,deadline_at,tasks_executed,source_calls,saturation_reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (run_id, target_url, "RUNNING", 0, now, now, max_tasks, max_searches, max_nodes, max_edges, deadline_at, 0, 0, None))
        domain = _domain(target_url)
        node_id = _stable_id("domain", domain)
        conn.execute("INSERT OR IGNORE INTO nodes(node_id,node_type,value,normalized_value,status,confidence,first_seen_round,frontier_state) VALUES(?,?,?,?,?,?,?,?)", (node_id, "domain", domain, domain, "OBSERVED", 1.0, 0, "OPEN"))
        from tools.companyintel_pivots import expand_pivots
        planned_specs = expand_pivots("domain", domain)
        planned_pivots = [spec.pivot_type for spec in planned_specs]
        priority = max(spec.priority for spec in planned_specs)
        conn.execute("INSERT OR REPLACE INTO frontier(node_id,priority,reason,planned_pivots,state) VALUES(?,?,?,?,?)", (node_id, priority, "seed domain inventory", _json(planned_pivots), "OPEN"))
        _enqueue_frontier_tasks(conn, run_id, node_id, priority, planned_pivots)
    _materialize(conn, run_id)
    _checkpoint(conn, run_id, "run_initialized")
    conn.close()
    return {"ok": True, "run_id": run_id, "status": "RUNNING", "seed_node_id": node_id}


def _inventory(args: dict) -> dict:
    """Run the bounded code-owned inventory and persist its findings."""
    from tools.companyintel_inventory import InventoryLimits, extract_inventory

    run_id = args.get("run_id", "")
    target_url = _normalize_url(args.get("target_url", ""))
    conn = _connect(run_id)
    existing = conn.execute("SELECT target_url FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    conn.close()
    if existing:
        if existing["target_url"] != target_url:
            raise ValueError("run_id is already bound to a different target_url")
        initialized = {"seed_node_id": _stable_id("domain", _domain(target_url))}
    else:
        initialized = _init_run(args)
    limits = InventoryLimits(
        max_urls=int(args.get("max_urls", 32)),
        max_bytes_per_url=int(args.get("max_bytes_per_url", 256 * 1024)),
        max_total_bytes=int(args.get("max_total_bytes", 2 * 1024 * 1024)),
    )
    inventory = extract_inventory(args.get("target_url", ""), limits=limits)
    inventory_path = _run_dir(run_id) / "inventory.json"
    _atomic_json(inventory_path, inventory)
    persisted = 0
    for finding in inventory.get("findings", []):
        result = _record_observation({
            "run_id": run_id,
            "node_type": finding["node_type"],
            "value": finding["value"],
            "source_url": finding["source_url"],
            "excerpt": finding["excerpt"],
            "relation_from_node_id": initialized["seed_node_id"],
            "relation": "inventory_discovered",
            "retrieval_method": "bounded_http_inventory",
            "access_status": "success",
        })
        if result.get("ok"):
            persisted += 1
    return {
        "ok": True,
        "run_id": run_id,
        "inventory_path": str(inventory_path),
        "findings": persisted,
        "fetched_urls": inventory["stats"]["fetched_urls"],
        "errors": len(inventory["errors"]),
        "inventory": {key: inventory[key] for key in ("metadata", "urls", "documents", "images", "scripts", "identifiers", "external_domains", "discovered_sources", "stats")},
    }


def _record_observation(args: dict) -> dict:
    run_id = args.get("run_id", "")
    node_type = args.get("node_type", "")
    if node_type not in _ALLOWED_NODE_TYPES:
        raise ValueError("unsupported node_type")
    value = _normalize_value(node_type, args.get("value", ""))
    source_url = _normalize_url(args.get("source_url", ""))
    excerpt = " ".join(str(args.get("excerpt", "")).split())[:2000]
    if not excerpt:
        raise ValueError("excerpt is required")
    conn = _connect(run_id)
    run = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    if run is None:
        conn.close()
        raise ValueError("run is not initialized")
    node_id = _stable_id(node_type, value)
    source_node_id = args.get("relation_from_node_id", "")
    if source_node_id and conn.execute("SELECT 1 FROM nodes WHERE node_id=?", (source_node_id,)).fetchone() is None:
        conn.close()
        raise ValueError("relation_from_node_id does not exist")
    evidence_id = _stable_id("evidence", source_url, excerpt)
    edge_id = _stable_id("edge", source_node_id, node_id, args.get("relation", "observed_in"), evidence_id)
    now = _now()
    with conn:
        conn.execute("INSERT OR IGNORE INTO evidence VALUES(?,?,?,?,?,?,?,?,?)", (evidence_id, source_url, args.get("source_title", ""), args.get("source_tier", "C"), now, excerpt, hashlib.sha256(excerpt.encode()).hexdigest(), args.get("retrieval_method", "browser"), args.get("access_status", "success")))
        conn.execute("INSERT OR IGNORE INTO nodes VALUES(?,?,?,?,?,?,?,?,?,?,?)", (node_id, node_type, value, value, "OBSERVED", float(args.get("confidence", 0.7)), _json([evidence_id]), _json([args.get("relation_from_node_id")] if args.get("relation_from_node_id") else []), run["round"], None, "OPEN"))
        if source_node_id:
            conn.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?,?,?,?,?)", (edge_id, source_node_id, node_id, args.get("relation", "observed_in"), float(args.get("confidence", 0.7)), _json([evidence_id]), run["round"]))
        from tools.companyintel_pivots import expand_pivots
        planned_specs = expand_pivots(node_type, value)
        pivots = [spec.pivot_type for spec in planned_specs]
        priority = max(spec.priority for spec in planned_specs)
        conn.execute("INSERT OR REPLACE INTO frontier(node_id,priority,reason,planned_pivots,state) VALUES(?,?,?,?,?)", (node_id, priority, f"typed pivots for {node_type}", _json(pivots), "OPEN"))
        _enqueue_frontier_tasks(conn, run_id, node_id, priority, pivots)
        conn.execute("UPDATE run_state SET updated_at=? WHERE run_id=?", (now, run_id))
    _materialize(conn, run_id)
    _checkpoint(conn, run_id, "observation_recorded")
    conn.close()
    return {"ok": True, "run_id": run_id, "node_id": node_id, "evidence_id": evidence_id, "edge_id": edge_id}


def _record_search(args: dict) -> dict:
    run_id = args.get("run_id", "")
    conn = _connect(run_id)
    run = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    if run is None:
        conn.close()
        raise ValueError("run is not initialized")
    search_id = args.get("search_id") or _stable_id("search", args.get("query", ""), args.get("source", ""), _now())
    with conn:
        conn.execute("INSERT OR REPLACE INTO search_log VALUES(?,?,?,?,?,?,?,?,?)", (search_id, run["round"], args.get("pivot_node_id", ""), args.get("pivot_type", ""), str(args.get("query", ""))[:1000], args.get("source", ""), args.get("status", "completed"), _json(args.get("new_node_ids", [])), args.get("limitation")))
        conn.execute("UPDATE run_state SET updated_at=? WHERE run_id=?", (_now(), run_id))
    _materialize(conn, run_id)
    count = conn.execute("SELECT COUNT(*) FROM search_log").fetchone()[0]
    if count % 5 == 0:
        _checkpoint(conn, run_id, f"retrieval_batch_{count}")
    conn.close()
    return {"ok": True, "run_id": run_id, "search_id": search_id, "retrievals": count, "checkpoint_created": count % 5 == 0}


def _execute_frontier(args: dict) -> dict:
    """Execute the first registered worker slice: bounded public exact search."""
    from tools.companyintel_pivots import get_pivot
    from tools.companyintel_public_search import SearchLimits, execute_public_search

    run_id = args.get("run_id", "")
    pivot_type = str(args.get("pivot_type", "exact_search")).strip() or "exact_search"
    spec = get_pivot(pivot_type)
    worker_configs = {
        "public_search": {"mode": "exact", "node_type": "url", "relation": "search_result", "source": "duckduckgo_html"},
        "maps_search": {"mode": "maps", "node_type": "place", "relation": "maps_result", "source": "duckduckgo_maps_search"},
        "marketplace_search": {"mode": "marketplace", "node_type": "marketplace_profile", "relation": "marketplace_result", "source": "duckduckgo_marketplace_search"},
        "document_search": {"mode": "document", "node_type": "document", "relation": "document_result", "source": "duckduckgo_document_search"},
    }
    worker_config = worker_configs.get(spec.worker if spec else "")
    if spec is None or worker_config is None:
        raise ValueError("pivot type is not backed by an implemented typed worker")
    worker_id = str(args.get("worker_id", "")).strip()
    claimed = _claim_frontier({**args, "pivot_type": pivot_type})
    task = claimed.get("task")
    if task is None:
        return {"ok": True, "outcome": "NO_TASK", "task": None, "persisted_results": 0}
    conn = _connect(run_id)
    node = conn.execute("SELECT node_type,value FROM nodes WHERE node_id=?", (task["node_id"],)).fetchone()
    conn.close()
    if node is None:
        _lease_mutation({**args, "task_id": task["task_id"], "worker_id": worker_id, "lease_token": task["lease_token"], "error": "pivot node not found"}, "fail")
        return {"ok": False, "outcome": "FAILED", "task_id": task["task_id"], "persisted_results": 0}
    result = execute_public_search(
        node["node_type"],
        node["value"],
        limits=SearchLimits(
            timeout_seconds=min(15.0, max(1.0, float(args.get("search_timeout_seconds", 8)))),
            max_bytes=min(512 * 1024, max(4096, int(args.get("search_max_bytes", 256 * 1024)))),
            max_results=min(20, max(1, int(args.get("search_max_results", 10)))),
        ),
        mode=worker_config["mode"],
    )
    if result["outcome"] in {"RETRYABLE_ERROR", "UNAVAILABLE"}:
        failure = _lease_mutation({
            **args,
            "task_id": task["task_id"],
            "worker_id": worker_id,
            "lease_token": task["lease_token"],
            "error": result.get("error") or result["outcome"],
            "retry_after_seconds": args.get("retry_after_seconds", 30),
        }, "fail")
        return {"ok": True, "outcome": result["outcome"], "task_id": task["task_id"], "state": failure["state"], "persisted_results": 0, "error": result.get("error")}
    new_node_ids = []
    persisted = 0
    for item in result["results"]:
        observation = _record_observation({
            "run_id": run_id,
            "node_type": worker_config["node_type"],
            "value": item["url"],
            "source_url": item["url"],
            "source_title": item["title"],
            "excerpt": item["snippet"] or item["title"] or item["url"],
            "relation_from_node_id": task["node_id"],
            "relation": worker_config["relation"],
            "retrieval_method": "public_search",
            "access_status": "success",
        })
        if observation.get("ok"):
            persisted += 1
            new_node_ids.append(observation["node_id"])
    _record_search({
        "run_id": run_id,
        "pivot_node_id": task["node_id"],
        "pivot_type": pivot_type,
        "query": result["query"],
        "source": worker_config["source"],
        "status": result["outcome"],
        "new_node_ids": new_node_ids,
        "limitation": None,
    })
    completed = _lease_mutation({
        **args,
        "task_id": task["task_id"],
        "worker_id": worker_id,
        "lease_token": task["lease_token"],
    }, "complete")
    return {"ok": True, "outcome": result["outcome"], "pivot_type": pivot_type, "task_id": task["task_id"], "state": completed["state"], "persisted_results": persisted, "new_node_ids": new_node_ids}


def _execute_inventory_frontier(args: dict) -> dict:
    run_id = args.get("run_id", "")
    worker_id = str(args.get("worker_id", "")).strip()
    claimed = _claim_frontier({**args, "pivot_type": "site_inventory"})
    task = claimed.get("task")
    if task is None:
        return {"ok": True, "outcome": "NO_TASK", "pivot_type": "site_inventory", "task": None, "persisted_results": 0}
    conn = _connect(run_id)
    run = conn.execute("SELECT target_url FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    conn.close()
    if run is None:
        raise ValueError("run is not initialized")
    try:
        result = _inventory({"run_id": run_id, "target_url": run["target_url"]})
        completed = _lease_mutation({
            **args,
            "task_id": task["task_id"],
            "worker_id": worker_id,
            "lease_token": task["lease_token"],
        }, "complete")
        return {
            "ok": True,
            "outcome": "COMPLETED_WITH_RESULTS" if result.get("findings", 0) else "COMPLETED_ZERO_RESULTS",
            "pivot_type": "site_inventory",
            "task_id": task["task_id"],
            "state": completed["state"],
            "persisted_results": result.get("findings", 0),
        }
    except Exception as exc:
        failure = _lease_mutation({
            **args,
            "task_id": task["task_id"],
            "worker_id": worker_id,
            "lease_token": task["lease_token"],
            "error": str(exc),
        }, "fail")
        return {"ok": True, "outcome": "RETRYABLE_ERROR", "pivot_type": "site_inventory", "task_id": task["task_id"], "state": failure["state"], "persisted_results": 0}


def _run_frontier(args: dict) -> dict:
    """Run a bounded automatic dispatch loop over supported typed pivots."""
    run_id = args.get("run_id", "")
    worker_id = str(args.get("worker_id", "")).strip()
    if not worker_id:
        raise ValueError("worker_id is required")
    max_tasks = min(50, max(1, int(args.get("max_tasks", 10))))
    outcomes = []
    supported = ("site_inventory", "exact_search", "maps", "marketplaces", "documents")
    while len(outcomes) < max_tasks:
        dispatched = None
        for pivot_type in supported:
            if pivot_type == "site_inventory":
                candidate = _execute_inventory_frontier(args)
            else:
                candidate = _execute_frontier({**args, "pivot_type": pivot_type})
            if candidate.get("outcome") != "NO_TASK":
                dispatched = candidate
                break
        if dispatched is None:
            break
        outcomes.append(dispatched)
    conn = _connect(run_id)
    run_row = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    if run_row and run_row["status"] == "RUNNING" and run_row["tasks_executed"] >= run_row["max_tasks"]:
        conn.execute("UPDATE run_state SET status='PARTIAL',saturation_reason='BUDGET_EXHAUSTED:max_tasks',updated_at=? WHERE run_id=?", (_now(), run_id))
        conn.commit()
        _materialize(conn, run_id)
        _checkpoint(conn, run_id, "budget_exhausted_max_tasks")
    conn.close()
    legal_identity = _legal_identity_exhaustion({"run_id": run_id})
    return {
        "ok": True,
        "run_id": run_id,
        "worker_id": worker_id,
        "tasks_executed": len(outcomes),
        "legal_identity_exhaustion": legal_identity,
        "outcomes": outcomes,
        "remaining_supported_tasks": sum(1 for item in outcomes if item.get("state") in {"RETRY_WAIT", "CLAIMED"}),
    }


_IDENTITY_MATCH_SCORES = {
    "exact_official_legal_id": 100,
    "domain_authoritative_record": 95,
    "legal_name_phone_address": 90,
    "exact_phone_email_marketplace_legal_card": 90,
    "name_director_country": 80,
    "brand_phone_email": 75,
    "name_only": 40,
    "logo_favicon_only": 20,
}


def _candidate_status(score: int, conflicts: list[str]) -> str:
    if conflicts:
        return "REJECTED" if "exact_official_legal_id" in conflicts else "AMBIGUOUS"
    if score >= 100:
        return "VERIFIED"
    if score >= 90:
        return "HIGH_CONFIDENCE"
    if score >= 75:
        return "PROBABLE"
    if score >= 40:
        return "AMBIGUOUS"
    return "REJECTED"


def _record_identity_candidate(args: dict) -> dict:
    run_id = args.get("run_id", "")
    candidate_id = str(args.get("candidate_id", "")).strip()
    if not candidate_id:
        raise ValueError("candidate_id is required")
    match_types = [str(item).strip() for item in args.get("match_types", []) if str(item).strip()]
    unknown = sorted(set(match_types) - set(_IDENTITY_MATCH_SCORES))
    if unknown:
        raise ValueError(f"unknown match_types: {', '.join(unknown)}")
    matching = sorted(set(match_types), key=lambda item: (-_IDENTITY_MATCH_SCORES[item], item))
    conflicts = sorted(set(str(item).strip() for item in args.get("conflicting_dimensions", []) if str(item).strip()))
    missing = sorted(set(str(item).strip() for item in args.get("missing_dimensions", []) if str(item).strip()))
    evidence_ids = sorted(set(str(item).strip() for item in args.get("evidence_ids", []) if str(item).strip()))
    score = max((_IDENTITY_MATCH_SCORES[item] for item in matching), default=0)
    status = _candidate_status(score, conflicts)
    conn = _connect(run_id)
    run = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    if run is None:
        conn.close()
        raise ValueError("run is not initialized")
    if evidence_ids:
        placeholders = ",".join("?" * len(evidence_ids))
        existing_evidence = {row[0] for row in conn.execute(f"SELECT evidence_id FROM evidence WHERE evidence_id IN ({placeholders})", evidence_ids).fetchall()}
        unknown_evidence = sorted(set(evidence_ids) - existing_evidence)
        if unknown_evidence:
            conn.close()
            raise ValueError(f"unknown evidence_ids: {', '.join(unknown_evidence)}")
    now = _now()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO identity_candidates(candidate_id,run_id,legal_name,legal_id,country,score,status,matching_dimensions,conflicting_dimensions,missing_dimensions,evidence_ids,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (candidate_id, run_id, str(args.get("legal_name", "")).strip(), str(args.get("legal_id", "")).strip(), str(args.get("country", "")).strip(), score, status, _json(matching), _json(conflicts), _json(missing), _json(evidence_ids), now, now),
        )
        conn.execute("UPDATE run_state SET updated_at=? WHERE run_id=?", (now, run_id))
    _materialize(conn, run_id)
    _checkpoint(conn, run_id, "identity_candidate_recorded")
    conn.close()
    return {"ok": True, "run_id": run_id, "candidate_id": candidate_id, "score": score, "status": status, "matching_dimensions": matching, "conflicting_dimensions": conflicts, "missing_dimensions": missing, "evidence_ids": evidence_ids}


def _legal_identity_coverage(conn: sqlite3.Connection, run_id: str) -> dict:
    required = (
        "structured_data", "phone", "email", "address", "domain_variants",
        "document_metadata", "javascript_identifiers", "marketplace", "maps",
        "archive", "social", "image_or_favicon", "official_registry",
        "procurement", "court_or_ip",
    )
    pivot_map = {
        "structured_data": ("site_inventory",), "phone": ("site_inventory",),
        "email": ("site_inventory",), "address": ("site_inventory",),
        "domain_variants": ("exact_search",), "document_metadata": ("documents",),
        "javascript_identifiers": ("site_inventory",), "marketplace": ("marketplaces",),
        "maps": ("maps",), "archive": ("archive",), "social": ("social",),
        "image_or_favicon": ("site_inventory",), "official_registry": ("official_registry",),
        "procurement": ("procurement",), "court_or_ip": ("court_or_ip",),
    }
    unsupported = {"archive", "social", "official_registry", "procurement", "court_or_ip"}
    coverage = {}
    for coverage_class in required:
        pivots = pivot_map[coverage_class]
        rows = conn.execute(
            "SELECT pivot_type,state,last_error FROM frontier_tasks WHERE run_id=? AND pivot_type IN ({}) ORDER BY updated_at DESC".format(",".join("?" * len(pivots))),
            (run_id, *pivots),
        ).fetchall()
        if not rows:
            if coverage_class == "maps":
                inventory_rows = conn.execute("SELECT state FROM frontier_tasks WHERE run_id=? AND pivot_type='site_inventory'", (run_id,)).fetchall()
                relevant_nodes = conn.execute("SELECT COUNT(*) FROM nodes WHERE node_type IN ('address','phone','place')").fetchone()[0]
                if inventory_rows and all(row["state"] == "COMPLETED" for row in inventory_rows) and relevant_nodes == 0:
                    coverage[coverage_class] = "NOT_APPLICABLE"
                    continue
            coverage[coverage_class] = "UNAVAILABLE" if coverage_class in unsupported else "MISSING"
            continue
        states = [row["state"] for row in rows]
        if any(state in {"OPEN", "CLAIMED", "RETRY_WAIT"} for state in states):
            coverage[coverage_class] = "PENDING"
            continue
        failures = [str(row["last_error"] or "").lower() for row in rows if row["state"] == "FAILED"]
        if failures:
            text = " ".join(failures)
            if "captcha" in text:
                coverage[coverage_class] = "BLOCKED_BY_CAPTCHA"
            elif "rate" in text:
                coverage[coverage_class] = "RATE_LIMITED"
            elif "budget" in text:
                coverage[coverage_class] = "BUDGET_EXHAUSTED"
            else:
                coverage[coverage_class] = "RETRYABLE_ERROR"
            continue
        logs = conn.execute(
            "SELECT new_node_ids,limitation,status FROM search_log WHERE pivot_type IN ({}) ORDER BY rowid DESC".format(",".join("?" * len(pivots))),
            pivots,
        ).fetchall()
        limitations = " ".join(str(row["limitation"] or "").lower() for row in logs)
        if "captcha" in limitations:
            coverage[coverage_class] = "BLOCKED_BY_CAPTCHA"
        elif "rate" in limitations:
            coverage[coverage_class] = "RATE_LIMITED"
        elif "budget" in limitations:
            coverage[coverage_class] = "BUDGET_EXHAUSTED"
        else:
            has_results = False
            for log in logs:
                try:
                    has_results = has_results or bool(json.loads(log["new_node_ids"] or "[]"))
                except json.JSONDecodeError:
                    pass
            coverage[coverage_class] = "COMPLETED_WITH_RESULTS" if has_results else "COMPLETED_ZERO_RESULTS"
    return coverage


def _legal_identity_exhaustion(args: dict) -> dict:
    run_id = args.get("run_id", "")
    conn = _connect(run_id)
    run = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    if run is None:
        conn.close()
        raise ValueError("run is not initialized")
    coverage = _legal_identity_coverage(conn, run_id)
    candidates = conn.execute("SELECT candidate_id,status,score FROM identity_candidates WHERE run_id=? ORDER BY score DESC,updated_at DESC", (run_id,)).fetchall()
    viable_candidates = [dict(row) for row in candidates if row["status"] != "REJECTED"]
    pending_states = {"PENDING", "MISSING"}
    blocked_states = {"UNAVAILABLE", "RATE_LIMITED", "BLOCKED_BY_CAPTCHA", "RETRYABLE_ERROR", "BUDGET_EXHAUSTED"}
    pending = [name for name, state in coverage.items() if state in pending_states]
    blocked = [name for name, state in coverage.items() if state in blocked_states]
    if viable_candidates:
        status = "PARTIAL"
        reason = "identity_candidate_present"
    elif pending:
        status = "PENDING"
        reason = "coverage_pending"
    elif blocked:
        status = "PARTIAL"
        reason = "coverage_blocked"
    else:
        status = "EXHAUSTED"
        reason = "coverage_complete"
    allowed = status == "EXHAUSTED" and run["status"] not in {"PARTIAL", "FAILED", "CANCELLED"}
    identity_status = "NOT_FOUND_ELIGIBLE" if allowed else "UNRESOLVED"
    with conn:
        conn.execute(
            "UPDATE run_state SET legal_identity_status=?,legal_identity_reason=?,legal_identity_checked_at=?,legal_identity_coverage=?,updated_at=? WHERE run_id=?",
            (status, reason, _now(), _json(coverage), _now(), run_id),
        )
    _materialize(conn, run_id)
    _checkpoint(conn, run_id, f"legal_identity_exhaustion_{status.lower()}")
    conn.close()
    return {"ok": True, "run_id": run_id, "status": status, "identity_status": identity_status, "allowed_not_found": allowed, "reason": reason, "coverage": coverage, "pending_classes": pending, "blocking_classes": blocked, "candidates": [dict(row) for row in candidates], "saturation_reason": run["saturation_reason"]}


def _exhaustion_gate(args: dict) -> dict:
    result = _legal_identity_exhaustion(args)
    result["reason"] = "coverage_complete" if result["allowed_not_found"] else "coverage_pending_or_missing"
    return result


def _resume_frontier(args: dict) -> dict:
    run_id = args.get("run_id", "")
    conn = _connect(run_id)
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("run is not initialized")
        row = _refresh_run_retry_state(conn, run_id, now, promote_ready=True)
        if row["status"] == "RETRY_WAIT":
            conn.commit()
            return {"ok": False, "run_id": run_id, "resumed": False, "status": "RETRY_WAIT", "retry_at": row["retry_at"], "reason": "retry_not_ready"}
        if row["status"] not in {"RESUMABLE", "PARTIAL"}:
            conn.commit()
            return {"ok": False, "run_id": run_id, "resumed": False, "status": row["status"], "reason": "run_not_resumable"}
        additional_tasks = min(500, max(1, int(args.get("additional_tasks", 25))))
        additional_searches = min(2000, max(1, int(args.get("additional_searches", additional_tasks))))
        deadline_seconds = min(86400.0, max(60.0, float(args.get("deadline_seconds", 3600))))
        conn.execute("UPDATE run_state SET status='RUNNING',max_tasks=max_tasks+?,max_searches=max_searches+?,deadline_at=?,saturation_reason=NULL,retry_at=NULL,retry_reason=NULL,resume_count=resume_count+1,updated_at=? WHERE run_id=?", (additional_tasks, additional_searches, now + deadline_seconds, _now(), run_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    result = _run_frontier(args)
    result["resumed"] = True
    return result


def _summary(args: dict) -> dict:
    run_id = args.get("run_id", "")
    conn = _connect(run_id)
    row = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        conn.close()
        raise ValueError("run is not initialized")
    identity_candidates = []
    for candidate in conn.execute("SELECT candidate_id,legal_name,legal_id,country,score,status,matching_dimensions,conflicting_dimensions,missing_dimensions,evidence_ids FROM identity_candidates WHERE run_id=? ORDER BY score DESC,updated_at DESC", (run_id,)).fetchall():
        item = dict(candidate)
        for key in ("matching_dimensions", "conflicting_dimensions", "missing_dimensions", "evidence_ids"):
            item[key] = json.loads(item[key] or "[]")
        identity_candidates.append(item)
    worker_checkpoints = []
    for checkpoint in conn.execute("SELECT checkpoint_id,run_id,worker_id,task_id,attempt,phase,status,cursor_json,created_at,updated_at FROM worker_checkpoints WHERE run_id=? ORDER BY updated_at DESC,checkpoint_id DESC", (run_id,)).fetchall():
        item = dict(checkpoint)
        item["cursor"] = json.loads(item.pop("cursor_json") or "{}")
        worker_checkpoints.append(item)
    result = {"ok": True, "run_id": run_id, "nodes": conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0], "edges": conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0], "evidence": conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0], "frontier_open": conn.execute("SELECT COUNT(*) FROM frontier WHERE state='OPEN'").fetchone()[0], "search_log": conn.execute("SELECT COUNT(*) FROM search_log").fetchone()[0], "round": row["round"], "status": row["status"], "saturation_reason": row["saturation_reason"], "retry_at": row["retry_at"], "retry_reason": row["retry_reason"], "retry_count": row["retry_count"], "resume_count": row["resume_count"], "legal_identity_exhaustion": {"status": row["legal_identity_status"], "reason": row["legal_identity_reason"], "checked_at": row["legal_identity_checked_at"], "coverage": json.loads(row["legal_identity_coverage"] or "{}")}, "identity_candidates": identity_candidates, "worker_checkpoints": worker_checkpoints, "budgets": {"max_tasks": row["max_tasks"], "max_searches": row["max_searches"], "max_nodes": row["max_nodes"], "max_edges": row["max_edges"], "deadline_at": row["deadline_at"]}, "usage": {"tasks_executed": row["tasks_executed"], "source_calls": row["source_calls"]}}
    conn.close()
    return result


def companyintel_graph(args: dict, **_kwargs) -> str:
    if not check_companyintel_requirements():
        return json.dumps({"ok": False, "error": "companyintel graph tool is profile-gated"})
    try:
        action = args.get("action", "")
        if action == "init_run": result = _init_run(args)
        elif action == "inventory": result = _inventory(args)
        elif action == "schedule_frontier": result = _schedule_frontier(args)
        elif action == "claim_frontier": result = _claim_frontier(args)
        elif action == "renew_frontier": result = _lease_mutation(args, "renew")
        elif action == "complete_frontier": result = _lease_mutation(args, "complete")
        elif action == "fail_frontier": result = _lease_mutation(args, "fail")
        elif action == "checkpoint_worker": result = _checkpoint_worker(args)
        elif action == "resume_worker": result = _resume_worker(args)
        elif action == "frontier_status": result = _frontier_status(args)
        elif action == "record_observation": result = _record_observation(args)
        elif action == "record_search": result = _record_search(args)
        elif action == "execute_frontier": result = _execute_frontier(args)
        elif action == "run_frontier": result = _run_frontier(args)
        elif action == "resume_frontier": result = _resume_frontier(args)
        elif action == "exhaustion_gate": result = _exhaustion_gate(args)
        elif action == "legal_identity_exhaustion": result = _legal_identity_exhaustion(args)
        elif action == "record_identity_candidate": result = _record_identity_candidate(args)
        elif action == "summary": result = _summary(args)
        else: raise ValueError("unsupported action")
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


_COMPANYINTEL_GRAPH_SCHEMA = {
    "name": "companyintel_graph",
    "description": "Deterministically persist companyintel graph nodes, evidence, edges and typed frontier. Required for graph state; profile-gated.",
    "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["init_run", "inventory", "schedule_frontier", "claim_frontier", "renew_frontier", "complete_frontier", "fail_frontier", "checkpoint_worker", "resume_worker", "frontier_status", "execute_frontier", "run_frontier", "resume_frontier", "exhaustion_gate", "legal_identity_exhaustion", "record_identity_candidate", "record_observation", "record_search", "summary"]}, "run_id": {"type": "string"}, "target_url": {"type": "string"}, "node_type": {"type": "string"}, "value": {"type": "string"}, "source_url": {"type": "string"}, "source_title": {"type": "string"}, "excerpt": {"type": "string"}, "relation_from_node_id": {"type": "string"}, "relation": {"type": "string"}, "worker_id": {"type": "string"}, "task_id": {"type": "string"}, "lease_token": {"type": "string"}, "phase": {"type": "string"}, "cursor": {"type": "object", "additionalProperties": True}, "lease_seconds": {"type": "number"}, "max_attempts": {"type": "integer"}, "retry_after_seconds": {"type": "number"}, "pivot_type": {"type": "string"}, "error": {"type": "string"}, "search_timeout_seconds": {"type": "number"}, "search_max_bytes": {"type": "integer"}, "search_max_results": {"type": "integer"}, "max_urls": {"type": "integer"}, "max_bytes_per_url": {"type": "integer"}, "max_total_bytes": {"type": "integer"}}, "required": ["action", "run_id"]},
}

registry.register(name="companyintel_graph", toolset="companyintel", schema=_COMPANYINTEL_GRAPH_SCHEMA, handler=companyintel_graph, check_fn=check_companyintel_requirements, description=_COMPANYINTEL_GRAPH_SCHEMA["description"], emoji="🕸️", max_result_size_chars=12000)
