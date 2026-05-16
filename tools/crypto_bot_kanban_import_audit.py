#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import datetime as dt
import json
import os
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "hermes.autonomy.crypto_bot_kanban_import_audit.v1"
DEFAULT_STATE_ROOT = Path("/Users/preston/.local/state/hermes-operator")
DEFAULT_KANBAN_HOME = Path("/Users/preston/.hermes")
DEFAULT_REPO_ROOT = Path("/Users/preston/robinhood/crypto_bot")
PASSING_CONCLUSIONS = {"PASS", "passed", "pass"}
BLOCKED_STATUSES = {"blocked", "review", "review_required", "blocked_remote_pr_missing"}
RUNNING_STATUSES = {"running", "in_progress", "in progress"}
VALID_IMPORT_CLASSIFICATIONS = {
    "IMPORT_VALID_SAFE_TO_REQUEST_S006_PR_PILOT",
    "IMPORT_VALID_REMOTE_LIFECYCLE_BLOCKED",
    "IMPORT_VALID_READY_FOR_NEXT_TASK",
}
_TEMP_DIRS: list[Path] = []


def cleanup_temp_dirs() -> None:
    for path in _TEMP_DIRS:
        shutil.rmtree(path, ignore_errors=True)


atexit.register(cleanup_temp_dirs)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return read_json(path)


def kanban_db_path(kanban_home: Path, board_slug: str) -> Path:
    if board_slug == "default":
        return kanban_home / "kanban.db"
    return kanban_home / "kanban" / "boards" / board_slug / "kanban.db"


def board_dir(kanban_home: Path, board_slug: str) -> Path:
    return kanban_home / "kanban" / "boards" / board_slug


def _connect_uri(uri: str) -> sqlite3.Connection:
    conn = sqlite3.connect(uri, uri=True, timeout=1.0)
    conn.execute("PRAGMA query_only = ON;")
    return conn


def _temp_copy(path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="hermes-kanban-audit-"))
    _TEMP_DIRS.append(temp_dir)
    dest = temp_dir / path.name
    shutil.copy2(path, dest)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, Path(str(dest) + suffix))
    return dest


def connect_readonly(path: Path) -> sqlite3.Connection:
    """Connect read-only, falling back to immutable/temp-copy snapshots."""
    if not path.exists():
        raise FileNotFoundError(f"Kanban DB not found: {path}")
    errors: list[str] = []
    for label, uri in (
        ("immutable", f"file:{path}?mode=ro&immutable=1"),
        ("read_only", f"file:{path}?mode=ro"),
    ):
        try:
            return _connect_uri(uri)
        except sqlite3.OperationalError as exc:
            errors.append(f"{label}: {exc}")
    try:
        snapshot = _temp_copy(path)
        return _connect_uri(f"file:{snapshot}?mode=ro&immutable=1")
    except sqlite3.OperationalError as exc:
        errors.append(f"temp_copy: {exc}")
        raise sqlite3.OperationalError("; ".join(errors)) from exc


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type='table' and name=?", (table,)
    ).fetchone()
    return row is not None


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not table_exists(conn, table):
        return set()
    return {str(row[1]) for row in conn.execute(f"pragma table_info({table})")}


def select_rows(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    if not table_exists(conn, table):
        return []
    cursor = conn.execute(f"select * from {table}")
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def preview_cards(preview: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cards = preview.get("cards") or preview.get("tasks") or []
    if isinstance(cards, dict):
        return {str(k): v for k, v in cards.items() if isinstance(v, dict)}
    result: dict[str, dict[str, Any]] = {}
    for card in cards:
        if not isinstance(card, dict):
            continue
        card_id = str(card.get("card_id") or card.get("id") or "")
        if card_id:
            result[card_id] = card
    return result


def expected_status(card: dict[str, Any]) -> str | None:
    for key in (
        "imported_status",
        "native_status",
        "kanban_status",
        "status",
        "initial_status",
    ):
        value = card.get(key)
        if value:
            return str(value)
    return None


def dependency_links(preview: dict[str, Any]) -> set[tuple[str, str]]:
    links: set[tuple[str, str]] = set()
    raw_links = preview.get("dependencies") or preview.get("dependency_links") or []
    for item in raw_links:
        if not isinstance(item, dict):
            continue
        parent = str(item.get("parent") or item.get("parent_id") or "")
        child = str(item.get("child") or item.get("child_id") or "")
        if parent and child:
            links.add((parent, child))
    for card_id, card in preview_cards(preview).items():
        for parent in card.get("dependencies") or []:
            if isinstance(parent, str) and parent:
                links.add((parent, card_id))
            elif isinstance(parent, dict):
                p = str(parent.get("parent") or parent.get("parent_id") or "")
                c = str(parent.get("child") or parent.get("child_id") or card_id)
                if p and c:
                    links.add((p, c))
        for item in card.get("parent_links") or []:
            if isinstance(item, dict):
                parent = str(item.get("parent") or item.get("parent_id") or "")
                child = str(item.get("child") or item.get("child_id") or card_id)
            else:
                parent = str(item)
                child = card_id
            if parent and child:
                links.add((parent, child))
    return links


def normalize_text(*parts: Any) -> str:
    return "\n".join(str(part or "") for part in parts)


def evidence_text(
    task: dict[str, Any] | None,
    comments_by_task: dict[str, list[dict[str, Any]]],
) -> str:
    if not task:
        return ""
    task_id = str(task.get("id") or "")
    comment_text = "\n".join(
        str(c.get("body") or "") for c in comments_by_task.get(task_id, [])
    )
    return normalize_text(task.get("body"), task.get("result"), comment_text)


def extract_json_paths(text: str) -> list[Path]:
    matches = re.findall(r"(/[^`'\"\s]+\.json)", text)
    return [Path(match.rstrip(".,);]")) for match in matches]


def completion_gate_path_for(
    task: dict[str, Any],
    comments_by_task: dict[str, list[dict[str, Any]]],
    preview_card: dict[str, Any] | None,
) -> Path | None:
    metadata = (preview_card or {}).get("evidence_metadata")
    if isinstance(metadata, dict):
        raw = metadata.get("completion_gate_json_path") or metadata.get(
            "completion_gate_path"
        )
        if raw:
            return Path(str(raw))
    text = evidence_text(task, comments_by_task)
    for path in extract_json_paths(text):
        if "completion-gates" in str(path) or "completion_gate" in path.name:
            return path
    paths = extract_json_paths(text)
    return paths[0] if paths else None


def gate_passed(gate: dict[str, Any]) -> bool:
    conclusion = str(gate.get("conclusion") or gate.get("final_conclusion") or "")
    return gate.get("gate_passed") is True and conclusion in PASSING_CONCLUSIONS


def gate_task_id(gate: dict[str, Any]) -> str | None:
    value = gate.get("task_id") or gate.get("session_id")
    return str(value) if value else None


def branch_head(gate: dict[str, Any]) -> tuple[str | None, str | None]:
    branch = gate.get("target_branch") or gate.get("branch")
    head = gate.get("target_full_head") or gate.get("target_head") or gate.get("head")
    return (str(branch) if branch else None, str(head) if head else None)


def audit_done_cards(
    *,
    tasks: dict[str, dict[str, Any]],
    comments_by_task: dict[str, list[dict[str, Any]]],
    preview_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    card_results: list[dict[str, Any]] = []
    blockers: list[str] = []
    for task_id, task in sorted(tasks.items()):
        status = str(task.get("status") or "").lower()
        if status != "done":
            continue
        preview_card = preview_by_id.get(task_id)
        gate_path = completion_gate_path_for(task, comments_by_task, preview_card)
        result: dict[str, Any] = {
            "card_id": task_id,
            "completion_gate_path": str(gate_path) if gate_path else None,
            "gate_json_exists": bool(gate_path and gate_path.exists()),
            "gate_pass": False,
            "task_id_matches": False,
            "branch_head_matches_preview": None,
            "sidecar_evidence_exists": None,
            "ok": False,
        }
        if not gate_path:
            blockers.append(f"{task_id} is done without a completion-gate JSON path")
            card_results.append(result)
            continue
        if not gate_path.exists():
            blockers.append(f"{task_id} completion-gate JSON is missing: {gate_path}")
            card_results.append(result)
            continue
        try:
            gate = read_json(gate_path)
        except json.JSONDecodeError as exc:
            blockers.append(
                f"{task_id} completion-gate JSON is invalid: {gate_path}: {exc}"
            )
            card_results.append(result)
            continue
        result["gate_pass"] = gate_passed(gate)
        result["task_id_matches"] = gate_task_id(gate) == task_id
        branch, head = branch_head(gate)
        preview_meta = (preview_card or {}).get("evidence_metadata")
        if isinstance(preview_meta, dict):
            expected_branch = preview_meta.get("branch")
            expected_head = preview_meta.get("head")
            if expected_branch or expected_head:
                result["branch_head_matches_preview"] = (
                    (not expected_branch or expected_branch == branch)
                    and (not expected_head or expected_head == head)
                )
        sidecar_path = None
        sidecar = gate.get("sidecar_result")
        if isinstance(sidecar, dict) and sidecar.get("path"):
            sidecar_path = Path(str(sidecar["path"]))
            result["sidecar_evidence_exists"] = sidecar_path.exists()
        if not result["gate_pass"]:
            blockers.append(f"{task_id} completion gate is not PASS: {gate_path}")
        if not result["task_id_matches"]:
            blockers.append(f"{task_id} completion gate task_id does not match card")
        if result["branch_head_matches_preview"] is False:
            blockers.append(
                f"{task_id} completion gate branch/head does not match card evidence"
            )
        if result["sidecar_evidence_exists"] is False:
            blockers.append(f"{task_id} sidecar evidence is missing: {sidecar_path}")
        result["ok"] = (
            bool(result["gate_pass"])
            and bool(result["task_id_matches"])
            and result["branch_head_matches_preview"] is not False
            and result["sidecar_evidence_exists"] is not False
        )
        card_results.append(result)
    return {
        "done_card_count": len(card_results),
        "cards": card_results,
        "ok": not blockers,
        "blockers": blockers,
    }


def s006_local_status(
    preview_by_id: dict[str, dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    comments_by_task: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    task = tasks.get("S006")
    preview_card = preview_by_id.get("S006", {})
    gate_path = completion_gate_path_for(
        task or {"id": "S006"},
        comments_by_task,
        preview_card,
    )
    gate = maybe_read_json(gate_path)
    return {
        "card_exists": task is not None,
        "card_status": task.get("status") if task else None,
        "completion_gate_path": str(gate_path) if gate_path else None,
        "gate_json_exists": bool(gate_path and gate_path.exists()),
        "gate_pass": bool(gate and gate_passed(gate)),
        "local_complete": bool(gate and gate_passed(gate)),
    }


def s006_remote_status(
    preview_by_id: dict[str, dict[str, Any]],
    remote_readiness: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = preview_by_id.get("S006", {}).get("evidence_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    pr_exists = bool(metadata.get("pr_exists"))
    ci_ready = False
    merge_ready = False
    if remote_readiness:
        ci_ready = bool(remote_readiness.get("ci_evidence_ready"))
        merge_ready = bool(
            remote_readiness.get("merge_readiness_ready")
            or remote_readiness.get("remote_readiness_ready")
            or remote_readiness.get("merge_ready")
        )
        pr_exists = bool(remote_readiness.get("pr_exists", pr_exists))
        state = remote_readiness.get("s006_remote_lifecycle_state")
        if state:
            remote_done = bool(pr_exists and ci_ready and merge_ready)
            return {
                "pr_exists": pr_exists,
                "ci_evidence_ready": ci_ready,
                "merge_readiness_ready": merge_ready,
                "remote_done": remote_done,
                "remote_lifecycle_state": str(state),
            }
    remote_done = bool(pr_exists and ci_ready and merge_ready)
    if not pr_exists:
        state = "pr_absent"
    elif not ci_ready:
        state = "pr_created_ci_pending"
    elif not merge_ready:
        state = "pr_created_ci_passed_merge_pending"
    else:
        state = "remote_lifecycle_complete"
    return {
        "pr_exists": pr_exists,
        "ci_evidence_ready": ci_ready,
        "merge_readiness_ready": merge_ready,
        "remote_done": remote_done,
        "remote_lifecycle_state": state,
    }


def explicit_remote_blocker_present(
    tasks: dict[str, dict[str, Any]],
    comments_by_task: dict[str, list[dict[str, Any]]],
    links: set[tuple[str, str]],
) -> bool:
    for task_id, task in tasks.items():
        haystack = evidence_text(task, comments_by_task).lower()
        title = str(task.get("title") or "").lower()
        status = str(task.get("status") or "").lower()
        mentions_remote = (
            "s006" in haystack + title
            and any(
                word in haystack + title
                for word in ("pr", "ci", "merge", "remote readiness")
            )
        )
        if mentions_remote and (
            status in BLOCKED_STATUSES or (task_id, "S007A") in links
        ):
            return True
    s007a = tasks.get("S007A")
    if s007a:
        text = evidence_text(s007a, comments_by_task).lower()
        return "s006" in text and any(
            word in text for word in ("pr", "ci", "merge", "remote readiness")
        )
    return False


def s006_card_safety(
    tasks: dict[str, dict[str, Any]],
    comments_by_task: dict[str, list[dict[str, Any]]],
    links: set[tuple[str, str]],
    remote: dict[str, Any],
) -> dict[str, Any]:
    task = tasks.get("S006")
    if not task:
        return {"safe": False, "reason": "S006 card is missing"}
    status = str(task.get("status") or "").lower()
    blocker_present = explicit_remote_blocker_present(tasks, comments_by_task, links)
    plain_done = status == "done"
    safe = True
    reason = "S006 is not plain done"
    if plain_done and not remote.get("remote_done") and not blocker_present:
        safe = False
        reason = "S006 is plain done without explicit remote lifecycle blocker"
    elif plain_done and blocker_present:
        reason = "S006 is local done with explicit remote lifecycle blocker"
    return {
        "safe": safe,
        "card_status": task.get("status"),
        "plain_done": plain_done,
        "explicit_remote_blocker_present": blocker_present,
        "reason": reason,
    }


def s007a_dispatch_safety(
    tasks: dict[str, dict[str, Any]],
    links: set[tuple[str, str]],
    runs_by_task: dict[str, list[dict[str, Any]]],
    remote: dict[str, Any],
) -> dict[str, Any]:
    task = tasks.get("S007A")
    if not task:
        return {
            "exists": False,
            "dispatch_safe": False,
            "reason": "S007A card is missing",
        }
    status = str(task.get("status") or "").lower()
    linked_to_s006 = ("S006", "S007A") in links
    has_runs = bool(runs_by_task.get("S007A"))
    blocked = status in BLOCKED_STATUSES
    dispatch_safe = blocked and not has_runs
    reason = "S007A is blocked and has no runs"
    if not remote.get("remote_done") and not blocked:
        dispatch_safe = False
        reason = "S007A is dispatchable before S006 remote lifecycle is complete"
    elif has_runs:
        dispatch_safe = False
        reason = "S007A has worker run history"
    return {
        "exists": True,
        "status": task.get("status"),
        "blocked": blocked,
        "linked_to_s006": linked_to_s006,
        "has_runs": has_runs,
        "dispatch_safe": dispatch_safe,
        "reason": reason,
    }


def blocked_card_safety(
    tasks: dict[str, dict[str, Any]],
    card_id: str,
) -> dict[str, Any]:
    task = tasks.get(card_id)
    if not task:
        return {
            "exists": False,
            "blocked": False,
            "status": None,
            "safe": False,
            "reason": f"{card_id} card is missing",
        }
    status = str(task.get("status") or "").lower()
    blocked = status in BLOCKED_STATUSES
    return {
        "exists": True,
        "blocked": blocked,
        "status": task.get("status"),
        "safe": blocked,
        "reason": (
            f"{card_id} is blocked"
            if blocked
            else f"{card_id} is not blocked: {task.get('status')}"
        ),
    }


def worker_dispatch_status(
    tasks: dict[str, dict[str, Any]],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    active_tasks = [
        task_id
        for task_id, task in tasks.items()
        if str(task.get("status") or "").lower() in RUNNING_STATUSES
        or task.get("current_run_id")
        or task.get("worker_pid")
    ]
    return {
        "worker_dispatch_detected": bool(runs or active_tasks),
        "run_count": len(runs),
        "active_task_ids": sorted(active_tasks),
    }


def evaluate_kanban_import_audit(
    *,
    preview_path: Path,
    board_slug: str = "crypto_bot",
    kanban_home: Path = DEFAULT_KANBAN_HOME,
    remote_readiness_path: Path | None = None,
    expected_card_count: int = 90,
    expected_dependency_count: int | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    preview = read_json(preview_path)
    preview_by_id = preview_cards(preview)
    expected_ids = set(preview_by_id) or {
        str(i) for i in preview.get("card_ids", [])
    }
    expected_links = dependency_links(preview)
    db_path = kanban_db_path(kanban_home, board_slug)
    bdir = board_dir(kanban_home, board_slug)
    board_exists = db_path.exists()
    tasks: dict[str, dict[str, Any]] = {}
    comments: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    actual_links: set[tuple[str, str]] = set()
    if not board_exists:
        blockers.append(f"Native Kanban board is missing: {board_slug} ({db_path})")
    else:
        try:
            with connect_readonly(db_path) as conn:
                rows = select_rows(conn, "tasks")
                tasks = {str(row.get("id")): row for row in rows if row.get("id")}
                comments = select_rows(conn, "task_comments")
                runs = select_rows(conn, "task_runs")
                actual_links = {
                    (str(row.get("parent_id")), str(row.get("child_id")))
                    for row in select_rows(conn, "task_links")
                    if row.get("parent_id") and row.get("child_id")
                }
        except sqlite3.Error as exc:
            blockers.append(f"Unable to read board DB read-only: {db_path}: {exc}")
            warnings.append("Kanban DB read failed; board audit is incomplete")
    actual_ids = set(tasks)
    missing_cards = sorted(expected_ids - actual_ids)
    unexpected_cards = sorted(actual_ids - expected_ids)
    missing_links = sorted(expected_links - actual_links)
    unexpected_links = sorted(actual_links - expected_links)
    if len(actual_ids) != expected_card_count:
        blockers.append(
            "Card count mismatch: "
            f"expected {expected_card_count}, found {len(actual_ids)}"
        )
    if (
        expected_dependency_count is not None
        and len(actual_links) != expected_dependency_count
    ):
        blockers.append(
            "Dependency count mismatch: "
            f"expected {expected_dependency_count}, found {len(actual_links)}"
        )
    elif (
        expected_dependency_count is None
        and len(actual_links) != len(expected_links)
    ):
        blockers.append(
            "Dependency count mismatch: "
            f"expected {len(expected_links)}, found {len(actual_links)}"
        )
    if missing_cards:
        blockers.append(f"Missing expected cards: {', '.join(missing_cards[:20])}")
    if unexpected_cards:
        blockers.append(f"Unexpected cards present: {', '.join(unexpected_cards[:20])}")
    if missing_links:
        blockers.append(f"Missing dependency links: {missing_links[:20]}")
    if unexpected_links:
        blockers.append(f"Unexpected dependency links: {unexpected_links[:20]}")
    status_mismatches = []
    for task_id in sorted(expected_ids & actual_ids):
        exp = expected_status(preview_by_id.get(task_id, {}))
        actual = str(tasks[task_id].get("status") or "")
        if exp and actual != exp:
            status_mismatches.append(
                {"card_id": task_id, "expected": exp, "actual": actual}
            )
    if status_mismatches:
        blockers.append(f"Imported status mismatches: {len(status_mismatches)}")
    comments_by_task: dict[str, list[dict[str, Any]]] = {}
    for comment in comments:
        comments_by_task.setdefault(str(comment.get("task_id") or ""), []).append(
            comment
        )
    runs_by_task: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        runs_by_task.setdefault(str(run.get("task_id") or ""), []).append(run)
    done_audit = audit_done_cards(
        tasks=tasks,
        comments_by_task=comments_by_task,
        preview_by_id=preview_by_id,
    )
    blockers.extend(done_audit["blockers"])
    remote_readiness = maybe_read_json(remote_readiness_path)
    s006_local = s006_local_status(preview_by_id, tasks, comments_by_task)
    s006_remote = s006_remote_status(preview_by_id, remote_readiness)
    s006_safety = s006_card_safety(tasks, comments_by_task, actual_links, s006_remote)
    if not s006_safety["safe"]:
        blockers.append(str(s006_safety["reason"]))
    s006_task = tasks.get("S006")
    s006_status = str((s006_task or {}).get("status") or "").lower()
    s006_lifecycle_status_ok = s006_status in {
        "review_required",
        "blocked_remote_lifecycle",
        "blocked_remote_pr_missing",
    } or (
        s006_status == "done"
        and s006_safety.get("explicit_remote_blocker_present") is True
    )
    if s006_task and not s006_lifecycle_status_ok:
        blockers.append(
            "S006 is not review_required or explicitly remote-lifecycle-blocked"
        )
    s006a_safety = blocked_card_safety(tasks, "S006A")
    if not s006a_safety["safe"]:
        blockers.append(str(s006a_safety["reason"]))
    s007a_safety = s007a_dispatch_safety(tasks, actual_links, runs_by_task, s006_remote)
    if not s007a_safety["dispatch_safe"]:
        blockers.append(str(s007a_safety["reason"]))
    dispatch = worker_dispatch_status(tasks, runs)
    if dispatch["worker_dispatch_detected"]:
        blockers.append("Worker dispatch/run history detected after import")
    if (
        not board_exists
        or len(actual_ids) != expected_card_count
        or missing_cards
        or unexpected_cards
        or missing_links
    ):
        classification = "IMPORT_FAILED_OR_PARTIAL"
    elif blockers:
        classification = "IMPORT_NEEDS_BOARD_STATE_REPAIR"
    elif s006_remote["remote_done"]:
        classification = "IMPORT_VALID_READY_FOR_NEXT_TASK"
    elif s006_remote["pr_exists"]:
        classification = "IMPORT_VALID_REMOTE_LIFECYCLE_BLOCKED"
    else:
        classification = "IMPORT_VALID_SAFE_TO_REQUEST_S006_PR_PILOT"
    if warnings and not board_exists and not blockers:
        classification = "UNKNOWN_DUE_TO_TOOLING_LIMITATION"
    status_counts: dict[str, int] = {}
    for task in tasks.values():
        status = str(task.get("status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
    recommended = {
        "IMPORT_VALID_SAFE_TO_REQUEST_S006_PR_PILOT": (
            "Request exact Operator approval for the S006 PR pilot retry."
        ),
        "IMPORT_NEEDS_BOARD_STATE_REPAIR": (
            "Repair native Kanban lifecycle state before autonomous work resumes."
        ),
        "IMPORT_FAILED_OR_PARTIAL": (
            "Reconcile or re-import the native crypto_bot board before dispatch "
            "or PR work."
        ),
        "UNKNOWN_DUE_TO_TOOLING_LIMITATION": (
            "Improve read-only board audit tooling, then rerun the audit."
        ),
    }[classification]
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "board_exists": board_exists,
        "board_slug": board_slug,
        "kanban_home": str(kanban_home),
        "board_dir": str(bdir),
        "board_db_path": str(db_path),
        "card_count": len(actual_ids),
        "dependency_count": len(actual_links),
        "preview_card_count": len(expected_ids),
        "preview_dependency_count": len(expected_links),
        "expected_card_count": expected_card_count,
        "expected_dependency_count": expected_dependency_count or len(expected_links),
        "columns_status_counts": status_counts,
        "unexpected_cards": unexpected_cards,
        "missing_cards": missing_cards,
        "unexpected_links": [{"parent": p, "child": c} for p, c in unexpected_links],
        "missing_links": [{"parent": p, "child": c} for p, c in missing_links],
        "status_mismatches": status_mismatches,
        "evidence_metadata_preserved": not missing_cards and not unexpected_cards,
        "done_card_evidence_audit": done_audit,
        "s006_local_completion_status": s006_local,
        "s006_remote_completion_status": s006_remote,
        "s006_card_state_safety": s006_safety,
        "s006_lifecycle_status_ok": s006_lifecycle_status_ok,
        "s006a_blocked_safety": s006a_safety,
        "s007a_blocked_dispatch_safety": s007a_safety,
        "worker_dispatch_status": dispatch,
        "blockers": blockers,
        "warnings": warnings,
        "recommended_next_action": recommended,
        "classification": classification,
    }


def durable_audit_path(state_root: Path, board_slug: str) -> Path:
    audit_dir = state_root / "kanban-import-audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = audit_dir / f"{timestamp}-{board_slug}-kanban-import-audit.json"
    if not base.exists():
        return base
    for idx in range(1, 100):
        candidate = audit_dir / (
            f"{timestamp}-{board_slug}-kanban-import-audit-{idx}.json"
        )
        if not candidate.exists():
            return candidate
    raise RuntimeError("unable to allocate durable Kanban audit path")


def write_durable_audit(payload: dict[str, Any], state_root: Path) -> Path:
    path = durable_audit_path(state_root, str(payload.get("board_slug") or "board"))
    payload["audit_json_path"] = str(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only audit for the native Hermes crypto_bot Kanban import."
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=DEFAULT_STATE_ROOT / "kanban-import-previews/crypto_bot-preview.json",
    )
    parser.add_argument("--board-slug", default="crypto_bot")
    parser.add_argument(
        "--kanban-home",
        type=Path,
        default=Path(os.environ.get("HERMES_KANBAN_HOME", DEFAULT_KANBAN_HOME)),
    )
    parser.add_argument("--remote-readiness-json", type=Path)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--no-write-audit", action="store_true")
    parser.add_argument("--expected-card-count", type=int, default=90)
    parser.add_argument("--expected-dependency-count", type=int)
    parser.add_argument("--format", choices=["json"], default="json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = evaluate_kanban_import_audit(
        preview_path=args.preview,
        board_slug=args.board_slug,
        kanban_home=args.kanban_home,
        remote_readiness_path=args.remote_readiness_json,
        expected_card_count=args.expected_card_count,
        expected_dependency_count=args.expected_dependency_count,
    )
    if not args.no_write_audit:
        write_durable_audit(payload, args.state_root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["classification"] in VALID_IMPORT_CLASSIFICATIONS else 1


if __name__ == "__main__":
    raise SystemExit(main())
