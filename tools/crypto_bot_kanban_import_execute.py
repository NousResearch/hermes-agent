#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterator


SCHEMA = "hermes.autonomy.crypto_bot_kanban_import_execute.v1"
PREVIEW_SCHEMA = "hermes.autonomy.crypto_bot_kanban_import_preview.v1"
DEFAULT_STATE_ROOT = Path("/Users/preston/.local/state/hermes-operator")
DEFAULT_KANBAN_HOME = Path("/Users/preston/.hermes")
DEFAULT_PREVIEW = DEFAULT_STATE_ROOT / "kanban-import-previews/crypto_bot-preview.json"
SAFE_S006_STATUSES = {"review_required", "blocked_remote_pr_missing", "blocked"}
BLOCKED_STATUSES = {"blocked", "review_required", "blocked_remote_pr_missing"}


class ImportBlocked(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def epoch_now() -> int:
    return int(dt.datetime.now(dt.timezone.utc).timestamp())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def board_dir(kanban_home: Path, board_slug: str) -> Path:
    return kanban_home / "kanban" / "boards" / board_slug


def board_db_path(kanban_home: Path, board_slug: str) -> Path:
    if board_slug == "default":
        return kanban_home / "kanban.db"
    return board_dir(kanban_home, board_slug) / "kanban.db"


def preview_cards(preview: dict[str, Any]) -> list[dict[str, Any]]:
    cards = preview.get("cards") or preview.get("tasks") or preview.get("items") or []
    if not isinstance(cards, list):
        raise ImportBlocked("preview cards must be a list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in cards:
        if not isinstance(item, dict):
            raise ImportBlocked("preview contains a non-object card")
        card_id = str(item.get("card_id") or item.get("id") or "").strip()
        if not card_id:
            raise ImportBlocked("preview contains a card without card_id")
        if card_id in seen:
            raise ImportBlocked(f"preview contains duplicate card_id: {card_id}")
        seen.add(card_id)
        result.append(item)
    return result


def dependency_links(preview: dict[str, Any]) -> set[tuple[str, str]]:
    raw_links = (
        preview.get("dependency_links")
        or preview.get("dependencies")
        or preview.get("links")
        or []
    )
    if not isinstance(raw_links, list):
        raise ImportBlocked("preview dependency links must be a list")
    links: set[tuple[str, str]] = set()
    for index, item in enumerate(raw_links):
        if isinstance(item, dict):
            parent = item.get("parent") or item.get("from") or item.get("depends_on")
            child = item.get("child") or item.get("to") or item.get("task")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            parent, child = item[0], item[1]
        else:
            raise ImportBlocked(f"dependency link {index} has unsupported shape")
        if parent is None or child is None:
            raise ImportBlocked(f"dependency link {index} is missing parent or child")
        parent_id = str(parent).strip()
        child_id = str(child).strip()
        if not parent_id or not child_id:
            raise ImportBlocked(f"dependency link {index} is missing parent or child")
        links.add((parent_id, child_id))
    return links


def s006_remote_done(card: dict[str, Any] | None) -> bool:
    metadata = (card or {}).get("evidence_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if metadata.get("remote_done") is True:
        return True
    return bool(
        metadata.get("pr_exists")
        and metadata.get("ci_evidence_ready")
        and (metadata.get("merge_ready") or metadata.get("merge_readiness_ready"))
    )


def import_status(
    card: dict[str, Any],
    cards_by_id: dict[str, dict[str, Any]],
) -> tuple[str, str | None]:
    card_id = str(card.get("card_id") or card.get("id") or "")
    status = str(card.get("initial_status") or card.get("status") or "blocked").strip()
    if card_id == "S006" and status == "done" and not s006_remote_done(card):
        return (
            "review_required",
            "S006 plain done downgraded because remote lifecycle is incomplete",
        )
    if card_id == "S006" and status == "done":
        return status, None
    if card_id == "S006" and status not in SAFE_S006_STATUSES:
        return "review_required", f"S006 unsafe status {status!r} downgraded"
    if card_id == "S007A" and not s006_remote_done(cards_by_id.get("S006")):
        if status not in BLOCKED_STATUSES:
            return (
                "blocked",
                "S007A blocked because S006 remote lifecycle is incomplete",
            )
    return status, None


def card_body(
    card: dict[str, Any],
    *,
    preview_path: Path,
    preview_sha256: str,
    effective_status: str,
) -> str:
    payload = {
        "preview_path": str(preview_path),
        "preview_sha256": preview_sha256,
        "card_id": card.get("card_id") or card.get("id"),
        "title": card.get("title") or card.get("summary"),
        "summary": card.get("summary"),
        "plan_status": card.get("plan_status"),
        "initial_status": card.get("initial_status"),
        "imported_status": effective_status,
        "status_reason": card.get("status_reason"),
        "assignee_lane": card.get("assignee_lane"),
        "operator_approval_required": card.get("operator_approval_required"),
        "dependencies": card.get("dependencies") or [],
        "parent_links": card.get("parent_links") or [],
        "allowed_write_scope": card.get("allowed_write_scope") or [],
        "forbidden_in_session": card.get("forbidden_in_session") or [],
        "validation_expectations": card.get("validation_expectations") or [],
        "workspace_strategy": card.get("workspace_strategy") or {},
        "evidence_metadata": card.get("evidence_metadata") or {},
        "unresolved_claims": card.get("unresolved_claims") or [],
    }
    return (
        "Imported from the verified crypto_bot native Kanban preview.\n"
        "No worker dispatch, product writes, Gitea mutation, PR/CI action, or merge "
        "was performed by this import.\n\n"
        + json.dumps(payload, indent=2, sort_keys=True)
    )


def comment_body(
    card: dict[str, Any],
    *,
    preview_sha256: str,
    effective_status: str,
) -> str:
    payload = {
        "preview_sha256": preview_sha256,
        "card_id": card.get("card_id") or card.get("id"),
        "imported_status": effective_status,
        "status_reason": card.get("status_reason"),
        "evidence_metadata": card.get("evidence_metadata") or {},
        "no_worker_dispatch": True,
    }
    return "Controlled crypto_bot board import metadata:\n" + json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )


def import_kanban_db_module() -> Any:
    try:
        module = importlib.import_module("hermes_cli.kanban_db")
    except TypeError as exc:
        if "unsupported operand type(s) for |" not in str(exc):
            raise
        raise ModuleNotFoundError(
            "installed Hermes native Kanban runtime is incompatible with this "
            "Python interpreter"
        ) from exc
    reason = kanban_runtime_incompatibility_reason(module)
    if reason:
        raise ModuleNotFoundError(reason)
    return module


def kanban_runtime_incompatibility_reason(module: Any) -> str | None:
    if sys.version_info >= (3, 10):
        return None
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return None
    runtime_root = Path(module_file).resolve().parents[1]
    registry_path = runtime_root / "tools" / "registry.py"
    if not registry_path.exists():
        return None
    registry_source = registry_path.read_text(encoding="utf-8")
    if "Callable | None" not in registry_source:
        return None
    return (
        "installed Hermes native Kanban runtime requires Python 3.10+ syntax "
        "in tools/registry.py"
    )


def load_kanban_db_module() -> Any:
    try:
        return import_kanban_db_module()
    except ModuleNotFoundError:
        candidate = Path.home() / ".hermes" / "hermes-agent"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
        return import_kanban_db_module()


@contextlib.contextmanager
def pinned_kanban_env(kanban_home: Path) -> Iterator[None]:
    keys = ("HERMES_KANBAN_HOME", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD")
    saved = {key: os.environ.get(key) for key in keys}
    os.environ["HERMES_KANBAN_HOME"] = str(kanban_home)
    os.environ.pop("HERMES_KANBAN_DB", None)
    os.environ.pop("HERMES_KANBAN_BOARD", None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def table_count(conn: sqlite3.Connection, table: str) -> int:
    exists = conn.execute(
        "select 1 from sqlite_master where type='table' and name=?", (table,)
    ).fetchone()
    if not exists:
        return 0
    return int(conn.execute(f"select count(*) from {table}").fetchone()[0])


def existing_board_summary(db_path: Path) -> dict[str, int]:
    if not db_path.exists():
        return {"tasks": 0, "links": 0, "comments": 0, "runs": 0, "events": 0}
    conn = sqlite3.connect(db_path)
    try:
        return {
            "tasks": table_count(conn, "tasks"),
            "links": table_count(conn, "task_links"),
            "comments": table_count(conn, "task_comments"),
            "runs": table_count(conn, "task_runs"),
            "events": table_count(conn, "task_events"),
        }
    finally:
        conn.close()


def validate_preview(
    *,
    preview_path: Path,
    expected_sha256: str,
    board_slug: str,
    expected_card_count: int,
    expected_dependency_count: int,
) -> dict[str, Any]:
    if not preview_path.exists():
        raise ImportBlocked(f"preview path is missing: {preview_path}")
    actual_sha = sha256_file(preview_path)
    if actual_sha != expected_sha256:
        raise ImportBlocked(
            f"preview SHA mismatch: expected {expected_sha256}, found {actual_sha}"
        )
    preview = read_json(preview_path)
    if preview.get("schema") != PREVIEW_SCHEMA:
        raise ImportBlocked(f"unexpected preview schema: {preview.get('schema')}")
    if preview.get("board_slug") != board_slug:
        raise ImportBlocked(
            "preview board slug mismatch: "
            f"expected {board_slug}, found {preview.get('board_slug')}"
        )
    safety = preview.get("import_safety") or {}
    if isinstance(safety, dict):
        if safety.get("blockers"):
            raise ImportBlocked(
                f"preview import_safety has blockers: {safety['blockers']}"
            )
        disallowed_flags = {
            "worker_dispatch_allowed": False,
            "product_file_writes_allowed": False,
            "gitea_mutation_allowed": False,
            "pr_creation_allowed": False,
            "merge_allowed": False,
        }
        for key, expected in disallowed_flags.items():
            if safety.get(key) is not expected:
                raise ImportBlocked(
                    f"preview import_safety.{key} must be {expected}"
                )
    cards = preview_cards(preview)
    if len(cards) != expected_card_count:
        raise ImportBlocked(
            f"card count mismatch: expected {expected_card_count}, found {len(cards)}"
        )
    links = dependency_links(preview)
    if len(links) != expected_dependency_count:
        raise ImportBlocked(
            "dependency count mismatch: "
            f"expected {expected_dependency_count}, found {len(links)}"
        )
    card_ids = {str(card.get("card_id") or card.get("id")) for card in cards}
    missing_link_cards = sorted(
        {item for link in links for item in link if item not in card_ids}
    )
    if missing_link_cards:
        raise ImportBlocked(
            f"dependency links reference unknown cards: {missing_link_cards}"
        )
    cards_by_id = {
        str(card.get("card_id") or card.get("id")): card for card in cards
    }
    status_overrides = []
    statuses: dict[str, str] = {}
    for card in cards:
        card_id = str(card.get("card_id") or card.get("id"))
        status, reason = import_status(card, cards_by_id)
        statuses[card_id] = status
        if reason:
            status_overrides.append(
                {"card_id": card_id, "status": status, "reason": reason}
            )
    if statuses.get("S006") == "done" and not s006_remote_done(
        cards_by_id.get("S006")
    ):
        raise ImportBlocked(
            "S006 would import as done while remote lifecycle is incomplete"
        )
    if statuses.get("S007A") not in BLOCKED_STATUSES and not s006_remote_done(
        cards_by_id.get("S006")
    ):
        raise ImportBlocked(
            "S007A would be dispatchable before S006 remote lifecycle completes"
        )
    return {
        "preview": preview,
        "preview_sha256": actual_sha,
        "cards": cards,
        "links": links,
        "statuses": statuses,
        "status_overrides": status_overrides,
    }


def insert_import_records(
    *,
    conn: sqlite3.Connection,
    cards: list[dict[str, Any]],
    links: set[tuple[str, str]],
    statuses: dict[str, str],
    preview_path: Path,
    preview_sha256: str,
) -> None:
    now = epoch_now()
    for index, card in enumerate(cards):
        card_id = str(card.get("card_id") or card.get("id"))
        status = statuses[card_id]
        title = str(card.get("title") or card.get("summary") or card_id)
        body = card_body(
            card,
            preview_path=preview_path,
            preview_sha256=preview_sha256,
            effective_status=status,
        )
        assignee = card.get("assignee_lane")
        conn.execute(
            """
            insert into tasks (
                id, title, body, assignee, status, priority, created_by, created_at,
                workspace_kind, workspace_path, tenant, result, idempotency_key
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                card_id,
                title,
                body,
                str(assignee) if assignee else None,
                status,
                index,
                "crypto_bot_kanban_import_execute",
                now,
                "scratch",
                None,
                "crypto_bot",
                None,
                f"crypto_bot:{card_id}",
            ),
        )
        conn.execute(
            """
            insert into task_comments (task_id, author, body, created_at)
            values (?, ?, ?, ?)
            """,
            (
                card_id,
                "crypto_bot_kanban_import_execute",
                comment_body(
                    card,
                    preview_sha256=preview_sha256,
                    effective_status=status,
                ),
                now,
            ),
        )
        event_payload = {
            "source": "crypto_bot_kanban_import_execute",
            "preview_sha256": preview_sha256,
            "status": status,
            "no_worker_dispatch": True,
        }
        conn.execute(
            """
            insert into task_events (task_id, run_id, kind, payload, created_at)
            values (?, NULL, ?, ?, ?)
            """,
            (
                card_id,
                "imported_from_preview",
                json.dumps(event_payload, sort_keys=True),
                now,
            ),
        )
    for parent, child in sorted(links):
        conn.execute(
            "insert into task_links (parent_id, child_id) values (?, ?)",
            (parent, child),
        )


def status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "select status, count(*) from tasks group by status"
    ).fetchall()
    return {str(status): int(count) for status, count in rows}


def execute_import(
    *,
    preview_path: Path,
    expected_sha256: str,
    board_slug: str,
    expected_card_count: int,
    expected_dependency_count: int,
    kanban_home: Path = DEFAULT_KANBAN_HOME,
    state_root: Path = DEFAULT_STATE_ROOT,
    execute: bool = False,
) -> dict[str, Any]:
    validation = validate_preview(
        preview_path=preview_path,
        expected_sha256=expected_sha256,
        board_slug=board_slug,
        expected_card_count=expected_card_count,
        expected_dependency_count=expected_dependency_count,
    )
    db_path = board_db_path(kanban_home, board_slug)
    summary_before = existing_board_summary(db_path)
    existing_content = sum(summary_before.values())
    if db_path.exists() and existing_content:
        raise ImportBlocked(
            "conflicting non-empty board DB already exists: "
            f"{db_path}: {summary_before}"
        )
    if not db_path.exists() and board_dir(kanban_home, board_slug).exists():
        board_json = board_dir(kanban_home, board_slug) / "board.json"
        allowed_files = {board_json} if board_json.exists() else set()
        present_files = {
            path
            for path in board_dir(kanban_home, board_slug).rglob("*")
            if path.is_file()
        }
        unexpected = sorted(str(path) for path in present_files - allowed_files)
        if unexpected:
            raise ImportBlocked(f"board directory has unexpected files: {unexpected}")
    mutation_performed = False
    final_counts = {
        "tasks": 0,
        "links": 0,
        "comments": 0,
        "runs": 0,
        "events": 0,
    }
    final_status_counts: dict[str, int] = {}
    if execute:
        kb = load_kanban_db_module()
        with pinned_kanban_env(kanban_home):
            kb.create_board(
                board_slug,
                name="Crypto Bot",
                description="Controlled native import from verified crypto_bot preview",
                icon="C",
                color="#2563eb",
            )
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("pragma foreign_keys=on")
                conn.execute("begin immediate")
                if table_count(conn, "tasks") or table_count(conn, "task_links"):
                    raise ImportBlocked(
                        "board became non-empty before import transaction"
                    )
                insert_import_records(
                    conn=conn,
                    cards=validation["cards"],
                    links=validation["links"],
                    statuses=validation["statuses"],
                    preview_path=preview_path,
                    preview_sha256=validation["preview_sha256"],
                )
                actual_cards = table_count(conn, "tasks")
                actual_links = table_count(conn, "task_links")
                actual_runs = table_count(conn, "task_runs")
                if actual_cards != expected_card_count:
                    raise ImportBlocked(
                        f"post-insert card count mismatch: {actual_cards}"
                    )
                if actual_links != expected_dependency_count:
                    raise ImportBlocked(
                        f"post-insert dependency count mismatch: {actual_links}"
                    )
                if actual_runs:
                    raise ImportBlocked("post-insert run history exists unexpectedly")
                final_status_counts = status_counts(conn)
                conn.commit()
                mutation_performed = True
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        final_counts = existing_board_summary(db_path)
    result = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "execute": execute,
        "mutation_performed": mutation_performed,
        "board_slug": board_slug,
        "kanban_home": str(kanban_home),
        "board_db_path": str(db_path),
        "preview_path": str(preview_path),
        "preview_sha256": validation["preview_sha256"],
        "expected_card_count": expected_card_count,
        "expected_dependency_count": expected_dependency_count,
        "validated_card_count": len(validation["cards"]),
        "validated_dependency_count": len(validation["links"]),
        "existing_board_summary_before": summary_before,
        "final_board_summary": final_counts,
        "final_status_counts": final_status_counts,
        "status_overrides": validation["status_overrides"],
        "worker_dispatch_performed": False,
        "product_file_writes_performed": False,
        "gitea_mutation_performed": False,
        "pr_or_ci_action_performed": False,
        "classification": "IMPORT_EXECUTED" if execute else "DRY_RUN_VALID",
    }
    imports_dir = state_root / "kanban-imports"
    imports_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = imports_dir / f"{stamp}-{board_slug}-import.json"
    result["import_result_path"] = str(out_path)
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled native Hermes Kanban import for the crypto_bot board."
    )
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--board-slug", required=True)
    parser.add_argument("--expected-card-count", type=int, required=True)
    parser.add_argument("--expected-dependency-count", type=int, required=True)
    parser.add_argument("--kanban-home", type=Path, default=DEFAULT_KANBAN_HOME)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--no-worker-dispatch", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--format", choices=["json"], default="json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.no_worker_dispatch:
        payload = {
            "schema": SCHEMA,
            "generated_at": utc_now(),
            "classification": "IMPORT_BLOCKED",
            "blocker": "--no-worker-dispatch is required",
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    try:
        payload = execute_import(
            preview_path=args.preview,
            expected_sha256=args.expected_sha256,
            board_slug=args.board_slug,
            expected_card_count=args.expected_card_count,
            expected_dependency_count=args.expected_dependency_count,
            kanban_home=args.kanban_home,
            state_root=args.state_root,
            execute=args.execute,
        )
    except ImportBlocked as exc:
        payload = {
            "schema": SCHEMA,
            "generated_at": utc_now(),
            "classification": "IMPORT_BLOCKED",
            "blocker": str(exc),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
