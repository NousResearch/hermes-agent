#!/usr/bin/env python3
"""Codex app-server bridge tool.

This module talks to Codex's native app-server protocol over stdio JSON-RPC.
State is persisted for status/recovery, but communication never uses mailbox,
inbox, or outbox files.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error


CODEX_BRIDGE_DB = "codex_bridge.db"
DEFAULT_APPROVAL_POLICY = "untrusted"
DEFAULT_SANDBOX = "read-only"
EVENT_TAIL_LIMIT = 20
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def check_codex_bridge_requirements() -> bool:
    return shutil.which("codex") is not None


def _now() -> float:
    return time.time()


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _text_input(text: str) -> List[Dict[str, Any]]:
    return [{"type": "text", "text": text, "text_elements": []}]


def _summarize_payload(payload: Any, max_chars: int = 1200) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        text = payload
    else:
        text = _json_dumps(payload)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "..."


def _normalize_status(method: str, params: Dict[str, Any]) -> Optional[str]:
    if method == "turn/started":
        return "working"
    if method == "turn/completed":
        turn = params.get("turn") or {}
        status = turn.get("status")
        if status == "failed":
            return "failed"
        if status == "cancelled":
            return "cancelled"
        return "completed"
    if method == "thread/status/changed":
        status = params.get("status") or {}
        if status.get("type") == "systemError":
            return "failed"
        if status.get("type") == "active":
            return "working"
    if method == "error":
        return "failed" if not params.get("willRetry") else "working"
    return None


class CodexBridgeStore:
    def __init__(self, db_path: Optional[Path] = None):
        home = get_hermes_home()
        self.db_path = db_path or (home / CODEX_BRIDGE_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS codex_bridge_tasks (
                    hermes_task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    prompt_summary TEXT,
                    codex_thread_id TEXT,
                    codex_turn_id TEXT,
                    cwd TEXT,
                    model TEXT,
                    sandbox TEXT,
                    approval_policy TEXT,
                    degraded_mode TEXT,
                    last_progress_summary TEXT,
                    final_summary TEXT,
                    error_summary TEXT,
                    notify_target TEXT,
                    notification_status TEXT,
                    notified_at REAL,
                    notification_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL
                );

                CREATE TABLE IF NOT EXISTS codex_bridge_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hermes_task_id TEXT NOT NULL,
                    codex_thread_id TEXT,
                    codex_turn_id TEXT,
                    source_event_type TEXT NOT NULL,
                    normalized_status TEXT,
                    payload_summary TEXT,
                    payload_json TEXT,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS codex_bridge_pending_requests (
                    request_id TEXT NOT NULL,
                    hermes_task_id TEXT NOT NULL,
                    method TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    resolved_at REAL,
                    response_json TEXT,
                    PRIMARY KEY (request_id, hermes_task_id)
                );
                """
            )
            self._ensure_task_columns(conn)

    def _ensure_task_columns(self, conn: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(codex_bridge_tasks)").fetchall()
        }
        migrations = {
            "notify_target": "ALTER TABLE codex_bridge_tasks ADD COLUMN notify_target TEXT",
            "notification_status": "ALTER TABLE codex_bridge_tasks ADD COLUMN notification_status TEXT",
            "notified_at": "ALTER TABLE codex_bridge_tasks ADD COLUMN notified_at REAL",
            "notification_error": "ALTER TABLE codex_bridge_tasks ADD COLUMN notification_error TEXT",
        }
        for column, statement in migrations.items():
            if column not in existing:
                conn.execute(statement)

    def upsert_task(self, task: "CodexBridgeTask") -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO codex_bridge_tasks (
                    hermes_task_id, status, prompt_summary, codex_thread_id,
                    codex_turn_id, cwd, model, sandbox, approval_policy,
                    degraded_mode, last_progress_summary, final_summary,
                    error_summary, notify_target, notification_status, notified_at,
                    notification_error, created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hermes_task_id) DO UPDATE SET
                    status=excluded.status,
                    codex_thread_id=excluded.codex_thread_id,
                    codex_turn_id=excluded.codex_turn_id,
                    last_progress_summary=excluded.last_progress_summary,
                    final_summary=excluded.final_summary,
                    error_summary=excluded.error_summary,
                    notify_target=excluded.notify_target,
                    notification_status=excluded.notification_status,
                    notified_at=excluded.notified_at,
                    notification_error=excluded.notification_error,
                    updated_at=excluded.updated_at,
                    completed_at=excluded.completed_at
                """,
                (
                    task.hermes_task_id,
                    task.status,
                    task.prompt_summary,
                    task.codex_thread_id,
                    task.codex_turn_id,
                    task.cwd,
                    task.model,
                    task.sandbox,
                    task.approval_policy,
                    task.degraded_mode,
                    task.last_progress_summary,
                    task.final_summary,
                    task.error_summary,
                    task.notify_target,
                    task.notification_status,
                    task.notified_at,
                    task.notification_error,
                    task.created_at,
                    task.updated_at,
                    task.completed_at,
                ),
            )

    def insert_event(
        self,
        task_id: str,
        thread_id: Optional[str],
        turn_id: Optional[str],
        method: str,
        normalized_status: Optional[str],
        payload: Any,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO codex_bridge_events (
                    hermes_task_id, codex_thread_id, codex_turn_id,
                    source_event_type, normalized_status, payload_summary,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    thread_id,
                    turn_id,
                    method,
                    normalized_status,
                    _summarize_payload(payload),
                    _json_dumps(payload),
                    _now(),
                ),
            )

    def upsert_pending_request(
        self,
        task_id: str,
        request_id: str,
        method: str,
        payload: Any,
        status: str = "pending",
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO codex_bridge_pending_requests (
                    request_id, hermes_task_id, method, payload_json,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id, hermes_task_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    status=excluded.status
                """,
                (request_id, task_id, method, _json_dumps(payload), status, _now()),
            )

    def resolve_pending_request(
        self,
        task_id: str,
        request_id: str,
        response: Any,
        status: str = "resolved",
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE codex_bridge_pending_requests
                SET status=?, resolved_at=?, response_json=?
                WHERE hermes_task_id=? AND request_id=?
                """,
                (status, _now(), _json_dumps(response), task_id, request_id),
            )

    def get_task_snapshot(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM codex_bridge_tasks WHERE hermes_task_id=?",
                (task_id,),
            ).fetchone()
            if not row:
                return None
            events = conn.execute(
                """
                SELECT source_event_type, normalized_status, payload_summary, created_at
                FROM codex_bridge_events
                WHERE hermes_task_id=?
                ORDER BY id DESC LIMIT ?
                """,
                (task_id, EVENT_TAIL_LIMIT),
            ).fetchall()
            pending = conn.execute(
                """
                SELECT request_id, method, payload_json, status, created_at
                FROM codex_bridge_pending_requests
                WHERE hermes_task_id=? AND status='pending'
                ORDER BY created_at ASC
                """,
                (task_id,),
            ).fetchall()
        snap = dict(row)
        snap["recent_events"] = [dict(r) for r in reversed(events)]
        snap["pending_requests"] = [
            {
                "request_id": r["request_id"],
                "method": r["method"],
                "payload": json.loads(r["payload_json"]),
                "status": r["status"],
                "created_at": r["created_at"],
            }
            for r in pending
        ]
        return snap

    def list_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT hermes_task_id, status, prompt_summary, codex_thread_id,
                       codex_turn_id, last_progress_summary, final_summary,
                       error_summary, notify_target, notification_status,
                       notified_at, notification_error, created_at, updated_at,
                       completed_at
                FROM codex_bridge_tasks
                ORDER BY updated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_completed_for_notification(self, limit: int = 10) -> List[Dict[str, Any]]:
        placeholders = ",".join("?" for _ in TERMINAL_STATUSES)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM codex_bridge_tasks
                WHERE status IN ({placeholders})
                  AND (
                    notification_status IS NULL
                    OR notification_status='pending'
                    OR notification_status='failed'
                  )
                ORDER BY completed_at ASC, updated_at ASC
                LIMIT ?
                """,
                (*sorted(TERMINAL_STATUSES), limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_notification_status(
        self,
        task_id: str,
        status: str,
        *,
        notified_at: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE codex_bridge_tasks
                SET notification_status=?, notified_at=?, notification_error=?, updated_at=?
                WHERE hermes_task_id=?
                """,
                (status, notified_at, error, _now(), task_id),
            )


@dataclass
class CodexBridgeTask:
    hermes_task_id: str
    prompt_summary: str
    cwd: str
    model: Optional[str]
    sandbox: str
    approval_policy: str
    status: str = "starting"
    codex_thread_id: Optional[str] = None
    codex_turn_id: Optional[str] = None
    degraded_mode: str = "none"
    last_progress_summary: Optional[str] = None
    final_summary: Optional[str] = None
    error_summary: Optional[str] = None
    notify_target: Optional[str] = None
    notification_status: Optional[str] = None
    notified_at: Optional[float] = None
    notification_error: Optional[str] = None
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    completed_at: Optional[float] = None
    pending_requests: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "hermes_task_id": self.hermes_task_id,
            "status": self.status,
            "codex_thread_id": self.codex_thread_id,
            "codex_turn_id": self.codex_turn_id,
            "prompt_summary": self.prompt_summary,
            "cwd": self.cwd,
            "model": self.model,
            "sandbox": self.sandbox,
            "approval_policy": self.approval_policy,
            "degraded_mode": self.degraded_mode,
            "last_progress_summary": self.last_progress_summary,
            "final_summary": self.final_summary,
            "error_summary": self.error_summary,
            "notify_target": self.notify_target,
            "notification_status": self.notification_status,
            "notified_at": self.notified_at,
            "notification_error": self.notification_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "pending_requests": list(self.pending_requests.values()),
        }


class CodexJsonRpcClient:
    def __init__(self, task_id: str, task: CodexBridgeTask, manager: "CodexBridgeManager"):
        self.task_id = task_id
        self.task = task
        self.manager = manager
        self._process: Optional[subprocess.Popen[str]] = None
        self._next_id = 1
        self._pending: Dict[int, queue.Queue] = {}
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._closed = False

    def start(self, *, codex_home: Optional[str] = None) -> None:
        env = os.environ.copy()
        if codex_home:
            Path(codex_home).mkdir(parents=True, exist_ok=True)
            env["CODEX_HOME"] = codex_home
        self._process = subprocess.Popen(
            ["codex", "app-server", "--listen", "stdio://"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self.task.cwd,
            env=env,
        )
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._reader_thread.start()
        self._stderr_thread.start()

    def initialize(self) -> Dict[str, Any]:
        result = self.request(
            "initialize",
            {
                "clientInfo": {"name": "hermes-codex-bridge", "version": "0.1.0"},
                "capabilities": {"experimentalApi": True},
            },
            timeout=10,
        )
        self.notify("initialized")
        return result

    def request(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 30) -> Dict[str, Any]:
        with self._pending_lock:
            request_id = self._next_id
            self._next_id += 1
            response_q: queue.Queue = queue.Queue(maxsize=1)
            self._pending[request_id] = response_q
        self._send({"id": request_id, "method": method, "params": params})
        try:
            msg = response_q.get(timeout=timeout)
        except queue.Empty:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise TimeoutError(f"Codex app-server request timed out: {method}")
        if "error" in msg:
            raise RuntimeError(msg["error"].get("message") or _json_dumps(msg["error"]))
        return msg.get("result") or {}

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        msg: Dict[str, Any] = {"method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)

    def respond(self, request_id: str, result: Dict[str, Any]) -> None:
        self._send({"id": request_id, "result": result})

    def close(self) -> None:
        self._closed = True
        process = self._process
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

    def _send(self, msg: Dict[str, Any]) -> None:
        if not self._process or not self._process.stdin:
            raise RuntimeError("Codex app-server process is not running.")
        with self._write_lock:
            self._process.stdin.write(json.dumps(msg) + "\n")
            self._process.stdin.flush()

    def _read_stdout(self) -> None:
        assert self._process and self._process.stdout
        for line in self._process.stdout:
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self.manager.record_event(self.task_id, "protocol/raw", {"line": line.rstrip()})
                continue
            if "id" in msg and "method" not in msg:
                with self._pending_lock:
                    response_q = self._pending.pop(msg["id"], None)
                if response_q:
                    response_q.put(msg)
                else:
                    self.manager.record_event(self.task_id, "protocol/response", msg)
                continue
            if "id" in msg and "method" in msg:
                self.manager.handle_server_request(self.task_id, self, msg)
            else:
                self.manager.record_event(self.task_id, msg.get("method", "unknown"), msg.get("params", {}))

    def _read_stderr(self) -> None:
        assert self._process and self._process.stderr
        for line in self._process.stderr:
            text = line.strip()
            if text:
                self.manager.record_event(self.task_id, "codex/stderr", {"message": text})


class CodexBridgeManager:
    def __init__(self, store: Optional[CodexBridgeStore] = None):
        self.store = store or CodexBridgeStore()
        self._tasks: Dict[str, CodexBridgeTask] = {}
        self._clients: Dict[str, CodexJsonRpcClient] = {}
        self._lock = threading.RLock()

    def start_task(
        self,
        prompt: str,
        *,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        sandbox: str = DEFAULT_SANDBOX,
        approval_policy: str = DEFAULT_APPROVAL_POLICY,
        codex_home: Optional[str] = None,
        notify_target: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not prompt or not prompt.strip():
            raise ValueError("codex_bridge start requires a non-empty prompt.")
        if sandbox == "danger-full-access":
            raise ValueError("codex_bridge refuses danger-full-access as a default bridge sandbox.")
        if approval_policy == "never":
            raise ValueError("codex_bridge refuses approval_policy=never.")
        task_id = f"codex-{uuid.uuid4().hex[:12]}"
        cwd = str(Path(cwd or os.getcwd()).resolve())
        task = CodexBridgeTask(
            hermes_task_id=task_id,
            prompt_summary=_summarize_payload(prompt, max_chars=300),
            cwd=cwd,
            model=model,
            sandbox=sandbox,
            approval_policy=approval_policy,
            notify_target=notify_target.strip() if notify_target and notify_target.strip() else None,
            notification_status="pending" if notify_target and notify_target.strip() else None,
        )
        client = CodexJsonRpcClient(task_id, task, self)
        with self._lock:
            self._tasks[task_id] = task
            self._clients[task_id] = client
        self.store.upsert_task(task)

        try:
            client.start(codex_home=codex_home)
            init = client.initialize()
            self.record_event(task_id, "bridge/initialized", init)
            thread_params: Dict[str, Any] = {
                "cwd": cwd,
                "sandbox": sandbox,
                "approvalPolicy": approval_policy,
                "approvalsReviewer": "user",
                "ephemeral": True,
                "sessionStartSource": "startup",
                "developerInstructions": (
                    "You are running under Hermes Codex Bridge. Do not treat your "
                    "own output as approval. Ask for approval through Codex app-server "
                    "requests when required."
                ),
            }
            if model:
                thread_params["model"] = model
            thread_result = client.request("thread/start", thread_params, timeout=15)
            thread = thread_result.get("thread") or {}
            task.codex_thread_id = thread.get("id")
            task.status = "starting"
            task.updated_at = _now()
            self.store.upsert_task(task)
            if not task.codex_thread_id:
                raise RuntimeError("Codex app-server did not return a thread id.")

            turn_params: Dict[str, Any] = {
                "threadId": task.codex_thread_id,
                "input": _text_input(prompt),
                "approvalPolicy": approval_policy,
                "approvalsReviewer": "user",
                "cwd": cwd,
            }
            if model:
                turn_params["model"] = model
            turn_result = client.request("turn/start", turn_params, timeout=15)
            turn = turn_result.get("turn") or {}
            task.codex_turn_id = turn.get("id")
            task.status = "working"
            task.updated_at = _now()
            self.store.upsert_task(task)
        except Exception:
            client.close()
            with self._lock:
                self._clients.pop(task_id, None)
            raise
        return {
            "success": True,
            "message": "Codex task started through app-server stdio JSON-RPC.",
            "task": task.snapshot(),
            "protocol": {
                "transport": "app-server stdio",
                "mailbox": False,
            },
        }

    def status(self, task_id: str) -> Dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
        if task:
            snap = task.snapshot()
            stored = self.store.get_task_snapshot(task_id)
            if stored:
                snap["recent_events"] = stored.get("recent_events", [])
                snap["pending_requests"] = stored.get("pending_requests", snap["pending_requests"])
                for key in ("notification_status", "notified_at", "notification_error"):
                    snap[key] = stored.get(key)
            return {"success": True, "task": snap}
        stored = self.store.get_task_snapshot(task_id)
        if stored:
            stored["active_connection"] = False
            return {"success": True, "task": stored}
        return {"success": False, "error": f"Unknown Codex bridge task: {task_id}"}

    def list_tasks(self, limit: int = 10) -> Dict[str, Any]:
        return {"success": True, "tasks": self.store.list_tasks(limit=limit)}

    def notify_completed(
        self,
        *,
        limit: int = 10,
        dry_run: bool = False,
        notifier: Optional[Callable[[str, str], Any]] = None,
    ) -> Dict[str, Any]:
        notifier = notifier or _default_completion_notifier
        candidates = self.store.list_completed_for_notification(limit=limit)
        results: List[Dict[str, Any]] = []
        for task in candidates:
            task_id = str(task["hermes_task_id"])
            target = str(task.get("notify_target") or "").strip()
            message = _completion_notification_message(task)
            if not target:
                result = {
                    "task_id": task_id,
                    "status": task.get("status"),
                    "notification_status": "no_target",
                    "sent": False,
                    "message": message,
                }
                if not dry_run:
                    self.store.update_notification_status(task_id, "no_target")
                results.append(result)
                continue

            result = {
                "task_id": task_id,
                "status": task.get("status"),
                "target": target,
                "notification_status": "dry_run" if dry_run else "pending",
                "sent": False,
                "message": message,
            }
            if dry_run:
                results.append(result)
                continue

            try:
                delivery = notifier(target, message)
            except Exception as exc:
                error = str(exc)
                self.store.update_notification_status(task_id, "failed", error=error)
                result["notification_status"] = "failed"
                result["error"] = error
                results.append(result)
                continue

            notified_at = _now()
            self.store.update_notification_status(task_id, "sent", notified_at=notified_at)
            result["notification_status"] = "sent"
            result["sent"] = True
            result["notified_at"] = notified_at
            result["delivery"] = delivery
            results.append(result)

        return {
            "success": True,
            "dry_run": dry_run,
            "processed": len(results),
            "notifications": results,
        }

    def steer(self, task_id: str, instruction: str) -> Dict[str, Any]:
        task, client = self._active(task_id)
        if not task.codex_thread_id or not task.codex_turn_id:
            raise RuntimeError("Task has no active Codex turn to steer.")
        result = client.request(
            "turn/steer",
            {
                "threadId": task.codex_thread_id,
                "expectedTurnId": task.codex_turn_id,
                "input": _text_input(instruction),
            },
            timeout=15,
        )
        self.record_event(task_id, "bridge/steer", {"instruction": instruction, "result": result})
        return {"success": True, "task": task.snapshot(), "result": result}

    def interrupt(self, task_id: str) -> Dict[str, Any]:
        task, client = self._active(task_id)
        if not task.codex_thread_id or not task.codex_turn_id:
            raise RuntimeError("Task has no active Codex turn to interrupt.")
        result = client.request(
            "turn/interrupt",
            {"threadId": task.codex_thread_id, "turnId": task.codex_turn_id},
            timeout=15,
        )
        task.status = "cancelled"
        task.completed_at = _now()
        task.updated_at = _now()
        self.store.upsert_task(task)
        self.record_event(task_id, "bridge/interrupt", result)
        return {"success": True, "task": task.snapshot(), "result": result}

    def respond(self, task_id: str, request_id: str, decision: str, answers: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        task, client = self._active(task_id)
        pending = task.pending_requests.get(str(request_id))
        if not pending:
            raise RuntimeError(f"No pending Codex request {request_id!r} for task {task_id}.")
        method = pending["method"]
        if method == "item/tool/requestUserInput":
            result = {"answers": answers or {}}
        elif method == "item/commandExecution/requestApproval":
            result = {"decision": decision}
        elif method == "item/fileChange/requestApproval":
            result = {"decision": decision}
        elif method == "item/permissions/requestApproval":
            result = {"decision": decision}
        else:
            result = {"decision": decision}
        client.respond(str(request_id), result)
        pending["status"] = "resolved"
        pending["response"] = result
        task.pending_requests.pop(str(request_id), None)
        task.status = "working"
        task.updated_at = _now()
        self.store.resolve_pending_request(task_id, str(request_id), result)
        self.store.upsert_task(task)
        return {"success": True, "task": task.snapshot(), "response": result}

    def handle_server_request(self, task_id: str, client: CodexJsonRpcClient, msg: Dict[str, Any]) -> None:
        request_id = str(msg.get("id"))
        method = msg.get("method", "unknown")
        params = msg.get("params") or {}
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            if method == "item/tool/requestUserInput":
                task.status = "waiting_for_user_input"
            else:
                task.status = "waiting_for_approval"
            task.updated_at = _now()
            task.pending_requests[request_id] = {
                "request_id": request_id,
                "method": method,
                "payload": params,
                "status": "pending",
                "created_at": task.updated_at,
            }
            self.store.upsert_task(task)
        self.store.upsert_pending_request(task_id, request_id, method, params)
        self.record_event(task_id, method, params)

    def record_event(self, task_id: str, method: str, params: Any) -> None:
        terminal = False
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            normalized = _normalize_status(method, params if isinstance(params, dict) else {})
            if method == "turn/completed" and task.status == "cancelled" and normalized == "completed":
                normalized = "cancelled"
            if normalized:
                task.status = normalized
            if method == "turn/completed":
                terminal = True
                task.completed_at = _now()
                turn = params.get("turn") if isinstance(params, dict) else {}
                if isinstance(turn, dict) and turn.get("error"):
                    task.error_summary = _summarize_payload(turn.get("error"), max_chars=500)
                else:
                    task.final_summary = _summarize_payload(params, max_chars=500)
            elif method == "error":
                task.error_summary = _summarize_payload(params, max_chars=500)
            elif method not in {"codex/stderr"}:
                task.last_progress_summary = f"{method}: {_summarize_payload(params, max_chars=300)}"
            task.updated_at = _now()
            thread_id = task.codex_thread_id
            turn_id = task.codex_turn_id
            self.store.upsert_task(task)
        self.store.insert_event(task_id, thread_id, turn_id, method, normalized, params)
        if terminal:
            with self._lock:
                client = self._clients.pop(task_id, None)
            if client:
                threading.Thread(target=client.close, daemon=True).start()

    def _active(self, task_id: str) -> tuple[CodexBridgeTask, CodexJsonRpcClient]:
        with self._lock:
            task = self._tasks.get(task_id)
            client = self._clients.get(task_id)
        if not task or not client:
            raise RuntimeError(f"Codex bridge task is not active in this process: {task_id}")
        return task, client


_MANAGER: Optional[CodexBridgeManager] = None
_MANAGER_LOCK = threading.Lock()


def _get_manager() -> CodexBridgeManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = CodexBridgeManager()
        return _MANAGER


def _completion_notification_message(task: Dict[str, Any]) -> str:
    task_id = task.get("hermes_task_id") or "unknown"
    status = task.get("status") or "unknown"
    prompt = task.get("prompt_summary") or "(no prompt summary)"
    summary = task.get("final_summary") or task.get("error_summary") or task.get("last_progress_summary") or ""
    lines = [
        f"Codex Bridge task {task_id} finished with status: {status}.",
        f"Prompt: {prompt}",
    ]
    if summary:
        lines.append(f"Summary: {_summarize_payload(summary, max_chars=700)}")
    return "\n".join(lines)


def _default_completion_notifier(target: str, message: str) -> Dict[str, Any]:
    if target == "local":
        return {"success": True, "target": target, "local": True}
    from tools.send_message_tool import send_message_tool

    raw = send_message_tool({"action": "send", "target": target, "message": message})
    try:
        result = json.loads(raw)
    except Exception:
        result = {"raw": raw}
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(str(result["error"]))
    return result if isinstance(result, dict) else {"result": result}


def codex_bridge(
    action: str,
    prompt: Optional[str] = None,
    task_id: Optional[str] = None,
    instruction: Optional[str] = None,
    decision: str = "decline",
    answers: Optional[Dict[str, Any]] = None,
    cwd: Optional[str] = None,
    model: Optional[str] = None,
    sandbox: str = DEFAULT_SANDBOX,
    approval_policy: str = DEFAULT_APPROVAL_POLICY,
    codex_home: Optional[str] = None,
    notify_target: Optional[str] = None,
    limit: int = 10,
    dry_run: bool = False,
) -> str:
    try:
        action = (action or "").strip().lower()
        if action == "start":
            result = _get_manager().start_task(
                prompt or "",
                cwd=cwd,
                model=model,
                sandbox=sandbox or DEFAULT_SANDBOX,
                approval_policy=approval_policy or DEFAULT_APPROVAL_POLICY,
                codex_home=codex_home,
                notify_target=notify_target,
            )
        elif action == "status":
            if not task_id:
                raise ValueError("codex_bridge status requires task_id.")
            result = _get_manager().status(task_id)
        elif action == "list":
            result = _get_manager().list_tasks(limit=limit)
        elif action == "notify_completed":
            result = _get_manager().notify_completed(limit=limit, dry_run=dry_run)
        elif action == "steer":
            if not task_id or not instruction:
                raise ValueError("codex_bridge steer requires task_id and instruction.")
            result = _get_manager().steer(task_id, instruction)
        elif action in {"interrupt", "cancel"}:
            if not task_id:
                raise ValueError("codex_bridge interrupt requires task_id.")
            result = _get_manager().interrupt(task_id)
        elif action == "respond":
            if not task_id or not instruction:
                raise ValueError("codex_bridge respond requires task_id and instruction=request_id.")
            result = _get_manager().respond(task_id, instruction, decision=decision, answers=answers)
        else:
            raise ValueError("action must be one of start, status, list, notify_completed, steer, interrupt, respond.")
        return _json_dumps(result)
    except Exception as exc:
        return tool_error(str(exc))


CODEX_BRIDGE_SCHEMA = {
    "name": "codex_bridge",
    "description": (
        "Start and control local Codex tasks through Codex app-server JSON-RPC. "
        "Uses stdio/WebSocket-capable app-server protocol semantics and never "
        "uses mailbox, inbox, or outbox files as the communication path. "
        "Actions: start, status, list, notify_completed, steer, interrupt, respond."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "status", "list", "notify_completed", "steer", "interrupt", "cancel", "respond"],
                "description": "Bridge operation to perform.",
            },
            "prompt": {"type": "string", "description": "Task prompt for action=start."},
            "task_id": {"type": "string", "description": "Hermes Codex task id for status/control actions."},
            "instruction": {
                "type": "string",
                "description": "Steering text for steer, or pending request id for respond.",
            },
            "decision": {
                "type": "string",
                "enum": ["accept", "acceptForSession", "decline", "cancel"],
                "description": "Approval decision for action=respond.",
            },
            "answers": {
                "type": "object",
                "description": "requestUserInput answers keyed by Codex question id, e.g. {'q1': {'answers': ['value']}}.",
            },
            "cwd": {"type": "string", "description": "Working directory for the Codex thread."},
            "model": {"type": "string", "description": "Optional Codex model override."},
            "sandbox": {
                "type": "string",
                "enum": ["read-only", "workspace-write"],
                "description": "Codex sandbox mode. danger-full-access is intentionally not exposed.",
            },
            "approval_policy": {
                "type": "string",
                "enum": ["untrusted", "on-request"],
                "description": "Codex approval policy. Default untrusted.",
            },
            "codex_home": {
                "type": "string",
                "description": "Optional CODEX_HOME override for testing or isolated runs.",
            },
            "notify_target": {
                "type": "string",
                "description": (
                    "Optional completion notification target for action=start, such as "
                    "'local', 'feishu:<chat_id>', or any send_message target."
                ),
            },
            "limit": {"type": "integer", "description": "List limit for action=list."},
            "dry_run": {
                "type": "boolean",
                "description": "For action=notify_completed, preview notifications without sending or marking tasks.",
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="codex_bridge",
    toolset="codex_bridge",
    schema=CODEX_BRIDGE_SCHEMA,
    handler=lambda args, **kw: codex_bridge(
        action=args.get("action", ""),
        prompt=args.get("prompt"),
        task_id=args.get("task_id"),
        instruction=args.get("instruction"),
        decision=args.get("decision", "decline"),
        answers=args.get("answers"),
        cwd=args.get("cwd"),
        model=args.get("model"),
        sandbox=args.get("sandbox", DEFAULT_SANDBOX),
        approval_policy=args.get("approval_policy", DEFAULT_APPROVAL_POLICY),
        codex_home=args.get("codex_home"),
        notify_target=args.get("notify_target"),
        limit=args.get("limit", 10),
        dry_run=bool(args.get("dry_run", False)),
    ),
    check_fn=check_codex_bridge_requirements,
    emoji="C",
)
