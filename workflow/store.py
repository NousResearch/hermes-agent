"""Documents and run state under ``HERMES_HOME/workflows``.

The desktop still keeps a local cache so a drag does not wait on a socket.
This is the copy the gateway, cron, and inbound webhooks read. A run's state
file is what the runner resumes from; its history is the run's Relay trace
(``workflow/trace.py``), not a log kept here.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_write_text

_lock = threading.RLock()

LIVE_STATUSES = frozenset({"running", "paused", "waiting_human", "waiting_world"})


def workflows_dir(home: Path | None = None) -> Path:
    path = (home or get_hermes_home()) / "workflows"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _docs_path(home: Path | None = None) -> Path:
    return workflows_dir(home) / "documents.json"


def _secrets_path() -> Path:
    return workflows_dir() / "secrets.json"


def _runs_dir() -> Path:
    path = workflows_dir() / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    return data


def _write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def load_documents(home: Path | None = None) -> dict[str, Any]:
    with _lock:
        raw = _read_json(_docs_path(home), {"docs": [], "currentId": None})
    if not isinstance(raw, dict):
        return {"docs": [], "currentId": None}
    docs = raw.get("docs")
    if not isinstance(docs, list):
        docs = []
    current = raw.get("currentId")
    if current is not None:
        current = str(current)
    return {"docs": [d for d in docs if isinstance(d, dict) and d.get("id")], "currentId": current}


def save_documents(docs: list[dict], current_id: str | None = None) -> dict[str, Any]:
    cleaned: list[dict] = []
    now = int(time.time() * 1000)
    for raw in docs:
        if not isinstance(raw, dict) or not raw.get("id"):
            continue
        item = dict(raw)
        item["id"] = str(item["id"])
        item["name"] = str(item.get("name") or item["id"])
        item["updatedAt"] = int(item.get("updatedAt") or now)
        if not isinstance(item.get("scenario"), dict):
            item["scenario"] = {"steps": [], "edges": []}
        cleaned.append(item)
    ids = {d["id"] for d in cleaned}
    if current_id and current_id not in ids:
        current_id = cleaned[-1]["id"] if cleaned else None
    payload = {"docs": cleaned, "currentId": current_id}
    with _lock:
        _write_json(_docs_path(), payload)
    return payload


def get_document(workflow_id: str) -> dict | None:
    wid = (workflow_id or "").strip()
    if not wid:
        return None
    for doc in load_documents()["docs"]:
        if doc["id"] == wid or str(doc.get("name") or "").lower() == wid.lower():
            return doc
    return None


def upsert_document(doc: dict) -> dict:
    docs = load_documents()
    found = False
    next_docs = []
    for existing in docs["docs"]:
        if existing["id"] == doc["id"]:
            merged = {**existing, **doc, "updatedAt": int(time.time() * 1000)}
            next_docs.append(merged)
            found = True
        else:
            next_docs.append(existing)
    if not found:
        incoming = dict(doc)
        incoming["updatedAt"] = int(time.time() * 1000)
        next_docs.append(incoming)
    current = docs["currentId"] if found else doc["id"]
    return save_documents(next_docs, current)


def remove_document(workflow_id: str) -> dict[str, Any]:
    docs = load_documents()
    next_docs = [d for d in docs["docs"] if d["id"] != workflow_id]
    current = docs["currentId"]
    if current == workflow_id:
        current = next_docs[-1]["id"] if next_docs else None
    return save_documents(next_docs, current)


def load_secrets() -> dict[str, str]:
    raw = _read_json(_secrets_path(), {})
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if k and v}


def put_secret(workflow_id: str, secret: str) -> None:
    with _lock:
        secrets = load_secrets()
        secrets[workflow_id] = secret
        _write_json(_secrets_path(), secrets)


def secret_for(workflow_id: str) -> str | None:
    return load_secrets().get(workflow_id)


def new_run_id() -> str:
    return f"run-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"


def run_path(run_id: str) -> Path:
    return _runs_dir() / f"{run_id}.json"


def save_run(state: dict) -> dict:
    with _lock:
        _write_json(run_path(state["runId"]), state)
    return state


def load_run(run_id: str) -> dict | None:
    raw = _read_json(run_path(run_id), None)
    return raw if isinstance(raw, dict) else None


def list_runs(workflow_id: str | None = None) -> list[dict]:
    out: list[dict] = []
    for path in sorted(_runs_dir().glob("*.json")):
        raw = _read_json(path, None)
        if not isinstance(raw, dict):
            continue
        if workflow_id and raw.get("workflowId") != workflow_id:
            continue
        out.append(raw)
    return out


def live_runs(workflow_id: str | None = None) -> list[dict]:
    return [r for r in list_runs(workflow_id) if r.get("status") in LIVE_STATUSES]


def active_run(workflow_id: str) -> dict | None:
    """The workflow's newest live run — the one the canvas follows."""
    found = live_runs(workflow_id)
    if not found:
        return None
    found.sort(key=lambda r: r.get("startedAt") or 0, reverse=True)
    return found[0]
