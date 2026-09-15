"""Organization RPC surface: durable pin ordering, batch mutation, and undo.

The More-pane "Pin, batch and undo" contract. Pin/archive state itself lives
in the session DB (``pinned`` / ``archived`` columns, lineage-aware via the
SessionDB setters); what this module adds is:

* batch mutation — one call flips many sessions, reporting per-session
  outcomes so a partial failure never looks like a clean run;
* a durable undo log — every batch records the previous state of each
  session it touched, so any batch can be reversed later, even after an
  app restart. The log lives at ``<profile home>/organization_undo.json``
  (atomic writes, capped history).

Undo restores exactly the recorded previous state; it does not try to be
smart about later user changes between the batch and the undo.
"""

from __future__ import annotations

import threading
from pathlib import Path

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method

_E_ORG = 5081
_E_ORG_ARG = 5082
_E_NOT_FOUND = 5083

_MAX_BATCHES = 50
_MAX_SESSIONS_PER_BATCH = 100

_LOCK = threading.Lock()


def _max_batches() -> int:
    """Read the cap through the defining module at call time — handlers are
    rebound onto server.py's globals by bind_module, so a monkeypatch of
    ``methods_organization._MAX_BATCHES`` is invisible to a bare global read
    (and ``__name__`` itself is rebound to the server module)."""
    import sys
    return getattr(sys.modules["tui_gateway.methods_organization"], "_MAX_BATCHES")


def _undo_path() -> Path:
    from pathlib import Path as _P
    from hermes_constants import get_hermes_home
    return _P(get_hermes_home()) / "organization_undo.json"


def _load_log() -> dict:
    import json
    try:
        with _undo_path().open(encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"version": 1, "batches": []}
    if not isinstance(data, dict) or not isinstance(data.get("batches"), list):
        return {"version": 1, "batches": []}
    return data


def _save_log(data: dict) -> None:
    import json
    import os
    import tempfile
    p = _undo_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".organization_undo.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _prev_state(row: dict) -> dict:
    return {
        "pinned": int(bool(row.get("pinned"))),
        "archived": int(bool(row.get("archived"))),
    }


def _batch_mutate(rid, params, column: str, value: bool) -> dict:
    """Flip one column across many sessions; record the batch for undo."""
    import time
    import uuid
    raw_ids = params.get("session_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        return _err(rid, _E_ORG_ARG, "session_ids must be a non-empty list")
    if len(raw_ids) > _MAX_SESSIONS_PER_BATCH:
        return _err(rid, _E_ORG_ARG,
                    f"at most {_MAX_SESSIONS_PER_BATCH} sessions per batch")
    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=_E_ORG)
        changes, applied, failed = [], [], []
        for raw in raw_ids:
            sid = str(raw or "").strip()
            if not sid:
                continue
            try:
                key = db.resolve_session_id(sid) if hasattr(db, "resolve_session_id") else sid
                row = db.get_session(key) if key else None
                if row is None:
                    failed.append({"session_id": sid, "error": "not found"})
                    continue
                prev = _prev_state(row)
                setter = db.set_session_pinned if column == "pinned" else db.set_session_archived
                setter(key, value)
                changes.append({"session_id": key, "prev": prev})
                applied.append(key)
            except Exception as exc:
                failed.append({"session_id": sid, "error": str(exc)})
        if not changes:
            return _err(rid, _E_NOT_FOUND, "no session could be mutated")
        batch_id = uuid.uuid4().hex[:12]
        with _LOCK:
            log = _load_log()
            log["batches"].append({
                "id": batch_id,
                "ts": time.time(),
                "action": column,
                "value": int(value),
                "changes": changes,
                "undone": False,
            })
            del log["batches"][:-_max_batches()]
            _save_log(log)
        return _ok(rid, {
            "batch_id": batch_id,
            "applied": applied,
            "failed": failed,
            "partial": bool(failed),
        })


@_registry.method("organization.pin")
@_registry.profile_scoped
def _(rid, params: dict) -> dict:
    return _batch_mutate(rid, params, "pinned",
                        bool(params.get("pinned", True)))


@_registry.method("organization.archive")
@_registry.profile_scoped
def _(rid, params: dict) -> dict:
    return _batch_mutate(rid, params, "archived",
                        bool(params.get("archived", True)))


@_registry.method("organization.history")
@_registry.profile_scoped
def _(rid, params: dict) -> dict:
    log = _load_log()
    batches = [{
        "id": b.get("id"),
        "ts": b.get("ts"),
        "action": b.get("action"),
        "value": b.get("value"),
        "count": len(b.get("changes", [])),
        "undone": bool(b.get("undone")),
    } for b in log["batches"]]
    return _ok(rid, {"batches": batches})


@_registry.method("organization.undo")
@_registry.profile_scoped
def _(rid, params: dict) -> dict:
    """Reverse one batch (by ``batch_id``, or the newest not-yet-undone)."""
    wanted = str(params.get("batch_id") or "").strip()
    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=_E_ORG)
        with _LOCK:
            log = _load_log()
            candidates = [
                b for b in log["batches"]
                if (b.get("id") == wanted if wanted else not b.get("undone"))
            ]
            if not candidates:
                return _err(rid, _E_NOT_FOUND,
                           "no such batch" if wanted else "nothing to undo")
            batch = candidates[-1]
            restored, failed = [], []
            for change in batch.get("changes", []):
                sid = str(change.get("session_id") or "")
                prev = change.get("prev") or {}
                try:
                    key = db.resolve_session_id(sid) if hasattr(db, "resolve_session_id") else sid
                    if db.get_session(key) is None:
                        failed.append({"session_id": sid, "error": "not found"})
                        continue
                    db.set_session_pinned(key, bool(prev.get("pinned")))
                    db.set_session_archived(key, bool(prev.get("archived")))
                    restored.append(key)
                except Exception as exc:
                    failed.append({"session_id": sid, "error": str(exc)})
            batch["undone"] = True
            _save_log(log)
        return _ok(rid, {
            "batch_id": batch.get("id"),
            "restored": restored,
            "failed": failed,
            "partial": bool(failed),
        })


def register(server) -> None:
    """Publish this module's handlers onto ``server`` (rebound to its globals)."""
    bind_module(globals(), server, skip=("_",))
