"""Spawn-tree snapshot handlers (``spawn_tree.save`` / ``list`` / ``load``; ``methods_session`` split).

Moved verbatim from ``tui_gateway/methods_session.py`` (file-line ratchet): the bodies close over
server.py globals (``_spawn_trees_root``, ``_spawn_tree_session_dir``, the index helpers) through
``method_ctx.bind_module`` exactly as before — publication still runs from the parent's ``register()``.
"""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


@method("spawn_tree.save")
def _(rid, params: dict) -> dict:
    session_id = _str_param(params, "session_id")
    subagents = params.get("subagents") or []
    if not isinstance(subagents, list) or not subagents:
        return _err(rid, 4000, "subagents list required")
    started_at, label = params.get("started_at"), str(params.get("label") or "")
    finished_at = float(params.get("finished_at") or time.time())
    d = _spawn_tree_session_dir(session_id or "default")
    path = d / f"{datetime.fromtimestamp(finished_at, timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    meta = {"session_id": session_id, "started_at": float(started_at) if started_at else None,
            "finished_at": finished_at, "label": label}
    try:
        path.write_text(json.dumps({**meta, "subagents": subagents}, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        return _err(rid, 5000, f"spawn_tree.save failed: {exc}")
    _append_spawn_tree_index(d, {"path": str(path), **meta, "count": len(subagents)})
    return _ok(rid, {"path": str(path), "session_id": session_id})


def _legacy_spawn_tree_entry(p, session_dir_name: str) -> dict | None:
    """Index-shaped entry for a pre-index snapshot file (None when unreadable)."""
    try:
        stat = p.stat()
    except OSError:
        return None
    raw = {}
    with contextlib.suppress(Exception):
        raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raw = {}
    subagents = raw.get("subagents") or []
    return {"path": str(p), "session_id": raw.get("session_id") or session_dir_name,
            "finished_at": raw.get("finished_at") or stat.st_mtime, "started_at": raw.get("started_at"),
            "label": raw.get("label") or "", "count": len(subagents) if isinstance(subagents, list) else 0}


@method("spawn_tree.list")
def _(rid, params: dict) -> dict:
    session_id = _str_param(params, "session_id")
    if bool(params.get("cross_session")):
        roots = [p for p in _spawn_trees_root().iterdir() if p.is_dir()]
    else:
        roots = [_spawn_tree_session_dir(session_id or "default")]
    entries: list[dict] = []
    for d in roots:
        if indexed := _read_spawn_tree_index(d):
            # Skip index entries whose snapshot file was manually deleted.
            entries.extend(e for e in indexed if (p := e.get("path")) and Path(p).exists())
        else:  # Legacy (pre-index) sessions: full scan, once per session until the next save.
            entries.extend(
                entry for p in d.glob("*.json")
                if p.name != _SPAWN_TREE_INDEX and (entry := _legacy_spawn_tree_entry(p, d.name)) is not None)
    entries.sort(key=lambda e: e.get("finished_at") or 0, reverse=True)
    return _ok(rid, {"entries": entries[:int(params.get("limit") or 50)]})


@method("spawn_tree.load")
def _(rid, params: dict) -> dict:
    if not (raw_path := _str_param(params, "path")):
        return _err(rid, 4000, "path required")
    try:
        (resolved := Path(raw_path).resolve()).relative_to(_spawn_trees_root().resolve())
    except (ValueError, OSError) as exc:
        return _err(rid, 4030, f"path outside spawn-trees root: {exc}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _err(rid, 5000, f"spawn_tree.load failed: {exc}")
    if not isinstance(payload, dict):
        return _err(rid, 5000, "spawn_tree.load failed: snapshot is not a JSON object")
    return _ok(rid, payload)


def register(server) -> None:
    """Publish this module's helpers onto ``server`` (rebound to its globals) and install handlers."""
    bind_module(globals(), server, skip=("_",))
