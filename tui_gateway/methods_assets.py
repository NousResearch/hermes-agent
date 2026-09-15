"""Assets RPC surface: the server-authoritative artifact/attachment index.

The More-pane "Assets" contract and the Project-detail Assets tab. Desktop
holds artifacts in renderer memory only; this index is derived from durable
server state so every device sees the same thing:

* ``attachment`` — files the user uploaded, staged under
  ``<profile home>/attachments/`` by ``prompt_attachments``;
* ``media`` — files a session delivered via a ``MEDIA:`` tag (the same
  extension-anchored matcher the messaging gateway dispatches with), resolved
  against the live filesystem;
* ``artifact`` — generated files inside a session's working directory that
  no MEDIA tag named (surfaced by directory scan of project folders).

Project grouping mirrors filing semantics: an asset belongs to the project
that owns its path (``projects_db.project_for_path``), falling back to the
owning session's cwd. Everything is best-effort: an unreadable path is
skipped, never fatal, and the whole surface answers ``available: false``
when no state.db is reachable.
"""

from __future__ import annotations

import contextlib
import os
import re
import time
from pathlib import Path

from .method_ctx import HandlerRegistry, bind_module

# bind_module rebinds our functions onto server.py's globals and skips module
# objects, so anything not already a server global must be imported at call
# time (mimetypes is the only one here).

_registry = HandlerRegistry()
method = _registry.method

_E_ASSETS = 5091

# Mirror of gateway/run.py::_TOOL_MEDIA_RE — a bare "MEDIA:" in prose never
# matches; the path must be absolute (or ~/) and carry a known media/office
# extension. Keep the two in step when either changes.
_MEDIA_TAG_RE = re.compile(
    r'MEDIA:((?:[A-Za-z]:[/\\]|/|~\/)\S+\.(?:png|jpe?g|gif|webp|'
    r'mp4|mov|avi|mkv|webm|ogg|opus|mp3|wav|m4a|'
    r'flac|epub|pdf|zip|rar|7z|docx?|xlsx?|pptx?|'
    r'txt|csv|apk|ipa))',
    re.IGNORECASE)

_MAX_ROWS = 500
_MAX_SCAN_FILES = 2000
# A delivered file may be deleted later; re-stat at most this often per path.
_STAT_TTL_S = 2.0
_STAT_CACHE: dict[str, tuple[float, "os.stat_result | None"]] = {}


def _stat_cached(path: str):
    now = time.time()
    hit = _STAT_CACHE.get(path)
    if hit is not None and now - hit[0] < _STAT_TTL_S:
        return hit[1]
    try:
        st = os.stat(path)
    except OSError:
        st = None
    _STAT_CACHE[path] = (now, st)
    if len(_STAT_CACHE) > 4096:
        _STAT_CACHE.clear()
        _STAT_CACHE[path] = (now, st)
    return st


def _attachments_root() -> Path:
    from hermes_constants import get_hermes_home
    return Path(get_hermes_home()) / "attachments"


def _kind_for(name: str) -> str:
    import mimetypes
    mime = mimetypes.guess_type(name)[0] or ""
    if mime.startswith("image/") or mime.startswith("video/") or mime.startswith("audio/"):
        return "media"
    return "artifact"


def _asset_id(path: str) -> str:
    # Stable across processes: builtin hash() is salted per interpreter run.
    import hashlib
    return "asset-" + hashlib.sha256(os.path.abspath(path).encode()).hexdigest()[:16]


def _row(path: str, *, kind: str, session_id: str, session_title: str,
         project_id: str | None, project_name: str | None,
         st=None) -> dict | None:
    st = st if st is not None else _stat_cached(path)
    if st is None or not os.path.isfile(path):
        return None
    import mimetypes
    return {
        "id": _asset_id(path),
        "kind": kind,
        "name": os.path.basename(path),
        "path": os.path.abspath(path),
        "mime_type": mimetypes.guess_type(path)[0] or "application/octet-stream",
        "size_bytes": int(st.st_size),
        "modified_at": float(st.st_mtime),
        "session_id": session_id,
        "session_title": session_title,
        "project_id": project_id,
        "project_name": project_name,
    }


def _project_lookup():
    """(ctx, lookup) where lookup(path) -> (id, name) | None over a closing
    projects.db handle. The caller must exit the ctx; lookup is safe after."""
    try:
        from hermes_cli import projects_db as pdb
    except Exception:
        return contextlib.nullcontext(None), lambda p: None

    state: dict = {}

    @contextlib.contextmanager
    def ctx():
        with pdb.connect_closing() as conn:
            state["conn"] = conn
            try:
                yield conn
            finally:
                state.pop("conn", None)

    def lookup(p: str):
        conn = state.get("conn")
        if conn is None:
            return None
        try:
            proj = pdb.project_for_path(conn, p)
        except Exception:
            return None
        return (proj.id, proj.name) if proj is not None else None

    return ctx(), lookup


def _media_rows(db, project_lookup) -> list[dict]:
    """MEDIA-tagged deliveries from stored assistant messages, newest first."""
    rows: list[dict] = []
    try:
        msg_rows = db._read_rows(
            "SELECT m.session_id, m.content, m.timestamp, s.title, s.cwd "
            "FROM messages m JOIN sessions s ON s.id = m.session_id "
            "WHERE m.role = 'assistant' AND m.active = 1 AND m.content LIKE '%MEDIA:%' "
            "ORDER BY m.timestamp DESC LIMIT ?",
            (_MAX_ROWS * 2,))
    except Exception:
        return rows
    seen: set[str] = set()
    for r in msg_rows:
        for raw in _MEDIA_TAG_RE.findall(str(r["content"] or "")):
            path = os.path.expanduser(raw)
            if path in seen:
                continue
            seen.add(path)
            proj = project_lookup(path) or project_lookup(str(r["cwd"] or ""))
            row = _row(path,
                       kind=_kind_for(path),
                       session_id=str(r["session_id"]),
                       session_title=str(r["title"] or ""),
                       project_id=proj[0] if proj else None,
                       project_name=proj[1] if proj else None)
            if row:
                rows.append(row)
            if len(rows) >= _MAX_ROWS:
                return rows
    return rows


def _attachment_rows(project_lookup) -> list[dict]:
    """User uploads staged under <profile home>/attachments/, newest first."""
    root = _attachments_root()
    if not root.is_dir():
        return []
    rows: list[dict] = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []
    for p in entries[:_MAX_SCAN_FILES]:
        if not p.is_file():
            continue
        proj = project_lookup(str(p))
        row = _row(str(p),
                   kind="attachment",
                   session_id="",
                   session_title="",
                   project_id=proj[0] if proj else None,
                   project_name=proj[1] if proj else None)
        if row:
            rows.append(row)
        if len(rows) >= _MAX_ROWS:
            break
    return rows


@_registry.method("assets.status")
def _(rid, params: dict) -> dict:
    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=_E_ASSETS)
        root = _attachments_root()
        return _ok(rid, {
            "available": True,
            "attachments_dir": str(root),
            "attachments_present": root.is_dir(),
        })


@_registry.method("assets.list")
@_registry.profile_scoped
def _(rid, params: dict) -> dict:
    kind = str(params.get("kind") or "").strip().lower() or None
    if kind not in (None, "artifact", "attachment", "media"):
        return _err(rid, _E_ASSETS, f"unknown asset kind '{kind}'")
    project_id = str(params.get("project_id") or "").strip() or None
    session_id = str(params.get("session_id") or "").strip() or None
    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=_E_ASSETS)
        ctx, lookup = _project_lookup()
        with ctx:
            rows = _media_rows(db, lookup)
            if kind in (None, "attachment"):
                rows += _attachment_rows(lookup)
    if kind:
        rows = [r for r in rows if r["kind"] == kind]
    if project_id:
        rows = [r for r in rows if r["project_id"] == project_id]
    if session_id:
        rows = [r for r in rows if r["session_id"] == session_id]
    rows.sort(key=lambda r: r["modified_at"], reverse=True)
    return _ok(rid, {"assets": rows[:_MAX_ROWS], "total": len(rows)})


def register(server) -> None:
    """Publish this module's handlers onto ``server`` (rebound to its globals)."""
    bind_module(globals(), server, skip=("_",))
