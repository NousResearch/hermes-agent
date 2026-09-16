#!/usr/bin/env python3
"""Project tools — the agent's INTENTIONAL handle on first-class Projects (per-profile
``projects.db``, the desktop sidebar's named workspaces). Creating/switching is an explicit
tool call, never a side effect of ``cd``. GUI-only: the `project` toolset stays off
``_HERMES_CORE_TOOLS``; the desktop/TUI gateway folds it in and wires
``set_project_workspace_callback`` so the live session's cwd and sidebar follow."""

import json
import os
from typing import Callable, Optional

from tools.registry import registry

# Set by the GUI gateway: ``(task_id, primary_path, project_name)`` re-anchors that session's
# workspace. ``None`` in CLI/messaging — the DB write still happens, nothing to move.
_workspace_callback: Optional[Callable[..., Optional[dict]]] = None


def set_project_workspace_callback(fn: Optional[Callable[..., Optional[dict]]]) -> None:
    global _workspace_callback
    _workspace_callback = fn


def _primary_path(proj) -> Optional[str]:
    if getattr(proj, "primary_path", None):
        return proj.primary_path
    for folder in proj.folders:
        if folder.is_primary:
            return folder.path
    return proj.folders[0].path if proj.folders else None


def _apply_workspace(task_id: Optional[str], path: Optional[str], name: str) -> None:
    cb = _workspace_callback
    if cb and task_id and path:
        try:
            cb(task_id, path, name)
        except Exception:
            pass


def _resolve(conn, token: str):
    from hermes_cli import projects_db as pdb
    token = (token or "").strip()
    if not token:
        return None
    projects = pdb.list_projects(conn, include_archived=True)
    # Exact id / slug / name first, then case-insensitive slug / name.
    for proj in projects:
        if token in (proj.id, proj.slug) or proj.name == token:
            return proj
    low = token.lower()
    for proj in projects:
        if proj.slug.lower() == low or proj.name.lower() == low:
            return proj
    return None


def _activated(proj, task_id: Optional[str]) -> str:
    primary = _primary_path(proj)
    _apply_workspace(task_id, primary, proj.name)
    return json.dumps({
        "success": True, "id": proj.id, "slug": proj.slug, "name": proj.name,
        "primary_path": primary})


def project_list(task_id: Optional[str] = None) -> str:
    from hermes_cli import projects_db as pdb
    with pdb.connect_closing() as conn:
        active = pdb.get_active_id(conn)
        projects = pdb.list_projects(conn)
    return json.dumps({
        "active_id": active,
        "projects": [
            {
                "id": p.id, "slug": p.slug, "name": p.name,
                "primary_path": _primary_path(p), "active": p.id == active}
            for p in projects]})


def project_create(name: str, path: Optional[str] = None, task_id: Optional[str] = None) -> str:
    name = (name or "").strip()
    if not name:
        return json.dumps({"success": False, "error": "name is required"})
    from hermes_cli import projects_db as pdb
    folder = (path or "").strip()
    if folder:
        folder = os.path.abspath(os.path.expanduser(folder))
    try:
        with pdb.connect_closing() as conn:
            existing = pdb.find_by_primary_path(conn, folder) if folder else None
            if existing is not None:
                # Idempotent create: duplicates would render N identical sidebar subtrees.
                # Idempotent create: the folder already belongs to a project. Re-activating it beats minting
                # a duplicate — duplicated projects render N identical sidebar subtrees (#75820).
                pdb.set_active(conn, existing.id)
                proj = existing
            else:
                pid = pdb.create_project(conn, name=name, folders=[folder] if folder else [], primary_path=folder or None)
                pdb.set_active(conn, pid)
                proj = pdb.get_project(conn, pid)
    except ValueError as exc:
        return json.dumps({"success": False, "error": str(exc)})
    if proj is None:
        return json.dumps({"success": False, "error": "project vanished after create"})
    return _activated(proj, task_id)


def project_switch(project: str, task_id: Optional[str] = None) -> str:
    from hermes_cli import projects_db as pdb
    with pdb.connect_closing() as conn:
        proj = _resolve(conn, project)
        if proj is None:
            return json.dumps({"success": False, "error": f"no project matching '{project}'"})
        pdb.set_active(conn, proj.id)
    return _activated(proj, task_id)


def project_move(project: str, session_key: str, task_id: Optional[str] = None) -> str:
    """Re-home a stored thread without activating its destination for the caller."""
    if not isinstance(session_key, str) or not session_key.strip():
        return json.dumps({"success": False, "error": "session_key is required for move"})
    if session_key != session_key.strip():
        return json.dumps({"success": False, "error": "session_key must be an exact stored session ID without surrounding whitespace"})
    cb = _workspace_callback
    if cb is None or not task_id:
        return json.dumps({"success": False, "error": "move requires a connected Desktop session"})
    from hermes_cli import projects_db as pdb
    with pdb.connect_closing() as conn:
        proj = _resolve(conn, project)
    if proj is None:
        return json.dumps({"success": False, "error": f"no project matching '{project}'"})
    primary = _primary_path(proj)
    if not primary:
        return json.dumps({"success": False, "error": "destination project has no folder"})
    result = cb(task_id, primary, proj.name, session_key=session_key)
    if not isinstance(result, dict):
        return json.dumps({"success": False, "error": "native move not acknowledged; verify before retrying"})
    return json.dumps({"id": proj.id, "name": proj.name, **result})


_ACTIONS = {
    "list": lambda args, tid: project_list(task_id=tid),
    "create": lambda args, tid: project_create(
        name=args.get("name", ""), path=args.get("path"), task_id=tid),
    "switch": lambda args, tid: project_switch(project=args.get("name", ""), task_id=tid),
    "move": lambda args, tid: project_move(
        project=args.get("name", ""), session_key=args.get("session_key", ""), task_id=tid)}


def _handle_project(args, **kw):
    action = _ACTIONS.get((args.get("action") or "").strip())
    if action is None:
        return json.dumps({"success": False, "error": "action must be one of: create, switch, list, move."})
    return action(args, kw.get("task_id"))


# One action enum instead of three tools: each re-taught "desktop Projects" (244 -> ~145 tok).
# Consolidated (#95681, maintainer-directed): project_list/create/switch each re-taught "desktop Projects
# (named workspaces)"; one action enum says it once (244 -> ~145 tok).
registry.register(
    name="desktop_project",
    toolset="project",
    schema={
        "name": "desktop_project",
        "description": (
            "Organize desktop Projects and sessions. create: one and switch "
            "this chat into it — pass path to anchor it to a repo/folder (the "
            "chat's workspace moves there, the sidebar follows). switch: move "
            "this chat into an existing project by name/slug/id — the "
            "intentional way to move the session, not `cd`. move: move an existing thread by exact session_key into name, in this profile, without switching this chat. list: projects + active."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create", "switch", "list", "move"]},
                "name": {"type": "string", "description": "create: human name. switch/move: name, slug, or id."},
                "path": {"type": "string", "description": "create: repo/folder to anchor to."},
                "session_key": {"type": "string", "description": "move: exact stored session ID, not title or runtime ID."},
            },
            "required": ["action"],
        },
    },
    handler=_handle_project,
)
