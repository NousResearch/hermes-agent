#!/usr/bin/env python3
"""Project tools — the agent's INTENTIONAL handle on first-class Projects (per-profile
``projects.db``, the desktop sidebar's named workspaces). Creating/switching is an explicit
tool call, never a side effect of ``cd``. GUI-only: the `project` toolset stays off
``_HERMES_CORE_TOOLS``; the desktop/TUI gateway folds it in and wires
``set_project_workspace_callback`` so the live session's cwd and sidebar follow. A live
session's switch re-anchors only that session, and create never moves it; neither may move the
profile-global Desktop selection shared by concurrent chats."""

import json
import logging
import os
from typing import Callable, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

# Set by the GUI gateway: ``(task_id, primary_path, project_name)`` re-anchors that session's
# workspace. ``None`` in CLI/messaging — the DB write still happens, nothing to move. It returns
# False when it found no live session for *task_id* (or the folder is gone), and ``"pending"`` when
# the move is committed at the end of the calling turn; ``None`` (an older callback with no verdict)
# counts as moved.
_workspace_callback: Optional[Callable[[str, str, str], object]] = None


def set_project_workspace_callback(fn: Optional[Callable[[str, str, str], object]]) -> None:
    global _workspace_callback
    _workspace_callback = fn


def get_project_workspace_callback() -> Optional[Callable[[str, str, str], object]]:
    """The installed callback, so a second GUI host can wrap it (``gateway/local_desktop_ui.py``)."""
    return _workspace_callback


def _primary_path(proj) -> Optional[str]:
    if getattr(proj, "primary_path", None):
        return proj.primary_path
    for folder in proj.folders:
        if folder.is_primary:
            return folder.path
    return proj.folders[0].path if proj.folders else None


def _apply_workspace(task_id: Optional[str], path: Optional[str], name: str) -> tuple[object, Optional[str]]:
    """Move the calling session into *path*: ``(moved, why_not)``. ``moved`` is True, ``"pending"``
    (the move lands when the calling turn ends) or False with the reason, reported to the model so
    it never keeps working in a folder it believes it left."""
    cb = _workspace_callback
    if not path:
        return False, "project has no folder"
    if not task_id:
        return False, "no calling session"
    if cb is None:
        return False, "no live GUI session to move (CLI/messaging): only the active project changed"
    try:
        verdict = cb(task_id, path, name)
    except Exception as exc:
        logger.warning("desktop_project workspace callback failed for %s", task_id, exc_info=True)
        return False, f"workspace move failed: {exc}"
    if verdict is False:
        return False, "no live GUI session owns this task"
    return ("pending" if verdict == "pending" else True), None


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
    """Move the live session into *proj*; one that cannot move (CLI/messaging, a pathless project,
    no session found) switches the profile-global pointer instead. A live session that moves owns
    its workspace, so it leaves the pointer alone (a background chat would otherwise redirect the
    sidebar)."""
    from hermes_cli import projects_db as pdb
    primary = _primary_path(proj)
    moved, reason = _apply_workspace(task_id, primary, proj.name)
    if not moved:
        with pdb.connect_closing() as conn:
            pdb.set_active(conn, proj.id)
    result = {"success": True, "id": proj.id, "slug": proj.slug, "name": proj.name,
              "primary_path": primary, "moved": moved}
    if reason is not None:
        result["not_moved_reason"] = reason
    if moved == "pending":
        result["note"] = (f"This chat moves to {primary} when this turn ends; commands in this turn still "
                          "run in the current folder. If the folder is gone by then, the chat stays put.")
    return json.dumps(result)


def _calling_session_project_id(conn, task_id: Optional[str]) -> tuple[bool, Optional[str]]:
    """``(scoped, project_id)`` for the calling session. The GUI gateway registers each session's
    workspace (``cwd_source``) in the terminal override table, and a Desktop or TUI chat the gateway
    owns runs its turn with its frozen policy bound (its ``cwd`` is the workspace); any other caller
    (CLI, scripts, kanban) is unscoped and falls back to the profile-global pointer."""
    from hermes_cli import projects_db as pdb
    from tools.terminal_tool import resolve_task_overrides
    overrides = resolve_task_overrides(task_id) if task_id else {}
    if "cwd_source" in overrides:
        cwd = overrides.get("cwd") if overrides["cwd_source"] == "session" else overrides.get("host_workspace")
    else:
        try:
            from gateway.session_policy import active_policy
        except ImportError:  # install without the gateway package: no gateway turn can be running
            return False, None
        policy = active_policy()
        if not task_id or policy is None or policy.source not in ("gui", "tui"):
            return False, None
        cwd = policy.cwd
    project = pdb.project_for_path(conn, cwd) if cwd else None
    return True, project.id if project else None


def project_list(task_id: Optional[str] = None) -> str:
    from hermes_cli import projects_db as pdb
    with pdb.connect_closing() as conn:
        # Another tab's switch moves the profile-global pointer; this chat's project is its own cwd.
        scoped, active = _calling_session_project_id(conn, task_id)
        if not scoped:
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
                # Idempotent create: the folder already belongs to a project. Reusing it beats minting
                # a duplicate — duplicated projects render N identical sidebar subtrees (#75820).
                proj = existing
            else:
                pid = pdb.create_project(conn, name=name, folders=[folder] if folder else [], primary_path=folder or None)
                proj = pdb.get_project(conn, pid)
    except ValueError as exc:
        return json.dumps({"success": False, "error": str(exc)})
    if proj is None:
        return json.dumps({"success": False, "error": "project vanished after create"})
    # Create only creates: this chat stays where it is and the active project is untouched; `switch` moves.
    return json.dumps({"success": True, "id": proj.id, "slug": proj.slug, "name": proj.name,
                       "primary_path": _primary_path(proj)})


def project_switch(project: str, task_id: Optional[str] = None) -> str:
    from hermes_cli import projects_db as pdb
    with pdb.connect_closing() as conn:
        proj = _resolve(conn, project)
        if proj is None:
            return json.dumps({"success": False, "error": f"no project matching '{project}'"})
    return _activated(proj, task_id)


_ACTIONS = {
    "list": lambda args, tid: project_list(task_id=tid),
    "create": lambda args, tid: project_create(
        name=args.get("name", ""), path=args.get("path"), task_id=tid),
    "switch": lambda args, tid: project_switch(project=args.get("name", ""), task_id=tid)}


def _handle_project(args, **kw):
    action = _ACTIONS.get((args.get("action") or "").strip())
    if action is None:
        return json.dumps({"success": False, "error": "action must be one of: create, switch, list."})
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
            "Create or switch desktop Projects (named workspaces). create: make one; pass path to "
            "anchor it to a repo/folder. This chat stays where it is and the active project is unchanged. "
            "switch: move this chat into an existing project by name/slug/id (the chat's workspace "
            "moves there, the sidebar follows); the intentional way to move the session, not `cd`. "
            "list: all projects + which one this chat is in."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create", "switch", "list"]},
                "name": {"type": "string", "description": "create: human name. switch: name, slug, or id."},
                "path": {"type": "string", "description": "create: repo/folder to anchor to."},
            },
            "required": ["action"],
        },
    },
    handler=_handle_project,
)
