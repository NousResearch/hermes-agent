"""``session.workspace.move`` (``methods_session`` split) with the explicit project binding.

Moved from ``tui_gateway/methods_session.py`` (file-line ratchet) and extended for
``sessions.project_id``: the body closes over server.py globals through ``method_ctx.bind_module``
exactly as before; registration still runs from the parent's ``register()``.

Binding rules (a session is in exactly ONE project):

* ``project_id`` = an active project's id/slug -> bind it; ``cwd`` defaults to the project folder.
* ``project_id`` = ``null`` / ``""`` -> unassign: clear the binding; ``cwd`` defaults to ``~``.
* ``project_id`` absent (legacy cwd-only move) -> the binding is kept while the new cwd stays inside
  the bound project's folders, and cleared when the session leaves them.
"""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


def _wsmove_projects_path(params: dict):
    """projects.db of the profile the request addresses (None = the launch profile's)."""
    home = _profile_home(params.get("profile"))
    return (home / "projects.db") if home is not None else None


def _wsmove_in_project(cwd: str, project) -> bool:
    """``cwd`` is one of ``project``'s folders or inside one (separator/case-normalized)."""
    import os
    target = os.path.normcase(os.path.abspath(os.path.expanduser(cwd))).rstrip("/\\")
    for folder in project.folders or []:
        root = os.path.normcase(os.path.abspath(os.path.expanduser(folder.path))).rstrip("/\\")
        if target == root or target.startswith(root + os.sep) or target.startswith(root + "/"):
            return True
    return False


def _wsmove_binding(rid, params: dict, db, target: str):
    """``(project, keep_or_clear, err)``: the project the move binds to (None = unbound) and whether
    ``sessions.project_id`` must be rewritten (``False`` = a legacy move that keeps the binding)."""
    from hermes_cli import projects_db as pdb
    if "project_id" in params:
        ref = str(params.get("project_id") or "").strip()
        if not ref:
            return None, True, None
        with pdb.connect_closing(_wsmove_projects_path(params)) as conn:
            project = pdb.resolve_session_project(conn, ref)
        if project is None:
            return None, True, _err(rid, 5062, f"unknown or archived project: {ref}")
        return project, True, None
    bound = ((db.get_session(target) if db is not None else None) or {}).get("project_id")
    if not bound:
        return None, False, None
    with pdb.connect_closing(_wsmove_projects_path(params)) as conn:
        return pdb.get_project(conn, bound), False, None


def _wsmove_target_cwd(params: dict, project) -> str:
    """The requested cwd, else the bound project's folder, else ``~`` (an unassign)."""
    from hermes_cli.projects_db import project_folder
    if raw := _str_param(params, "cwd"):
        return raw
    if project is not None and "project_id" in params:
        return project_folder(project) or ""
    return "~" if "project_id" in params else ""


def _wsmove_result(target_cwd: str, branch, root, project) -> dict:
    from hermes_cli.projects_db import session_project_ref
    return {"cwd": target_cwd, "branch": branch, "git_repo_root": root,
            "project_id": project.id if project is not None else None,
            "project": session_project_ref(project) if project is not None else None}


@method("session.workspace.move")
def _(rid, params: dict) -> dict:
    """Re-home a STORED session's workspace (by ``session_key``; no live agent required). git branch/root are
    REPLACED (a stale ``git_repo_root`` kept the session under the project it left); a live agent follows even
    mid-turn (refusing made the UI claim success while state.db kept the old cwd). ``project_id`` moves the
    explicit project binding with it (see the module docstring)."""
    if not (target := _str_param(params, "session_key")):
        return _err(rid, 4007, "session_key required")
    import sqlite3

    from hermes_constants import translate_cwd_for_wsl_backend
    from hermes_state_session_projects import set_session_project
    # Snapshot under the lock — concurrent RPCs mutate _sessions.
    with _sessions_lock:
        live_sid, live = next(
            ((sid, sess) for sid, sess in list(_sessions.items()) if sess.get("session_key") == target), ("", None))
    # The live session's profile decides (as _set_session_cwd below does); an ssh workspace is never host-validated.
    home = live.get("profile_home") if live is not None else _profile_home(params.get("profile"))
    with _profile_db(params, writer=True) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        project, rebind, err = _wsmove_binding(rid, params, db, target)
        if err:
            return err
        if not (raw := _wsmove_target_cwd(params, project)):
            return _err(rid, 4016, "cwd required")
        try:
            target_cwd = _workspace_cwd(home, translate_cwd_for_wsl_backend(raw))
        except ValueError:
            return _err(rid, 4017, f"working directory does not exist: {raw}")
        if project is not None and not rebind and not _wsmove_in_project(target_cwd, project):
            project, rebind = None, True  # a cwd-only move out of the bound project leaves it
        branch, root = git_probe.branch(target_cwd), git_probe.common_repo_root(target_cwd)
        # A draft has no row yet; the live re-home still applies (row inherits cwd on write).
        if not db.get_session(target):
            if live is None:
                return _err(rid, 4007, "session not found")
        else:
            try:
                db.update_session_cwd(target, target_cwd, branch, root, replace_git_meta=True)
                if rebind:
                    set_session_project(db, target, project.id if project is not None else None)
            except (sqlite3.Error, ValueError, OSError) as e:
                return _err(rid, 5007, f"move failed: {e}")
    if live is not None:
        try:
            _set_session_cwd(live, target_cwd)
        except ValueError as e:
            return _err(rid, 4017, str(e))
        _emit("session.info", live_sid, _cwd_info(live, target_cwd, branch=branch))
    return _ok(rid, _wsmove_result(target_cwd, branch, root, project))


def register(server) -> None:
    """Publish this module's helpers onto ``server`` (rebound to its globals) and install handlers."""
    bind_module(globals(), server, skip=("_",))
