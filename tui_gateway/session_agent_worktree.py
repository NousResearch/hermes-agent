"""Agent worktree: the linked git worktree a session's terminal activity settled in, when that tree is NOT the
session's own workspace. Display-only — never re-homes the session cwd (a per-command ``workdir`` is transient by
contract, and a launch-artifact desktop cwd stays what it is). Adopted from terminal tool completions, persisted
in the row's ``model_config.agent_worktree``, restored on resume, revalidated at turn settle. Bodies are rebound
onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

from __future__ import annotations

import contextlib
import json

from tui_gateway import git_probe

from .method_ctx import bind_module

_TERMINAL_TOOL_NAMES = ("terminal",)


def _terminal_activity_cwd(session: dict, args: dict | None, result) -> str:
    """Where one terminal call ran: explicit ``workdir`` > the result's cwd echo (a ``cd`` landed somewhere) >
    terminal_tool's per-session record. Empty when nothing is known."""
    if isinstance(args, dict) and str(args.get("workdir") or "").strip():
        return str(args["workdir"]).strip()
    payload = result
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except Exception:
            payload = None
    if isinstance(payload, dict) and str(payload.get("cwd") or "").strip():
        return str(payload["cwd"]).strip()
    with contextlib.suppress(Exception):
        from tools.terminal_tool import get_session_cwd
        return str(get_session_cwd(session.get("session_key") or "") or "")
    return ""


def _linked_worktree_for(path: str) -> dict | None:
    """``{cwd, branch, repoRoot, projectName}`` when ``path`` sits inside a LINKED worktree (its own toplevel differs
    from the common repo root); None for the main checkout, a non-repo, or a missing dir."""
    resolved = os.path.abspath(os.path.expanduser(str(path or "")))
    if not resolved or not os.path.isdir(resolved):
        return None
    root = git_probe.repo_root(resolved)
    if not root:
        return None
    common = git_probe.common_repo_root(resolved)
    if not common or root.replace(os.sep, "/") == common:
        return None
    root_native = os.path.normpath(root)
    return {
        "cwd": root_native, "branch": git_probe.branch(root_native) or None, "repoRoot": common,
        "projectName": os.path.basename(common.rstrip("/")) or common,
    }


def _same_tree(a: str, b: str) -> bool:
    with contextlib.suppress(Exception):
        return os.path.realpath(a) == os.path.realpath(b)
    return a == b


def _agent_worktree_is_own_workspace(session: dict, tree_cwd: str) -> bool:
    """The badge is redundant once the session's own workspace IS that tree (explicit choice, settle-follow, or a
    Projects coding binding)."""
    own = str(session.get("cwd") or "")
    if own and _same_tree(own, tree_cwd):
        return True
    binding = session.get("coding_workspace")
    return bool(isinstance(binding, dict) and binding.get("cwd") and _same_tree(str(binding["cwd"]), tree_cwd))


def _persist_agent_worktree(session: dict) -> None:
    key = session.get("session_key")
    if not key:
        return
    with contextlib.suppress(Exception):
        with _session_db(session) as db:
            if db is not None and hasattr(db, "patch_session_model_config"):
                db.patch_session_model_config(key, {"agent_worktree": session.get("agent_worktree")})


def _set_agent_worktree(session: dict, value: dict | None) -> bool:
    """Returns changed."""
    if session.get("agent_worktree") == value:
        return False
    session["agent_worktree"] = value
    _persist_agent_worktree(session)
    return True


def _observe_terminal_activity(session: dict | None, args: dict | None, result) -> bool:
    """Adopt (or switch) the agent worktree from one terminal completion. Returns True when the badge changed.
    Sticky: activity outside any linked worktree (browsing ``/tmp``, the main checkout, another repo) neither adopts
    nor clears — turn settle revalidates. A coding-workspace binding owns the identity; nothing competes with it."""
    if not session or not _is_local_terminal_backend() or session.get("coding_workspace"):
        return False
    activity = _terminal_activity_cwd(session, args, result)
    if not activity:
        return False
    tree = _linked_worktree_for(activity)
    if tree is None or _agent_worktree_is_own_workspace(session, tree["cwd"]):
        return False
    return _set_agent_worktree(session, tree)


def _observe_tool_activity(session: dict | None, name: str, args: dict | None, result) -> bool:
    """tool.complete hook: only terminal calls carry a working directory."""
    if name not in _TERMINAL_TOOL_NAMES:
        return False
    return _observe_terminal_activity(session, args, result)


def _revalidate_agent_worktree(session: dict | None) -> bool:
    """Turn settle / resume: drop a badge whose tree is gone or has become the session's own workspace; refresh the
    branch otherwise. Returns True when the badge changed."""
    if not session or not isinstance(session.get("agent_worktree"), dict):
        return False
    current = session["agent_worktree"]
    tree = _linked_worktree_for(str(current.get("cwd") or "")) if _is_local_terminal_backend() else None
    if tree is None or _agent_worktree_is_own_workspace(session, tree["cwd"]):
        return _set_agent_worktree(session, None)
    return _set_agent_worktree(session, tree)


def _restore_agent_worktree(session: dict, model_config: dict | None) -> None:
    """Resume: adopt the persisted badge, then validate it against the disk (a removed worktree is dropped and the
    row is patched so the stale value does not come back next resume)."""
    stored = (model_config or {}).get("agent_worktree") if isinstance(model_config, dict) else None
    if not isinstance(stored, dict) or not stored.get("cwd"):
        return
    session["agent_worktree"] = stored
    _revalidate_agent_worktree(session)


def register(server) -> None:
    """Publish this module's helpers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
