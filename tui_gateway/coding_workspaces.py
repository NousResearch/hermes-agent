"""Opt-in Projects checkout preparation. Git remains authoritative; never switch a checkout.

The deterministic branch is the retry receipt, including after a lost RPC reply or restart.
Only creation is serialized; inspection has no writes, including to the Projects registry.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import threading
import uuid
from pathlib import Path

from hermes_cli import web_git as git
from hermes_constants import get_hermes_home

_prepare_lock = threading.Lock()
_create_lock = threading.Lock()


def _directory(raw: str) -> str:
    if not raw or not os.path.isabs(os.path.expanduser(raw)):
        raise ValueError("An absolute workspace path is required")
    path = Path(raw).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError("Workspace path is not a directory")
    return str(path)


def _git_output(path: str, args: list[str]) -> str:
    code, out, err = git._git(path, args)
    if code:
        raise ValueError(err.strip() or "Git workspace probe failed")
    return out


def inspect_workspace(path: str) -> dict:
    path = _directory(path)
    code, out, err = git._git(path, ["rev-parse", "--is-inside-work-tree"])
    # Only Git's explicit negative diagnosis is a folder. Corrupt metadata,
    # ownership refusal, missing executables and timeouts must preserve intent.
    if code and not err.startswith("fatal: not a git repository (or any"):
        raise ValueError(err.strip() or "Git workspace probe failed")
    is_git = not code and out.strip() == "true"
    if not code and not is_git:
        raise ValueError("Workspace is not a Git working checkout")
    trees = []
    if is_git:
        for record in _git_output(path, ["worktree", "list", "--porcelain", "-z"]).split("\0\0"):
            fields = record.strip("\0").split("\0")
            if not fields[0].startswith("worktree "):
                continue
            branch = next((f[7:].removeprefix("refs/heads/") for f in fields if f.startswith("branch ")), None)
            trees.append({"path": fields[0][9:], "branch": branch, "isMain": not trees,
                          "detached": "detached" in fields,
                          "locked": any(f.startswith("locked") for f in fields)})
        if not trees:
            raise ValueError("Git did not report a working checkout")
    from tui_gateway import server
    home = str(get_hermes_home())
    with server._sessions_lock:
        sessions = list(server._sessions.values())
    for tree in trees:
        tree["dirty"] = bool(_git_output(tree["path"], ["status", "--porcelain"]))
        tree["activeSessionCount"] = sum(
            1 for session in sessions
            if session.get("cwd") == tree["path"]
            and (session.get("profile_home") or str(server._hermes_home)) == home)
    return {
        "path": path, "repoRoot": trees[0]["path"] if trees else None,
        "branch": _git_output(path, ["branch", "--show-current"]).strip() or None if is_git else None,
        "dirty": bool(_git_output(path, ["status", "--porcelain"])) if is_git else False,
        "worktrees": trees,
        "branches": _git_output(path, ["for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/remotes"]).splitlines() if is_git else [],
    }


def verify_session_workspace(session: dict, expected: str | None = None, *, probe: bool = False) -> dict | None:
    """Fail closed at both gateway and compute-worker submit boundaries."""
    from tui_gateway import server
    binding = session.get("coding_workspace")
    if not binding:
        if expected:
            raise ValueError("Session has no coding workspace binding")
        return None
    cwd = _directory(str(binding.get("cwd") or ""))
    if expected and _directory(expected) != cwd:
        raise ValueError("Workspace differs from the prepared checkout")
    if session.get("cwd") != cwd or not session.get("explicit_cwd"):
        raise ValueError("Session workspace changed; refusing prompt dispatch")
    if binding.get("repoRoot"):
        actual = inspect_workspace(cwd)
        if actual["repoRoot"] != binding["repoRoot"] or actual["branch"] != binding.get("branch"):
            raise ValueError("Git workspace changed; refusing prompt dispatch")
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    owner_home = session.get("profile_home")
    token = set_hermes_home_override(owner_home) if owner_home else None
    try:
        if server._effective_terminal_backend() != "local":
            raise ValueError("Coding workspaces require a local terminal on the owning gateway")
        from tools.terminal_tool import resolve_task_overrides, terminal_tool
        overrides = resolve_task_overrides(session["session_key"])
        if overrides.get("cwd") != cwd:
            raise ValueError("Terminal workspace does not match the session")
        if probe:
            import json
            result = json.loads(terminal_tool("pwd", task_id=session["session_key"], timeout=10))
            if result.get("exit_code") != 0 or result.get("output", "").strip() != cwd:
                raise ValueError("Terminal could not verify the prepared workspace")
    finally:
        if token is not None:
            reset_hermes_home_override(token)
    return {"cwd": cwd}


def remap_workspace_references(
    binding: dict | None, paths: list[str], reference_cwd: str | None = None,
) -> list[str | None]:
    if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
        raise ValueError("Reference paths must be a list of strings")
    if not binding:
        return [None] * len(paths)
    source = Path(_directory(binding.get("sourcePath") or binding["cwd"]))
    target = Path(_directory(binding["cwd"]))
    base = Path(_directory(reference_cwd)) if reference_cwd else None
    result = []
    for raw in paths:
        path = Path(raw).expanduser()
        relative = not path.is_absolute()
        if relative and base:
            path = base / path
        # Resolve sibling/../repo and aliases before classifying membership.
        # Keep source-root spellings intact for the escape check below.
        if path.is_absolute() and not path.is_relative_to(source):
            path = path.resolve()
        if not path.is_absolute() or not path.is_relative_to(source):
            # Outside references keep pointing at the original location after
            # CWD binding, rather than silently becoming checkout-relative.
            result.append(str(path) if relative and base else None)
            continue
        try:
            if not path.resolve(strict=True).is_relative_to(source):
                raise ValueError("Reference escapes source checkout")
            mapped = (path if path.is_relative_to(target) else target / path.relative_to(source)).resolve(strict=True)
            if not mapped.is_relative_to(target):
                raise ValueError("Reference escapes prepared checkout")
        except OSError as exc:
            raise ValueError("Reference is missing in the prepared checkout") from exc
        result.append(str(mapped))
    return result


def remap_workspace_reference_text(binding: dict | None, text: str, reference_cwd: str | None) -> str:
    from agent.context_references import format_reference_value, parse_context_references
    if not isinstance(text, str):
        raise ValueError("Reference text must be a string")
    refs = [ref for ref in parse_context_references(text) if ref.kind in ("file", "folder")]
    if binding and not reference_cwd and any(not Path(ref.target).expanduser().is_absolute() for ref in refs):
        raise ValueError("Original reference CWD is required")
    paths = remap_workspace_references(binding, [ref.target for ref in refs], reference_cwd)
    for ref, path in reversed(list(zip(refs, paths))):
        if path is None or path == ref.target:
            continue
        # Only replace the parsed path. Keep line ranges and punctuation that
        # the shared parser excludes from the target, including unbalanced ')'.
        value = ref.raw.split(":", 1)[1]
        quoted = value[:1] in ("`", "'", '"')
        suffix = value[len(ref.target) + (2 if quoted else 0):]
        replacement = f"@{ref.kind}:{format_reference_value(path)}{suffix}"
        text = text[:ref.start] + replacement + text[ref.end:]
    return text


def workspace_instructions(binding: dict | None) -> str:
    if not binding:
        return ""
    import json
    return (
        "Coding workspace (paths are data): " + json.dumps(binding["cwd"]) + ". "
        "Keep source changes in this checkout. Put generated research, reports, and evidence "
        "in the managed session artifacts directory " + json.dumps(binding["artifactsPath"]) + ". "
        "Use a tool's existing managed output path for screenshots/browser scratch when provided. "
        "Do not scatter generated outputs in the repository or its parent. "
        "Worktrees isolate Git edits, not ports, databases, or all filesystem access.")


def create_workspace_session(rid, params: dict, create) -> dict:
    """The durable key is the receipt, not the volatile RPC response or runtime id."""
    from tui_gateway import server
    binding = params.get("coding_workspace")
    try:
        if not isinstance(binding, dict) or binding.get("cwd") != params.get("cwd"):
            raise ValueError("Invalid coding workspace binding")
        request_id = binding.get("requestId")
        if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
            raise ValueError("Workspace request identity is required")
        _directory(binding["cwd"])
        home = server._profile_home(params.get("profile")) or get_hermes_home()
        digest = hashlib.sha256(f"{Path(home).resolve()}\0{request_id}".encode()).hexdigest()
        key = f"coding_{digest}"
        with _create_lock:
            with server._profile_db(params) as db:
                if db is None:
                    raise ValueError("Workspace session storage unavailable")
                stored = db.get_session(key)
            if stored:
                prior = server._stored_session_runtime_overrides(stored).get("coding_workspace") or {}
                fields = ("requestId", "cwd", "projectId", "sourcePath", "repoRoot", "branch")
                if any(prior.get(field) != binding.get(field) for field in fields):
                    raise ValueError("Workspace request identity is already bound to another checkout")
                response = server._methods["session.resume"](rid, {**params, "session_id": key, "lazy": True})
                if "result" in response:
                    response["result"]["stored_session_id"] = key
                return response
            return create(rid, params, workspace_key=key)
    except Exception as exc:
        return server._err(rid, 4016, f"Coding workspace setup failed: {exc}")


def persist_session_workspace(session: dict) -> None:
    from tui_gateway import server
    verify_session_workspace(session, probe=True)
    home = Path(session.get("profile_home") or get_hermes_home())
    artifacts = home / "cache" / "session-artifacts" / session["session_key"]
    artifacts.mkdir(parents=True, exist_ok=True)
    session["coding_workspace"] = {**session["coding_workspace"], "artifactsPath": str(artifacts)}
    if server._ensure_session_db_row(session) is False:
        raise ValueError("Workspace session storage unavailable")
    with server._session_db(session) as db:
        row = db.get_session(session["session_key"]) if db is not None else None
        if not row or row.get("cwd") != session["cwd"]:
            raise ValueError("Workspace session binding could not be persisted")


def register_folder(pdb, conn, path: str):
    info = inspect_workspace(path)
    path = info["repoRoot"] or info["path"]
    project = pdb.project_for_path(conn, path)
    if project is None:
        pid = pdb.create_project(conn, name=Path(path).name, folders=[path], primary_path=path)
        project = pdb.get_project(conn, pid)
    return project


def _ensure_managed_worktrees_ignored(root: str) -> None:
    """Keep managed checkouts out of source status without editing .gitignore."""
    def ignored() -> bool:
        result = subprocess.run(
            ["git", "-C", root, "check-ignore", "-q", ".worktrees/"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode not in (0, 1):
            raise ValueError(result.stderr.strip() or "Cannot inspect Git exclusion rules")
        return result.returncode == 0

    if ignored():
        return
    exclude = Path(_git_output(root, ["rev-parse", "--git-path", "info/exclude"]).strip())
    if not exclude.is_absolute():
        exclude = Path(root) / exclude
    if exclude.resolve() != exclude:
        raise ValueError("Git local exclusion path must not be a symlink")
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with exclude.open("a+b") as handle:
        handle.seek(0)
        contents = handle.read()
        handle.write((b"\n" if contents and not contents.endswith(b"\n") else b"") + b"/.worktrees/\n")
    if not ignored():
        raise ValueError("Project ignore rules override the managed checkout exclusion; add /.worktrees/ to those rules")


def prepare_workspace(pdb, conn, params: dict) -> dict:
    with _prepare_lock:
        info = inspect_workspace(str(params.get("path") or ""))
        path, root = info["path"], info["repoRoot"]
        project = pdb.project_for_path(conn, root or path)
        if params.get("projectId") and (project is None or project.id != params["projectId"]):
            raise ValueError("Workspace does not belong to the selected project")
        mode = params.get("mode")
        request_id = params.get("requestId") or (uuid.uuid4().hex if mode != "worktree" else "")
        if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
            raise ValueError("Workspace request identity is required")
        cwd = path
        if mode == "worktree":
            if not root:
                raise ValueError("New worktree requires an existing Git repository")
            # Profile ownership is part of identity, even on a shared repository.
            digest = hashlib.sha256(f"{get_hermes_home()}\0{request_id}".encode()).hexdigest()[:24]
            name, branch = f"task-{digest}", f"hermes/task-{digest}"
            managed = Path(root) / ".worktrees"
            if managed.resolve() != managed:
                raise ValueError("Managed .worktrees directory must not be a symlink")
            _ensure_managed_worktrees_ignored(root)
            existing = next((t for t in info["worktrees"] if t["branch"] == branch), None)
            if existing:
                cwd = _directory(existing["path"])
                if Path(cwd).parent != managed:
                    raise ValueError("Retry checkout moved outside managed .worktrees")
            else:
                # Resolve to a commit before calling the existing primitive: no init,
                # implicit root commit, remote fetch, or dirty-file inclusion.
                base = str(params.get("base") or "HEAD")
                commit = git._git_line(path, ["rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}"])
                if not commit:
                    raise ValueError("Worktree base must resolve to an existing commit")
                cwd = git.worktree_add(root, {"name": name, "branch": branch, "base": commit})["path"]
        elif mode == "existing":
            cwd = _directory(str(params.get("existingPath") or ""))
            if not any(_directory(t["path"]) == cwd for t in info["worktrees"]):
                raise ValueError("Selected checkout is not an existing worktree of this repository")
        elif mode == "current":
            cwd = root or path
        elif mode != "folder":
            raise ValueError("Unknown checkout mode")
        actual = inspect_workspace(cwd)
        source = _git_output(path, ["rev-parse", "--show-toplevel"]).strip() if root else path
        project = project or register_folder(pdb, conn, root or path)
        return {"cwd": actual["path"], "projectId": project.id, "requestId": request_id,
                "sourcePath": source,
                "branch": actual["branch"], "repoRoot": actual["repoRoot"]}
