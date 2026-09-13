"""GitHub repo-reference JSON-RPC handler for the desktop transcript: resolves the repo
behind a session's cwd to ``{host, owner, repo}`` so ``#123`` prose refs can render as
github.com links. Runs where the gateway runs (local Electron backend or remote gateway),
so the origin always reads from the filesystem the sessions actually work in. Bodies are
rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped

_E_GIT, _E_ARG = 5065, 4065


def _owner_repo_from_origin(url: str) -> tuple[str, str]:
    """``owner``/``repo`` from an HTTPS or SSH GitHub origin URL, or ``("", "")``.
    ``.git`` suffixes and trailing slashes are stripped; non-github hosts return empty
    (the desktop renders only github.com refs by design)."""
    raw = (url or "").strip()
    if raw.startswith("https://"):
        rest = raw[len("https://") :].split("/", 1)
    elif raw.startswith("git@"):
        rest = raw[len("git@") :].split(":", 1)
    else:
        return "", ""
    if len(rest) != 2 or "/" not in rest[1]:
        return "", ""
    host = rest[0].strip().lower()
    if host not in ("github.com", "www.github.com"):
        return "", ""
    owner, _, repo = rest[1].partition("/")
    repo = repo.rsplit("/", 1)[0].removesuffix(".git").strip("/")
    return (owner.strip(), repo) if owner and repo else ("", "")


def _session_repo_refs(cwd: str) -> dict | None:
    """``{host, owner, repo}`` for ``cwd``'s repo, or None. Cached per repo root —
    an origin URL survives branch switches, and re-running ``git remote`` per chip
    would multiply one git spawn by every `#N` in the transcript."""
    from tui_gateway import git_probe

    root = git_probe.common_repo_root(cwd)
    if not root:
        return None
    refs = _refs_cache.get(root)
    if refs is None:
        url = git_probe.run_git(root, "remote", "get-url", "origin")
        owner, repo = _owner_repo_from_origin(url)
        refs = (
            {"host": "github.com", "owner": owner, "repo": repo, "repo_root": root}
            if owner
            else None
        )
        _refs_cache[root] = refs
    return refs


# repo root -> refs (or None for a github-less repo). Process-lifetime like git_probe's
# root cache: a re-pointed origin shows up after a gateway/backend restart, which is the
# same refresh story every other cwd-derived surface (sidebar lanes, Projects) has.
_refs_cache: dict[str, dict | None] = {}


@method("git.repo_refs")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """GitHub coordinates for a session's cwd: ``{available, host, owner, repo}``.
    ``available`` is false outside a git repo, without a github.com origin, or when
    the desktop kill-switch (``desktop.autolink_issue_refs``) is off — callers then
    render the prose untouched."""
    sid = str(params.get("session_id") or "")
    if not sid:
        return _err(rid, _E_ARG, "session_id required")
    if not _autolink_issue_refs_enabled():
        return _ok(rid, {"available": False})
    session = _sessions.get(sid)
    if session is None:
        return _ok(rid, {"available": False})
    cwd = str(session.get("cwd") or "")
    refs = _session_repo_refs(cwd) if cwd else None
    if refs is None:
        return _ok(rid, {"available": False})
    return _ok(
        rid,
        {
            "available": True,
            "host": refs["host"],
            "owner": refs["owner"],
            "repo": refs["repo"],
            "repo_root": refs["repo_root"],
            "cwd": cwd,
        },
    )


def _autolink_issue_refs_enabled() -> bool:
    """Desktop kill-switch: ``desktop.autolink_issue_refs`` (default true)."""
    try:
        from hermes_cli.config import load_config

        desktop = load_config().get("desktop") or {}
    except Exception:
        return True
    return desktop.get("autolink_issue_refs") is not False


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
