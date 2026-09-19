"""Projects RPC surface: per-profile multi-folder workspaces, repo discovery, sidebar tree. Reaches server.py state through ``srv`` (method_ctx.py)."""

from __future__ import annotations

from .method_ctx import HandlerRegistry, bind_module
from .contracts.base import Params
from .contracts.projects_pets import (
    ActiveIdResult, OptionalProjectResult, ProjectFolderParams, ProjectIdParams, ProjectResult,
    ProjectsAddFolderParams, ProjectsArchiveParams, ProjectsCreateParams, ProjectsForCwdParams,
    ProjectsForCwdResult, ProjectsPayload, ProjectsSetActiveParams, ProjectsUpdateParams,
)
import contextlib
import json
import os
import logging

logger = logging.getLogger("tui_gateway.server")  # siblings log as the gateway facade (operators and caplog filter on it)

_registry = HandlerRegistry()
method = _registry.method


# JSON-RPC error codes: generic failure / id resolved to nothing / invalid argument.
_E_PROJECTS, _E_NO_PROJECT, _E_PROJECT_ARG = 5061, 5062, 5063


class _NoProject(Exception):
    """Raised inside a projects handler when ``params['id']`` resolves to None."""


def _project_info(project):
    from tui_gateway.contracts.projects_pets import ProjectInfo
    return ProjectInfo.model_validate(project.to_dict())


def _projects_payload(conn) -> ProjectsPayload:
    from hermes_cli import projects_db as pdb
    return ProjectsPayload(
        projects=[srv._project_info(project) for project in pdb.list_projects(conn, include_archived=True)],
        active_id=pdb.get_active_id(conn),
    )


def _projects_method(name: str):
    """Register a projects RPC, injecting (pdb, conn) and unifying error mapping; profile-scoped
    so app-global remote mode reads that profile's ``projects.db``."""
    def decorator(fn):
        @method(name)
        @_registry.profile_scoped
        def handler(rid, params) -> dict:
            try:
                from hermes_cli import projects_db as pdb
                with pdb.connect_closing() as conn:
                    return fn(rid, params, pdb, conn)
            except _NoProject:
                return srv._err(rid, _E_NO_PROJECT, "no such project")
            except ValueError as e:
                return srv._err(rid, _E_PROJECT_ARG, str(e))
            except Exception as e:
                return srv._err(rid, _E_PROJECTS, str(e))
        return handler
    return decorator


def _require_project(pdb, conn, project_id: str):
    """The project named by ``project_id`` (or raise ``_NoProject``)."""
    proj = pdb.get_project(conn, project_id)
    if proj is None:
        raise _NoProject
    return proj


@_projects_method("projects.update")
def _(rid, params: ProjectsUpdateParams, pdb, conn) -> ProjectResult | dict:
    project = srv._require_project(pdb, conn, params.id)
    pdb.update_project(
        conn, project.id, name=params.name, description=params.description, icon=params.icon,
        color=params.color, board_slug=params.board_slug)
    return ProjectResult(project=srv._project_info(pdb.get_project(conn, project.id)))


@_projects_method("projects.add_folder")
def _(rid, params: ProjectsAddFolderParams, pdb, conn) -> ProjectResult | dict:
    project = srv._require_project(pdb, conn, params.id)
    pdb.add_folder(conn, project.id, params.path, label=params.label, is_primary=params.is_primary)
    return ProjectResult(project=srv._project_info(pdb.get_project(conn, project.id)))


@_projects_method("projects.remove_folder")
def _(rid, params: ProjectFolderParams, pdb, conn) -> ProjectResult | dict:
    project = srv._require_project(pdb, conn, params.id)
    pdb.remove_folder(conn, project.id, params.path)
    return ProjectResult(project=srv._project_info(pdb.get_project(conn, project.id)))


@_projects_method("projects.set_primary")
def _(rid, params: ProjectFolderParams, pdb, conn) -> ProjectResult | dict:
    project = srv._require_project(pdb, conn, params.id)
    pdb.set_primary(conn, project.id, params.path)
    return ProjectResult(project=srv._project_info(pdb.get_project(conn, project.id)))


@_projects_method("projects.list")
def _(rid, params: Params, pdb, conn) -> ProjectsPayload | dict:
    return srv._projects_payload(conn)


@_projects_method("projects.get")
def _(rid, params: ProjectIdParams, pdb, conn) -> ProjectResult | dict:
    return ProjectResult(project=srv._project_info(srv._require_project(pdb, conn, params.id)))


@_projects_method("projects.create")
def _(rid, params: ProjectsCreateParams, pdb, conn) -> OptionalProjectResult | dict:
    pid = pdb.create_project(
        conn, name=params.name, folders=params.folders or [], slug=params.slug,
        primary_path=params.primary_path, description=params.description, icon=params.icon,
        color=params.color, board_slug=params.board_slug)
    if params.use:
        pdb.set_active(conn, pid)
    proj = pdb.get_project(conn, pid)
    return OptionalProjectResult(project=srv._project_info(proj) if proj else None)


@_projects_method("projects.archive")
def _(rid, params: ProjectsArchiveParams, pdb, conn) -> ProjectsPayload | dict:
    proj = srv._require_project(pdb, conn, params.id)
    (pdb.restore_project if params.restore else pdb.archive_project)(conn, proj.id)
    return srv._projects_payload(conn)


@_projects_method("projects.delete")
def _(rid, params: ProjectIdParams, pdb, conn) -> ProjectsPayload | dict:
    pdb.delete_project(conn, srv._require_project(pdb, conn, params.id).id)
    return srv._projects_payload(conn)


@_projects_method("projects.set_active")
def _(rid, params: ProjectsSetActiveParams, pdb, conn) -> ActiveIdResult | dict:
    pdb.set_active(conn, srv._require_project(pdb, conn, params.id).id if params.id else None)
    return ActiveIdResult(active_id=pdb.get_active_id(conn))


@_projects_method("projects.for_cwd")
def _(rid, params: ProjectsForCwdParams, pdb, conn) -> ProjectsForCwdResult | dict:
    cwd = srv._completion_cwd(cwd=params.cwd.strip() if params.cwd else None)
    proj = pdb.project_for_path(conn, cwd)
    return ProjectsForCwdResult(
        project=srv._project_info(proj) if proj else None, cwd=cwd, branch=srv.git_probe.branch(cwd))


def _non_workspace_dirs() -> set[str]:
    """Never-a-workspace dirs: ``/``, the user's home, the dir homes live in, plus both POSIX
    spellings on every host (remote shells hand back Linux paths; promoting one mints a
    catch-all project)."""
    home = os.path.realpath(os.path.expanduser("~"))
    candidates = (os.sep, home, os.path.dirname(home), "/home", "/Users")
    return {os.path.normcase(os.path.realpath(path)) for path in candidates if path}


def _is_repo_junk(root: str) -> bool:
    """A git root never auto-surfaced as a project: a non-workspace dir or anything under
    HERMES_HOME. User-created projects pointing there are still honored."""
    if not root:
        return True
    from hermes_constants import get_hermes_home
    real = os.path.realpath(root)
    hermes_home = os.path.realpath(str(get_hermes_home()))
    return (
        os.path.normcase(real) in srv._non_workspace_dirs()
        or real == hermes_home
        or real.startswith(hermes_home + os.sep))


def _is_session_cwd_junk(cwd: str) -> bool:
    """A non-git cwd that stays in flat Recents. A DESCENDANT of HERMES_HOME may be an
    intentional prose/data workspace, so only HERMES_HOME itself is excluded here."""
    if not cwd:
        return True
    from hermes_constants import get_hermes_home
    real = os.path.normcase(os.path.realpath(cwd))
    hermes_home = os.path.normcase(os.path.realpath(str(get_hermes_home())))
    return real in srv._non_workspace_dirs() or real == hermes_home


def _repo_discovery_policy(raw: dict | None = None) -> dict:
    """Return the effective, profile-local Desktop repository scan policy."""
    from hermes_cli.config import DEFAULT_CONFIG
    defaults = DEFAULT_CONFIG["desktop"]
    source = raw if isinstance(raw, dict) else (srv._load_cfg().get("desktop") or {})
    if not isinstance(source, dict):
        source = {}

    def _get(short: str, long: str):
        return source.get(short, source.get(long, defaults[long]))

    def _paths(short: str, long: str) -> list[str]:
        values = _get(short, long)
        if not isinstance(values, list):
            return list(defaults[long])
        return [v.strip() for v in values if isinstance(v, str) and v.strip()]
    enabled = _get("enabled", "repo_scan_enabled")
    return {
        "enabled": enabled if isinstance(enabled, bool) else defaults["repo_scan_enabled"],
        "roots": _paths("roots", "repo_scan_roots"),
        "exclude_paths": _paths("exclude_paths", "repo_scan_exclude_paths")}


def _repo_discovery_policy_key(policy: dict) -> str:
    def _paths(values: list[str]) -> list[str]:
        home = os.path.expanduser("~")
        return sorted({
            os.path.normcase(os.path.abspath(os.path.join(home, os.path.expanduser(v))))
            for v in values})
    canonical = {
        "enabled": bool(policy["enabled"]), "roots": _paths(policy["roots"]),
        "exclude_paths": _paths(policy["exclude_paths"])}
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _repo_discovery_policy_is_default(policy: dict) -> bool:
    from hermes_cli.config import DEFAULT_CONFIG
    return srv._repo_discovery_policy_key(policy) == srv._repo_discovery_policy_key(
        srv._repo_discovery_policy(DEFAULT_CONFIG["desktop"]))


def _scan_discovered_repos_remote(conn, policy: dict) -> bool:
    """Backend-side disk scan of the policy roots into the discovery cache. Best-effort:
    failures log and leave the cache untouched. True only when the scan is authoritative
    (every root walked to completion, cap not hit) — only then is the cache write
    ``replace=True``; a partial/errored scan must MERGE, or a failed refresh blanks the sidebar.

    The desktop's native repo scan only runs on the local filesystem. On a remote gateway connection the
    host must scan its own disk so repos with zero Hermes sessions still appear in the sidebar (#81723).
    Mirrors the desktop's behavior: walk each root (bounded depth), find `.git` directories, record (root,
    label) pairs into the discovery cache.
    See #81723.
    """
    from hermes_cli import projects_db as pdb
    roots = policy.get("roots") or []
    excludes = policy.get("exclude_paths") or []
    pairs: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    authoritative = True

    def _is_excluded(path: str) -> bool:
        return any(
            path == ex or path.startswith(ex.rstrip("/\\") + os.sep) for ex in excludes if ex)
    for root in roots:
        if not os.path.isdir(root):
            # `os.walk` on a missing root yields nothing; an unmounted volume must not wipe.
            authoritative = False
            logger.debug("discover_repos scan root missing, skipping: %s", root)
            continue
        try:
            for dirpath, dirnames, _filenames in os.walk(root):
                if _is_excluded(dirpath):
                    dirnames[:] = []
                elif ".git" in dirnames:  # check BEFORE pruning hidden dirs — `.git` is hidden
                    if dirpath not in seen:
                        seen.add(dirpath)
                        pairs.append((dirpath, os.path.basename(dirpath)))
                    dirnames[:] = []  # don't hunt nested repos inside a repo
                else:
                    dirnames[:] = [
                        d for d in dirnames if not d.startswith(".") and d != "node_modules"]
                if len(pairs) >= 500:
                    break
        except Exception:
            authoritative = False
            logger.debug("discover_repos scan failed for root %s", root, exc_info=True)
        if len(pairs) >= 500:  # cap hit: the walk didn't cover the full roots
            authoritative = False
            break
    if pairs:
        try:
            pdb.record_discovered_repos(
                conn, pairs, replace=authoritative, policy_key=srv._repo_discovery_policy_key(policy))
        except Exception:
            logger.debug("discover_repos cache write failed", exc_info=True)
            authoritative = False
    return authoritative


def _discover_repos_payload(
    db, *, conn=None, backfill: bool = True, include_cached: bool = True) -> list[dict]:
    """Merge cached filesystem-scanned repos with session-derived roots, junk-filtered, with
    session totals. ``backfill`` persists resolved roots onto session rows — kept OFF the
    per-turn tree path and done only on explicit refresh."""
    repos: dict[str, dict] = {}

    def _agg(root: str) -> dict:
        return repos.setdefault(
            root, {"root": root, "label": "", "sessions": 0, "last_active": 0.0})
    cwd_rows = list(db.distinct_session_cwds())
    # Parallel-warm the per-cwd git probes so a cold first paint doesn't serialize them.
    srv.git_probe.warm_roots(str(r.get("cwd") or "") for r in cwd_rows)
    cwd_to_root: dict[str, str] = {}
    for row in cwd_rows:
        cwd = str(row.get("cwd") or "")
        root = srv.git_probe.common_repo_root(cwd)
        if not root:
            continue
        cwd_to_root[cwd] = root
        if srv._is_repo_junk(root):
            continue
        agg = _agg(root)
        agg["sessions"] += int(row.get("sessions") or 0)
        agg["last_active"] = max(agg["last_active"], float(row.get("last_active") or 0))
    # A read-only handle (foreign-profile RPC) must not attempt the persistence write: it would
    # raise and be swallowed here, silently dropping the backfill. That profile's own gateway
    # backfills on its own refreshes.
    if backfill and not getattr(db, "read_only", False):
        try:
            db.backfill_repo_roots(cwd_to_root)
        except Exception:
            logger.debug("failed to backfill repo roots", exc_info=True)
    if include_cached:
        # `last_seen` is scan time, not user activity — never fold it into `last_active`.
        try:
            from hermes_cli import projects_db as pdb
            with (contextlib.nullcontext(conn) if conn is not None else pdb.connect_closing()) as c:
                for entry in pdb.list_discovered_repos(c):
                    root = str(entry.get("root") or "")
                    if root and not srv._is_repo_junk(root):
                        agg = _agg(root)
                        if entry.get("label"):
                            agg["label"] = entry["label"]
        except Exception:
            logger.debug("failed to read discovered repo cache", exc_info=True)
    out = sorted(repos.values(), key=lambda r: r["last_active"], reverse=True)
    for r in out:
        r["label"] = r["label"] or os.path.basename(r["root"].rstrip("/\\")) or r["root"]
    return out


# Not user conversations; subagent/compression children are dropped by include_children=False.
_PROJECT_TREE_EXCLUDED_SOURCES = ["cron", "kanban", "oneshot"]


def _project_tree_row(r: dict) -> dict:
    """Project a SessionDB row to the minimal shape the sidebar renders (grouping fields +
    what ``SidebarSessionRow`` reads), minus the heavy columns."""
    row = {k: r.get(k) for k in (
        "id", "_lineage_root_id", "_lineage_ids", "parent_session_id", "title", "preview")}
    row.update(
        started_at=r.get("started_at") or 0, ended_at=r.get("ended_at"),
        last_active=r.get("last_active") or r.get("started_at") or 0,
        source=r.get("source"), archived=bool(r.get("archived")),
        **{k: r.get(k) or 0 for k in (
            "message_count", "tool_call_count", "input_tokens", "output_tokens")},
        **{k: r.get(k) for k in ("actual_cost_usd", "estimated_cost_usd", "model")},
        is_active=False, **{k: r.get(k) for k in ("cwd", "git_branch", "git_repo_root")})
    return row


def _project_tree_inputs(
    db, session_limit: int, *, include_discovered: bool
) -> tuple[list[dict], list[dict], list[dict], str | None]:
    """Gather (sessions, projects, discovered_repos, active_id) for build_tree.
    ``include_discovered`` is the zero-session-repo overview tier; drill-in skips it (and
    the distinct-cwd scan + git probes) on that per-turn path."""
    # compact_rows: selecting the system-prompt blob only to drop it costs tens of MB of reads.
    rows = db.list_sessions_rich(
        limit=session_limit, offset=0, order_by_last_active=True, min_message_count=1,
        include_children=False, exclude_sources=srv._PROJECT_TREE_EXCLUDED_SOURCES,
        include_archived=False, compact_rows=True)
    sessions = [srv._project_tree_row(r) for r in rows]
    # Parallel-warm the git cache so build_tree's resolver doesn't cold-probe each cwd in turn.
    srv.git_probe.warm_roots(s["cwd"] for s in sessions if s.get("cwd"))
    from hermes_cli import projects_db as pdb
    policy = srv._repo_discovery_policy()
    policy_key = srv._repo_discovery_policy_key(policy)
    with pdb.connect_closing() as conn:
        if include_discovered:
            pdb.reconcile_discovered_repos_policy(
                conn, policy_key, preserve_unversioned=srv._repo_discovery_policy_is_default(policy))
        projects = [p.to_dict() for p in pdb.list_projects(conn)]
        active_id = pdb.get_active_id(conn)
        # backfill stays off the hot tree path — grouping uses the live resolver.
        discovered = []
        if include_discovered:
            discovered = srv._discover_repos_payload(
                db, conn=conn, backfill=False, include_cached=policy["enabled"])
    return sessions, projects, discovered, active_id


# Per-build memo for `_dir_exists_cached`; cleared by every `_build_project_tree`.
_DIR_EXISTS_CACHE: dict[str, bool] = {}


def _dir_exists_cached(path: str) -> bool:
    """``os.path.isdir`` memoized per build — ``build_tree`` asks per SESSION, not per path."""
    hit = srv._DIR_EXISTS_CACHE.get(path)
    if hit is None:
        hit = srv._DIR_EXISTS_CACHE[path] = os.path.isdir(path)
    return hit


def _build_project_tree(
    db, *, preview_limit: int, hydrate: bool, session_limit: int, include_discovered: bool
) -> tuple[dict, str | None]:
    """Gather inputs and run the one authoritative builder. Returns (tree, active_id)."""
    from tui_gateway import project_tree
    srv._DIR_EXISTS_CACHE.clear()
    sessions, projects, discovered, active_id = srv._project_tree_inputs(
        db, session_limit, include_discovered=include_discovered)
    # build_tree also resolves declared project folders and discovered roots — warm them too.
    srv.git_probe.warm_roots(
        [str(f.get("path") or "") for p in projects for f in (p.get("folders") or [])]
        + [str(r.get("root") or "") for r in discovered])
    tree = project_tree.build_tree(
        projects, sessions, discovered, srv.git_probe.resolve, preview_limit=preview_limit,
        hydrate=hydrate, is_junk_root=srv._is_repo_junk, is_junk_cwd=srv._is_session_cwd_junk,
        exists=srv._dir_exists_cached)
    return tree, active_id


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server`` and install its handlers."""
    bind_module(globals(), server)

# Bound last, after every definition, so importing this module first (tests, the gateway process)
# lets server.py's own tail import see a complete module — the same tail-import idiom server.py uses.
from tui_gateway import server as srv  # noqa: E402
