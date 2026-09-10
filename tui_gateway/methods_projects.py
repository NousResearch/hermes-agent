"""Projects RPC surface: per-profile multi-folder workspaces, repo discovery, sidebar tree.
Bodies are rebound onto server.py's globals at install (method_ctx.bind_module)."""

from __future__ import annotations

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


# JSON-RPC error codes: generic failure / id resolved to nothing / invalid argument.
_E_PROJECTS, _E_NO_PROJECT, _E_PROJECT_ARG = 5061, 5062, 5063


class _NoProject(Exception):
    """Raised inside a projects handler when ``params['id']`` resolves to None."""


def _projects_payload(conn) -> dict:
    from hermes_cli import projects_db as pdb
    return {
        "projects": [p.to_dict() for p in pdb.list_projects(conn, include_archived=True)],
        "active_id": pdb.get_active_id(conn)}


def _projects_method(name: str):
    """Register a projects RPC, injecting (pdb, conn) and unifying error mapping; profile-scoped
    so app-global remote mode reads that profile's ``projects.db``."""
    def decorator(fn):
        @method(name)
        @_registry.profile_scoped
        def handler(rid, params: dict) -> dict:
            try:
                from hermes_cli import projects_db as pdb
                with pdb.connect_closing() as conn:
                    return fn(rid, params, pdb, conn)
            except _NoProject:
                return _err(rid, _E_NO_PROJECT, "no such project")
            except ValueError as e:
                return _err(rid, _E_PROJECT_ARG, str(e))
            except Exception as e:
                return _err(rid, _E_PROJECTS, str(e))
        return handler
    return decorator


def _require_project(pdb, conn, params: dict):
    """The project named by ``params['id']`` (or raise ``_NoProject``)."""
    proj = pdb.get_project(conn, str(params.get("id") or ""))
    if proj is None:
        raise _NoProject
    return proj


def _pick(params: dict, *keys: str) -> dict:
    return {k: params.get(k) for k in keys}


def _register_project_mutator(suffix: str, fn_name: str, takes_path: bool, kwargs_of) -> None:
    """``projects.<suffix>``: resolve ``params['id']`` (5062 when missing), call
    ``pdb.<fn_name>(conn, id[, path], **kwargs_of(params))``, answer with the refreshed project."""
    @_projects_method(f"projects.{suffix}")
    def _(rid, params, pdb, conn) -> dict:
        proj = _require_project(pdb, conn, params)
        args = (str(params.get("path") or ""),) if takes_path else ()
        getattr(pdb, fn_name)(conn, proj.id, *args, **kwargs_of(params))
        return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


_register_project_mutator(
    "update", "update_project", False,
    lambda p: _pick(p, "name", "description", "icon", "color", "board_slug"))
_register_project_mutator(
    "add_folder", "add_folder", True,
    lambda p: {"label": p.get("label"), "is_primary": bool(p.get("is_primary"))})
_register_project_mutator("remove_folder", "remove_folder", True, lambda p: {})
_register_project_mutator("set_primary", "set_primary", True, lambda p: {})


@_projects_method("projects.list")
def _(rid, params, pdb, conn) -> dict:
    return _ok(rid, _projects_payload(conn))


@_projects_method("projects.get")
def _(rid, params, pdb, conn) -> dict:
    return _ok(rid, {"project": _require_project(pdb, conn, params).to_dict()})


@_projects_method("projects.create")
def _(rid, params, pdb, conn) -> dict:
    pid = pdb.create_project(
        conn, name=str(params.get("name") or ""), folders=params.get("folders") or [],
        **_pick(params, "slug", "primary_path", "description", "icon", "color", "board_slug"))
    if params.get("use"):
        pdb.set_active(conn, pid)
    proj = pdb.get_project(conn, pid)
    return _ok(rid, {"project": proj.to_dict() if proj else None})


@_projects_method("projects.archive")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    (pdb.restore_project if params.get("restore") else pdb.archive_project)(conn, proj.id)
    return _ok(rid, _projects_payload(conn))


@_projects_method("projects.delete")
def _(rid, params, pdb, conn) -> dict:
    pdb.delete_project(conn, _require_project(pdb, conn, params).id)
    return _ok(rid, _projects_payload(conn))


@_projects_method("projects.set_active")
def _(rid, params, pdb, conn) -> dict:
    pdb.set_active(conn, _require_project(pdb, conn, params).id if params.get("id") else None)
    return _ok(rid, {"active_id": pdb.get_active_id(conn)})


@_projects_method("projects.for_cwd")
def _(rid, params, pdb, conn) -> dict:
    cwd = _completion_cwd(
        {"cwd": str(params.get("cwd") or "").strip()} if params.get("cwd") else {})
    proj = pdb.project_for_path(conn, cwd)
    return _ok(rid, {
        "project": proj.to_dict() if proj else None, "cwd": cwd,
        "branch": git_probe.branch(cwd)})


def _non_workspace_dirs() -> set[str]:
    """Never-a-workspace dirs: ``/``, the user's home, the dir homes live in, plus both POSIX
    spellings on every host (remote shells hand back Linux paths; promoting one mints a
    catch-all project)."""
    home = os.path.realpath(os.path.expanduser("~"))
    candidates = (os.sep, home, os.path.dirname(home), "/home", "/Users")
    return {os.path.normcase(os.path.realpath(path)) for path in candidates if path}


_EXCLUDE_CACHE: tuple[tuple[str, ...], tuple[str, ...]] | None = None


def _normalised_excludes() -> tuple[str, ...]:
    """``desktop.repo_scan_exclude_paths`` normalised once per config generation
    (expanduser, home-relative abspath, realpath, normcase) - so `~/x`, `x` and
    `/home/u/x` spell the same exclusion. The config loader is already
    mtime-cached; this extra cache spares the per-session tree loop from
    re-running realpath on every entry for every row. Empty entries are dropped;
    a filesystem-root entry excludes its children too (rstrip keeps the prefix
    join at exactly one separator)."""
    global _EXCLUDE_CACHE
    # _repo_discovery_policy() already coerces the value to list[str] via
    # _paths(); guard anyway - the failure mode (TypeError inside a per-session
    # junk check) breaks every tree build.
    raw_value = _repo_discovery_policy().get("exclude_paths") or []
    raw_entries = tuple(raw_value) if isinstance(raw_value, list) else ()
    if _EXCLUDE_CACHE is not None and _EXCLUDE_CACHE[0] == raw_entries:
        return _EXCLUDE_CACHE[1]
    home = os.path.expanduser("~")
    norm: list[str] = []
    for raw in raw_entries:
        if not isinstance(raw, str) or not raw.strip():
            continue
        ex = os.path.normcase(os.path.realpath(
            os.path.abspath(os.path.join(home, os.path.expanduser(raw)))))
        if ex:
            norm.append(ex.rstrip(os.sep) or ex)
    _EXCLUDE_CACHE = (raw_entries, tuple(norm))
    return _EXCLUDE_CACHE[1]


def _policy_excluded(path: str) -> bool:
    """True when ``path`` sits at/under a ``desktop.repo_scan_exclude_paths`` entry."""
    if not path:
        return False
    real = os.path.normcase(os.path.realpath(path)).rstrip(os.sep) or path
    for ex in _normalised_excludes():
        if real == ex or real.startswith(ex + os.sep):
            return True
    return False


def _is_repo_junk(root: str) -> bool:
    """A git root never auto-surfaced as a project: a non-workspace dir, anything under
    HERMES_HOME, or a policy-excluded path. User-created projects pointing there are
    still honored (explicit folders bypass the junk filter)."""
    if not root:
        return True
    from hermes_constants import get_hermes_home
    real = os.path.realpath(root)
    hermes_home = os.path.realpath(str(get_hermes_home()))
    return (
        os.path.normcase(real) in _non_workspace_dirs()
        or real == hermes_home
        or real.startswith(hermes_home + os.sep)
        or _policy_excluded(root))


def _is_session_cwd_junk(cwd: str) -> bool:
    """A non-git cwd that never auto-surfaces. Junk cwds fall to the Home bucket
    (``_auto_buckets`` routes them to homeless), so a policy-excluded non-git
    folder - e.g. a deleted explicit project's folder - lands its sessions in
    Home instead of re-minting a path-only auto project. Explicit projects are
    unaffected: folder-index ownership runs before any junk filter. A DESCENDANT
    of HERMES_HOME may be an intentional prose/data workspace, so only
    HERMES_HOME itself is excluded here."""
    if not cwd:
        return True
    from hermes_constants import get_hermes_home
    real = os.path.normcase(os.path.realpath(cwd))
    hermes_home = os.path.normcase(os.path.realpath(str(get_hermes_home())))
    return real in _non_workspace_dirs() or real == hermes_home or _policy_excluded(cwd)


def _repo_discovery_policy(raw: dict | None = None) -> dict:
    """Return the effective, profile-local Desktop repository scan policy."""
    from hermes_cli.config import DEFAULT_CONFIG
    defaults = DEFAULT_CONFIG["desktop"]
    source = raw if isinstance(raw, dict) else (_load_cfg().get("desktop") or {})
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
    return _repo_discovery_policy_key(policy) == _repo_discovery_policy_key(
        _repo_discovery_policy(DEFAULT_CONFIG["desktop"]))


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
                conn, pairs, replace=authoritative, policy_key=_repo_discovery_policy_key(policy))
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
    git_probe.warm_roots(str(r.get("cwd") or "") for r in cwd_rows)
    cwd_to_root: dict[str, str] = {}
    for row in cwd_rows:
        cwd = str(row.get("cwd") or "")
        root = git_probe.common_repo_root(cwd)
        if not root:
            continue
        cwd_to_root[cwd] = root
        if _is_repo_junk(root):
            continue
        agg = _agg(root)
        agg["sessions"] += int(row.get("sessions") or 0)
        agg["last_active"] = max(agg["last_active"], float(row.get("last_active") or 0))
    if backfill:
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
                    if root and not _is_repo_junk(root):
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
_PROJECT_TREE_EXCLUDED_SOURCES = ["cron", "kanban"]


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


# ── Folder-scoped project queries (no recency window) ─────────────────────────
# The tree builder walks the N most recent sessions, so on busy accounts an old
# project's sessions fall out of the window and read as zero. These two RPCs query
# by folder membership directly: cost follows the project's own history, not the
# account's. ``session_rows`` is the paginated drill-in; ``exact_counts`` feeds the
# overview badges.

# Hard cap on candidate rows per drill-in call: the WHERE already bounds the
# scan to the project's own history, this only bounds the transfer. total
# comes from the grouped scan, so counts stay exact past the cap; the plugin
# hides Load more once a page returns zero rows.
_SCOPED_CANDIDATE_CAP = 50_000


def _scoped_folder_clause(folders: list) -> tuple:
    """(sql, params) matching sessions whose cwd OR stored repo root sits at/under any
    project folder - a SUPERSET of the tree's members for this project (rows whose
    deepest owner is another project are filtered out in Python afterwards). LIKE
    wildcards in paths are escaped (paths legitimately contain ``_``/``%``)."""
    from hermes_state_common import escape_like
    arms: list = []
    params: list = []
    for folder in folders:
        base = str(folder or "").rstrip("/\\")
        if not base:
            continue
        esc = escape_like(base)
        for col in ("s.cwd", "s.git_repo_root"):
            arms.append(f"({col} = ? OR {col} LIKE ? ESCAPE '\\' OR {col} LIKE ? ESCAPE '\\')")
            params.extend([base, f"{esc}/%", f"{esc}\\\\%"])
    if not arms:
        return "1 = 0", []
    return "(" + " OR ".join(arms) + ")", params


def _scoped_base_where() -> tuple:
    """Base visibility filters - the tree's own set (archived/hidden excluded, min 1
    message, no cron/kanban, no subagent/compression children)."""
    from hermes_state_sessions import _session_filter_where, _where_sql
    clauses, params = _session_filter_where(
        exclude_children=True, exclude_sources=list(_PROJECT_TREE_EXCLUDED_SOURCES),
        min_message_count=1)
    clauses.append("s.hidden = 0")
    return _where_sql(clauses, " "), params


def _scoped_row(r) -> dict:
    """Shape a raw scoped-query row like the tree's own rows (preview + projection)."""
    from hermes_state_common import _shape_preview
    d = dict(r)
    d["preview"] = _shape_preview(d.pop("_preview_raw", ""))
    return _project_tree_row(d)


def _scoped_owner(cwd, repo, index):
    """The tree's ownership rule: longest folder match wins over candidates
    [cwd, repo_root]; ties keep cwd (max() keeps the first maximum). Returns
    the owning project dict or None."""
    candidates = [cwd, repo] if repo and repo != cwd else [cwd]
    hits = [index.match(t) for t in candidates if t]
    if not hits:
        return None
    return max(hits, key=lambda hit: hit[1])[0]


def _scoped_grouped_counts(db, pdb, conn) -> tuple:
    """ONE grouped scan of (cwd, repo_root) pairs across ALL history, plus the folder
    index over explicit projects - cost follows distinct workspaces, not session or
    project counts. Reuses the injected projects.db connection (no second open)."""
    from tui_gateway.project_tree import _FolderIndex

    where, where_params = _scoped_base_where()
    rows = db._read_rows(
        f"""SELECT s.cwd AS cwd, s.git_repo_root AS repo, COUNT(*) AS n
            FROM sessions s {where}
            GROUP BY s.cwd, s.git_repo_root""", where_params)
    projects = [p.to_dict() for p in pdb.list_projects(conn)]
    return rows, _FolderIndex(projects)


@_projects_method("projects.exact_counts")
def _(rid, params, pdb, conn) -> dict:
    """{project_id: exact session count} across ALL history for every explicit project,
    using the tree's ownership rule so badges and drill-in can never disagree."""
    counts: dict = {}
    with _profile_db(params) as db:
        if db is None:
            return _ok(rid, {"counts": counts})
        rows, index = _scoped_grouped_counts(db, pdb, conn)
        for r in rows:
            owner = _scoped_owner(r["cwd"], r["repo"], index)
            if owner is not None:
                counts[owner["id"]] = counts.get(owner["id"], 0) + int(r["n"])
    return _ok(rid, {"counts": counts})


@_projects_method("projects.session_rows")
def _(rid, params, pdb, conn) -> dict:
    """Paginated session rows for ONE project, folder-scoped - exact regardless of how
    many sessions the account holds, and ownership-faithful to the tree (deepest folder
    match over cwd/repo candidates). ``id`` may be an explicit project id/slug or an
    auto-project's repo path; the Home bucket refuses (the windowed drill-in covers it)."""
    project_id = str(params.get("id") or params.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("id required")
    raw_limit = params.get("limit")
    limit = min(max(int(raw_limit) if raw_limit is not None else 50, 1), 500)
    offset = max(int(params.get("offset") or 0), 0)

    if project_id == "__no_project__":
        raise ValueError("home bucket is not folder-scoped")

    proj = pdb.get_project(conn, project_id)
    if proj is not None:
        proj_id = proj.id
        folders = [f.path for f in proj.folders if f.path]
        if not folders:
            return _ok(rid, {"sessions": [], "total": 0, "limit": limit, "offset": offset})
    elif project_id.startswith("p_"):
        raise _NoProject
    else:
        proj_id = None  # auto project: tree keys it by repo root path
        folders = [project_id]

    from tui_gateway.project_tree import _FolderIndex
    from hermes_state_sessions import _PREVIEW_COL_SQL
    from hermes_state_common import _sql_session_last_active

    with _profile_db(params) as db:
        if db is None:
            return _ok(rid, {"sessions": [], "total": 0, "limit": limit, "offset": offset})

        # Grouped scan for the exact total + folder index for ownership filtering.
        grouped, index = _scoped_grouped_counts(db, pdb, conn)
        total = 0
        for r in grouped:
            owner = _scoped_owner(r["cwd"], r["repo"], index)
            if proj_id is not None:
                if owner is not None and owner["id"] == proj_id:
                    total += int(r["n"])
            elif owner is None:
                # Auto project: the tree only places UNOWNED sessions there, so
                # rows claimed by an explicit project never count toward it.
                # Both candidates must count: the candidate SQL matches on
                # cwd OR repo, so a row whose cwd sits outside the auto folder
                # but whose repo root sits inside still belongs to the page.
                if _cand_under(r["cwd"], folders) or _cand_under(r["repo"], folders):
                    total += int(r["n"])

        # Candidate rows via scoped SQL (no LIMIT: bounded by this project's own
        # history - the design premise), then filter + paginate in Python.
        where, where_params = _scoped_base_where()
        clause, clause_params = _scoped_folder_clause(folders)
        where += f" AND {clause}"
        rows = db._read_rows(
            f"""SELECT {db._compact_session_cols()}, {_PREVIEW_COL_SQL},
                        {_sql_session_last_active("s")} AS last_active
                FROM sessions s {where}
                ORDER BY last_active DESC, s.started_at DESC, s.id DESC
                LIMIT {_SCOPED_CANDIDATE_CAP}""",
            [*where_params, *clause_params])
        keep = []
        for r in rows:
            owner = _scoped_owner(r["cwd"], r["git_repo_root"], index)
            if proj_id is not None:
                if owner is not None and owner["id"] == proj_id:
                    keep.append(r)
            elif owner is None:
                keep.append(r)
        sessions = [_scoped_row(r) for r in keep[offset:offset + limit]]
        return _ok(rid, {"sessions": sessions, "total": total, "limit": limit, "offset": offset})


def _cand_under(cand: str, folders: list) -> bool:
    """candidate path at/under any auto-project folder (normalised separators)."""
    norm = lambda p: str(p or "").replace("\\", "/").rstrip("/")
    c = norm(cand)
    return bool(c) and any(c == norm(f) or c.startswith(norm(f) + "/") for f in folders)


def _project_tree_inputs(
    db, session_limit: int, *, include_discovered: bool
) -> tuple[list[dict], list[dict], list[dict], str | None]:
    """Gather (sessions, projects, discovered_repos, active_id) for build_tree.
    ``include_discovered`` is the zero-session-repo overview tier; drill-in skips it (and
    the distinct-cwd scan + git probes) on that per-turn path."""
    # compact_rows: selecting the system-prompt blob only to drop it costs tens of MB of reads.
    rows = db.list_sessions_rich(
        limit=session_limit, offset=0, order_by_last_active=True, min_message_count=1,
        include_children=False, exclude_sources=_PROJECT_TREE_EXCLUDED_SOURCES,
        include_archived=False, compact_rows=True)
    sessions = [_project_tree_row(r) for r in rows]
    # Parallel-warm the git cache so build_tree's resolver doesn't cold-probe each cwd in turn.
    git_probe.warm_roots(s["cwd"] for s in sessions if s.get("cwd"))
    from hermes_cli import projects_db as pdb
    policy = _repo_discovery_policy()
    policy_key = _repo_discovery_policy_key(policy)
    with pdb.connect_closing() as conn:
        if include_discovered:
            pdb.reconcile_discovered_repos_policy(
                conn, policy_key, preserve_unversioned=_repo_discovery_policy_is_default(policy))
        projects = [p.to_dict() for p in pdb.list_projects(conn)]
        active_id = pdb.get_active_id(conn)
        # backfill stays off the hot tree path — grouping uses the live resolver.
        discovered = []
        if include_discovered:
            discovered = _discover_repos_payload(
                db, conn=conn, backfill=False, include_cached=policy["enabled"])
    return sessions, projects, discovered, active_id


# Per-build memo for `_dir_exists_cached`; cleared by every `_build_project_tree`.
_DIR_EXISTS_CACHE: dict[str, bool] = {}


def _dir_exists_cached(path: str) -> bool:
    """``os.path.isdir`` memoized per build — ``build_tree`` asks per SESSION, not per path."""
    hit = _DIR_EXISTS_CACHE.get(path)
    if hit is None:
        hit = _DIR_EXISTS_CACHE[path] = os.path.isdir(path)
    return hit


def _build_project_tree(
    db, *, preview_limit: int, hydrate: bool, session_limit: int, include_discovered: bool
) -> tuple[dict, str | None]:
    """Gather inputs and run the one authoritative builder. Returns (tree, active_id)."""
    from tui_gateway import project_tree
    _DIR_EXISTS_CACHE.clear()
    sessions, projects, discovered, active_id = _project_tree_inputs(
        db, session_limit, include_discovered=include_discovered)
    # build_tree also resolves declared project folders and discovered roots — warm them too.
    git_probe.warm_roots(
        [str(f.get("path") or "") for p in projects for f in (p.get("folders") or [])]
        + [str(r.get("root") or "") for r in discovered])
    tree = project_tree.build_tree(
        projects, sessions, discovered, git_probe.resolve, preview_limit=preview_limit,
        hydrate=hydrate, is_junk_root=_is_repo_junk, is_junk_cwd=_is_session_cwd_junk,
        exists=_dir_exists_cached)
    return tree, active_id


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
