"""Authoritative project -> repo -> lane -> session tree builder.

This is the single source of truth for how the desktop sidebar groups sessions
into projects, repos, and lanes. It is pure (all git resolution is injected via
``resolve``) so it can be unit-tested with fixtures and reused by the gateway's
``projects.tree`` / ``projects.project_sessions`` RPCs.

It deliberately mirrors the desktop's former client-side grouping (the old
``workspace-groups.ts``) so the emitted ids and lane keys stay byte-compatible
with the renderer's persisted state (pins, manual ordering, dismissal), which
all key off these exact strings:

  - explicit project id .......... ``p_<hex>`` (from projects.db)
  - auto/discovered project id ... the repo root path
  - repo node id ................. the repo root path
  - main branch lane id .......... ``<repoRoot>::branch::<branch>`` (or ``::branch::``)
  - kanban bucket lane id ........ ``<repoRoot>::kanban``
  - linked worktree lane id ...... the worktree path

The one correctness upgrade over the client version: linked worktrees are folded
under their MAIN repo via a git common-dir probe (injected as ``resolve``),
instead of being treated as separate repos (``git rev-parse --show-toplevel``
returns the worktree's own root, which is why the client double-counted them).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

# A cwd -> git identity resolver. Returns ``{"repo_root", "worktree_root"}`` where
# ``repo_root`` is the COMMON (main) repo root shared across worktrees and
# ``worktree_root`` is this cwd's own checkout root. Returns ``None`` when the
# cwd is not in a git repo (or cannot be probed, e.g. a remote backend).
Resolve = Callable[[str], Optional[dict]]

# Only KANBAN-TASK worktrees (`<repo>/.worktrees/t_<hex>`, the `t_…` id kanban_db
# mints) collapse into one lane; user-named "New worktree" dirs under
# `.worktrees/` stay as their own lanes.
_KANBAN_DIR_RE = re.compile(r"^(.*[/\\]\.worktrees)[/\\]t_[0-9a-f]+[/\\]?$")
_TRUNK_BRANCHES = {"main", "master", "trunk", "develop"}
DEFAULT_BRANCH_LABEL = "main"


def _branch_lane_id(repo_root: str, branch: str = "") -> str:
    """The one definition of a main-checkout lane id (must match the desktop)."""
    return f"{repo_root}::branch::{(branch or '').strip()}"


def _kanban_lane_id(repo_root: str) -> str:
    return f"{repo_root}::kanban"


# ---------------------------------------------------------------------------
# Path helpers (match the TS segment logic so labels/ids line up)
# ---------------------------------------------------------------------------


def _segments(path: str) -> list[str]:
    return [s for s in re.split(r"[/\\]", (path or "").rstrip("/\\")) if s]


def _is_windows_path(path: str) -> bool:
    value = (path or "").strip()
    # Drive-letter (`C:\…`), UNC (`\\srv`, `//srv`), or any backslash-rooted path
    # — the root-relative `\wsl.localhost\…` / `\Users\…` spellings included. A
    # single leading `/` stays POSIX (case-sensitive).
    return bool(re.match(r"^[A-Za-z]:[/\\]", value)) or value.startswith(("\\", "//"))


def _comparison_segments(path: str) -> list[str]:
    """Path segments suitable for identity comparisons on any host.

    Windows paths remain case-insensitive even when tests or remote backends run
    on POSIX. Display paths and emitted IDs keep their original spelling.
    """
    segs = _segments(path)
    return [segment.casefold() for segment in segs] if _is_windows_path(path) else segs


def _path_key(path: str) -> str:
    """Canonical comparison key (separator/trailing-slash agnostic)."""
    return "/".join(_comparison_segments(path))


def _lane_key(path_or_lane: str) -> str:
    """Canonicalize only the path portion of a lane id.

    Branch labels remain byte-preserved; repo/worktree paths follow platform path
    identity so equivalent Windows spellings do not create duplicate lanes.
    """
    for marker in ("::branch::", "::kanban"):
        if marker in path_or_lane:
            root, suffix = path_or_lane.split(marker, 1)
            return f"{_path_key(root)}{marker}{suffix}"
    return _path_key(path_or_lane)


def base_name(path: str) -> str:
    segs = _segments(path)
    return segs[-1] if segs else ""


def kanban_worktree_dir(path: str) -> Optional[str]:
    """The ``<repo>/.worktrees`` dir for a ``.../.worktrees/<task>`` path, else None."""
    m = _KANBAN_DIR_RE.match(path or "")
    return m.group(1) if m else None


def _is_path_under(folder: str, target: str) -> bool:
    """True when ``target`` equals ``folder`` or is nested under it (segment-wise)."""
    f = _comparison_segments(folder)
    t = _comparison_segments(target)
    if not f or len(f) > len(t):
        return False
    return all(f[i] == t[i] for i in range(len(f)))


def _with_base_name(path: str, name: str) -> str:
    stripped = re.sub(r"[/\\]+$", "", path)
    return re.sub(r"[^/\\]+$", name, stripped)


# ---------------------------------------------------------------------------
# Lane placement
# ---------------------------------------------------------------------------


def _placement(
    repo_root: str,
    lane_key: str,
    lane_label: str,
    lane_path: str,
    is_main: bool,
    is_kanban: bool,
) -> dict:
    return {
        "repo_key": repo_root,
        "repo_label": base_name(repo_root) or repo_root,
        "repo_path": repo_root,
        "lane_key": lane_key,
        "lane_label": lane_label,
        "lane_path": lane_path,
        "is_main": is_main,
        "is_kanban": is_kanban,
    }


def _place_by_heuristic(path: str) -> Optional[dict]:
    """Path-only fallback when there is no git probe and no persisted root."""
    base = base_name(path)
    if not base:
        return None

    kanban_dir = kanban_worktree_dir(path)
    if kanban_dir:
        repo_path = re.sub(r"[/\\]+$", "", _with_base_name(kanban_dir, ""))
        return _placement(repo_path, _kanban_lane_id(repo_path), "kanban", kanban_dir, False, True)

    m = re.match(r"^(.+)-wt-(.+)$", base)
    if m:
        repo_path = _with_base_name(path, m.group(1))
        return _placement(repo_path, path, m.group(2), path, False, False)

    return _placement(path, path, base, path, True, False)


def _place(cwd: str, branch: str, resolve: Optional[Resolve], persisted_root: str) -> Optional[dict]:
    info = resolve(cwd) if resolve else None

    if info and info.get("repo_root") and info.get("worktree_root"):
        repo_root = info["repo_root"]
        worktree_root = info["worktree_root"]
        is_main = worktree_root == repo_root or bool(info.get("is_main"))

        if is_main:
            # Unrecorded branch folds into the one trunk lane, so a repo never
            # shows two "main" lanes (recorded "main" + the empty-branch bucket).
            b = (branch or "").strip() or DEFAULT_BRANCH_LABEL
            return _placement(repo_root, _branch_lane_id(repo_root, b), b, repo_root, True, False)

        kanban_dir = kanban_worktree_dir(worktree_root)
        if kanban_dir:
            return _placement(repo_root, _kanban_lane_id(repo_root), "kanban", kanban_dir, False, True)

        label = base_name(worktree_root) or worktree_root
        return _placement(repo_root, worktree_root, label, worktree_root, False, False)

    # No live probe: trust the backend-persisted root (group by it, split main by
    # the session's recorded branch). Kanban tasks still collapse by path shape.
    if persisted_root:
        kanban_dir = kanban_worktree_dir(cwd)
        if kanban_dir:
            return _placement(persisted_root, _kanban_lane_id(persisted_root), "kanban", kanban_dir, False, True)
        b = (branch or "").strip() or DEFAULT_BRANCH_LABEL
        return _placement(persisted_root, _branch_lane_id(persisted_root, b), b, persisted_root, True, False)

    return _place_by_heuristic(cwd)


def _session_repo_root(session: dict, resolve: Optional[Resolve]) -> str:
    """The COMMON repo root a session belongs to (folds linked worktrees)."""
    cwd = (session.get("cwd") or "").strip()
    if cwd and resolve:
        info = resolve(cwd)
        if info and info.get("repo_root"):
            return info["repo_root"]
    return (session.get("git_repo_root") or "").strip()


# ---------------------------------------------------------------------------
# Ordering + label disambiguation (parity with the old client tree)
# ---------------------------------------------------------------------------


def _lane_sort_key(group: dict) -> tuple:
    # Trunk pins to the top; the kanban aggregate sinks to the bottom; the rest
    # (branches + linked worktrees) sort by most-recent activity, then label.
    is_trunk = bool(group.get("isMain")) and group["label"].lower() in _TRUNK_BRANCHES
    is_kanban = bool(group.get("isKanban"))
    activity = max((_session_time(s) for s in group.get("sessions") or []), default=0.0)
    return (
        0 if is_trunk else 1,
        1 if is_kanban else 0,
        -activity,
        group["label"].lower(),
    )


def _sort_lanes(groups: list[dict]) -> list[dict]:
    return sorted(groups, key=_lane_sort_key)


def _disambiguate_labels(items: list[dict]) -> None:
    """Grow colliding basenames into path-prefixed labels (in place)."""
    by_label: dict[str, list[dict]] = {}
    for item in items:
        by_label.setdefault(item["label"], []).append(item)

    for bucket in by_label.values():
        pathed = [g for g in bucket if g.get("path")]
        if len(pathed) < 2:
            continue

        parents = {id(g): _segments(g["path"])[:-1] for g in pathed}
        max_depth = max(len(p) for p in parents.values())
        depth = 1
        while depth <= max_depth:
            counts: dict[str, int] = {}
            for g in pathed:
                segs = parents[id(g)]
                prefix = "/".join(segs[-depth:]) if depth else ""
                base = base_name(g["path"]) or g["path"]
                g["label"] = f"{prefix}/{base}" if prefix else base
                counts[g["label"]] = counts.get(g["label"], 0) + 1
            if all(c == 1 for c in counts.values()):
                break
            depth += 1


# ---------------------------------------------------------------------------
# Repo subtree assembly
# ---------------------------------------------------------------------------


def _session_time(session: dict) -> float:
    return float(session.get("last_active") or session.get("started_at") or 0)


def _build_repos(sessions: list[dict], resolve: Optional[Resolve], hydrate: bool) -> list[dict]:
    """Build the ``repo -> lane -> sessions`` subtree for a set of sessions."""
    lanes: dict[str, dict] = {}  # lane_key -> {group, repo_key, repo_label, repo_path}

    for session in sessions:
        cwd = (session.get("cwd") or "").strip()
        if not cwd:
            continue

        placement = _place(
            cwd,
            (session.get("git_branch") or "").strip(),
            resolve,
            (session.get("git_repo_root") or "").strip(),
        )
        if not placement:
            continue

        lane_identity = _lane_key(placement["lane_key"])
        entry = lanes.get(lane_identity)
        if entry is None:
            entry = {
                "group": {
                    "id": placement["lane_key"],
                    "label": placement["lane_label"],
                    "path": placement["lane_path"],
                    "isMain": placement["is_main"],
                    "isKanban": placement["is_kanban"],
                    "sessions": [],
                },
                "repo_key": placement["repo_key"],
                "repo_label": placement["repo_label"],
                "repo_path": placement["repo_path"],
            }
            lanes[lane_identity] = entry
        entry["group"]["sessions"].append(session)

    repos: dict[str, dict] = {}
    for entry in lanes.values():
        group = entry["group"]
        group["sessions"].sort(key=_session_time, reverse=True)
        count = len(group["sessions"])
        if not hydrate:
            group["sessions"] = []

        repo_identity = _path_key(entry["repo_key"])
        repo = repos.get(repo_identity)
        if repo is None:
            repo = {
                "id": entry["repo_key"],
                "label": entry["repo_label"],
                "path": entry["repo_path"],
                "groups": [],
                "sessionCount": 0,
            }
            repos[repo_identity] = repo
        repo["groups"].append(group)
        repo["sessionCount"] += count

    repo_list = list(repos.values())
    for repo in repo_list:
        repo["groups"] = _sort_lanes(repo["groups"])
        _disambiguate_labels(repo["groups"])
    _disambiguate_labels(repo_list)
    return repo_list


def _seed_folder_repos(
    repos: list[dict], folders: list[dict], resolve: Optional[Resolve]
) -> list[dict]:
    """Ensure every declared project folder shows as a repo, even with 0 sessions.

    A brand-new project (or any project whose sessions haven't loaded yet) has an
    empty session-derived ``repos`` list. That breaks two things on the desktop:
    the entered-project view renders blank (it early-returns on no repos), and the
    optimistic live-session overlay has no lane to drop a freshly-created session
    into — so a new session in the project only appears after a full tree refresh.
    Seeding each folder as an empty repo fixes both: the overlay matches a new
    session's cwd under the folder root, and the drill-in renders a real (if
    empty) project body. Folders already covered by a session-derived repo (same
    git root) are left untouched.
    """
    seen = {
        _path_key(value)
        for repo in repos
        for value in (repo.get("id"), repo.get("path"))
        if value
    }
    seeded = list(repos)

    for folder in folders or []:
        raw = (folder.get("path") or "").strip()
        if not raw:
            continue
        info = resolve(raw) if resolve else None
        root = (info or {}).get("repo_root") or re.sub(r"[/\\]+$", "", raw)
        root_key = _path_key(root)
        if not root_key or root_key in seen:
            continue
        seeded.append({"id": root, "label": base_name(root) or root, "path": root, "groups": [], "sessionCount": 0})
        seen.add(root_key)

    if len(seeded) != len(repos):
        _disambiguate_labels(seeded)

    return seeded


# ---------------------------------------------------------------------------
# Explicit-project ownership
# ---------------------------------------------------------------------------


class _FolderIndex:
    """Maps a normalized folder path → (owning project, depth), so a session is
    matched to its project by walking its cwd's ancestors (O(path depth) dict
    lookups) instead of scanning every project × folder per session — the
    difference between O(sessions × projects) and O(sessions) at power-user scale.
    """

    def __init__(self, projects: list[dict]) -> None:
        self._by_path: dict[str, tuple[dict, int]] = {}
        for project in projects:
            for folder in project.get("folders") or []:
                segs = _comparison_segments(folder.get("path") or "")
                if not segs:
                    continue
                key = "/".join(segs)
                depth = len(segs)
                # Deepest folder wins; ties keep the first project (scan order).
                existing = self._by_path.get(key)
                if existing is None or depth > existing[1]:
                    self._by_path[key] = (project, depth)

    def match(self, target: str) -> tuple[Optional[dict], int]:
        """Owning project for ``target`` by longest ancestor folder, + its depth."""
        segs = _comparison_segments(target or "")
        # Longest prefix first → deepest (most specific) folder wins.
        for end in range(len(segs), 0, -1):
            hit = self._by_path.get("/".join(segs[:end]))
            if hit:
                return hit
        return None, -1


def _project_for_path(index: _FolderIndex, target: str) -> Optional[dict]:
    return index.match(target)[0]


def _project_for_session(session: dict, index: _FolderIndex, resolve: Optional[Resolve]) -> Optional[dict]:
    cwd = (session.get("cwd") or "").strip()
    if not cwd:
        return None
    repo_root = _session_repo_root(session, resolve)
    candidates = [cwd, repo_root] if repo_root and repo_root != cwd else [cwd]

    best: Optional[dict] = None
    best_len = -1
    for target in candidates:
        match, length = index.match(target)
        if match and length > best_len:
            best_len = length
            best = match
    return best


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def _project_node(
    *,
    pid: str,
    label: str,
    path: Optional[str],
    repos: list[dict],
    session_count: int,
    last_active: float,
    preview_sessions: list[dict],
    color: Any = None,
    icon: Any = None,
    is_auto: bool = False,
) -> dict:
    return {
        "id": pid,
        "label": label,
        "path": path,
        "color": color,
        "icon": icon,
        "isAuto": is_auto,
        "sessionCount": session_count,
        "lastActive": last_active,
        "repos": repos,
        "previewSessions": preview_sessions,
    }


def build_tree(
    projects: list[dict],
    sessions: list[dict],
    discovered_repos: list[dict],
    resolve: Optional[Resolve] = None,
    *,
    preview_limit: int = 3,
    hydrate: bool = False,
    is_junk_root: Optional[Callable[[str], bool]] = None,
    is_junk_cwd: Optional[Callable[[str], bool]] = None,
) -> dict:
    """Build the authoritative project tree.

    ``projects`` are ``projects_db.Project.to_dict()`` shapes (non-archived).
    ``sessions`` are projected session-row dicts (must carry ``id``, ``cwd``,
    ``git_branch``, ``git_repo_root``, ``started_at``, ``last_active``).
    ``discovered_repos`` are ``{"root", "label", "sessions", "last_active"}``.
    ``is_junk_root`` flags git roots that must never become an AUTO project (the
    bare home dir, the HERMES_HOME subtree). ``is_junk_cwd`` is the narrower
    policy for non-git session folders: selected descendants may be intentional
    workspaces even when their parent tree contains Hermes state. User-created
    projects are honored regardless.

    Returns ``{"projects": [...], "scoped_session_ids": [...]}``. When
    ``hydrate`` is False (overview), lane ``sessions`` arrays are emptied but
    every count is preserved and each project carries up to ``preview_limit``
    ``previewSessions``. When True (drill-in), lanes carry full session rows.
    """
    active_projects = [p for p in projects if not p.get("archived")]
    _junk = is_junk_root or (lambda _root: False)
    _junk_cwd = is_junk_cwd or (lambda _cwd: False)
    folder_index = _FolderIndex(active_projects)

    by_project: dict[str, list[dict]] = {}
    unowned: list[dict] = []
    for session in sessions:
        owner = _project_for_session(session, folder_index, resolve)
        if owner:
            by_project.setdefault(owner["id"], []).append(session)
        else:
            unowned.append(session)

    scoped_ids: list[str] = []

    def _previews(project_sessions: list[dict]) -> list[dict]:
        if preview_limit <= 0:
            return []
        ordered = sorted(project_sessions, key=_session_time, reverse=True)
        return ordered[:preview_limit]

    def _last_active(project_sessions: list[dict]) -> float:
        return max((_session_time(s) for s in project_sessions), default=0.0)

    result: list[dict] = []

    # Tier 1: explicit, user-created projects (always shown, even with 0 sessions).
    for project in active_projects:
        psessions = by_project.get(project["id"], [])
        scoped_ids.extend(s["id"] for s in psessions if s.get("id"))
        repos = _seed_folder_repos(
            _build_repos(psessions, resolve, hydrate), project.get("folders") or [], resolve
        )
        result.append(
            _project_node(
                pid=project["id"],
                label=project.get("name") or project["id"],
                path=project.get("primary_path"),
                color=project.get("color"),
                icon=project.get("icon"),
                repos=repos,
                session_count=len(psessions),
                last_active=_last_active(psessions),
                preview_sessions=_previews(psessions),
            )
        )

    # Tier 2: auto projects from leftover sessions. Prefer the common git repo
    # root, then fall back to the session cwd for historical/non-git workspaces.
    # The pre-Projects desktop grouped every non-empty cwd; keeping that fallback
    # prevents upgrades from flattening those sessions into Recents.
    by_auto_root: dict[str, dict] = {}

    def _add_auto(root: str, session: dict) -> None:
        key = _path_key(root)
        if not key:
            return
        bucket = by_auto_root.setdefault(key, {"root": root, "sessions": []})
        bucket["sessions"].append(session)

    for session in unowned:
        root = _session_repo_root(session, resolve)
        if root:
            # A real git root uses the stricter repo policy. Do not reinterpret a
            # filtered internal repo as a cwd-only project.
            if not _junk(root):
                _add_auto(root, session)
            continue

        cwd = (session.get("cwd") or "").strip()
        if not cwd or _junk_cwd(cwd):
            continue
        placement = _place(
            cwd,
            (session.get("git_branch") or "").strip(),
            resolve,
            (session.get("git_repo_root") or "").strip(),
        )
        if placement:
            _add_auto(placement["repo_key"], session)

    seen: set[str] = set()
    for bucket in by_auto_root.values():
        auto_root = bucket["root"]
        auto_sessions = bucket["sessions"]
        auto_key = _path_key(auto_root)
        repos = _build_repos(auto_sessions, resolve, hydrate)
        repo_node = next(
            (
                repo
                for repo in repos
                if _path_key(repo.get("id") or repo.get("path") or "") == auto_key
            ),
            None,
        )
        if repo_node is None:
            continue
        seen.add(auto_key)
        scoped_ids.extend(s["id"] for s in auto_sessions if s.get("id"))
        result.append(
            _project_node(
                pid=auto_root,
                label=base_name(auto_root) or auto_root,
                path=auto_root,
                repos=repos,
                session_count=repo_node["sessionCount"],
                last_active=_last_active(auto_sessions),
                preview_sessions=_previews(auto_sessions),
                is_auto=True,
            )
        )

    # Tier 3: repos discovered from full history / disk scan with no loaded
    # sessions, folded to their common root and not owned by an explicit project.
    for repo in discovered_repos or []:
        raw_root = (repo.get("root") or "").strip()
        if not raw_root:
            continue
        info = resolve(raw_root) if resolve else None
        root = (info or {}).get("repo_root") or raw_root
        root_key = _path_key(root)
        if root_key in seen or _junk(root) or _project_for_path(folder_index, root):
            continue
        seen.add(root_key)
        label = repo.get("label") or base_name(root) or root
        result.append(
            _project_node(
                pid=root,
                label=label,
                path=root,
                repos=[{"id": root, "label": label, "path": root, "groups": [], "sessionCount": 0}],
                session_count=int(repo.get("sessions") or 0),
                last_active=float(repo.get("last_active") or 0),
                preview_sessions=[],
                is_auto=True,
            )
        )

    # Auto projects are labelled by repo basename, which can collide (two "app"
    # repos in different parents). Grow path prefixes so each is distinct.
    # Explicit projects keep their user-chosen names untouched.
    _disambiguate_labels([p for p in result if p.get("isAuto")])

    return {"projects": result, "scoped_session_ids": scoped_ids}
