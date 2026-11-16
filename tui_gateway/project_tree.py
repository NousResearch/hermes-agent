"""Authoritative project -> repo -> lane -> session tree builder (pure; git via ``resolve``).

Emitted ids/lane keys must stay byte-compatible with the renderer's persisted state
(pins, ordering, dismissal), which keys off: explicit project id ``p_<hex>``; auto
project id / repo node id = repo root path; home bucket ``__no_project__``; main lane
``<repoRoot>::branch::<branch>``; kanban lane ``<repoRoot>::kanban``; linked worktree
lane = the worktree path. Linked worktrees fold under their MAIN repo (common-dir probe).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, Optional

# cwd -> ``{"repo_root", "worktree_root"}`` (COMMON main root / this cwd's checkout root);
# ``None`` when not in git or unprobeable (remote backend).
Resolve = Callable[[str], Optional[dict]]
# "does this directory still exist?"; always-True default keeps remote-host projects visible.
Exists = Callable[[str], bool]
# Physical spelling of a local path (symlinks resolved). It feeds IDENTITY comparisons only and
# is never emitted: ids and display paths are always a spelling that arrived on a row.
Canonical = Callable[[str], str]

# Only KANBAN-TASK worktrees (`<repo>/.worktrees/t_<hex>`, the id kanban_db mints)
# collapse into one lane; user-named dirs under `.worktrees/` stay their own lanes.
_KANBAN_DIR_RE = re.compile(r"^(.*[/\\]\.worktrees)[/\\]t_[0-9a-f]+[/\\]?$")
_TRUNK_BRANCHES = {"main", "master", "trunk", "develop"}
DEFAULT_BRANCH_LABEL = "main"

# Synthetic bucket for every session no project claimed (no cwd, bare home, deleted workspace).
NO_PROJECT_ID = "__no_project__"
NO_PROJECT_LABEL = "Home"

# Sibling probes when recovering a deleted worktree's parent repo (each miss is a git call).
_MAX_SIBLING_PROBES = 4


def stamp_profile(projects: list[dict], profile: str) -> None:
    """Stamp every session row with the request-scope profile (authoritative even for legacy
    rows whose ``profile_name`` is NULL) for cross-profile routing."""
    for project in projects:
        lanes = [g for repo in project.get("repos") or [] for g in repo.get("groups") or []]
        for session in (project.get("previewSessions") or []) + [
                s for g in lanes for s in g.get("sessions") or []]:
            session["profile"] = profile


def _branch_lane_id(repo_root: str, branch: str = "") -> str:
    """The one definition of a main-checkout lane id (must match the desktop)."""
    return f"{repo_root}::branch::{(branch or '').strip()}"


def _kanban_lane_id(repo_root: str) -> str:
    return f"{repo_root}::kanban"


def _segments(path: str) -> list[str]:
    return [s for s in re.split(r"[/\\]", (path or "").rstrip("/\\")) if s]


def _is_windows_path(path: str) -> bool:
    # Drive-letter, UNC (`\\srv`, `//srv`) or backslash-rooted; a single leading `/` stays POSIX.
    value = (path or "").strip()
    return bool(re.match(r"^[A-Za-z]:[/\\]", value)) or value.startswith(("\\", "//"))


def _comparison_segments(path: str, canonical: Optional[Canonical] = None) -> list[str]:
    """Path segments suitable for identity comparisons on any host.

    Windows paths remain case-insensitive even when tests or remote backends run
    on POSIX. Display paths and emitted IDs keep their original spelling.

    Segments are NFC-normalized before comparing: the same on-disk folder can
    reach us as NFC (typed paths, os.getcwd()) or NFD (macOS file pickers,
    HFS+/APFS round-trips), and ``casefold()`` does not unify the two forms —
    an accented project folder would otherwise render empty (#65014). Mirrors
    ``comparisonSegments`` in the desktop's ``workspace-groups.ts``.

    ``canonical`` first maps the path to its physical spelling, so a folder reached
    through a symlinked parent (a renamed home that left a compatibility link) is
    one identity when no git probe unifies the two spellings. The renderer cannot
    stat, so this step has no mirror there: its live overlay leaves a row it cannot
    match by spelling where this builder placed it. Case-insensitivity follows the
    spelling that arrived, so resolving never case-folds a POSIX path.
    """
    windows = _is_windows_path(path)
    if canonical and path:
        path = canonical(path)
    segs = [unicodedata.normalize("NFC", segment) for segment in _segments(path)]
    return [segment.casefold() for segment in segs] if windows else segs


def _path_key(path: str, canonical: Optional[Canonical] = None) -> str:
    """Canonical comparison key (separator/trailing-slash agnostic)."""
    return "/".join(_comparison_segments(path, canonical))


def _split_lane(path_or_lane: str) -> tuple[str, str]:
    """``(path, tail)`` of a lane id; the tail is its ``::branch::<label>`` / ``::kanban`` part."""
    marker = next((m for m in ("::branch::", "::kanban") if m in path_or_lane), None)
    if marker is None:
        return path_or_lane, ""
    root, suffix = path_or_lane.split(marker, 1)
    return root, marker + suffix


def _lane_key(path_or_lane: str, canonical: Optional[Canonical] = None) -> str:
    """Canonicalize only the path portion of a lane id (branch labels stay byte-preserved)."""
    root, tail = _split_lane(path_or_lane)
    return _path_key(root, canonical) + tail


def _shown(spellings: dict[str, str], identity: str) -> str:
    """The ONE spelling a folded identity is emitted under, out of ``{lexical key: spelling}``:
    the physical one when a row carries it, else the smallest. Never the most recent: rows
    arrive by recency, and an id that followed them would orphan the renderer's persisted
    state (dismissed auto projects, pins, order) each time the other spelling is used."""
    return spellings[min(spellings, key=lambda key: (key != identity, key))]


def _respeller(paths: list[str], canonical: Optional[Canonical]) -> Optional[Callable[[str], str]]:
    """``None`` unless ``canonical`` identifies some of ``paths`` that differ lexically; then a
    function mapping a path or lane id onto the spelling its folded identity is shown under.
    Spellings that already compare equal keep today's first-seen representative."""
    if not canonical:
        return None
    seen: dict[str, dict[str, str]] = {}  # identity -> {lexical key: first spelling}
    for path in paths:
        seen.setdefault(_path_key(path, canonical), {}).setdefault(_path_key(path), path)
    shown = {
        key: _shown(spellings, identity)
        for identity, spellings in seen.items() if len(spellings) > 1 for key in spellings}
    if not shown:
        return None

    def respell(path_or_lane: str) -> str:
        root, tail = _split_lane(path_or_lane)
        return shown.get(_path_key(root), root) + tail
    return respell


def base_name(path: str) -> str:
    segs = _segments(path)
    return segs[-1] if segs else ""


def kanban_worktree_dir(path: str) -> Optional[str]:
    """The ``<repo>/.worktrees`` dir for a ``.../.worktrees/<task>`` path, else None."""
    m = _KANBAN_DIR_RE.match(path or "")
    return m.group(1) if m else None


def _with_base_name(path: str, name: str = "") -> str:
    """Swap the last segment for ``name`` (``""`` -> the parent dir, ``""`` past the root)."""
    return re.sub(r"[^/\\]+$", name, (path or "").rstrip("/\\")).rstrip("" if name else "/\\")


def _field(row: dict, key: str) -> str:
    return (row.get(key) or "").strip()


def _session_time(session: dict) -> float:
    return float(session.get("last_active") or session.get("started_at") or 0)


def _last_active(sessions: list[dict]) -> float:
    return max((_session_time(s) for s in sessions), default=0.0)


def _placement(
    repo_root: str, lane_key: str, lane_label: str, lane_path: str, is_main: bool, is_kanban: bool,
    is_git: bool = True,
) -> dict:
    return {
        "repo_key": repo_root, "repo_label": base_name(repo_root) or repo_root,
        "lane_key": lane_key, "lane_label": lane_label, "lane_path": lane_path,
        "is_main": is_main, "is_kanban": is_kanban, "is_git": is_git}


def _trunk_placement(repo_root: str, branch: str) -> dict:
    # An unrecorded branch folds into the trunk lane so a repo never shows two "main" lanes.
    b = (branch or "").strip() or DEFAULT_BRANCH_LABEL
    return _placement(repo_root, _branch_lane_id(repo_root, b), b, repo_root, True, False)


def _kanban_placement(repo_root: str, kanban_dir: str) -> dict:
    return _placement(repo_root, _kanban_lane_id(repo_root), "kanban", kanban_dir, False, True)


def _probe_sibling_worktree(cwd: str, resolve: Resolve) -> str:
    """The parent repo root of a deleted ``<repo>-<suffix>`` worktree, else ``""``: trim one
    ``-<segment>`` at a time off each ancestor's name (the cwd is often a SUBDIR of the dead
    worktree), deepest first, returning the first sibling that resolves; probes are bounded."""
    probes = 0
    path = (cwd or "").rstrip("/\\")
    while path and probes < _MAX_SIBLING_PROBES:
        parts = base_name(path).split("-")
        for i in range(len(parts) - 1, 0, -1):
            if probes >= _MAX_SIBLING_PROBES:
                break
            probes += 1
            info = resolve(_with_base_name(path, "-".join(parts[:i])))
            if info and info.get("repo_root"):
                return (info["repo_root"] or "").strip()
        path = _with_base_name(path)
    return ""


def _place_by_heuristic(path: str) -> Optional[dict]:
    """Path-only fallback when there is no git probe and no persisted root."""
    base = base_name(path)
    if not base:
        return None
    kanban_dir = kanban_worktree_dir(path)
    if kanban_dir:
        return _kanban_placement(_with_base_name(kanban_dir), kanban_dir)
    m = re.match(r"^(.+)-wt-(.+)$", base)
    if m:
        return _placement(_with_base_name(path, m.group(1)), path, m.group(2), path, False, False)
    # No git knowledge at all: the lane is the folder itself, not a branch —
    # is_git=False tells the renderer never to `git switch` it (#61362).
    return _placement(path, _branch_lane_id(path, DEFAULT_BRANCH_LABEL), base, path, True, False, is_git=False)


def _place(
        cwd: str, branch: str, resolve: Optional[Resolve], persisted_root: str) -> Optional[dict]:
    info = resolve(cwd) if resolve else None
    if info and info.get("repo_root") and info.get("worktree_root"):
        repo_root, worktree_root = info["repo_root"], info["worktree_root"]
        if _path_key(worktree_root) == _path_key(repo_root) or info.get("is_main"):
            return _trunk_placement(repo_root, branch)
        kanban_dir = kanban_worktree_dir(worktree_root)
        if kanban_dir:
            return _kanban_placement(repo_root, kanban_dir)
        label = base_name(worktree_root) or worktree_root
        return _placement(repo_root, worktree_root, label, worktree_root, False, False)

    # No live probe: trust the persisted root; kanban tasks still collapse by path shape.
    if persisted_root:
        kanban_dir = kanban_worktree_dir(cwd)
        if kanban_dir:
            return _kanban_placement(persisted_root, kanban_dir)
        return _trunk_placement(persisted_root, branch)

    # Unresolvable cwd: a deleted ``<repo>-<suffix>`` worktree still belongs to its parent.
    sibling_root = _probe_sibling_worktree(cwd, resolve) if resolve else ""
    if sibling_root:
        return _trunk_placement(sibling_root, branch)
    return _place_by_heuristic(cwd)


def _place_session(session: dict, resolve: Optional[Resolve]) -> Optional[dict]:
    """``_place`` for a session row, anchored on its cwd or else its persisted repo root;
    ``None`` only when it has neither (the renderer's ``isDetachedSession``)."""
    root = _field(session, "git_repo_root")
    anchor = _field(session, "cwd") or root
    if not anchor:
        return None
    return _place(anchor, _field(session, "git_branch"), resolve, root)


def _session_repo_root(session: dict, resolve: Optional[Resolve]) -> str:
    """The COMMON repo root a session belongs to (folds linked worktrees)."""
    cwd = _field(session, "cwd")
    if cwd and resolve:
        info = resolve(cwd)
        if info and info.get("repo_root"):
            return info["repo_root"]
    return _field(session, "git_repo_root")


def _lane_sort_key(group: dict) -> tuple:
    # Trunk pins to the top, the kanban aggregate to the bottom; the rest by recency, then label.
    is_trunk = bool(group.get("isMain")) and group["label"].lower() in _TRUNK_BRANCHES
    return (0 if is_trunk else 1, 1 if group.get("isKanban") else 0,
            -_last_active(group.get("sessions") or []), group["label"].lower())


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
        for depth in range(1, max(len(p) for p in parents.values()) + 1):
            for g in pathed:
                prefix = "/".join(parents[id(g)][-depth:])
                base = base_name(g["path"]) or g["path"]
                g["label"] = f"{prefix}/{base}" if prefix else base
            if len({g["label"] for g in pathed}) == len(pathed):
                break


# Lane group wire fields <- placement keys (same order).
_LANE_FIELDS = ("id", "label", "path", "isMain", "isKanban", "isGit")
_PLACEMENT_LANE_KEYS = ("lane_key", "lane_label", "lane_path", "is_main", "is_kanban", "is_git")


def _repo_node(root: str, label: str) -> dict:
    return {"id": root, "label": label, "path": root, "groups": [], "sessionCount": 0}


def _build_repos(
        sessions: list[dict], resolve: Optional[Resolve], hydrate: bool,
        canonical: Optional[Canonical]) -> list[dict]:
    """Build the ``repo -> lane -> sessions`` subtree for a set of sessions. Roots and lane
    paths that only ``canonical`` identifies fold into one row under the spelling ``_shown``
    picks, with its lane ids re-rooted on it (``<repoRoot>::…`` stays true of the node)."""
    placed = [(s, p) for s in sessions if (p := _place_session(s, resolve))]
    respell = _respeller(
        [p[key] for _, p in placed for key in ("repo_key", "lane_path")], canonical)
    lanes: dict[str, tuple[dict, dict]] = {}  # lane identity -> (group, placement)
    for session, placement in placed:
        if respell:
            root = respell(placement["repo_key"])
            lane_path = respell(placement["lane_path"])
            label = placement["lane_label"]
            # A lane named after its folder (non-git, linked worktree) follows the spelling
            # shown; a trunk lane is named after its branch, even one called like the folder.
            if label == base_name(placement["lane_path"]) and not (
                    placement["is_main"] and placement["is_git"]):
                label = base_name(lane_path) or lane_path
            placement = {
                **placement, "repo_key": root, "repo_label": base_name(root) or root,
                "lane_key": respell(placement["lane_key"]), "lane_label": label,
                "lane_path": lane_path}
        lane_identity = _lane_key(placement["lane_key"], canonical)
        if lane_identity not in lanes:
            group = dict(zip(_LANE_FIELDS, (placement[k] for k in _PLACEMENT_LANE_KEYS)))
            group["sessions"] = []
            lanes[lane_identity] = (group, placement)
        lanes[lane_identity][0]["sessions"].append(session)
    repos: dict[str, dict] = {}
    for group, placement in lanes.values():
        group["sessions"].sort(key=_session_time, reverse=True)
        repo_key = placement["repo_key"]
        repo = repos.setdefault(
            _path_key(repo_key, canonical), _repo_node(repo_key, placement["repo_label"]))
        repo["groups"].append(group)
        repo["sessionCount"] += len(group["sessions"])
    repo_list = list(repos.values())
    for repo in repo_list:
        repo["groups"] = sorted(repo["groups"], key=_lane_sort_key)
        _disambiguate_labels(repo["groups"])
        # Drop per-lane rows only AFTER sorting (_lane_sort_key reads them); counts stay.
        if not hydrate:
            for group in repo["groups"]:
                group["sessions"] = []
    _disambiguate_labels(repo_list)
    return repo_list


def _seed_folder_repos(
        repos: list[dict], folders: list[dict], resolve: Optional[Resolve],
        canonical: Optional[Canonical]) -> list[dict]:
    """Ensure every declared project folder shows as a repo, even with 0 sessions (else the
    entered-project view renders blank); folders covered by a session-derived repo are untouched."""
    seen = {
        _path_key(v, canonical) for repo in repos for v in (repo.get("id"), repo.get("path")) if v}
    seeded = list(repos)
    for folder in folders or []:
        raw = _field(folder, "path")
        if not raw:
            continue
        info = resolve(raw) if resolve else None
        root = (info or {}).get("repo_root") or raw.rstrip("/\\")
        root_key = _path_key(root, canonical)
        if not root_key or root_key in seen:
            continue
        seeded.append(_repo_node(root, base_name(root) or root))
        seen.add(root_key)
    if len(seeded) != len(repos):
        _disambiguate_labels(seeded)
    return seeded


class _FolderIndex:
    """Normalized folder path -> (owning project, depth); matched by walking cwd ancestors.
    A target walks the folders as declared under its own spelling, exactly as it does
    without ``canonical``. Only when that finds no owner is its physical spelling tried:
    against the declared folders (the owner the same row has under that spelling), then
    against the targets of folders declared through a link. So for ONE target a link can
    supply a match but never replaces the one its own spelling makes: a cwd reached
    through a link that points OUT of its project stays in it. A row still takes the
    deeper of its cwd's and its repo root's matches (``_project_for_session``), so a match
    supplied for one of them can outrank the other's. A folder's depth is always its
    declared spelling's, whichever way it was found."""

    def __init__(self, projects: list[dict], canonical: Optional[Canonical]) -> None:
        self._canonical = canonical
        self._by_path: dict[str, tuple[dict, int]] = {}
        # Where a folder declared through a link points: the last resort of ``match``.
        self._by_target: dict[str, tuple[dict, int]] = {}
        for project in projects:
            for folder in project.get("folders") or []:
                segs = _comparison_segments(folder.get("path") or "")
                self._add(self._by_path, segs, project, len(segs))
                physical = _comparison_segments(folder.get("path") or "", canonical)
                if physical != segs:
                    self._add(self._by_target, physical, project, len(segs))

    @staticmethod
    def _add(
            index: dict[str, tuple[dict, int]], segs: list[str], project: dict, depth: int) -> None:
        # Deepest folder wins; ties keep the first project (scan order).
        if segs and depth > index.get("/".join(segs), (None, -1))[1]:
            index["/".join(segs)] = (project, depth)

    @staticmethod
    def _walk(index: dict[str, tuple[dict, int]], segs: list[str]) -> tuple[Optional[dict], int]:
        for end in range(len(segs), 0, -1):
            hit = index.get("/".join(segs[:end]))
            if hit:
                return hit
        return None, -1

    def match(self, target: str) -> tuple[Optional[dict], int]:
        """Owning project for ``target`` by longest ancestor folder, + its depth."""
        segs = _comparison_segments(target or "")
        hit = self._walk(self._by_path, segs)
        # No declared folders: nothing to resolve a cwd for.
        if hit[0] is not None or not self._canonical or not self._by_path:
            return hit
        physical = _comparison_segments(target or "", self._canonical)
        if physical != segs:
            hit = self._walk(self._by_path, physical)
        return hit if hit[0] is not None else self._walk(self._by_target, physical)


def _project_for_session(
        session: dict, index: _FolderIndex, resolve: Optional[Resolve]) -> Optional[dict]:
    cwd = _field(session, "cwd")
    repo_root = _session_repo_root(session, resolve)
    # A root-only row (empty cwd) still belongs to the project owning its root.
    candidates = [t for t in dict.fromkeys((cwd, repo_root)) if t]
    if not candidates:
        return None
    # Longest folder match wins; ties keep the cwd match (max() keeps the first maximum).
    return max((index.match(t) for t in candidates), key=lambda hit: hit[1])[0]


def _project_node(
    pid: str, label: str, path: Optional[str], repos: list[dict], session_count: int,
    last_active: float, preview_sessions: list[dict], sessions: Optional[list[dict]] = None,
    **flags: Any) -> dict:
    """``flags`` overrides ``color``/``icon``/``isAuto``/``isNoProject``; key order = wire shape."""
    rows = sessions or []
    node = {
        "id": pid, "label": label, "path": path, "color": None, "icon": None,
        "isAuto": False, "isNoProject": False,
        "sessionCount": session_count, "lastActive": last_active,
        # Totals over the same sessions `sessionCount` counts (billed cost, else estimated).
        "totalTokens": sum(
            (s.get("input_tokens") or 0) + (s.get("output_tokens") or 0) for s in rows),
        "totalCostUsd": sum(
            float(s.get("actual_cost_usd") or s.get("estimated_cost_usd") or 0) for s in rows),
        "repos": repos, "previewSessions": preview_sessions,
        # Every claimed row, not just the preview window: the renderer's live overlay
        # uses this as the ONE owner (a sibling worktree row with git_repo_root NULL
        # re-classifies to an ancestor project by cwd alone).
        "sessionIds": [s["id"] for s in rows if s.get("id")]}
    node.update(flags)
    return node


def _auto_buckets(
    unowned: list[dict], resolve: Optional[Resolve], junk: Callable, junk_cwd: Callable,
    exists: Callable, canonical: Optional[Canonical]) -> tuple[dict[str, dict], list[dict]]:
    """Group leftover sessions by auto-project root (common git root, else the session cwd
    for non-git workspaces); the rest go to the Home bucket."""
    by_auto_root: dict[str, dict] = {}
    homeless: list[dict] = []
    for session in unowned:
        root = _session_repo_root(session, resolve)
        if root:
            # Stricter repo policy for real git roots; a root gone from disk is stale and
            # must not resurrect as a project (never reinterpret it as a cwd-only project).
            if junk(root) or not exists(root):
                root = ""
        elif (cwd := _field(session, "cwd")) and not junk_cwd(cwd):
            # A path-only heuristic placement whose dir is gone from disk would mint a phantom
            # project that can only be dismissed by hand -> Home.
            placement = _place_session(session, resolve)
            if placement and exists(placement["repo_key"]):
                root = placement["repo_key"]
        key = _path_key(root, canonical) if root else ""
        if key:
            bucket = by_auto_root.setdefault(key, {"spellings": {}, "sessions": []})
            bucket["spellings"].setdefault(_path_key(root), root)
            bucket["sessions"].append(session)
        else:
            homeless.append(session)
    for key, bucket in by_auto_root.items():
        bucket["root"] = _shown(bucket.pop("spellings"), key)
    return by_auto_root, homeless


def _home_project(homeless: list[dict], hydrate: bool, previews: list[dict]) -> dict:
    """The synthetic Home bucket: no folder => no repo/lane structure, one lane carries the rows."""
    lane = {
        "id": NO_PROJECT_ID, "label": NO_PROJECT_LABEL, "path": None, "isMain": False,
        "isKanban": False, "sessions": homeless if hydrate else []}
    home_repo = {
        "id": NO_PROJECT_ID, "label": NO_PROJECT_LABEL, "path": None, "groups": [lane],
        "sessionCount": len(homeless)}
    return _project_node(
        NO_PROJECT_ID, NO_PROJECT_LABEL, None, [home_repo], len(homeless), _last_active(homeless),
        previews, homeless, isNoProject=True)


def build_tree(
    projects: list[dict], sessions: list[dict], discovered_repos: list[dict],
    resolve: Optional[Resolve] = None, *, preview_limit: int = 3, hydrate: bool = False,
    is_junk_root: Optional[Callable[[str], bool]] = None,
    is_junk_cwd: Optional[Callable[[str], bool]] = None, exists: Optional[Exists] = None,
    canonical: Optional[Canonical] = None) -> dict:
    """Build the authoritative project tree -> ``{"projects", "scoped_session_ids"}``.

    ``is_junk_root`` flags git roots that must never become an AUTO project; ``is_junk_cwd``
    is the narrower non-git policy (explicit projects are honored regardless); ``exists``
    keeps a DELETED workspace from becoming a phantom AUTO project (omit on remote backends).
    ``canonical`` folds two spellings of one local folder (a symlinked parent) into one row
    when no git probe does. The row is emitted under one of them, picked without regard to
    row order (``_shown``); a folder that arrives under a single spelling keeps it.
    ``hydrate`` False empties lane ``sessions`` but keeps counts + ``previewSessions``.
    """
    active_projects = [p for p in projects if not p.get("archived")]
    _junk = is_junk_root or (lambda _root: False)
    _junk_cwd = is_junk_cwd or (lambda _cwd: False)
    _exists = exists or (lambda _path: True)
    folder_index = _FolderIndex(active_projects, canonical)
    by_project: dict[str, list[dict]] = {}  # explicit project id -> owned rows
    unowned: list[dict] = []
    for session in sessions:
        owner = _project_for_session(session, folder_index, resolve)
        (by_project.setdefault(owner["id"], []) if owner else unowned).append(session)

    scoped_ids: list[str] = []
    result: list[dict] = []

    def _previews(project_sessions: list[dict]) -> list[dict]:
        if preview_limit <= 0:
            return []
        return sorted(project_sessions, key=_session_time, reverse=True)[:preview_limit]

    def _scope(project_sessions: list[dict]) -> None:
        scoped_ids.extend(s["id"] for s in project_sessions if s.get("id"))
    # Tier 1: explicit, user-created projects (always shown, even with 0 sessions).
    for project in active_projects:
        psessions = by_project.get(project["id"], [])
        _scope(psessions)
        repos = _build_repos(psessions, resolve, hydrate, canonical)
        repos = _seed_folder_repos(repos, project.get("folders") or [], resolve, canonical)
        result.append(_project_node(
            project["id"], project.get("name") or project["id"], project.get("primary_path"), repos,
            len(psessions), _last_active(psessions), _previews(psessions), psessions,
            color=project.get("color"), icon=project.get("icon")))

    # Tier 2: auto projects from leftover sessions.
    by_auto_root, homeless = _auto_buckets(
        unowned, resolve, _junk, _junk_cwd, _exists, canonical)
    seen: set[str] = set()
    for bucket in by_auto_root.values():
        auto_root, auto_sessions = bucket["root"], bucket["sessions"]
        auto_key = _path_key(auto_root, canonical)
        repos = _build_repos(auto_sessions, resolve, hydrate, canonical)
        repo_node = next(
            (r for r in repos
             if _path_key(r.get("id") or r.get("path") or "", canonical) == auto_key), None)
        if repo_node is None:
            homeless.extend(auto_sessions)
            continue
        seen.add(auto_key)
        _scope(auto_sessions)
        result.append(_project_node(
            auto_root, base_name(auto_root) or auto_root, auto_root, repos,
            repo_node["sessionCount"], _last_active(auto_sessions), _previews(auto_sessions),
            auto_sessions, isAuto=True))

    # Tier 3: discovered repos with no loaded sessions, folded to their common root.
    found: list[tuple[dict, str]] = []
    for repo in discovered_repos or []:
        raw_root = _field(repo, "root")
        if raw_root:
            info = resolve(raw_root) if resolve else None
            found.append((repo, (info or {}).get("repo_root") or raw_root))
    respell = _respeller([root for _, root in found], canonical)
    for repo, root in found:
        root = respell(root) if respell else root
        root_key = _path_key(root, canonical)
        if root_key in seen or _junk(root) or folder_index.match(root)[0]:
            continue
        seen.add(root_key)
        label = repo.get("label") or base_name(root) or root
        result.append(_project_node(
            root, label, root, [_repo_node(root, label)], int(repo.get("sessions") or 0),
            float(repo.get("last_active") or 0), [], isAuto=True))

    # Auto-project basename labels can collide; explicit projects keep their user-chosen names.
    _disambiguate_labels([p for p in result if p.get("isAuto")])

    # Tier 0: whatever the tiers above could not place. Leads the list; omitted when empty.
    if homeless:
        homeless.sort(key=_session_time, reverse=True)
        _scope(homeless)
        result.insert(0, _home_project(homeless, hydrate, _previews(homeless)))

    return {"projects": result, "scoped_session_ids": scoped_ids}
