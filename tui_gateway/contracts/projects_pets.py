"""Projects (``projects.*`` — per-profile multi-folder workspaces, repo discovery, sidebar tree)
and the pet mascot surface (``pet.*`` — gallery, sprite payloads, adopt/remove/rename).

Every method here is profile-scoped: the desktop's ``projectParams`` / ``petRpc`` wrappers add
``profile`` so app-global remote mode reads the focused profile's ``projects.db`` / ``config.yaml``.
The pet wire predates the snake_case rule and travels camelCase (``displayName``,
``spritesheetBase64``); the field names below are those wire keys verbatim.
"""

from __future__ import annotations

from pydantic import Field

from .base import MethodParams, Params, Result
from .common import OkResult, StoredSessionRow
from .registry import method

# ── projects: stored rows ─────────────────────────────────────────────────────────────────────


class ProjectFolder(Result):
    """``hermes_cli/projects_db.py::ProjectFolder.to_dict``."""

    path: str
    label: str | None = None
    is_primary: bool = False
    added_at: float = 0.0


class ProjectInfo(Result):
    """``hermes_cli/projects_db.py::Project.to_dict`` — one stored project with its folders."""

    id: str
    slug: str
    name: str
    description: str | None = None
    icon: str | None = None
    color: str | None = None
    board_slug: str | None = None
    primary_path: str | None = None
    archived: bool = False
    created_at: float
    folders: list[ProjectFolder] = Field(default_factory=list)


class ProjectsPayload(Result):
    """``methods_projects._projects_payload``: every project (archived included) + the active id."""

    projects: list[ProjectInfo]
    active_id: str | None = None


class ProjectResult(Result):
    project: ProjectInfo


class OptionalProjectResult(Result):
    project: ProjectInfo | None = None


class ProjectIdParams(MethodParams):
    """A method addressed at one stored project (``5062`` when the id resolves to nothing)."""

    id: str


method("projects.list", params=MethodParams, result=ProjectsPayload,
       doc="Every project of the profile (archived included) plus which one is active.")
method("projects.get", params=ProjectIdParams, result=ProjectResult,
       doc="One stored project with its folders.")


class ProjectsCreateParams(MethodParams):
    """``use`` also activates the new project."""

    name: str
    folders: list[str] | None = None
    slug: str | None = None
    primary_path: str | None = None
    description: str | None = None
    icon: str | None = None
    color: str | None = None
    board_slug: str | None = None
    use: bool = False


method("projects.create", params=ProjectsCreateParams, result=OptionalProjectResult,
       doc="Create a project from a name + folders; duplicate primary paths are refused (5063).")


class ProjectsUpdateParams(ProjectIdParams):
    """Absent keys are left untouched; ``''`` clears ``color`` / ``icon``."""

    name: str | None = None
    description: str | None = None
    icon: str | None = None
    color: str | None = None
    board_slug: str | None = None


method("projects.update", params=ProjectsUpdateParams, result=ProjectResult,
       doc="Patch a project's display fields; answers the refreshed project.")


class ProjectsAddFolderParams(ProjectIdParams):
    path: str
    label: str | None = None
    is_primary: bool = False


method("projects.add_folder", params=ProjectsAddFolderParams, result=ProjectResult,
       doc="Attach a folder to a project (optionally as its primary path).")


class ProjectFolderParams(ProjectIdParams):
    path: str


method("projects.remove_folder", params=ProjectFolderParams, result=ProjectResult,
       doc="Detach a folder from a project.")
method("projects.set_primary", params=ProjectFolderParams, result=ProjectResult,
       doc="Make one attached folder the project's primary path.")


class ProjectsArchiveParams(ProjectIdParams):
    restore: bool = False


method("projects.archive", params=ProjectsArchiveParams, result=ProjectsPayload,
       doc="Archive (or with ``restore`` un-archive) a project; answers the full listing.")
method("projects.delete", params=ProjectIdParams, result=ProjectsPayload,
       doc="Delete a project and its folders; answers the full listing.")


class ProjectsSetActiveParams(MethodParams):
    """No ``id`` (or null) clears the active project."""

    id: str | None = None


class ActiveIdResult(Result):
    active_id: str | None = None


method("projects.set_active", params=ProjectsSetActiveParams, result=ActiveIdResult,
       doc="Switch (or clear) the active project for the profile.")


class ProjectsForCwdParams(MethodParams):
    """Absent ``cwd`` resolves the gateway's default completion cwd."""

    cwd: str | None = None


class ProjectsForCwdResult(Result):
    project: ProjectInfo | None = None
    cwd: str
    branch: str = ""


method("projects.for_cwd", params=ProjectsForCwdParams, result=ProjectsForCwdResult,
       doc="Which project (if any) owns a directory, plus the resolved cwd and its git branch.")


# ── projects: repo discovery ──────────────────────────────────────────────────────────────────


class RepoDiscoveryPolicy(Result):
    """``methods_projects._repo_discovery_policy`` — the effective ``desktop.repo_scan_*`` config."""

    enabled: bool
    roots: list[str]
    exclude_paths: list[str]


class RepoDiscoveryPolicyParams(Params):
    """The policy the desktop scanned under (short or ``repo_scan_*`` long keys both accepted)."""

    enabled: bool | None = None
    roots: list[str] | None = None
    exclude_paths: list[str] | None = None
    repo_scan_enabled: bool | None = None
    repo_scan_roots: list[str] | None = None
    repo_scan_exclude_paths: list[str] | None = None


class DiscoveredRepo(Result):
    """``methods_projects._discover_repos_payload`` row: a git root with session totals."""

    root: str
    label: str = ""
    sessions: int = 0
    last_active: float = 0.0


class ProjectsDiscoverReposParams(MethodParams):
    """``scan`` asks the host to walk the policy roots itself (remote-gateway desktop)."""

    scan: bool = False


class ProjectsDiscoverReposResult(Result):
    repos: list[DiscoveredRepo]
    discovery_policy: RepoDiscoveryPolicy | None = None


method("projects.discover_repos", params=ProjectsDiscoverReposParams, result=ProjectsDiscoverReposResult,
       doc="Repos for the desktop overview: scanned-from-disk (cached) ∪ session-derived.")


class RecordRepoItem(Params):
    root: str
    label: str | None = None


class ProjectsRecordReposParams(MethodParams):
    """Repos as ``{root, label}`` objects or bare root strings; entries without a root are skipped."""

    repos: list[RecordRepoItem | str] | None = None
    discovery_policy: RepoDiscoveryPolicyParams | None = None


class ProjectsRecordReposResult(ProjectsDiscoverReposResult):
    accepted: bool


method("projects.record_repos", params=ProjectsRecordReposParams, result=ProjectsRecordReposResult,
       doc="Persist repo roots found by the client's (desktop-side) scan; return the merged list.")


# ── projects: sidebar tree ────────────────────────────────────────────────────────────────────


class ProjectTreeSession(StoredSessionRow):
    """``methods_projects._project_tree_row`` + ``project_tree.stamp_profile``: the minimal row the
    sidebar renders, stamped with the profile it belongs to."""

    profile: str | None


class ProjectTreeLane(Result):
    """One branch / worktree / kanban lane inside a repo; ``sessions`` is empty unless hydrated."""

    id: str
    label: str
    path: str | None
    isMain: bool
    isKanban: bool
    sessions: list[ProjectTreeSession]


class ProjectTreeRepo(Result):
    id: str
    label: str
    path: str | None
    groups: list[ProjectTreeLane]
    sessionCount: int


class ProjectTreeNode(Result):
    """``project_tree._project_node`` — closed project → repo → lane → session graph."""

    id: str
    label: str
    path: str | None
    color: str | None
    icon: str | None
    isAuto: bool
    isNoProject: bool
    sessionCount: int
    lastActive: float
    totalTokens: int
    totalCostUsd: float
    repos: list[ProjectTreeRepo]
    previewSessions: list[ProjectTreeSession]


class ProjectsTreeParams(MethodParams):
    preview_limit: int | None = None
    session_limit: int | None = None


class ProjectsTreeResult(Result):
    projects: list[ProjectTreeNode]
    active_id: str | None
    scoped_session_ids: list[str]


method("projects.tree", params=ProjectsTreeParams, result=ProjectsTreeResult,
       doc="Project → repo → lane overview with counts and a few preview sessions per project.")


class ProjectsProjectSessionsParams(MethodParams):
    project_id: str
    session_limit: int | None = None


class ProjectsProjectSessionsResult(Result):
    project: ProjectTreeNode | None


method("projects.project_sessions", params=ProjectsProjectSessionsParams, result=ProjectsProjectSessionsResult,
       doc="Fully hydrated lanes for one project, from the same grouping as projects.tree.")


# ── pet: active mascot ────────────────────────────────────────────────────────────────────────


class PetInfoParams(MethodParams):
    """``knownRevision``: the spritesheet revision the caller already holds (send-once bytes)."""

    knownRevision: str | None = None


class PetInfoResult(Result):
    """``server._pet_sprite_payload`` behind ``enabled``; sprite fields are absent when display is off
    and ``spritesheetBase64`` is absent when ``spritesheetUnchanged`` is true."""

    enabled: bool
    slug: str | None
    displayName: str | None  # noqa: N815 - wire key
    mime: str | None
    # The producer pops spritesheetBase64 when the client's revision matches and only then sets spritesheetUnchanged.
    spritesheetBase64: str | None = None  # noqa: N815 - wire key
    spritesheetRevision: str | None  # noqa: N815 - wire key
    spritesheetUnchanged: bool | None = None  # noqa: N815 - wire key
    frameW: int | None  # noqa: N815 - wire key
    frameH: int | None  # noqa: N815 - wire key
    framesPerState: int | None  # noqa: N815 - wire key
    framesByState: dict[str, int] | None  # noqa: N815 - wire key
    framesByRow: dict[str, int] | None  # noqa: N815 - wire key
    loopMs: int | None  # noqa: N815 - wire key
    scale: float | None
    stateRows: list[str] | None  # noqa: N815 - wire key


method("pet.info", params=PetInfoParams, result=PetInfoResult,
       doc="Active pet for sprite renderers: spritesheet (base64) + frame geometry + state-row taxonomy.")


class PetInfoMetaResult(Result):
    enabled: bool
    slug: str | None
    displayName: str | None  # noqa: N815 - wire key
    scale: float | None
    spritesheetRevision: str | None  # noqa: N815 - wire key


method("pet.info.meta", params=MethodParams, result=PetInfoMetaResult,
       doc="Cheap active-pet metadata used to avoid full payload refreshes.")


class PetCellsParams(MethodParams):
    """``graphics`` opts into the kitty payload when the TTY speaks it; ``cols`` overrides the width."""

    state: str | None = None
    cols: int | None = None
    graphics: bool = False


class PetCellsResult(Result):
    """Unicode cells or kitty transmit escapes from methods_session.py:1268-1290."""

    enabled: bool
    slug: str | None
    displayName: str | None  # noqa: N815 - wire key
    state: str | None
    cols: int | None
    frameMs: float | None  # noqa: N815 - wire key
    frames: list[list[list[list[int]]]] | list[str] | None
    scale: float | None
    graphics: str | None
    imageId: int | None  # noqa: N815 - wire key
    color: str | None
    rows: int | None
    placeholder: list[str] | None


method("pet.cells", params=PetCellsParams, result=PetCellsResult,
       doc="Half-block cell frames (or a kitty placement) for one pet state.")


# ── pet: gallery / picker ─────────────────────────────────────────────────────────────────────


class PetGalleryParams(MethodParams):
    localOnly: bool = False


class PetGalleryEntry(Result):
    slug: str
    displayName: str  # noqa: N815 - wire key
    installed: bool
    spritesheetUrl: str  # noqa: N815 - wire key
    curated: bool | None
    generated: bool


class PetGalleryResult(Result):
    enabled: bool
    active: str
    pets: list[PetGalleryEntry]


method("pet.gallery", params=PetGalleryParams, result=PetGalleryResult,
       doc="Petdex gallery + local install state (installed-only offline); localOnly skips the remote manifest.")


class PetSlugParams(MethodParams):
    slug: str


class PetSlugResult(Result):
    ok: bool
    slug: str
    displayName: str | None  # noqa: N815 - wire key


method("pet.select", params=PetSlugParams, result=PetSlugResult,
       doc="Adopt a pet: install (if needed) + activate; writes display.pet.* to config.")
method("pet.remove", params=PetSlugParams, result=PetSlugResult,
       doc="Uninstall a pet (delete its directory); if it was active, turn the display off.")


class PetRenameParams(PetSlugParams):
    name: str


method("pet.rename", params=PetRenameParams, result=PetSlugResult,
       doc="Rename a pet's display name + realign its slug/dir; follows the active slug in config.")


class PetExportResult(Result):
    ok: bool
    filename: str
    zipBase64: str  # noqa: N815 - wire key


method("pet.export", params=PetSlugParams, result=PetExportResult,
       doc="Export an installed pet as a re-importable .zip.")


class PetThumbParams(PetSlugParams):
    """``url``: spritesheet source for a not-yet-installed pet."""

    url: str | None = None


class PetThumbResult(Result):
    ok: bool
    slug: str
    dataUri: str | None  # noqa: N815 - wire key


method("pet.thumb", params=PetThumbParams, result=PetThumbResult,
       doc="Idle-frame PNG data URI for the picker (desktop CSP breaks CDN <img>).")

method("pet.disable", params=MethodParams, result=OkResult,
       doc="Turn the pet display off from the desktop picker.")


class PetScaleParams(MethodParams):
    scale: float | str | None = None


class PetScaleResult(Result):
    ok: bool
    scale: float


method("pet.scale", params=PetScaleParams, result=PetScaleResult,
       doc="Persist display.pet.scale (clamped to engine bounds) from the desktop slider.")
