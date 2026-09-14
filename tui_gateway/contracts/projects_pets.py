"""Projects (``projects.*`` — per-profile multi-folder workspaces, repo discovery, sidebar tree)
and the pet mascot surface (``pet.*`` — gallery, sprite payloads, adopt/remove/rename).

Every method here is profile-scoped: the desktop's ``projectParams`` / ``petRpc`` wrappers add
``profile`` so app-global remote mode reads the focused profile's ``projects.db`` / ``config.yaml``.
The pet wire predates the snake_case rule and travels camelCase (``displayName``,
``spritesheetBase64``); the field names below are those wire keys verbatim.
"""

from __future__ import annotations

from pydantic import Field

from .base import JsonValue, Params, Result
from .common import OkResult, OpenModel, ProfileParams, StoredSessionRow
from .registry import method

# ── projects: stored rows ─────────────────────────────────────────────────────────────────────


class ProjectFolder(Result):
    """``hermes_cli/projects_db.py::ProjectFolder.to_dict``."""

    path: str
    label: str | None = None
    is_primary: bool = False
    added_at: int | None = None


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
    auto_pull: bool = False
    archived: bool = False
    created_at: int
    folders: list[ProjectFolder] = Field(default_factory=list)


class ProjectsPayload(Result):
    """``methods_projects._projects_payload``: every project (archived included) + the active id."""

    projects: list[ProjectInfo]
    active_id: str | None = None


class ProjectResult(Result):
    project: ProjectInfo


class OptionalProjectResult(Result):
    project: ProjectInfo | None = None


class ProjectIdParams(ProfileParams):
    """Any method addressed at one stored project (``5062`` when the id resolves to nothing)."""

    id: str


method("projects.list", params=ProfileParams, result=ProjectsPayload,
       doc="Every project of the profile (archived included) plus which one is active.")
method("projects.get", params=ProjectIdParams, result=ProjectResult,
       doc="One stored project with its folders.")


class ProjectsCreateParams(ProfileParams):
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

    auto_pull: bool | None = None


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


class ProjectsSetActiveParams(ProfileParams):
    """No ``id`` (or null) clears the active project."""

    id: str | None = None


class ActiveIdResult(Result):
    active_id: str | None = None


method("projects.set_active", params=ProjectsSetActiveParams, result=ActiveIdResult,
       doc="Switch (or clear) the active project for the profile.")


class ProjectsForCwdParams(ProfileParams):
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


class ProjectsDiscoverReposParams(ProfileParams):
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


class ProjectsRecordReposParams(ProfileParams):
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

    profile: str | None = None


class ProjectTreeLane(Result):
    """One branch / worktree / kanban lane inside a repo; ``sessions`` is empty unless hydrated."""

    id: str
    label: str
    path: str | None = None
    isMain: bool = False
    isKanban: bool = False
    sessions: list[ProjectTreeSession] = Field(default_factory=list)


class ProjectTreeRepo(Result):
    id: str
    label: str
    path: str | None = None
    groups: list[ProjectTreeLane] = Field(default_factory=list)
    sessionCount: int = 0


class ProjectTreeNode(Result):
    """``project_tree._project_node`` — explicit, auto (git root) or the synthetic Home bucket."""

    id: str
    label: str
    path: str | None = None
    color: str | None = None
    icon: str | None = None
    auto_pull: bool = False
    isAuto: bool = False
    isNoProject: bool = False
    sessionCount: int = 0
    lastActive: float = 0.0
    totalTokens: int = 0
    totalCostUsd: float = 0.0
    repos: list[ProjectTreeRepo] = Field(default_factory=list)
    previewSessions: list[ProjectTreeSession] = Field(default_factory=list)


class ProjectsTreeParams(ProfileParams):
    preview_limit: int | None = None
    session_limit: int | None = None


class ProjectsTreeResult(Result):
    projects: list[ProjectTreeNode]
    active_id: str | None = None
    scoped_session_ids: list[str] = Field(default_factory=list)


method("projects.tree", params=ProjectsTreeParams, result=ProjectsTreeResult,
       doc="Project → repo → lane overview with counts and a few preview sessions per project.")


class ProjectsProjectSessionsParams(ProfileParams):
    project_id: str
    session_limit: int | None = None


class ProjectsProjectSessionsResult(Result):
    project: ProjectTreeNode | None = None


method("projects.project_sessions", params=ProjectsProjectSessionsParams, result=ProjectsProjectSessionsResult,
       doc="Fully hydrated lanes for one project, from the same grouping as projects.tree.")


# ── pet: active mascot ────────────────────────────────────────────────────────────────────────


class PetInfoParams(ProfileParams):
    """``knownRevision``: the spritesheet revision the caller already holds (send-once bytes)."""

    knownRevision: str | None = None


class PetInfoResult(OpenModel):
    """``server._pet_sprite_payload`` behind ``enabled``; every sprite field is absent when the pet
    display is off, ``spritesheetBase64`` is elided when ``spritesheetUnchanged``."""

    enabled: bool
    slug: str | None = None
    displayName: str | None = None
    mime: str | None = None
    spritesheetBase64: str | None = None
    spritesheetRevision: str | None = None
    spritesheetUnchanged: bool | None = None
    frameW: int | None = None
    frameH: int | None = None
    framesPerState: int | None = None
    framesByState: dict[str, int] | None = None
    framesByRow: dict[str, int] | None = None
    loopMs: int | None = None
    scale: float | None = None
    stateRows: list[str] | None = None


method("pet.info", params=PetInfoParams, result=PetInfoResult,
       doc="Active pet for sprite renderers: spritesheet (base64) + frame geometry + state-row taxonomy.")


class PetInfoMetaResult(Result):
    enabled: bool
    slug: str | None = None
    displayName: str | None = None
    scale: float | None = None
    spritesheetRevision: str | None = None


method("pet.info.meta", params=ProfileParams, result=PetInfoMetaResult,
       doc="Cheap active-pet metadata used to avoid full payload refreshes.")


class PetCellsParams(ProfileParams):
    """``graphics`` opts into the kitty payload when the TTY speaks it; ``cols`` overrides the width."""

    state: str | None = None
    cols: int | None = None
    graphics: bool = False


class PetCellsResult(Result):
    """Unicode: ``frames`` is frame → row → cell ``[tr,tg,tb,ta, br,bg,bb,ba]``; kitty (``graphics``
    set): ``frames`` are transmit escapes and ``placeholder`` the text grid."""

    enabled: bool
    slug: str | None = None
    displayName: str | None = None
    state: str | None = None
    cols: int | None = None
    frameMs: float | None = None
    frames: list[list[list[list[int]]]] | list[str] | None = None
    scale: float | None = None
    graphics: str | None = None
    imageId: int | None = None
    color: str | None = None
    rows: int | None = None
    placeholder: list[str] | None = None


method("pet.cells", params=PetCellsParams, result=PetCellsResult,
       doc="Half-block cell frames (or a kitty placement) for one pet state.")


# ── pet: gallery / picker ─────────────────────────────────────────────────────────────────────


class PetGalleryParams(ProfileParams):
    localOnly: bool = False


class PetGalleryEntry(Result):
    slug: str
    displayName: str
    installed: bool
    spritesheetUrl: str = ""
    curated: bool | None = None
    generated: bool = False


class PetGalleryResult(Result):
    enabled: bool
    active: str = ""
    pets: list[PetGalleryEntry] = Field(default_factory=list)


method("pet.gallery", params=PetGalleryParams, result=PetGalleryResult,
       doc="Petdex gallery + local install state (installed-only offline); localOnly skips the remote manifest.")


class PetSlugParams(ProfileParams):
    slug: str


class PetSlugResult(Result):
    ok: bool
    slug: str
    displayName: str | None = None


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
    zipBase64: str


method("pet.export", params=PetSlugParams, result=PetExportResult,
       doc="Export an installed pet as a re-importable .zip.")


class PetThumbParams(PetSlugParams):
    """``url``: spritesheet source for a not-yet-installed pet."""

    url: str | None = None


class PetThumbResult(Result):
    ok: bool
    slug: str
    dataUri: str | None = None


method("pet.thumb", params=PetThumbParams, result=PetThumbResult,
       doc="Idle-frame PNG data URI for the picker (desktop CSP breaks CDN <img>).")

method("pet.disable", params=ProfileParams, result=OkResult,
       doc="Turn the pet display off from the desktop picker.")


class PetScaleParams(ProfileParams):
    scale: JsonValue = None  # number or numeric string; a non-number answers 4004


class PetScaleResult(Result):
    ok: bool
    scale: float


method("pet.scale", params=PetScaleParams, result=PetScaleResult,
       doc="Persist display.pet.scale (clamped to engine bounds) from the desktop slider.")
