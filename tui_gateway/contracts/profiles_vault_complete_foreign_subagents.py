"""Contracts for ``methods_profiles``, ``methods_vault``, ``methods_complete``,
``methods_session_foreign`` and ``methods_subagents``.

Profiles are the ws twin of the dashboard's ``/api/profiles``; the vault handlers are the
Desktop's Settings → Credential Vault door (metadata only — a secret never appears in a result);
completions feed the composer popovers; ``session.foreign.*`` browses Claude Code / Codex
histories on the serving backend; ``subagent.*`` is the session-scoped roster of live children.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import JsonValue, Params, Result, WireEnum
from .common import OpenModel, ProfileParams, SessionParams, SubagentStatus
from .config_free_tier_control import ModelOptionProvider
from .registry import method

# ── completions / paste / model keys (methods_complete) ───────────────────────────────────────


class CompletionItem(Result):
    """One popover row; ``kind`` rides only on slash completions (command vs skill)."""

    text: str
    display: str = ""
    meta: str = ""
    kind: str | None = None


class CompletionItemsResult(Result):
    items: list[CompletionItem] = Field(default_factory=list)


class CompletePathParams(ProfileParams):
    """``word`` is the token under the cursor (``@`` prefix = context reference); ``cwd`` /
    ``session_id`` pick the directory the listing resolves against."""

    word: str | None = None
    cwd: str | None = None
    session_id: str | None = None


method("complete.path", params=CompletePathParams, result=CompletionItemsResult,
       doc="Path / @-reference completions for the composer (files, folders, profiles, plugin providers).")


class CompleteSlashParams(Params):
    """``session_id`` binds skill completions to that session's profile and workspace (project skills)."""

    text: str | None = None
    session_id: str | None = None


class CompleteSlashResult(Result):
    """``replace_from`` is the column the accepted item replaces from."""

    items: list[CompletionItem] = Field(default_factory=list)
    replace_from: int | None = None


method("complete.slash", params=CompleteSlashParams, result=CompleteSlashResult,
       doc="Ranked slash-command / skill completions for a ``/`` token.")


class PasteCollapseParams(Params):
    text: str | None = None


class PasteCollapseResult(Result):
    placeholder: str
    path: str
    lines: int


method("paste.collapse", params=PasteCollapseParams, result=PasteCollapseResult,
       doc="Spill a large paste to a file and hand back the inline placeholder.")


class ModelSaveKeyParams(Params):
    slug: str
    api_key: str
    session_id: str | None = None


class ModelSaveKeyResult(Result):
    provider: ModelOptionProvider


method("model.save_key", params=ModelSaveKeyParams, result=ModelSaveKeyResult,
       doc="Save an API key for a provider and return its refreshed inventory row.")


class ModelDisconnectParams(Params):
    slug: str
    session_id: str | None = None


class ModelDisconnectResult(Result):
    slug: str
    name: str
    disconnected: bool


method("model.disconnect", params=ModelDisconnectParams, result=ModelDisconnectResult,
       doc="Remove every credential (env keys and OAuth state) for a provider.")


# ── profiles (methods_profiles) ───────────────────────────────────────────────────────────────


class ProfileSessionPreview(Result):
    """Newest human-facing session of a profile (``_latest_profile_session_rows``)."""

    id: str
    title: str = ""
    preview: str = ""
    started_at: float | int = 0
    last_active: float | int = 0
    message_count: int = 0


class ProfileWorkerSession(Result):
    """Newest kanban/tool worker row, so rosters can show a profile as working."""

    id: str
    source: str = ""
    title: str = ""
    last_active: float | int = 0


class ProfileCanonicalSession(Result):
    """The profile's "Bot Chat" registry row; ``resolved_id`` is the live compression tip."""

    id: str
    resolved_id: str
    root_title: str = ""
    title: str = ""
    preview: str = ""
    started_at: float | int = 0
    last_active: float | int = 0
    message_count: int = 0


class ProfileRow(Result):
    """One roster row; the session fields are present only with ``include_sessions``."""

    name: str
    path: str
    is_default: bool = False
    model: str | None = None
    provider: str | None = None
    description: str = ""
    display_name: str = ""
    # Rename history from ``profile.yaml``: Bot Mode re-seats a renamed member's persisted
    # handle/group membership on it, so it is consumed and must be part of the wire contract.
    previous_names: list[str] = Field(default_factory=list)
    skill_count: int = 0
    last_session: ProfileSessionPreview | None = None
    worker_session: ProfileWorkerSession | None = None
    canonical_session: ProfileCanonicalSession | None = None
    ui_meta_revisions: dict[str, int] = Field(default_factory=dict)
    ui_meta: dict[str, JsonValue] | None = None
    has_avatar: bool = False


class ProfilesListParams(ProfileParams):
    include_sessions: bool | str | None = None


class ProfilesListResult(Result):
    """``bot_mode_protocol`` tells clients this backend injects the teammate protocol itself."""

    profiles: list[ProfileRow] = Field(default_factory=list)
    bot_mode_protocol: bool = True


method("profiles.list", params=ProfilesListParams, result=ProfilesListResult,
       doc="Roster of profiles with previews so a client paints without N follow-up calls.")


class ProfilesCreateParams(ProfileParams):
    """``clone_from`` omitted = fresh profile + bundled skills; ``mirror_credentials`` defaults on
    so a headless bot has a provider."""

    name: str
    description: str | None = None
    clone_from: str | None = None
    clone_all: bool | str | None = None
    clone_channels: bool | str | None = None
    no_skills: bool | str | None = None
    no_alias: bool | str | None = None
    soul: str | None = None
    model: str | None = None
    provider: str | None = None
    share_auth: bool | str | None = None
    mirror_credentials: bool | str | None = None


class ProfileMirrored(Result):
    """What was copied from the launch profile; ``auth`` is ``"shared"`` under ``share_auth``."""

    env: bool = False
    auth: bool | Literal["shared"] = False
    model_inherited: bool = False
    voice: bool = False


class ProfilesCreateResult(Result):
    ok: bool = True
    name: str
    path: str
    soul_written: bool = False
    model_set: bool = False
    mirrored: ProfileMirrored


method("profiles.create", params=ProfilesCreateParams, result=ProfilesCreateResult,
       doc="Create a profile (ws twin of POST /api/profiles), mirroring launch credentials by default.")


class ProfileNameParams(ProfileParams):
    name: str | None = None


class CapabilityEntry(Result):
    name: str
    enabled: bool = True


class ToolsetEntry(CapabilityEntry):
    label: str = ""
    description: str = ""
    tool_count: int = 0


class McpServerEntry(CapabilityEntry):
    transport: str = "stdio"


class ProfileModelPin(Result):
    provider: str = ""
    default: str = ""


class ProfilesDescribeResult(Result):
    """Editor snapshot; ``toolsets_pinned`` says whether ``tools.enabled_toolsets`` is explicit."""

    name: str
    description: str = ""
    soul: str = ""
    model: ProfileModelPin
    skills: list[CapabilityEntry] = Field(default_factory=list)
    toolsets: list[ToolsetEntry] = Field(default_factory=list)
    toolsets_pinned: bool = False
    mcp_servers: list[McpServerEntry] = Field(default_factory=list)


method("profiles.describe", params=ProfileNameParams, result=ProfilesDescribeResult,
       doc="Everything the profile editor shows: soul, model pin, skills, toolsets, MCP servers.")


class ProfilesConfigureParams(ProfileParams):
    """Sections are independent; ``ui_meta_expected_revisions`` is a per-key compare-and-swap."""

    name: str | None = None
    ui_meta: dict[str, JsonValue] | None = None
    ui_meta_expected_revisions: dict[str, int] | None = None
    soul: str | None = None
    description: str | None = None
    model: str | None = None
    provider: str | None = None
    confirm_expensive_model: bool | str | None = None
    disabled_skills: list[str] | None = None
    enabled_toolsets: list[str] | None = None
    enabled_mcp_servers: list[str] | None = None


class UiMetaConflict(Result):
    expected: JsonValue = None
    actual: int = 0


class ProfilesConfigureApplied(Result):
    """Per-section outcome; only the sections the request carried are present."""

    ui_meta: bool | None = None
    ui_meta_revisions: dict[str, int] | None = None
    ui_meta_conflicts: dict[str, UiMetaConflict] | None = None
    soul: bool | None = None
    description: bool | None = None
    model: bool | None = None
    skills: bool | None = None
    toolsets: bool | None = None
    mcp_servers: bool | None = None


class ProfilesConfigureResult(Result):
    """``confirm_required`` mirrors ``config.set``: a guarded model pick wrote nothing yet."""

    ok: bool
    applied: ProfilesConfigureApplied
    confirm_required: bool | None = None
    confirm_message: str | None = None


method("profiles.configure", params=ProfilesConfigureParams, result=ProfilesConfigureResult,
       doc="Editor Save: apply any subset of a profile's sections and report each one.")


class ProfilesSetAssetParams(ProfileParams):
    """``data`` is a data URL or bare base64 (PNG/JPEG/WebP, sniffed); ``clear`` deletes instead."""

    name: str | None = None
    asset: str | None = None
    data: str | None = None
    clear: bool | str | None = None


class ProfilesSetAssetResult(Result):
    ok: bool = True
    asset: str
    size: int = 0
    removed: int | None = None


method("profiles.set_asset", params=ProfilesSetAssetParams, result=ProfilesSetAssetResult,
       doc="Store or clear a profile asset (avatar) atomically.")


class ProfilesGetAssetParams(ProfileParams):
    name: str | None = None
    asset: str | None = None


class ProfilesGetAssetResult(Result):
    """Absent is ``found: false``, not an error."""

    found: bool
    mime: str | None = None
    size: int | None = None
    data: str | None = None


method("profiles.get_asset", params=ProfilesGetAssetParams, result=ProfilesGetAssetResult,
       doc="A profile asset as a data URL.")


class OnboardingAnswers(Params):
    """``tui_gateway/onboarding_personalization.py`` — the facts agreed during onboarding."""

    name: str | None = None
    context: str | None = None
    theme: str | None = None
    accent: str | None = None
    layout: str | None = None
    focus: list[str] | None = None
    connectors: list[str] | None = None
    # The onboarding store may carry extra UI-only keys; the writer ignores unknown ones.
    model_config = Params.model_config | {"extra": "allow"}


class ProfilesRememberOnboardingParams(ProfileParams):
    answers: OnboardingAnswers | None = None


class ProfilesRememberOnboardingResult(Result):
    saved: bool = True
    profile: str = "default"
    target: str = "user"


method("profiles.remember_onboarding", params=ProfilesRememberOnboardingParams,
       result=ProfilesRememberOnboardingResult,
       doc="Write the onboarding facts into the default profile's user memory and confirm they landed.")


# ── vault (methods_vault) ─────────────────────────────────────────────────────────────────────


class VaultKind(WireEnum):
    login = "login"
    payment = "payment"
    address = "address"


class VaultItem(Result):
    """Metadata-only view (``VaultItemMeta.to_dict`` + ``backend``); never a secret."""

    id: str
    kind: str
    label: str
    origin: str | None = None
    created_at: str = ""
    identifier: str | None = None
    identifier_type: str | None = None
    has_otp: bool | None = None
    backend: str


class VaultListResult(Result):
    items: list[VaultItem] = Field(default_factory=list)


method("vault.list", params=ProfileParams, result=VaultListResult,
       doc="Metadata-only listing across the local vault and every unlocked password manager.")


class VaultSource(Result):
    name: str
    display_name: str
    enabled: bool
    needs_unlock: bool
    unlocked: bool
    installed: bool


class VaultSourcesResult(Result):
    sources: list[VaultSource] = Field(default_factory=list)


method("vault.sources", params=ProfileParams, result=VaultSourcesResult,
       doc="Status of every login source (local vault + detected password managers).")


class VaultSourceSetParams(ProfileParams):
    name: str | None = None
    enabled: bool | None = None


class VaultSourceSetResult(Result):
    name: str
    enabled: bool


method("vault.source.set", params=VaultSourceSetParams, result=VaultSourceSetResult,
       doc="Enable or disable an external password manager (disabling also locks it).")


class VaultUnlockParams(ProfileParams):
    """The master password is consumed by the manager CLI and never stored or logged."""

    name: str | None = None
    password: str | None = None


class VaultUnlockResult(Result):
    name: str
    unlocked: bool = True


method("vault.unlock", params=VaultUnlockParams, result=VaultUnlockResult,
       doc="Unlock a password manager for this session with its master password.")


class VaultLockParams(ProfileParams):
    name: str | None = None


class VaultLockResult(Result):
    locked: bool = True


method("vault.lock", params=VaultLockParams, result=VaultLockResult,
       doc="Forget a manager's session token (every manager when no name is given).")


class VaultAddParams(ProfileParams):
    """``secret`` goes straight into the encrypted store; the result carries only the new id."""

    kind: VaultKind | None = None
    label: str | None = None
    origin: str | None = None
    secret: dict[str, JsonValue] | None = None


class VaultAddResult(Result):
    id: str


method("vault.add", params=VaultAddParams, result=VaultAddResult,
       doc="Add a login / payment / address item to the local vault.")


class VaultRemoveParams(ProfileParams):
    id: str | None = None


class VaultRemoveResult(Result):
    removed: bool


method("vault.remove", params=VaultRemoveParams, result=VaultRemoveResult,
       doc="Remove a local vault item by id.")


# ── foreign histories (methods_session_foreign) ───────────────────────────────────────────────


class ForeignSource(WireEnum):
    claude = "claude"
    codex = "codex"


class ForeignSessionRow(Result):
    """``hermes_cli/foreign_sessions_browser.py::list_foreign_sessions`` — ``id`` is an opaque
    handle, never a path."""

    id: str
    source: ForeignSource
    label: str
    title: str = ""
    cwd: str | None = None
    mtime: float
    turn_count: int = 0
    excerpt: str = ""


class SessionForeignListParams(ProfileParams):
    source: ForeignSource | None = None
    offset: int | None = None
    limit: int | None = None


class SessionForeignListResult(Result):
    """``unreadable`` counts logs on this page that failed to parse."""

    sessions: list[ForeignSessionRow] = Field(default_factory=list)
    next_offset: int | None = None
    host: str
    unreadable: int = 0


method("session.foreign.list", params=SessionForeignListParams, result=SessionForeignListResult,
       doc="One page of Claude Code / Codex sessions found on the serving backend.")


class ForeignTurn(Result):
    role: str
    content: str


class SessionForeignIdParams(ProfileParams):
    id: str | None = None


class SessionForeignPreviewResult(Result):
    """Bounded to the last 40 turns / 8000 chars each; ``already_imported`` is the local id."""

    messages: list[ForeignTurn] = Field(default_factory=list)
    total: int = 0
    truncated: bool = False
    already_imported: str | None = None
    cwd: str | None = None


method("session.foreign.preview", params=SessionForeignIdParams, result=SessionForeignPreviewResult,
       doc="Preview a foreign session's tail before importing it.")


class SessionForeignImportResult(Result):
    session_id: str
    already_imported: bool = False


method("session.foreign.import", params=SessionForeignIdParams, result=SessionForeignImportResult,
       doc="Import a foreign session into this profile's history (idempotent per origin).")


# ── subagents (methods_subagents) ─────────────────────────────────────────────────────────────


class SubagentSnapshot(Result):
    """``methods_subagents._SUBAGENT_SNAPSHOT_FIELDS`` projection of one live child record."""

    subagent_id: str
    parent_id: str | None = None
    depth: int | None = None
    goal: str | None = None
    delegation_id: str | None = None
    model: str | None = None
    started_at: float | None = None
    status: SubagentStatus | None = None
    tool_count: int | None = None
    last_tool: str | None = None
    accepting_steer: bool | None = None


class SubagentListResult(Result):
    """``delegations`` is reserved for async delegation records and is currently always empty."""

    subagents: list[SubagentSnapshot] = Field(default_factory=list)
    delegations: list[dict[str, JsonValue]] = Field(default_factory=list)


method("subagent.list", params=SessionParams, result=SubagentListResult,
       doc="Live children owned by this session (other sessions' children never leak).")


class SubagentIdParams(SessionParams):
    subagent_id: str


class SubagentInterruptResult(Result):
    found: bool
    subagent_id: str


method("subagent.interrupt", params=SubagentIdParams, result=SubagentInterruptResult,
       doc="Hard-interrupt one owned child; ``found`` is false when it already finished.")


class SubagentTailResult(Result):
    """``available`` is false while the child has no live transcript yet (or it was cleaned up)."""

    subagent_id: str
    available: bool = False
    text: str = ""
    truncated: bool = False


method("subagent.tail", params=SubagentIdParams, result=SubagentTailResult,
       doc="Last 16KB of an owned child's live transcript.")
