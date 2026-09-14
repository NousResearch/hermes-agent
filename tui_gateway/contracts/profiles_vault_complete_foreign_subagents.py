"""Contracts for ``methods_profiles``, ``methods_vault``, ``methods_complete``,
``methods_session_foreign`` and ``methods_subagents``.

Profiles are the ws twin of the dashboard's ``/api/profiles``; the vault handlers are the
Desktop's Settings → Credential Vault door (metadata only — a secret never appears in a result);
completions feed the composer popovers; ``session.foreign.*`` browses Claude Code / Codex
histories on the serving backend; ``subagent.*`` is the session-scoped roster of live children.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from .base import Params, Result, WireEnum
from .common import SessionParams
from .registry import method


# The final enum belongs in common.py; local while common/events are migrated in parallel.
class SubagentSnapshotStatus(WireEnum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    error = "error"
    timeout = "timeout"
    interrupted = "interrupted"


class SavedKeyModelPricing(Result):
    """``hermes_cli/inventory.py::_apply_pricing`` formatted provider-model pricing."""

    input: str
    output: str
    cache: str | None = None
    free: bool
    discount_percent: int | None = None
    was_input: str | None = None
    was_output: str | None = None


class SavedKeyModelCapabilities(Result):
    """``hermes_cli/inventory.py::_apply_capabilities`` per-model capability row."""

    fast: bool
    reasoning: bool
    can_disable_reasoning: bool | None = None


class SavedKeyProvider(Result):
    """``methods_complete.py`` snapshot after saving a provider API key."""

    slug: str
    name: str
    models: list[str]
    total_models: int | None = None
    is_current: bool | None = None
    is_user_defined: bool | None = None
    source: str | None = None
    aliases: list[str] | None = None
    api_url: str | None = None
    auth_type: str | None = None
    authenticated: bool | None = None
    key_env: str | None = None
    warning: str | None = None
    featured_models: list[str] | None = None
    capabilities: dict[str, SavedKeyModelCapabilities] | None = None
    pricing: dict[str, SavedKeyModelPricing] | None = None
    pricing_pending: bool | None = None
    free_tier: bool | None = None
    free_tier_pending: bool | None = None
    free_tier_row: bool | None = None
    unavailable_models: list[str] | None = None


# ── completions / paste / model keys (methods_complete) ───────────────────────────────────────


class CompletionItem(Result):
    """One popover row; ``kind`` rides only on slash completions (command vs skill)."""

    text: str
    display: str
    meta: str
    kind: str | None = None


class CompletionItemsResult(Result):
    items: list[CompletionItem]


class CompletePathParams(Params):
    """``word`` is the token under the cursor (``@`` prefix = context reference); ``cwd`` /
    ``session_id`` pick the directory the listing resolves against."""

    word: str = ""
    cwd: str | None = None
    session_id: str | None = None


method("complete.path", params=CompletePathParams, result=CompletionItemsResult,
       doc="Path / @-reference completions for the composer (files, folders, profiles, plugin providers).")


class CompleteSlashParams(Params):
    text: str = ""


class CompleteSlashResult(Result):
    """``replace_from`` is the column the accepted item replaces from."""

    items: list[CompletionItem]
    replace_from: int | None = None


method("complete.slash", params=CompleteSlashParams, result=CompleteSlashResult,
       doc="Ranked slash-command / skill completions for a ``/`` token.")


class PasteCollapseParams(Params):
    text: str = ""


class PasteCollapseResult(Result):
    placeholder: str
    path: str
    lines: int


method("paste.collapse", params=PasteCollapseParams, result=PasteCollapseResult,
       doc="Spill a large paste to a file and hand back the inline placeholder.")


class ModelSaveKeyParams(Params):
    slug: str
    api_key: str
    session_id: str = ""


class ModelSaveKeyResult(Result):
    provider: SavedKeyProvider


method("model.save_key", params=ModelSaveKeyParams, result=ModelSaveKeyResult,
       doc="Save an API key for a provider and return its refreshed inventory row.")


class ModelDisconnectParams(Params):
    slug: str


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
    title: str
    preview: str
    started_at: float
    last_active: float
    message_count: int


class ProfileWorkerSession(Result):
    """Newest kanban/tool worker row, so rosters can show a profile as working."""

    id: str
    source: str
    title: str
    last_active: float


class ProfileCanonicalSession(Result):
    """The profile's "Bot Chat" registry row; ``resolved_id`` is the live compression tip."""

    id: str
    resolved_id: str
    root_title: str
    title: str
    preview: str
    started_at: float
    last_active: float
    message_count: int


class ProfileRow(Result):
    """One roster row; the session fields are present only with ``include_sessions``."""

    name: str
    path: str
    is_default: bool
    model: str | None
    provider: str | None
    description: str
    display_name: str
    skill_count: int
    last_session: ProfileSessionPreview | None = None
    worker_session: ProfileWorkerSession | None = None
    canonical_session: ProfileCanonicalSession | None = None
    ui_meta_revisions: dict[str, int]
    # Client-owned opaque blob from methods_profiles._profile_ui_meta_fields.
    ui_meta: dict[str, JsonValue] | None = None
    has_avatar: bool


class ProfilesListParams(Params):
    include_sessions: bool | str = True


class ProfilesListResult(Result):
    """``bot_mode_protocol`` tells clients this backend injects the teammate protocol itself."""

    profiles: list[ProfileRow]
    bot_mode_protocol: bool


method("profiles.list", params=ProfilesListParams, result=ProfilesListResult,
       doc="Roster of profiles with previews so a client paints without N follow-up calls.")


class ProfilesCreateParams(Params):
    """``clone_from`` omitted = fresh profile + bundled skills; ``mirror_credentials`` defaults on
    so a headless bot has a provider."""

    name: str
    description: str | None = None
    clone_from: str | None = None
    clone_all: bool | str = False
    clone_channels: bool | str = False
    no_skills: bool | str = False
    no_alias: bool | str = False
    soul: str | None = None
    model: str | None = None
    provider: str | None = None
    share_auth: bool | str = False
    mirror_credentials: bool | str = True


class ProfileMirrored(Result):
    """What was copied from the launch profile; ``auth`` is ``"shared"`` under ``share_auth``."""

    env: bool
    auth: bool | Literal["shared"]
    model_inherited: bool
    voice: bool


class ProfilesCreateResult(Result):
    ok: bool
    name: str
    path: str
    soul_written: bool
    model_set: bool
    mirrored: ProfileMirrored


method("profiles.create", params=ProfilesCreateParams, result=ProfilesCreateResult,
       doc="Create a profile (ws twin of POST /api/profiles), mirroring launch credentials by default.")


class ProfileNameParams(Params):
    name: str


class CapabilityEntry(Result):
    name: str
    enabled: bool


class ToolsetEntry(CapabilityEntry):
    label: str
    description: str
    tool_count: int


class McpServerEntry(CapabilityEntry):
    transport: str


class ProfileModelPin(Result):
    provider: str
    default: str


class ProfilesDescribeResult(Result):
    """Editor snapshot; ``toolsets_pinned`` says whether ``tools.enabled_toolsets`` is explicit."""

    name: str
    description: str
    soul: str
    model: ProfileModelPin
    skills: list[CapabilityEntry]
    toolsets: list[ToolsetEntry]
    toolsets_pinned: bool
    mcp_servers: list[McpServerEntry]


method("profiles.describe", params=ProfileNameParams, result=ProfilesDescribeResult,
       doc="Everything the profile editor shows: soul, model pin, skills, toolsets, MCP servers.")


class ProfilesConfigureParams(Params):
    """Sections are independent; ``ui_meta_expected_revisions`` is a per-key compare-and-swap."""

    name: str | None = None
    # Client-owned opaque blob merged by methods_profiles._configure_ui_meta.
    ui_meta: dict[str, JsonValue] | None = None
    ui_meta_expected_revisions: dict[str, int] | None = None
    soul: str | None = None
    description: str | None = None
    model: str | None = None
    provider: str | None = None
    confirm_expensive_model: bool | str = False
    disabled_skills: list[str] | None = None
    enabled_toolsets: list[str] | None = None
    enabled_mcp_servers: list[str] | None = None


class UiMetaConflict(Result):
    # A missing requested revision is reported as null by _configure_ui_meta.
    expected: int | None
    actual: int


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


class ProfilesSetAssetParams(Params):
    """``data`` is a data URL or bare base64 (PNG/JPEG/WebP, sniffed); ``clear`` deletes instead."""

    name: str
    asset: str = "avatar"
    data: str | None = None
    clear: bool | str = False


class ProfilesGetAssetParams(Params):
    name: str
    asset: str = "avatar"


class ProfilesGetAssetResult(Result):
    """Absent is ``found: false``, not an error."""

    found: bool
    mime: str | None = None
    size: int | None = None
    data: str | None = None


method("profiles.get_asset", params=ProfilesGetAssetParams, result=ProfilesGetAssetResult,
       doc="A profile asset as a data URL.")


class ProfilesSetAssetResult(Result):
    ok: bool
    asset: str
    size: int
    removed: int | None = None


method("profiles.set_asset", params=ProfilesSetAssetParams, result=ProfilesSetAssetResult,
       doc="Store or clear a profile asset (avatar) atomically.")


class OnboardingAnswers(Params):
    """``tui_gateway/onboarding_personalization.py`` — the facts agreed during onboarding."""

    name: str | None = None
    context: str | None = None
    theme: str | None = None
    accent: str | None = None
    layout: str | None = None
    focus: list[str] = Field(default_factory=list)
    connectors: list[str] = Field(default_factory=list)


class ProfilesRememberOnboardingParams(Params):
    answers: OnboardingAnswers


class ProfilesRememberOnboardingResult(Result):
    saved: bool
    profile: str
    target: str


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
    origin: str | None
    created_at: str
    identifier: str | None = None
    identifier_type: str | None = None
    has_otp: bool | None = None
    backend: str


class VaultListResult(Result):
    items: list[VaultItem]


method("vault.list", params=Params, result=VaultListResult,
       doc="Metadata-only listing across the local vault and every unlocked password manager.")


class VaultSource(Result):
    name: str
    display_name: str
    enabled: bool
    needs_unlock: bool
    unlocked: bool
    installed: bool


class VaultSourcesResult(Result):
    sources: list[VaultSource]


method("vault.sources", params=Params, result=VaultSourcesResult,
       doc="Status of every login source (local vault + detected password managers).")


class VaultSourceSetParams(Params):
    name: str
    enabled: bool


class VaultSourceSetResult(Result):
    name: str
    enabled: bool


method("vault.source.set", params=VaultSourceSetParams, result=VaultSourceSetResult,
       doc="Enable or disable an external password manager (disabling also locks it).")


class VaultUnlockParams(Params):
    """The master password is consumed by the manager CLI and never stored or logged."""

    name: str
    password: str


class VaultUnlockResult(Result):
    name: str
    unlocked: bool


method("vault.unlock", params=VaultUnlockParams, result=VaultUnlockResult,
       doc="Unlock a password manager for this session with its master password.")


class VaultLockParams(Params):
    name: str | None = None


class VaultLockResult(Result):
    locked: bool


method("vault.lock", params=VaultLockParams, result=VaultLockResult,
       doc="Forget a manager's session token (every manager when no name is given).")


class VaultAddParams(Params):
    """``secret`` goes straight into the encrypted store; the result carries only the new id."""

    kind: VaultKind
    label: str
    origin: str | None = None
    secret: dict[str, str]


class VaultAddResult(Result):
    id: str


method("vault.add", params=VaultAddParams, result=VaultAddResult,
       doc="Add a login / payment / address item to the local vault.")


class VaultRemoveParams(Params):
    id: str


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
    title: str
    cwd: str | None
    mtime: float
    turn_count: int
    excerpt: str


class SessionForeignListParams(Params):
    source: ForeignSource | None = None
    offset: int = 0
    limit: int = 25


class SessionForeignListResult(Result):
    """``unreadable`` counts logs on this page that failed to parse."""

    sessions: list[ForeignSessionRow]
    next_offset: int | None
    host: str
    unreadable: int


method("session.foreign.list", params=SessionForeignListParams, result=SessionForeignListResult,
       doc="One page of Claude Code / Codex sessions found on the serving backend.")


class ForeignTurn(Result):
    role: str
    content: str


class SessionForeignIdParams(Params):
    id: str


class SessionForeignPreviewResult(Result):
    """Bounded to the last 40 turns / 8000 chars each; ``already_imported`` is the local id."""

    messages: list[ForeignTurn]
    total: int
    truncated: bool
    already_imported: str | None
    cwd: str | None


method("session.foreign.preview", params=SessionForeignIdParams, result=SessionForeignPreviewResult,
       doc="Preview a foreign session's tail before importing it.")


class SessionForeignImportResult(Result):
    session_id: str
    already_imported: bool


method("session.foreign.import", params=SessionForeignIdParams, result=SessionForeignImportResult,
       doc="Import a foreign session into this profile's history (idempotent per origin).")


# ── subagents (methods_subagents) ─────────────────────────────────────────────────────────────


class SubagentSnapshot(Result):
    """``methods_subagents._SUBAGENT_SNAPSHOT_FIELDS`` projection of one live child record."""

    subagent_id: str
    parent_id: str | None
    depth: int
    goal: str
    delegation_id: str | None
    model: str | None
    started_at: float
    status: SubagentSnapshotStatus
    tool_count: int
    last_tool: str | None
    accepting_steer: bool


class SubagentDelegationSnapshot(Result):
    """Reserved async-delegation row; methods_subagents currently emits no rows."""

    delegation_id: str
    goal: str
    role: str
    model: str | None
    status: str
    dispatched_at: float
    completed_at: float | None
    subagent_ids: list[str]


class SubagentListResult(Result):
    """``delegations`` is reserved for async delegation records and is currently always empty."""

    subagents: list[SubagentSnapshot]
    delegations: list[SubagentDelegationSnapshot]


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
    available: bool
    text: str
    truncated: bool


method("subagent.tail", params=SubagentIdParams, result=SubagentTailResult,
       doc="Last 16KB of an owned child's live transcript.")
