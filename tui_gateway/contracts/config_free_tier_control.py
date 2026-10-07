"""Contracts: config, setup readiness, free tier, model inventory, connectors, diagnostics,
image generation and structured session control.

Handlers: ``tui_gateway/methods_config.py`` (``config.get``, ``setup.*``, ``diagnostics.share_nous``),
``methods_config_set.py`` (``config.set``), ``methods_free_tier.py``, ``methods_shared_metrics.py``,
``methods_complete.py``
(``model.options``), ``methods_connectors.py``, ``methods_images.py``, ``methods_session_control.py``
and ``methods_session.py`` (``verification.status``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import JsonValue, Params, Result, WireEnum
from .common import OkResult, OpenModel, ProfileParams, SessionLiveInfo
from .registry import method

# ── config.get ────────────────────────────────────────────────────────────────────────────────


class ConfigGetParams(ProfileParams):
    """``key`` selects one getter from ``_CONFIG_GETTERS``; ``cwd`` feeds the ``project`` getter,
    ``session_id`` lets ``reasoning`` / ``fast`` answer with the session's live pin."""

    key: str
    cwd: str | None = None
    session_id: str | None = None


class ConfigProviderRef(OpenModel):
    """``hermes_cli/models.py::list_available_providers`` row."""

    id: str
    label: str
    aliases: list[str] = Field(default_factory=list)
    authenticated: bool = False


class ConfigGetResult(Result):
    """Union of every getter's payload: ``value`` for the simple words, ``config`` for ``full``,
    ``mtime`` / ``mcp_rev`` for the poller, ``model`` / ``provider`` / ``providers`` for ``provider``,
    ``home`` / ``display`` for ``profile``, ``cwd`` / ``branch`` for ``project``, ``prompt``."""

    value: str | None = None
    display: str | None = None
    tool_progress: str | None = None
    model: str | None = None
    provider: str | None = None
    providers: list[ConfigProviderRef] | None = None
    home: str | None = None
    cwd: str | None = None
    branch: str | None = None
    config: dict[str, JsonValue] | None = None
    prompt: str | None = None
    mtime: float | None = None
    mcp_rev: str | None = None


method("config.get", params=ConfigGetParams, result=ConfigGetResult,
       doc="Read one normalised config value (or the whole effective config) the way the UIs render it.")


# ── config.set ────────────────────────────────────────────────────────────────────────────────


class ConfigSetScope(WireEnum):
    session = "session"
    global_ = "global"
    once = "once"


class ConfigSetParams(ProfileParams):
    """``key`` picks the setter (``_CONFIG_SETTERS``, ``details_mode.<section>``, display toggles);
    ``value`` is the raw word/string the setter normalises (falsy non-strings are reported back in
    the error). ``scope`` applies to ``yolo`` / ``reasoning``; ``confirm_expensive_model`` to ``model``."""

    key: str
    value: JsonValue = ""
    session_id: str | None = None
    scope: str | None = None
    confirm_expensive_model: bool = False


class ConfigSetResult(Result):
    """``{key, value}`` plus the setter's extras: model switches add ``warning`` /
    ``confirm_required`` / ``confirm_message`` / ``scope`` / ``deferred``; ``focus`` adds
    ``tool_progress``; ``cwd`` adds ``cwd`` / ``branch``; ``personality`` adds ``history_reset`` /
    ``info``; ``yolo`` reports its ``scope``. ``value`` is a bool only for the display toggles."""

    key: str
    value: str | bool | None = None
    warning: str | None = None
    confirm_required: bool | None = None
    confirm_message: str | None = None
    scope: str | None = None
    deferred: bool | None = None
    tool_progress: str | None = None
    cwd: str | None = None
    branch: str | None = None
    history_reset: bool | None = None
    info: SessionLiveInfo | None = None


method("config.set", params=ConfigSetParams, result=ConfigSetResult,
       doc="Change one config key (persisted or session-scoped) and read back the normalised value.")


# ── setup readiness ───────────────────────────────────────────────────────────────────────────


class SetupStatusResult(Result):
    """``provider_configured`` is the loose answer; the boot record's fields (``ready``,
    ``free_tier_account``, ``free_tier_route``, ``other_providers``, ``inference_provider``) ride along
    on the launch profile. An unknown ``profile`` answers ``ok=False`` + ``error``."""

    provider_configured: bool | None = None
    ready: bool | None = None
    free_tier_account: bool | None = None
    free_tier_route: bool | None = None
    other_providers: bool | None = None
    inference_provider: str | None = None
    profile: str | None = None
    ok: bool | None = None
    error: str | None = None
    error_code: str | None = None
    retryable: bool | None = None
    retry_after: int | None = None


method("setup.status", params=ProfileParams, result=SetupStatusResult,
       doc="Loose provider check: is ANY provider auth state discoverable for the (launch or named) profile.")


class SetupRuntimeCheckParams(ProfileParams):
    provider: str | None = None


class SetupRuntimeCheckResult(Result):
    """``ok=False`` + ``error`` when the resolved model can't be served; ``free_tier_route`` says the
    selected route is the welcome host."""

    ok: bool
    provider: str | None = None
    model: str | None = None
    source: str | None = None
    error: str | None = None
    free_tier_route: bool | None = None
    profile: str | None = None


method("setup.runtime_check", params=SetupRuntimeCheckParams, result=SetupRuntimeCheckResult,
       doc="Strict provider check through the same runtime resolution the agent uses on session creation.")


# ── diagnostics.share_nous ────────────────────────────────────────────────────────────────────


class DiagnosticsShareNousParams(Params):
    error_context: str | None = None
    extra_files: dict[str, str] | None = None
    log_lines: int | None = None


class DiagnosticsShareNousResult(Result):
    """Structured envelope: ``ok=False`` + ``error`` renders inline instead of failing the RPC."""

    ok: bool
    view_url: str | None = None
    upload_id: str | None = None
    expires_at: str | None = None
    error: str | None = None


method("diagnostics.share_nous", params=DiagnosticsShareNousParams, result=DiagnosticsShareNousResult,
       doc="Upload a force-redacted debug bundle to Nous-internal diagnostics storage.")


# ── free tier ─────────────────────────────────────────────────────────────────────────────────


class FreeTierChallengePayload(OpenModel):
    """``hermes_cli/anon_challenge.py::BrowserChallenge.as_payload``: the ``free_tier.challenge``
    event, and ``free_tier.status``'s ``challenge`` field for a client that connected after it."""

    type: Literal["browser"]
    url: str
    # False = the account service is measuring, not enforcing: run it hidden, never reveal it.
    required: bool
    expires_in: int
    message: str
    attempt: int = 0


class FreeTierStatusResult(Result):
    """``available`` = an identity exists AND the tier is on; whether inference runs on it is
    ``setup.runtime_check.free_tier_route``'s question."""

    has_guest: bool
    enabled: bool
    available: bool
    notice_pending: bool
    model: str
    label: str
    error: str | None = None
    error_code: str | None = None
    retryable: bool | None = None
    retry_after: int | None = None
    # A browser challenge the account service is waiting on (``hermes_cli/anon_challenge.py``).
    challenge: FreeTierChallengePayload | None = None


method("free_tier.status", params=ProfileParams, result=FreeTierStatusResult,
       doc="Pure read of the focused profile's free-tier identity state (no network, no side effects).")


class FreeTierChallengeResultParams(ProfileParams):
    url: str
    attempt: int = 0
    outcome: Literal["done", "failed", "closed", "timeout", "refused", "error", "unsupported"]


class FreeTierChallengeResult(Result):
    accepted: bool


method("free_tier.challenge_result", params=FreeTierChallengeResultParams, result=FreeTierChallengeResult,
       doc="Report a browser window outcome for the matching pending attempt; mint remains authoritative.")


class FreeTierProvisionResult(Result):
    has_guest: bool
    enabled: bool
    error: str | None = None
    error_code: str | None = None
    retryable: bool | None = None
    retry_after: int | None = None


method("free_tier.provision", params=ProfileParams, result=FreeTierProvisionResult,
       doc="Explicit retry of the free-tier identity mint when the boot bootstrap could not create it.")


class FreeTierAckNoticeResult(Result):
    acked: bool


method("free_tier.ack_notice", params=ProfileParams, result=FreeTierAckNoticeResult,
       doc="Mark the one-time availability notice as shown on the free-tier identity.")


# ── shared metrics consent ────────────────────────────────────────────────────────────────────


class SharedMetricsConsentResult(Result):
    """The focused profile's ``telemetry.shared_metrics`` opt-ins. ``send`` is never true while
    ``enabled`` is false; ``decided`` = either key is written in config.yaml (the shipped defaults
    are not an answer) and it is not a ``reask``: an "off" from before the type-ahead fix, offered
    once more with the reason."""

    enabled: bool
    send: bool
    decided: bool
    reask: bool = False


method("shared_metrics.status", params=ProfileParams, result=SharedMetricsConsentResult,
       doc="Pure read of the focused profile's shared-metrics opt-ins (collection, upload, answered).")


class SharedMetricsSetParams(ProfileParams):
    """``send`` is ignored unless ``enabled``; ``first_run`` marks the Desktop first-run answer."""

    enabled: bool
    send: bool = False
    first_run: bool = False


method("shared_metrics.set", params=SharedMetricsSetParams, result=SharedMetricsConsentResult,
       doc="Write both shared-metrics opt-ins at once (send requires collection) and reconcile consent windows.")


class SharedMetricsSlashCommandParams(ProfileParams):
    """``command`` is the raw typed name (no leading ``/``, no args); the backend canonicalizes it
    against the published registry. ``session_id`` scopes the count to that session's profile."""

    command: str
    session_id: str | None = None


class SharedMetricsSlashCommandResult(Result):
    ok: bool


method("shared_metrics.slash_command", params=SharedMetricsSlashCommandParams,
       result=SharedMetricsSlashCommandResult,
       doc="Count one user-typed slash command (fire-and-forget; a no-op unless shared metrics are on).")


class SharedMetricsStartupLatencyParams(ProfileParams):
    """``elapsed_ms`` = the client's own launch (TUI process start / Desktop app start) to ready
    (TUI gateway ready / Desktop backend attached), measured once per launch by the client. The
    client names its surface because a Desktop may attach to a URL/cloud backend where
    ``HERMES_DESKTOP`` is unset; without it the backend falls back to its own client detection.
    ``launch_id`` is an opaque per-launch token the backend latches on (never recorded), so a
    reconnect re-sending the same launch counts once while a new launch counts again."""

    elapsed_ms: float
    surface: Literal["desktop_attach", "tui"] | None = None
    launch_id: str | None = None


class SharedMetricsStartupLatencyResult(Result):
    ok: bool


method("shared_metrics.startup_latency", params=SharedMetricsStartupLatencyParams,
       result=SharedMetricsStartupLatencyResult,
       doc="Record one client launch-to-ready latency (fire-and-forget; a no-op unless shared metrics are on).")


# ---- v4 reliability ----
class SharedMetricsUpdateRunParams(ProfileParams):
    """One Desktop PACKAGED self-update (electron-updater / App Installer / Store). Source-checkout
    hand-offs run ``hermes update`` and are counted from its receipt, never here. Raw words; the
    backend buckets them: ``outcome`` success|failed|noop|refused, ``failed_stage``
    download|verify|apply|restart, ``mechanism`` the updater strategy kind, ``duration_ms`` wall
    time, ``from_commit_date`` the updated-from build's commit time (epoch seconds) when known."""

    outcome: str
    failed_stage: str | None = None
    mechanism: str | None = None
    duration_ms: float | None = None
    from_commit_date: float | None = None


class SharedMetricsUpdateRunResult(Result):
    ok: bool


method("shared_metrics.update_run", params=SharedMetricsUpdateRunParams, result=SharedMetricsUpdateRunResult,
       doc="Count one Desktop packaged self-update outcome (fire-and-forget; a no-op unless shared metrics are on).")
# ---- end v4 reliability ----


# ---- v5 desktop ----
class SharedMetricsDesktopFeatureUseParams(ProfileParams):
    """``area`` is a Desktop surface id (``command_palette``, ``terminal_pane``, ``settings_<view>`` …);
    the backend collapses anything outside its closed set to ``other``."""

    area: str


method("shared_metrics.desktop_feature_use", params=SharedMetricsDesktopFeatureUseParams,
       result=OkResult,
       doc="Count one Desktop area used today (fire-and-forget; once per area per UTC day; a no-op unless on).")


class SharedMetricsDesktopFrictionParams(ProfileParams):
    """``kind`` notice_dismissed|error_toast|renderer_crash|backend_disconnect|slow_frame; ``detail`` a
    closed code-defined word for that kind (notice id, error category, crash reason, drop reason, frame
    duration bucket), never message text."""

    kind: str
    detail: str


method("shared_metrics.desktop_friction", params=SharedMetricsDesktopFrictionParams,
       result=OkResult,
       doc="Count one Desktop friction event (fire-and-forget; capped per day; a no-op unless on).")


class SharedMetricsDesktopOnboardingParams(ProfileParams):
    """``step`` a Desktop first-run step id; ``event`` reached|completed|abandoned."""

    step: str
    event: str


method("shared_metrics.desktop_onboarding", params=SharedMetricsDesktopOnboardingParams,
       result=OkResult,
       doc="Count one Desktop first-run step transition (fire-and-forget; once per step+event; a no-op unless on).")


class SharedMetricsDesktopDislikeParams(ProfileParams):
    """``signal`` quick_close|cancelled|setting_off_default|rage_click|undo|feature_disabled; ``target`` a
    closed code-defined id for that signal (area, flow, action, undo path, feature toggle); ``setting`` a
    config key for setting_off_default only (the value is never sent — the backend compares it to the
    default)."""

    signal: str
    target: str = ""
    setting: str | None = None


method("shared_metrics.desktop_dislike", params=SharedMetricsDesktopDislikeParams,
       result=OkResult,
       doc="Count one Desktop dislike signal (fire-and-forget; capped per signal per day; a no-op unless on).")


class SharedMetricsDesktopModeDay(Params):
    mode: Literal["bots", "sessions"]
    active_ms: float = 0
    messages_sent: int = 0


class SharedMetricsDesktopActionDay(Params):
    action: str
    via: Literal["click", "menu", "palette", "shortcut"]
    count: int


class SharedMetricsDesktopDailyParams(ProfileParams):
    """One finished UTC day of Desktop use, aggregated on the client. ``day`` (YYYY-MM-DD) only latches
    a resend and is never recorded; the raw counts are bucketed by the backend."""

    day: str
    bot_count: int = 0
    modes: list[SharedMetricsDesktopModeDay] = Field(default_factory=list)
    actions: list[SharedMetricsDesktopActionDay] = Field(default_factory=list)


class SharedMetricsDesktopDailyResult(Result):
    recorded: bool


method("shared_metrics.desktop_daily", params=SharedMetricsDesktopDailyParams,
       result=SharedMetricsDesktopDailyResult,
       doc="Record one finished Desktop day (mode use + button presses); recorded=false keeps it for a retry.")
# ---- end v5 desktop ----


# ── model.options ─────────────────────────────────────────────────────────────────────────────


class ModelOptionsParams(ProfileParams):
    session_id: str | None = None
    explicit_only: bool = False
    include_unconfigured: bool = False
    refresh: bool = False


# TODO(common): ModelPricing / ModelCapabilities / ModelOptionProvider are also the row shape of
# ``model.save_key``'s ``provider`` — the parent consolidates into contracts/common.py.
class ModelPricing(Result):
    """``hermes_cli/inventory.py::_apply_pricing`` — formatted $/Mtok strings (``""`` unknown,
    ``"free"``); the sale fields are Nous Portal-only."""

    input: str
    output: str
    cache: str | None = None
    free: bool
    discount_percent: int | None = None
    was_input: str | None = None
    was_output: str | None = None


class ModelCapabilities(Result):
    """``hermes_cli/inventory.py::_apply_capabilities``."""

    fast: bool
    ultrafast: bool = False
    reasoning: bool
    can_disable_reasoning: bool | None = None


class ProviderLimit(Result):
    """``hermes_cli/inventory.py::_apply_limits`` — ``account``: the whole login is rate-limited until
    ``resets_at`` (ISO, absent when unknown); ``models``: only these models are, each until its time."""

    scope: Literal["account", "models"]
    resets_at: str | None = None
    models: dict[str, str] | None = None


class ProviderUsageWindow(Result):
    """One subscription usage window (``agent/account_usage.py::AccountUsageWindow``): e.g. the 5-hour
    session or the weekly cap, with how much of it is spent and when it rolls over (ISO).

    ``scope``: ``account`` — exhausting the window exhausts the whole login (Codex session/weekly,
    so a limited account's resets_at must wait for it); ``model`` — the window caps only one model
    family (Anthropic Opus/Sonnet weekly) and can never imply the account itself is out of quota."""

    label: str
    used_percent: float
    resets_at: str | None = None
    scope: Literal["account", "model"] = "account"


class ProviderUsageAccount(Result):
    """One account of a provider's credential pool (``hermes_cli/inventory.py::_pool_usage_accounts``).
    ``id`` is a stable non-secret account identity (never a key or URL); ``label`` may be empty (UI
    falls back to a localized "Account N"). ``windows`` is empty while the account's usage is not
    yet known — state carries the meaning, never a fabricated gauge.

    ``state``: ``ready`` — live quota below the cap (numeric windows present); ``limited`` — a live
    credential-wide cooldown or exhausted account-scoped quota windows; ``unknown`` — no live
    numeric windows (failed/empty fetch, stale snapshot, provider without a usage API);
    ``unavailable`` — DEAD auth row (kept visible, never a quota row).

    ``resets_at``: for a limited account, the LATEST of its exhausted account-scoped windows (or a
    live cooldown when later); ``None`` when unknown (the frontend renders its own advisory, e.g.
    the earliest limited sibling)."""

    id: str
    label: str = ""
    windows: list[ProviderUsageWindow] = Field(default_factory=list)
    state: Literal["ready", "limited", "unknown", "unavailable"]
    resets_at: str | None = None


class ProviderUsage(Result):
    """``hermes_cli/inventory.py::_apply_usage`` — the provider's subscription usage, from cache.

    Multi-entry credential pools carry ``accounts`` (one row per account; the legacy ``windows``
    stays EMPTY there — a provider-wide percentage across different logins would be fabricated).
    Single-account providers keep the legacy ``windows`` gauge."""

    windows: list[ProviderUsageWindow] = Field(default_factory=list)
    accounts: list[ProviderUsageAccount] | None = None


class ModelOptionProvider(OpenModel):
    """One ``hermes_cli/inventory.py::build_models_payload`` provider row (the union of every field
    the builder sets; ``pricing_pending`` / ``free_tier_pending`` mark the cached-only path)."""

    slug: str
    name: str
    models: list[str] = Field(default_factory=list)
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
    capabilities: dict[str, ModelCapabilities] | None = None
    pricing: dict[str, ModelPricing] | None = None
    pricing_pending: bool | None = None
    free_tier: bool | None = None
    free_tier_pending: bool | None = None
    free_tier_row: bool | None = None
    unavailable_models: list[str] | None = None
    limit: ProviderLimit | None = None
    usage: ProviderUsage | None = None


class ModelOptionsResult(Result):
    providers: list[ModelOptionProvider]
    model: str = ""
    provider: str = ""


method("model.options", params=ModelOptionsParams, result=ModelOptionsResult,
       doc="Provider/model inventory for the picker, layered over the session's live provider when given.")


# ── image.generate ────────────────────────────────────────────────────────────────────────────


class ImageGenerateParams(Params):
    prompt: str | None = None
    aspect_ratio: str | None = None
    probe: JsonValue | None = None  # truthy word/flag: availability check only
    max_bytes: int | None = None


class ImageGenerateResult(Result):
    """``probe`` answers ``{available}`` alone; ``image_data`` (data URL) is omitted when the
    download failed or exceeded ``max_bytes`` so callers fall back to ``image``."""

    available: bool
    success: bool | None = None
    image: str | None = None
    image_data: str | None = None
    error: str | None = None


method("image.generate", params=ImageGenerateParams, result=ImageGenerateResult,
       doc="Generate an image through the tool's provider dispatcher and hand the renderer a data URL.")


# ── session.control ───────────────────────────────────────────────────────────────────────────


class GoalContractSnapshot(Result):
    """``hermes_cli/goals.py::GoalContract.to_dict``."""

    outcome: str = ""
    verification: str = ""
    constraints: str = ""
    boundaries: str = ""
    stop_when: str = ""


class GoalGateSnapshot(Result):
    command: str
    timeout_seconds: int
    max_retries: int
    attempts: int
    last_exit_code: int | None = None


class WaitBarrierUntil(Result):
    type: Literal["until"]
    until_at: float
    reason: str = ""


class WaitBarrierTarget(Result):
    type: Literal["session", "pid"]
    target: str | int
    reason: str = ""


class GoalSnapshot(Result):
    """``methods_session_control.py::_safe_goal_snapshot`` — the frontend-safe GoalState subset."""

    title: str
    status: str
    turns_used: int
    max_turns: int
    contract: GoalContractSnapshot
    subgoals: list[str]
    gates: list[GoalGateSnapshot]
    created_at: float | None = None
    updated_at: float | None = None
    paused_reason: str | None = None
    last_verdict: str | None = None
    last_reason: str | None = None
    wait_barrier: WaitBarrierUntil | WaitBarrierTarget | None = Field(default=None, discriminator="type")


class LoopSnapshot(Result):
    """``_safe_loop_snapshot`` — persisted LoopState fields, never its route."""

    prompt: str
    status: str
    mode: str
    interval_seconds: float
    current_delay: float
    times: int
    until: str
    max_ticks: int
    ticks_fired: int
    created_at: float
    last_fired_at: float
    next_due_at: float
    awaiting_response: bool
    deferred_by_goal: bool
    paused_reason: str | None = None
    last_stop_reason: str | None = None


class HeartbeatSnapshot(Result):
    prompt: str
    status: str
    interval_seconds: int
    created_at: float
    last_fired_at: float
    fire_count: int


class SessionControlSnapshot(Result):
    """``_snapshot_control`` — ``revision`` is a hash of the visible state (``""`` when empty);
    ``updated_at`` is the newest persisted timestamp (``0`` when none)."""

    goal: GoalSnapshot | None
    loop: LoopSnapshot | None
    heartbeat: HeartbeatSnapshot | None
    revision: str
    updated_at: float


class SessionControlReadParams(ProfileParams):
    session_id: str


class SessionControlReadResult(Result):
    control: SessionControlSnapshot


method("session.control.read", params=SessionControlReadParams, result=SessionControlReadResult,
       doc="Stable, allowlisted snapshot of one live session's goal / loop / heartbeat state.")


class SessionControlAction(WireEnum):
    goal_pause = "goal.pause"
    goal_resume = "goal.resume"
    goal_clear = "goal.clear"
    goal_unwait = "goal.unwait"
    loop_pause = "loop.pause"
    loop_resume = "loop.resume"
    loop_stop = "loop.stop"
    subgoal_add = "subgoal.add"
    subgoal_remove = "subgoal.remove"
    subgoal_clear = "subgoal.clear"
    heartbeat_pause = "heartbeat.pause"
    heartbeat_resume = "heartbeat.resume"
    heartbeat_clear = "heartbeat.clear"


class SessionControlArgs(Params):
    """``subgoal.add`` reads ``text``; ``subgoal.remove`` reads the 1-based ``index``."""

    text: str | None = None
    index: int | None = None


class SessionControlParams(ProfileParams):
    """``action`` is validated by the handler (unknown / gate actions answer ``4004``), so it stays a
    string on the wire; ``SessionControlAction`` lists the accepted set."""

    session_id: str
    action: str
    args: SessionControlArgs | None = None


class SessionControlDispatch(Result):
    """``_dispatch_envelope`` — the command result's user-visible envelope, every key always present."""

    type: str | None
    output: str | None
    notice: str | None
    message: str | None
    display: str | None


class SessionControlResult(Result):
    control: SessionControlSnapshot
    dispatch: SessionControlDispatch


method("session.control", params=SessionControlParams, result=SessionControlResult,
       doc="Run one allowlisted goal / loop / subgoal / heartbeat action and return the exact resulting snapshot.")


# ── verification.status ───────────────────────────────────────────────────────────────────────


class VerificationStatusParams(ProfileParams):
    session_id: str | None = None
    session_key: str | None = None
    cwd: str | None = None


class VerificationEvidenceRow(OpenModel):
    """One ``verification_events`` row (``agent/verification_evidence.py``)."""

    id: int | None = None
    created_at: str | None = None
    session_id: str | None = None
    cwd: str | None = None
    root: str | None = None
    command: str | None = None
    canonical_command: str | None = None
    kind: str | None = None
    scope: str | None = None
    status: str | None = None
    exit_code: int | None = None
    output_summary: str | None = None


class VerificationStatusInfo(Result):
    """``verification_status()``: ``disabled`` / ``not_applicable`` / ``unverified`` / ``stale`` or the
    latest event's own status; ``root`` and friends only once a workspace was identified."""

    status: str
    evidence: VerificationEvidenceRow | None = None
    root: str | None = None
    session_id: str | None = None
    changed_paths: list[str] | None = None


class VerificationStatusResult(Result):
    verification: VerificationStatusInfo


method("verification.status", params=VerificationStatusParams, result=VerificationStatusResult,
       doc="Best known verification evidence for a cwd/session; read-only, never runs checks.")
