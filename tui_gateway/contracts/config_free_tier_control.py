"""Contracts: config, setup readiness, free tier, model inventory, connectors, diagnostics,
image generation and structured session control.

Handlers: ``tui_gateway/methods_config.py`` (``config.get``, ``setup.*``, ``diagnostics.share_nous``),
``methods_config_set.py`` (``config.set``), ``methods_free_tier.py``, ``methods_complete.py``
(``model.options``), ``methods_connectors.py``, ``methods_images.py``, ``methods_session_control.py``
and ``methods_session.py`` (``verification.status``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictInt

from .base import JsonValue, Params, Result, WireEnum
from .connectors_operation import ConnectionOperationStatus
from .tools_commands import DispatchType
from .registry import method

# ── config.get ────────────────────────────────────────────────────────────────────────────────


class ConfigGetParams(Params):
    """``key`` selects one getter from ``_CONFIG_GETTERS``; ``cwd`` feeds the ``project`` getter,
    ``session_id`` lets ``reasoning`` / ``fast`` answer with the session's live pin."""

    key: str = ""
    cwd: str = ""
    session_id: str = ""


class ConfigProviderRef(Result):
    """``hermes_cli/models.py::list_available_providers`` emits every field."""

    id: str
    label: str
    aliases: list[str]
    authenticated: bool


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
    # WHY JsonValue: effective YAML is user-owned and extensible.
    config: JsonValue | None = None
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


class ConfigSetParams(Params):
    """``key`` picks the setter (``_CONFIG_SETTERS``, ``details_mode.<section>``, display toggles);
    ``value`` is the raw word/string the setter normalises (falsy non-strings are reported back in
    the error). ``scope`` applies to ``yolo`` / ``reasoning``; ``confirm_expensive_model`` to ``model``."""

    key: str
    # WHY JsonValue: config setters preserve raw YAML fragments until each key normalizes them.
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
    # WHY JsonValue: setters echo normalized strings, booleans, and raw value-compatible results.
    value: JsonValue
    warning: str | None = None
    confirm_required: bool | None = None
    confirm_message: str | None = None
    scope: str | None = None
    deferred: bool | None = None
    tool_progress: str | None = None
    cwd: str | None = None
    branch: str | None = None
    history_reset: bool | None = None
    # WHY JsonValue: ``methods_config_set._set_personality`` returns the producer-owned session snapshot.
    info: JsonValue | None = None


method("config.set", params=ConfigSetParams, result=ConfigSetResult,
       doc="Change one config key (persisted or session-scoped) and read back the normalised value.")


# ── setup readiness ───────────────────────────────────────────────────────────────────────────


class SetupStatusResult(Result):
    """``provider_configured`` is the loose answer; the boot record's fields (``ready``,
    ``free_tier``, ``other_providers``, ``inference_provider``) ride along on the launch profile.
    An unknown ``profile`` answers ``ok=False`` + ``error``."""

    provider_configured: bool | None = None
    ready: bool | None = None
    free_tier: bool | None = None
    other_providers: bool | None = None
    inference_provider: str | None = None
    profile: str | None = None
    # Present only when the last free-tier mint failed (``anon_auth.MintFailure.as_payload``), flat —
    # the same block ``setup.status`` / ``setup.ready`` carry, so a client keys on ``error_code`` alike.
    error_code: str | None = None
    retryable: bool | None = None
    retry_after: int | None = None
    ok: bool | None = None
    error: str | None = None


method("setup.status", params=Params, result=SetupStatusResult,
       doc="Loose provider check: is ANY provider auth state discoverable for the (launch or named) profile.")


class SetupRuntimeCheckParams(Params):
    provider: str | None = None


class SetupRuntimeCheckResult(Result):
    """``ok=False`` + ``error`` when the resolved model can't be served; ``free_tier`` says the
    selected route is the welcome host."""

    ok: bool
    provider: str | None = None
    model: str | None = None
    source: str | None = None
    error: str | None = None
    free_tier: bool | None = None
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


class FreeTierModel(WireEnum):
    welcome = "nous/welcome"


class FreeTierLabel(WireEnum):
    free_tier = "Nous · free tier"


class FreeTierStatusResult(Result):
    """``available`` = an identity exists AND the tier is on; whether inference runs on it is
    ``setup.runtime_check.free_tier``'s question."""

    has_guest: bool
    enabled: bool
    available: bool
    notice_pending: bool
    model: FreeTierModel
    label: FreeTierLabel
    # Present only when the last free-tier mint failed (``anon_auth.MintFailure.as_payload``), flat —
    # the same block ``setup.status`` / ``setup.ready`` carry, so a client keys on ``error_code`` alike.
    error: str | None = None
    error_code: str | None = None
    retryable: bool | None = None
    retry_after: int | None = None


method("free_tier.status", params=Params, result=FreeTierStatusResult,
       doc="Pure read of the focused profile's free-tier identity state (no network, no side effects).")


class FreeTierProvisionResult(Result):
    has_guest: bool
    enabled: bool
    # Present only when the last free-tier mint failed (``anon_auth.MintFailure.as_payload``), flat —
    # the same block ``setup.status`` / ``setup.ready`` carry, so a client keys on ``error_code`` alike.
    error: str | None = None
    error_code: str | None = None
    retryable: bool | None = None
    retry_after: int | None = None


method("free_tier.provision", params=Params, result=FreeTierProvisionResult,
       doc="Explicit retry of the free-tier identity mint when the boot bootstrap could not create it.")


class FreeTierAckNoticeResult(Result):
    acked: bool


method("free_tier.ack_notice", params=Params, result=FreeTierAckNoticeResult,
       doc="Mark the one-time availability notice as shown on the free-tier identity.")


# ── model.options ─────────────────────────────────────────────────────────────────────────────


class ModelOptionsParams(Params):
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
    reasoning: bool
    can_disable_reasoning: bool | None = None


class ModelOptionProvider(Result):
    """Closed ``hermes_cli/inventory.py::build_model_options_payload`` provider row."""

    slug: str
    name: str
    models: list[str]
    total_models: int
    is_current: bool
    is_user_defined: bool
    source: str
    aliases: list[str] | None = None
    api_url: str | None = None
    native_catalog_empty: bool | None = None
    auth_type: str | None = None
    authenticated: bool | None = None
    key_env: str | None = None
    warning: str | None = None
    featured_models: list[str] | None = None
    # WHY JsonValue: provider model IDs are dynamic map keys from inventory.py.
    capabilities: JsonValue | None = None
    # WHY JsonValue: provider model IDs are dynamic map keys from inventory.py.
    pricing: JsonValue | None = None
    pricing_pending: bool | None = None
    free_tier: bool | None = None
    free_tier_pending: bool | None = None
    free_tier_row: bool | None = None
    unavailable_models: list[str] | None = None


class ModelOptionsResult(Result):
    providers: list[ModelOptionProvider]
    model: str
    provider: str


method("model.options", params=ModelOptionsParams, result=ModelOptionsResult,
       doc="Provider/model inventory for the picker, layered over the session's live provider when given.")


# ── connectors ────────────────────────────────────────────────────────────────────────────────


class ConnectorsListParams(Params):
    session_id: str


class ConnectorRow(Result):
    """Closed connector catalog row emitted by ``ConnectorClient.list_connectors``."""

    connector: str
    connected: bool
    enabled: bool
    connectionStatus: str | None = None
    name: str | None = None
    description: str | None = None


class ConnectorsListResult(Result):
    available: bool
    connectors: list[ConnectorRow]


method("connectors.list", params=ConnectorsListParams, result=ConnectorsListResult,
       doc="Connector catalog + connection state for one owned session (``available=False`` when the toolset is off).")


class ConnectorsConnectParams(Params):
    session_id: str
    connectors: list[str]
    reconnect: bool = False


class ConnectorsConnectResult(ConnectionOperationStatus):
    """The operation the connect opened (or re-minted on): ``tools/connectors/managed.py``
    ``_off_desktop_result`` / ``methods_connectors._reissue``. ``status``/``note`` ride along from
    the tool result when the call ran through ``manage_connections``."""

    status: str | None = None
    note: str | None = None


method("connectors.connect", params=ConnectorsConnectParams, result=ConnectorsConnectResult,
       doc="Start (or re-initiate) authorization for named connectors on the session's connection operation.")


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


class WaitBarrierSession(Result):
    type: Literal["session"]
    target: str
    reason: str = ""


class WaitBarrierPid(Result):
    type: Literal["pid"]
    target: int
    reason: str = ""


class GoalStatus(WireEnum):
    """``hermes_cli/goals.py::GoalState.status`` minus ``cleared``, which the snapshot drops."""

    active = "active"
    paused = "paused"
    done = "done"


class GoalVerdict(WireEnum):
    """``hermes_cli/goals.py::GoalState.last_verdict``."""

    done = "done"
    blocked = "blocked"
    continue_ = "continue"
    wait = "wait"
    skipped = "skipped"


class GoalSnapshot(Result):
    """``methods_session_control.py::_safe_goal_snapshot`` — the frontend-safe GoalState subset."""

    title: str
    status: GoalStatus
    turns_used: int
    max_turns: int
    contract: GoalContractSnapshot
    subgoals: list[str]
    gates: list[GoalGateSnapshot]
    created_at: float | None = None
    updated_at: float | None = None
    paused_reason: str | None = None
    last_verdict: GoalVerdict | None = None
    last_reason: str | None = None
    wait_barrier: WaitBarrierUntil | WaitBarrierSession | WaitBarrierPid | None = Field(default=None, discriminator="type")


class LoopStatus(WireEnum):
    """``hermes_cli/loops.py::LoopState.status`` minus ``cleared``."""

    active = "active"
    paused = "paused"
    done = "done"


class LoopMode(WireEnum):
    interval = "interval"
    self_paced = "self_paced"


class LoopSnapshot(Result):
    """``_safe_loop_snapshot`` — persisted LoopState fields, never its route."""

    prompt: str
    status: LoopStatus
    mode: LoopMode
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


class HeartbeatStatus(WireEnum):
    """``hermes_cli/heartbeat.py::HeartbeatState.status`` minus ``cleared``."""

    active = "active"
    paused = "paused"


class HeartbeatSnapshot(Result):
    prompt: str
    status: HeartbeatStatus
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


class SessionControlReadParams(Params):
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
    """``subgoal.add`` reads ``text``; ``subgoal.remove`` reads the 1-based ``index``. Strict: a
    string or float index is a client bug, not a value to coerce."""

    text: str | None = None
    index: StrictInt | None = None


class SessionControlParams(Params):
    """``action`` is validated by the handler (unknown / gate actions answer ``4004``), so it stays a
    string on the wire; ``SessionControlAction`` lists the accepted set."""

    session_id: str
    action: str
    args: SessionControlArgs | None = None


class SessionControlDispatch(Result):
    """``_dispatch_envelope`` always serializes all user-visible directive fields."""

    type: DispatchType | None
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


class VerificationStatusParams(Params):
    session_id: str | None = None
    session_key: str | None = None
    cwd: str | None = None


class VerificationKind(WireEnum):
    test = "test"
    lint = "lint"
    typecheck = "typecheck"
    build = "build"
    format = "format"
    check = "check"
    verify = "verify"
    ad_hoc = "ad_hoc"


class VerificationScope(WireEnum):
    full = "full"
    targeted = "targeted"


class VerificationOutcome(WireEnum):
    passed = "passed"
    failed = "failed"


class VerificationStatus(WireEnum):
    disabled = "disabled"
    not_applicable = "not_applicable"
    unverified = "unverified"
    stale = "stale"
    passed = "passed"
    failed = "failed"
    unknown = "unknown"


class VerificationEvidenceRow(Result):
    """Closed SQLite ``verification_events`` row from ``agent/verification_evidence.py:51-63``."""

    id: int
    created_at: str
    session_id: str
    cwd: str
    root: str
    command: str
    canonical_command: str
    kind: VerificationKind
    scope: VerificationScope
    status: VerificationOutcome
    exit_code: int
    output_summary: str


class VerificationStatusInfo(Result):
    """``verification_status`` produces its closed status set at ``agent/verification_evidence.py:533-566``."""

    status: VerificationStatus
    evidence: VerificationEvidenceRow | None = None
    root: str | None = None
    session_id: str | None = None
    changed_paths: list[str] | None = None


class VerificationStatusResult(Result):
    verification: VerificationStatusInfo


method("verification.status", params=VerificationStatusParams, result=VerificationStatusResult,
       doc="Best known verification evidence for a cwd/session; read-only, never runs checks.")
