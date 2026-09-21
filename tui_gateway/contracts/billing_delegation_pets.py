"""Contracts: billing / subscription / usage envelopes, delegation controls, handoff, message
reactions, pet generation (``tui_gateway/methods_session_pet.py``) and ``project.facts`` (handlers in
``tui_gateway/methods_session.py``).

Billing routes are FAIL-OPEN: a logged-out / unreachable portal answers an ``ok`` result whose
``error`` carries the typed code (``_serialize_billing_error``) — never a JSON-RPC error — so the
client always resolves and branches on the envelope. ``BillingEnvelope`` is that shared shape.
"""

from __future__ import annotations

from pydantic import Field

from .base import JsonValue, Params, Result, WireEnum
from .common import MessageReaction, OpenModel, ProfileParams, SessionParams
from .registry import method

# ── billing envelope ──────────────────────────────────────────────────────────────────────────


class BillingEnvelope(Result):
    """``tui_gateway/billing_view.py::_serialize_billing_error`` on failure (``ok`` false + typed
    ``error``); success routes add their own fields. ``payload`` is the raw portal body (error
    extras such as ``remainingUsd``, or the NAS success body on pending-change routes)."""

    ok: bool
    error: str | None = None
    message: str | None = None
    portal_url: str | None = None
    retry_after: int | float | None = None
    payload: dict[str, JsonValue] | None = None
    actor: str | None = None
    code: str | None = None
    recovery: str | None = None


# ── usage.bars (shared two-bar dollar model) ──────────────────────────────────────────────────


class UsageBarKind(WireEnum):
    plan = "plan"
    topup = "topup"


class UsageBar(Result):
    """``_serialize_usage_bar``: one bar, magnitudes pre-formatted; ``pct_used`` only for ``plan``."""

    kind: UsageBarKind
    remaining_display: str
    total_display: str
    spent_display: str
    pct_used: int | None = None
    fill_fraction: float


class UsageModel(Result):
    """``_serialize_usage_model`` — also embedded as ``usage`` in the billing / subscription states,
    where the fail-open form is a bare ``{available: false}`` (no ``ok``)."""

    ok: bool | None = None
    available: bool
    status: str | None = None
    plan_name: str | None = None
    renews_at: str | None = None
    renews_display: str | None = None
    subscription_remaining_display: str | None = None
    topup_remaining_display: str | None = None
    total_spendable_display: str | None = None
    has_topup: bool | None = None
    plan_bar: UsageBar | None = None
    topup_bar: UsageBar | None = None


method("usage.bars", params=ProfileParams, result=UsageModel,
       doc="Two-bar dollar usage view shared by /usage, /topup and /subscription; fail-open to unavailable.")


# ── billing.state ─────────────────────────────────────────────────────────────────────────────


class BillingCardInfo(Result):
    brand: str
    last4: str
    masked: str
    display: str | None = None
    resolved_via: str | None = None


class PaymentMethodKind(WireEnum):
    card = "card"
    link = "link"
    unknown = "unknown"


class BillingPaymentMethod(Result):
    """``_serialize_payment_method``: each kind emits only its own fields (a ``card`` never carries
    ``email``; ``unknown`` carries what the server called it in ``raw_kind``)."""

    kind: PaymentMethodKind
    brand: str | None = None
    last4: str | None = None
    wallet: str | None = None
    email: str | None = None
    raw_kind: str | None = None
    resolved_via: str | None = None


class BillingMonthlyCap(Result):
    limit_usd: str | None = None
    limit_display: str
    spent_this_month_usd: str | None = None
    spent_display: str
    is_default_ceiling: bool


class AutoReloadCardKind(WireEnum):
    canonical = "canonical"
    distinct = "distinct"
    none = "none"


class BillingAutoReloadCard(Result):
    """Only ``distinct`` carries the payment-method identity."""

    kind: AutoReloadCardKind
    payment_method_id: str | None = None
    brand: str | None = None
    last4: str | None = None


class BillingAutoReload(Result):
    enabled: bool
    threshold_usd: str | None = None
    threshold_display: str
    reload_to_usd: str | None = None
    reload_to_display: str
    card: BillingAutoReloadCard | None = None


class BillingStateResult(Result):
    """``_serialize_billing_state`` (money as strings); the ``except`` fallback emits only
    ``ok / logged_in / free_tier / error``, so everything else is optional."""

    ok: bool
    logged_in: bool
    free_tier: bool = False
    free_tier_model: str | None = None
    org_name: str | None = None
    org_slug: str | None = None
    role: str | None = None
    is_admin: bool | None = None
    can_change_plan: bool | None = None
    can_charge: bool | None = None
    balance_usd: str | None = None
    balance_display: str | None = None
    cli_billing_enabled: bool | None = None
    charge_presets: list[str] | None = None
    charge_presets_display: list[str] | None = None
    min_usd: str | None = None
    max_usd: str | None = None
    card: BillingCardInfo | None = None
    payment_method: BillingPaymentMethod | None = None
    monthly_cap: BillingMonthlyCap | None = None
    auto_reload: BillingAutoReload | None = None
    portal_url: str | None = None
    error: str | None = None
    usage: UsageModel | None = None


method("billing.state", params=ProfileParams, result=BillingStateResult,
       doc="Read-only billing view (no scope); the Nous free tier is answered locally without a portal call.")


# ── subscription.state / preview / change / resume / upgrade ──────────────────────────────────


class SubscriptionContext(WireEnum):
    personal = "personal"
    team = "team"


class CurrentSubscription(Result):
    tier_id: str | None = None
    tier_name: str | None = None
    monthly_credits: str | None = None
    credits_remaining: str | None = None
    cycle_ends_at: str | None = None
    pending_downgrade_tier_name: str | None = None
    pending_downgrade_at: str | None = None
    pending_downgrade_display: str | None = None
    cancel_at_period_end: bool
    cancellation_effective_at: str | None = None
    cancellation_effective_display: str | None = None


class SubscriptionTierOption(Result):
    tier_id: str
    name: str
    tier_order: int
    dollars_per_month_display: str
    monthly_credits: str | None = None
    is_current: bool
    is_enabled: bool


class SubscriptionStateResult(Result):
    """``_serialize_subscription_state``; the view's fallback emits only ``ok / logged_in / error``."""

    ok: bool
    logged_in: bool
    is_admin: bool | None = None
    can_change_plan: bool | None = None
    org_name: str | None = None
    org_id: str | None = None
    role: str | None = None
    context: SubscriptionContext | None = None
    current: CurrentSubscription | None = None
    tiers: list[SubscriptionTierOption] | None = None
    portal_url: str | None = None
    error: str | None = None
    usage: UsageModel | None = None


method("subscription.state", params=ProfileParams, result=SubscriptionStateResult,
       doc="Current plan, tier catalog and usage for the picker; fail-open when logged out.")


class SubscriptionPreviewParams(ProfileParams):
    subscription_type_id: str | None = None


class SubscriptionChangeEffect(WireEnum):
    charge_now = "charge_now"
    scheduled = "scheduled"
    no_op = "no_op"
    blocked = "blocked"


class SubscriptionPreviewResult(BillingEnvelope):
    """``_serialize_subscription_preview`` on success; ``effect`` drives the confirm copy."""

    effect: SubscriptionChangeEffect | None = None
    reason: str | None = None
    current_tier_id: str | None = None
    current_tier_name: str | None = None
    target_tier_id: str | None = None
    target_tier_name: str | None = None
    monthly_credits_delta: str | None = None
    amount_due_now_cents: int | None = None
    effective_at: str | None = None


method("subscription.preview", params=SubscriptionPreviewParams, result=SubscriptionPreviewResult,
       doc="Chargeless quote of what a plan change would do (billing:manage).")


class SubscriptionChangeParams(ProfileParams):
    """Either a target tier (downgrade / same-price change) or ``cancel`` (period-end cancellation)."""

    subscription_type_id: str | None = None
    cancel: bool = False


class BillingPendingChangeResult(BillingEnvelope):
    """``_billing_pending_change``: ``message`` + the raw NAS body in ``payload`` on success."""


method("subscription.change", params=SubscriptionChangeParams, result=BillingPendingChangeResult,
       doc="Schedule a downgrade / same-price change or a period-end cancellation.")
method("subscription.resume", params=ProfileParams, result=BillingPendingChangeResult,
       doc="Clear a scheduled downgrade / cancellation (re-enables recurring spend).")


class SubscriptionUpgradeParams(ProfileParams):
    subscription_type_id: str | None = None
    idempotency_key: str | None = None


class SubscriptionUpgradeResult(BillingEnvelope):
    """The money route: ``status`` separates a completed upgrade from an SCA / decline that must
    finish in the portal at ``recovery_url``; ``idempotency_key`` is echoed (also on error) so a
    retry reuses it."""

    status: str | None = None
    target_tier_name: str | None = None
    recovery_url: str | None = None
    reason: str | None = None
    idempotency_key: str | None = None


method("subscription.upgrade", params=SubscriptionUpgradeParams, result=SubscriptionUpgradeResult,
       doc="Prorate, charge and flip the plan (billing:manage, idempotent).")


# ── billing.charge / charge_status / auto_reload / step_up ───────────────────────────────────


class BillingChargeParams(ProfileParams):
    amount_usd: float | str | None = None
    idempotency_key: str | None = None


class BillingChargeResult(BillingEnvelope):
    """``202 {chargeId}`` — money is not confirmed yet; poll ``billing.charge_status``."""

    charge_id: str | None = None
    idempotency_key: str | None = None


method("billing.charge", params=BillingChargeParams, result=BillingChargeResult,
       doc="Start a one-off top-up charge (billing:manage, idempotent).")


class BillingChargeStatusParams(ProfileParams):
    charge_id: str | None = None


class BillingChargeStatusResult(BillingEnvelope):
    """Single status read (pending | settled | failed); the caller drives the poll cadence."""

    status: str | None = None
    amount_usd: str | float | None = None
    settled_at: str | None = None
    reason: str | None = None


method("billing.charge_status", params=BillingChargeStatusParams, result=BillingChargeStatusResult,
       doc="Poll one charge by id.")


class BillingAutoReloadParams(ProfileParams):
    enabled: bool = False
    threshold: float | str | None = None
    top_up_amount: float | str | None = None


class BillingMutationResult(BillingEnvelope):
    """A write with no success payload beyond ``ok``."""


method("billing.auto_reload", params=BillingAutoReloadParams, result=BillingMutationResult,
       doc="Enable/disable auto top-up with its threshold and reload amount (billing:manage).")


class BillingStepUpParams(ProfileParams):
    session_id: str | None = None


class BillingStepUpResult(BillingEnvelope):
    """``granted`` false when the server downscopes (also on every error envelope)."""

    granted: bool | None = None


method("billing.step_up", params=BillingStepUpParams, result=BillingStepUpResult,
       doc="Run the billing:manage device flow; the URL/code arrive via billing.step_up.verification.")


# ── delegation / subagent.steer ───────────────────────────────────────────────────────────────


class ActiveSubagent(OpenModel):
    """One live child from ``tools/delegate_tool_registry.py::list_active_subagents`` (the record
    is extended by the child runner — ``missed_steer`` etc. — so it stays open)."""

    subagent_id: str
    parent_id: str | None = None
    depth: int | None = None
    goal: str | None = None
    delegation_id: str | None = None
    model: str | None = None
    started_at: float | None = None
    status: str | None = None
    tool_count: int | None = None
    owner_agent_session_id: str | None = None


class DelegationStatusResult(Result):
    active: list[ActiveSubagent]
    paused: bool
    max_spawn_depth: int
    max_concurrent_children: int


method("delegation.status", params=ProfileParams, result=DelegationStatusResult,
       doc="Running subagent tree plus the spawn pause flag and limits.")


class DelegationPauseParams(ProfileParams):
    paused: bool = True


class DelegationPauseResult(Result):
    paused: bool


method("delegation.pause", params=DelegationPauseParams, result=DelegationPauseResult,
       doc="Block/unblock NEW spawns globally (active children keep running); returns the new state.")


class SubagentSteerParams(SessionParams):
    subagent_id: str
    text: str


class SteerStatus(WireEnum):
    queued = "queued"
    rejected = "rejected"


class SubagentSteerResult(Result):
    """``queued`` is not ``delivered``: a child past its final tool batch surfaces ``missed_steer``."""

    status: SteerStatus
    subagent_id: str
    text: str


method("subagent.steer", params=SubagentSteerParams, result=SubagentSteerResult,
       doc="Queue steering text into a live delegated child owned by this session.")


# ── handoff ───────────────────────────────────────────────────────────────────────────────────


class HandoffRequestParams(SessionParams):
    platform: str


class HandoffRequestResult(Result):
    queued: bool
    session_key: str
    platform: str
    home_name: str


method("handoff.request", params=HandoffRequestParams, result=HandoffRequestResult,
       doc="Queue a handoff to a messaging platform's home channel; the gateway watcher claims it.")


class HandoffStateResult(Result):
    """``state`` is pending | running | completed | failed, or '' when nothing was requested."""

    state: str
    platform: str
    error: str


method("handoff.state", params=SessionParams, result=HandoffStateResult,
       doc="Poll the handoff row for this session.")


class HandoffFailParams(SessionParams):
    error: str | None = None


class HandoffFailResult(Result):
    """``failed`` false when the watcher already claimed the row; ``state`` is what it is now."""

    failed: bool
    state: str


method("handoff.fail", params=HandoffFailParams, result=HandoffFailResult,
       doc="Fail a not-yet-claimed handoff (client poll timeout); CAS against the watcher.")


# ── message.react ─────────────────────────────────────────────────────────────────────────────


class ReactionAuthor(WireEnum):
    user = "user"
    agent = "agent"


class MessageReactParams(SessionParams):
    """``row_id`` is ``messages.id``; a not-yet-persisted live message names ``newest_role`` instead.
    ``emoji`` null clears; the same emoji again retracts."""

    row_id: int | None = None
    newest_role: str | None = None
    emoji: str | None = None
    author: ReactionAuthor | None = None


class MessageReactResult(Result):
    row_id: int
    reactions: list[MessageReaction]


method("message.react", params=MessageReactParams, result=MessageReactResult,
       doc="Set/clear one author's emoji reaction on a message; returns the row's full reaction list.")


# ── pets: generate / hatch / cancel / status ──────────────────────────────────────────────────


class PetCancelParams(ProfileParams):
    token: str | None = None


class PetCancelResult(Result):
    ok: bool


method("pet.cancel", params=PetCancelParams, result=PetCancelResult,
       doc="Stop an in-flight pet generate/hatch by token (idempotent).")


class PetGenProvider(Result):
    """``agent/pet/generate/imagegen.py::list_sprite_providers`` row."""

    name: str
    label: str
    default: bool


class PetGenerateStatusResult(Result):
    available: bool
    providers: list[PetGenProvider]


method("pet.generate.status", params=ProfileParams, result=PetGenerateStatusResult,
       doc="Whether pet generation is possible (a reference-capable image backend) and which providers.")


class PetGenerateParams(ProfileParams):
    """``prompt`` or a ``referenceImage`` data URL is required (the handler answers 4004 without one)."""

    prompt: str | None = None
    referenceImage: str | None = None  # noqa: N815 - wire key
    count: int | None = None
    style: str | None = None
    provider: str | None = None


class PetDraft(Result):
    index: int
    dataUri: str  # noqa: N815 - wire key


class PetGenerateResult(Result):
    ok: bool
    token: str
    drafts: list[PetDraft]


method("pet.generate", params=PetGenerateParams, result=PetGenerateResult,
       doc="Candidate base looks for a new pet (draft step); drafts also stream via pet.generate.progress.")


class PetHatchParams(ProfileParams):
    token: str
    name: str
    cancelToken: str | None = None  # noqa: N815 - wire key
    index: int | None = None
    description: str | None = None
    prompt: str | None = None
    style: str | None = None
    provider: str | None = None


# TODO(common): ``tui_gateway/server.py::_pet_sprite_payload`` is one shape for ``pet.info`` and
# ``pet.hatch``; consolidate with the pets contract module. Every field optional: ``pet.hatch``
# emits ``{}`` when the installed pet cannot be reloaded.
class PetSpritePayload(Result):
    slug: str | None = None
    displayName: str | None = None  # noqa: N815 - wire key
    mime: str | None = None
    spritesheetBase64: str | None = None  # noqa: N815 - wire key
    spritesheetRevision: str | None = None  # noqa: N815 - wire key
    frameW: int | None = None  # noqa: N815 - wire key
    frameH: int | None = None  # noqa: N815 - wire key
    framesPerState: int | None = None  # noqa: N815 - wire key
    framesByState: dict[str, int] | None = None  # noqa: N815 - wire key
    framesByRow: dict[str, int] | None = None  # noqa: N815 - wire key
    loopMs: int | None = None  # noqa: N815 - wire key
    scale: float | None = None
    stateRows: list[str] | None = None  # noqa: N815 - wire key


class PetHatchResult(Result):
    """The hatched pet is installed but NOT active (``pet.select`` adopts, ``pet.remove`` discards)."""

    ok: bool
    slug: str
    displayName: str  # noqa: N815 - wire key
    warnings: list[JsonValue] = Field(default_factory=list)
    pet: PetSpritePayload


method("pet.hatch", params=PetHatchParams, result=PetHatchResult,
       doc="Turn a base draft into a full spritesheet pet; progress streams via pet.hatch.progress.")


# ── project.facts ─────────────────────────────────────────────────────────────────────────────


class ProjectFactsParams(ProfileParams):
    cwd: str | None = None


class ProjectFacts(Result):
    """``agent/coding_context.py::project_facts_for`` — the system prompt's coding-context detection."""

    root: str
    manifests: list[str]
    packageManagers: list[str]  # noqa: N815 - wire key
    verifyCommands: list[str]  # noqa: N815 - wire key
    contextFiles: list[str]  # noqa: N815 - wire key


class ProjectFactsResult(Result):
    """``facts`` null outside a workspace (or when detection failed)."""

    facts: ProjectFacts | None = None


method("project.facts", params=ProjectFactsParams, result=ProjectFactsResult,
       doc="Structured project facts for a cwd so UIs don't re-sniff the workspace.")
