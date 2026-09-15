"""Contracts: billing / subscription / usage envelopes, delegation controls, handoff, message
reactions, pet generation and ``project.facts`` (handlers in ``tui_gateway/methods_session.py``).

Billing routes are FAIL-OPEN: a logged-out / unreachable portal answers an ``ok`` result whose
``error`` carries the typed code (``_serialize_billing_error``) — never a JSON-RPC error — so the
client always resolves and branches on the envelope. ``BillingEnvelope`` is that shared shape.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field

from .base import JsonValue, MethodParams, Params, Result, WireEnum
from .common import MessageReaction, SessionParams, SubagentStatus
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
    # JsonValue: billing_view.py:37 forwards provider-defined NAS bodies; desktop reads remainingUsd.
    payload: JsonValue | None = None
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
    where the fail-open form carries ``available: false`` and leaves every other field null."""

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


method("usage.bars", params=MethodParams, result=UsageModel,
       doc="Two-bar dollar usage view shared by /usage, /topup and /subscription; fail-open to unavailable.")


# ── billing.state ─────────────────────────────────────────────────────────────────────────────


class BillingCardInfo(Result):
    brand: str
    last4: str
    masked: str
    display: str | None = None
    resolved_via: str | None = None


class BillingCardPaymentMethod(Result):
    """``_serialize_payment_method`` card arm (billing_view.py:48-52)."""

    kind: Literal["card"]
    brand: str | None
    last4: str | None
    wallet: str | None
    resolved_via: str | None


class BillingLinkPaymentMethod(Result):
    """``_serialize_payment_method`` link arm (billing_view.py:53-54)."""

    kind: Literal["link"]
    email: str | None
    resolved_via: str | None


class BillingUnknownPaymentMethod(Result):
    """``_serialize_payment_method`` fallback arm (billing_view.py:55)."""

    kind: Literal["unknown"]
    raw_kind: str | None
    resolved_via: str | None


BillingPaymentMethod = Annotated[
    Union[BillingCardPaymentMethod, BillingLinkPaymentMethod, BillingUnknownPaymentMethod],
    Field(discriminator="kind"),
]


class BillingMonthlyCap(Result):
    limit_usd: str | None
    limit_display: str
    spent_this_month_usd: str | None
    spent_display: str
    is_default_ceiling: bool


class BillingCanonicalAutoReloadCard(Result):
    kind: Literal["canonical"]


class BillingDistinctAutoReloadCard(Result):
    kind: Literal["distinct"]
    payment_method_id: str | None
    brand: str | None
    last4: str | None


class BillingNoAutoReloadCard(Result):
    kind: Literal["none"]


BillingAutoReloadCard = Annotated[
    Union[BillingCanonicalAutoReloadCard, BillingDistinctAutoReloadCard, BillingNoAutoReloadCard],
    Field(discriminator="kind"),
]


class BillingAutoReload(Result):
    enabled: bool
    threshold_usd: str | None
    threshold_display: str
    reload_to_usd: str | None
    reload_to_display: str
    card: BillingAutoReloadCard | None


class BillingStateResult(Result):
    """``_serialize_billing_state`` (money as strings); the ``except`` fallback emits only
    ``ok / logged_in / free_tier / error``, so the normal-state additions may be absent."""

    ok: bool
    logged_in: bool
    free_tier: bool
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


method("billing.state", params=MethodParams, result=BillingStateResult,
       doc="Read-only billing view (no scope); the Nous free tier is answered locally without a portal call.")


# ── subscription.state / preview / change / resume / upgrade ──────────────────────────────────


class SubscriptionContext(WireEnum):
    personal = "personal"
    team = "team"


class CurrentSubscription(Result):
    tier_id: str | None
    tier_name: str | None
    monthly_credits: str | None
    credits_remaining: str | None
    cycle_ends_at: str | None
    pending_downgrade_tier_name: str | None
    pending_downgrade_at: str | None
    pending_downgrade_display: str | None
    cancel_at_period_end: bool
    cancellation_effective_at: str | None
    cancellation_effective_display: str | None


class SubscriptionTierOption(Result):
    tier_id: str
    name: str
    tier_order: int
    dollars_per_month_display: str
    monthly_credits: str | None
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


method("subscription.state", params=MethodParams, result=SubscriptionStateResult,
       doc="Current plan, tier catalog and usage for the picker; fail-open when logged out.")


class SubscriptionPreviewParams(MethodParams):
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


class SubscriptionChangeParams(MethodParams):
    """Either a target tier (downgrade / same-price change) or ``cancel`` (period-end cancellation)."""

    subscription_type_id: str | None = None
    cancel: bool | None = None


class BillingPendingChangeResult(BillingEnvelope):
    """``_billing_pending_change``: ``message`` + the raw NAS body in ``payload`` on success."""


class SubscriptionChangeResult(BillingPendingChangeResult):
    pass


class SubscriptionResumeResult(BillingPendingChangeResult):
    pass


method("subscription.change", params=SubscriptionChangeParams, result=SubscriptionChangeResult,
       doc="Schedule a downgrade / same-price change or a period-end cancellation.")
method("subscription.resume", params=MethodParams, result=SubscriptionResumeResult,
       doc="Clear a scheduled downgrade / cancellation (re-enables recurring spend).")


class SubscriptionUpgradeParams(MethodParams):
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


class BillingChargeParams(MethodParams):
    amount_usd: float | str | None = None
    idempotency_key: str | None = None


class BillingChargeResult(BillingEnvelope):
    """``202 {chargeId}`` — money is not confirmed yet; poll ``billing.charge_status``."""

    charge_id: str | None = None
    idempotency_key: str | None = None


method("billing.charge", params=BillingChargeParams, result=BillingChargeResult,
       doc="Start a one-off top-up charge (billing:manage, idempotent).")


class BillingChargeStatusParams(MethodParams):
    charge_id: str | None = None


class BillingChargeStatusResult(BillingEnvelope):
    """Single status read (pending | settled | failed); the caller drives the poll cadence."""

    status: str | None = None
    amount_usd: str | float | None = None
    settled_at: str | None = None
    reason: str | None = None


method("billing.charge_status", params=BillingChargeStatusParams, result=BillingChargeStatusResult,
       doc="Poll one charge by id.")


class BillingAutoReloadParams(MethodParams):
    enabled: bool | None = None
    threshold: float | str | None = None
    top_up_amount: float | str | None = None


class BillingMutationResult(BillingEnvelope):
    """A write with no success payload beyond ``ok``."""


method("billing.auto_reload", params=BillingAutoReloadParams, result=BillingMutationResult,
       doc="Enable/disable auto top-up with its threshold and reload amount (billing:manage).")


class BillingStepUpParams(MethodParams):
    session_id: str | None = None


class BillingStepUpResult(BillingEnvelope):
    """``granted`` false when the server downscopes (also on every error envelope)."""

    granted: bool


method("billing.step_up", params=BillingStepUpParams, result=BillingStepUpResult,
       doc="Run the billing:manage device flow; the URL/code arrive via billing.step_up.verification.")


# ── delegation / subagent.steer ───────────────────────────────────────────────────────────────


class ActiveSubagent(Result):
    """One ``list_active_subagents`` record (tools/delegate_tool_registry.py:173-177)."""

    subagent_id: str
    parent_id: str | None
    depth: int
    goal: str
    delegation_id: str | None
    model: str | None
    started_at: float
    status: SubagentStatus
    tool_count: int
    last_tool: str | None = None
    owner_agent_session_id: str | None


class DelegationStatusResult(Result):
    active: list[ActiveSubagent]
    paused: bool
    max_spawn_depth: int
    max_concurrent_children: int


method("delegation.status", params=MethodParams, result=DelegationStatusResult,
       doc="Running subagent tree plus the spawn pause flag and limits.")


class DelegationPauseParams(MethodParams):
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


class PetCancelParams(MethodParams):
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


method("pet.generate.status", params=MethodParams, result=PetGenerateStatusResult,
       doc="Whether pet generation is possible (a reference-capable image backend) and which providers.")


class PetGenerateParams(MethodParams):
    """``prompt`` or a ``referenceImage`` data URL is required (the handler answers 4004 without one)."""

    prompt: str | None = None
    referenceImage: str | None = None  # noqa: N815 - wire key
    count: int = 4
    style: str = "auto"
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


class PetHatchParams(MethodParams):
    token: str
    name: str
    cancelToken: str | None = None  # noqa: N815 - wire key
    index: int = 0
    description: str | None = None
    prompt: str | None = None
    style: str = "auto"
    provider: str | None = None


class PetSpritePayload(Result):
    """``server._pet_sprite_payload``; ``pet.hatch`` emits ``{}`` if reload fails."""

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
    warnings: list[str]
    pet: PetSpritePayload


method("pet.hatch", params=PetHatchParams, result=PetHatchResult,
       doc="Turn a base draft into a full spritesheet pet; progress streams via pet.hatch.progress.")


# ── project.facts ─────────────────────────────────────────────────────────────────────────────


class ProjectFactsParams(MethodParams):
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
