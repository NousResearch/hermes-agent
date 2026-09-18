import { randomUUID } from 'node:crypto'

import { Box, Text, useInput } from '@hermes/ink'
import { useEffect, useRef, useState } from 'react'

import type {
  SubscriptionOverlayState,
  SubscriptionPendingChange,
  SubscriptionResult,
  SubscriptionStepUpRetry
} from '../app/interfaces.js'
import type { SubscriptionStateResponse, SubscriptionTierOption, SubscriptionUpgradeResponse } from '../gatewayTypes.js'
import { getTranslations, useTranslations } from '../i18n/index.js'
import type { Theme } from '../theme.js'

import { ActionRow, footer, MenuRow, type MenuRowSpec, UsageBars, useMenu } from './overlayPrimitives.js'

const UPGRADE_CONFIRM_INTERVAL_MS = 2000
const UPGRADE_CONFIRM_ATTEMPTS = 15

interface SubscriptionOverlayProps {
  /** Close the overlay entirely. */
  onClose: () => void
  /** Merge a partial into the overlay state (screen transitions + pending/result). */
  onPatch: (next: Partial<SubscriptionOverlayState>) => void
  overlay: SubscriptionOverlayState
  t: Theme
}

/**
 * The /subscription modal — an in-terminal plan-change flow (V3). A small state
 * machine: overview → picker → confirm → result, with a stepup screen spliced in
 * when a mutation needs remote spending. Downgrades / cancellations / resume are
 * chargeless; an upgrade charges the card on the subscription, and an SCA/decline
 * is handed off to the portal. Starting a NEW subscription still deep-links (needs
 * a fresh card). All RPCs live in subscription.ts, reached via `overlay.ctx`.
 */
export function SubscriptionOverlay({ onClose, onPatch, overlay, t }: SubscriptionOverlayProps) {
  useTranslations()
  const { screen, state: s } = overlay

  // Teams have no personal subscription — dead-end to /topup, no picker.
  if (s.context === 'team') {
    return (
      <Box borderColor={t.color.accent} borderStyle="round" flexDirection="column" paddingX={1}>
        <TeamContextScreen onClose={onClose} s={s} t={t} />
      </Box>
    )
  }

  return (
    <Box borderColor={t.color.accent} borderStyle="round" flexDirection="column" paddingX={1}>
      {screen === 'picker' && <PickerScreen onClose={onClose} onPatch={onPatch} overlay={overlay} t={t} />}
      {screen === 'confirm' && <ConfirmScreen onClose={onClose} onPatch={onPatch} overlay={overlay} t={t} />}
      {screen === 'result' && <ResultScreen onClose={onClose} overlay={overlay} t={t} />}
      {screen === 'stepup' && <StepUpScreen onClose={onClose} onPatch={onPatch} overlay={overlay} t={t} />}
      {screen === 'overview' && <OverviewScreen onClose={onClose} onPatch={onPatch} overlay={overlay} t={t} />}
    </Box>
  )
}

// ── Shared helpers ───────────────────────────────────────────────────

interface ScreenProps {
  onClose: () => void
  onPatch: (next: Partial<SubscriptionOverlayState>) => void
  overlay: SubscriptionOverlayState
  t: Theme
}

/** ISO datetime → YYYY-MM-DD for display, or a soft fallback. */
function shortDate(iso?: null | string): string {
  return iso && iso.length >= 10 ? iso.slice(0, 10) : getTranslations().subscription.theEndOfTheBillingPeriod
}

/** Integer cents → "$X.YY", or null when no amount is quoted. */
function centsDisplay(cents?: null | number): null | string {
  return typeof cents === 'number' ? `$${(cents / 100).toFixed(2)}` : null
}

/** True when a response is the insufficient_scope denial (route to step-up). */
function isScopeDenial(r: { error?: string; ok?: boolean } | null): boolean {
  return !!r && !r.ok && r.error === 'insufficient_scope'
}

/**
 * Map a failed RPC envelope to a result. (insufficient_scope is intercepted
 * earlier and routed to the step-up screen, so it should not reach here.)
 */
function errorResult(r: { error?: string; message?: string; portal_url?: null | string } | null): SubscriptionResult {
  return {
    message: r?.message || r?.error || getTranslations().subscription.somethingWentWrongTryAgainOrManage,
    ok: false,
    recoveryUrl: r?.portal_url ?? null
  }
}

/** Map a chargeless pending-change mutation (schedule / cancel / resume). */
function mutationResult(r: null | { message?: string; ok?: boolean }, okMessage: string): SubscriptionResult {
  return r?.ok ? { message: r.message || okMessage, ok: true } : errorResult(r)
}

/** Map an upgrade response, routing SCA / decline to a portal recovery. */
function upgradeResult(r: null | SubscriptionUpgradeResponse, pendingTierId?: null | string): SubscriptionResult {
  if (!r) {
    // null = a transport failure (WS drop / request timeout) on the CHARGING
    // route — NAS may have already prorated + charged. Report it as ambiguous and
    // steer to a safe re-check, never a blind retry (which #2's dedup can't cover
    // once the key is lost).
    return {
      message: getTranslations().subscription.couldnTConfirmTheUpgradeYourCard,
      ok: false
    }
  }

  if (r.reason === 'authentication_required' || r.reason === 'subscription_payment_intent_requires_action') {
    return {
      message: getTranslations().subscription.pleaseVerifyYourCardInThePortal,
      ok: false,
      recoveryUrl: r.recovery_url ?? null
    }
  }

  if (r.reason === 'card_declined') {
    return {
      message: getTranslations().subscription.yourCardWasDeclinedTryADifferent,
      ok: false,
      recoveryUrl: r.recovery_url ?? null
    }
  }

  if (r.ok && r.status === 'already_on_tier') {
    return {
      message: getTranslations().subscription.alreadySubscribed(
        String(r.target_tier_name ?? getTranslations().subscription.thisPlan)
      ),
      ok: true
    }
  }

  if (r.ok && r.status === 'upgraded') {
    return {
      message: getTranslations().subscription.upgraded(
        String(r.target_tier_name ?? getTranslations().subscription.yourNewPlan)
      ),
      ok: true,
      pendingTierId: pendingTierId ?? null
    }
  }

  if (r.status === 'requires_action') {
    return {
      message: getTranslations().subscription.thisUpgradeNeedsExtraVerification3dsFinish,
      ok: false,
      recoveryUrl: r.recovery_url ?? null
    }
  }

  if (r.status === 'payment_failed') {
    return {
      message: getTranslations().subscription.yourCardWasDeclinedUpdateYourPayment,
      ok: false,
      recoveryUrl: r.recovery_url ?? null
    }
  }

  return errorResult(r)
}

/** Map a failed remote-spending step-up to the right recovery copy (typed). */
function stepUpDenialResult(res: { error?: string; message?: string }): SubscriptionResult {
  if (res.error === 'session_revoked') {
    return { message: getTranslations().subscription.yourSessionExpiredRunPortalToLog, ok: false }
  }

  if (res.error === 'remote_spending_revoked') {
    return {
      message: res.message || getTranslations().subscription.remoteSpendingWasStoppedForThisTerminal,
      ok: false
    }
  }

  if (res.error === 'rate_limited') {
    return { message: getTranslations().subscription.tooManyAttemptsWaitAMomentThen, ok: false }
  }

  return {
    message: res.message || getTranslations().subscription.remoteSpendingWasNotAllowedSomeoneWith,
    ok: false
  }
}

// ── Scope-aware routing (shared by the picker, confirm, overview + step-up) ──

// A REPEAT scope denial during a post-grant replay must NOT route back to the
// stepup screen: we're already mounted there (in the 'resuming' phase), so an
// onPatch({screen:'stepup'}) is a no-op that never remounts → the screen freezes.
// Post-grant replays pass allowStepUp=false and surface this instead (mirrors the
// CLI's allow_stepup=False cap).
const scopeStillDeniedResult: SubscriptionResult = {
  message: getTranslations().subscription.remoteSpendingStillIsnTActiveFor,
  ok: false
}

/** Preview a tier and route: confirm (ok), stepup (scope), or result (other error). */
function previewAndRoute(
  ctx: SubscriptionOverlayState['ctx'],
  tierId: string,
  onPatch: ScreenProps['onPatch'],
  allowStepUp = true
): Promise<void> {
  return ctx.preview(tierId).then(p => {
    if (!p) {
      return onPatch({
        result: { message: getTranslations().subscription.couldNotPreviewThatChange, ok: false },
        screen: 'result'
      })
    }

    if (!p.ok) {
      if (isScopeDenial(p)) {
        return allowStepUp
          ? onPatch({ screen: 'stepup', stepUpRetry: { kind: 'preview', tierId } })
          : onPatch({ result: scopeStillDeniedResult, screen: 'result' })
      }

      return onPatch({ result: errorResult(p), screen: 'result' })
    }

    // charge_now ⇒ an upgrade (charges now); everything else schedules at period
    // end. blocked/no_op still go to confirm, which shows why + no apply.
    const kind = p.effect === 'charge_now' ? 'upgrade' : 'tier_change'

    // Mint the upgrade idempotency key HERE so it rides `pending` into confirm AND
    // the step-up replay — a re-submit / post-grant replay dedups server-side
    // (mirrors billingOverlay's pendingCharge.idempotencyKey).
    const pending: SubscriptionPendingChange =
      kind === 'upgrade'
        ? { idempotencyKey: randomUUID(), kind, preview: p, targetTierId: tierId }
        : { kind, preview: p, targetTierId: tierId }

    onPatch({ pending, screen: 'confirm' })
  })
}

/** Apply the confirmed pending change and route: result (ok/err) or stepup (scope). */
function applyPendingAndRoute(
  ctx: SubscriptionOverlayState['ctx'],
  pending: null | SubscriptionPendingChange,
  onPatch: ScreenProps['onPatch'],
  allowStepUp = true
): Promise<void> {
  if (!pending) {
    // Nothing to apply (defensive) — return to the overview rather than stranding.
    onPatch({ screen: 'overview' })

    return Promise.resolve()
  }

  const toStepUp = () =>
    allowStepUp
      ? onPatch({ screen: 'stepup', stepUpRetry: { kind: 'apply' } })
      : onPatch({ result: scopeStillDeniedResult, screen: 'result' })

  const finish = (result: SubscriptionResult) => onPatch({ result, screen: 'result' })

  if (pending.kind === 'cancellation') {
    return ctx
      .scheduleCancellation()
      .then(r =>
        isScopeDenial(r)
          ? toStepUp()
          : finish(mutationResult(r, getTranslations().subscription.scheduledYourPlanStaysActiveUntilThe))
      )
  }

  if (pending.kind === 'upgrade') {
    return ctx
      .upgrade(pending.targetTierId ?? '', pending.idempotencyKey)
      .then(r => (isScopeDenial(r) ? toStepUp() : finish(upgradeResult(r, pending.targetTierId))))
  }

  return ctx
    .scheduleChange(pending.targetTierId ?? '')
    .then(r =>
      isScopeDenial(r)
        ? toStepUp()
        : finish(mutationResult(r, getTranslations().subscription.scheduledYourPlanDoesnTChangeToday))
    )
}

/** Resume (undo the pending change) and route: result (ok/err) or stepup (scope). */
function resumeAndRoute(
  ctx: SubscriptionOverlayState['ctx'],
  onPatch: ScreenProps['onPatch'],
  allowStepUp = true
): Promise<void> {
  return ctx.resume().then(r => {
    if (isScopeDenial(r)) {
      return allowStepUp
        ? onPatch({ screen: 'stepup', stepUpRetry: { kind: 'resume' } })
        : onPatch({ result: scopeStillDeniedResult, screen: 'result' })
    }

    return onPatch({
      result: mutationResult(r, getTranslations().subscription.yourPendingChangeWasUndoneYouStay),
      screen: 'result'
    })
  })
}

// ── The pending scheduled change (drives the banner + status echo) ──

interface PendingTransition {
  to: string
  when: string
}

/** The scheduled downgrade/cancel as a from→to transition, or null. */
function pendingTransition(c: SubscriptionStateResponse['current']): null | PendingTransition {
  if (!c) {
    return null
  }

  if (c.cancel_at_period_end) {
    return {
      to: getTranslations().subscription.cancels,
      when: c.cancellation_effective_display ?? shortDate(c.cancellation_effective_at)
    }
  }

  if (c.pending_downgrade_tier_name) {
    return { to: c.pending_downgrade_tier_name, when: c.pending_downgrade_display ?? shortDate(c.pending_downgrade_at) }
  }

  return null
}

// ── Screen: Overview (plan + usage + entry to the change flow) ────────

/** Status line — dollars-only, and echoes a pending "Ultra → Plus" transition. */
function statusLine(s: SubscriptionStateResponse): string {
  const u = s.usage
  const c = s.current
  const plan = c?.tier_name ?? u?.plan_name ?? null
  const trans = pendingTransition(c)
  const flip = plan && trans ? ` → ${trans.to}` : ''
  const renewsRaw = u?.renews_display ?? null
  const renews = renewsRaw ? getTranslations().subscription.renewalDate(String(renewsRaw)) : ''
  const viewOnly = !s.can_change_plan

  if (!plan) {
    return getTranslations().subscription.planFreeFreeModelsOnly
  }

  if (u?.status === 'low' && u.total_spendable_display) {
    return getTranslations().subscription.lowBalancePlan(String(plan), String(flip), String(u.total_spendable_display))
  }

  const left = u?.total_spendable_display
    ? getTranslations().subscription.remainingBalance(String(u.total_spendable_display))
    : ''

  return getTranslations().subscription.planStatus(
    String(plan),
    String(flip),
    String(left),
    String(viewOnly ? getTranslations().subscription.viewOnly : renews)
  )
}

function OverviewScreen({ onClose, onPatch, overlay, t }: ScreenProps) {
  const { ctx, state: s } = overlay
  const c = s.current
  const isFree = !c?.tier_id
  const currentName = c?.tier_name ?? getTranslations().subscription.yourPlan
  const trans = pendingTransition(c)
  const hasPendingChange = !!trans
  // Admin/owner on a personal paid plan can change it in-terminal; otherwise the
  // portal enforces who can act (members) / starting a new sub needs a card.
  const canChange = s.can_change_plan && !isFree

  // On Free the catalog renders inline; picking a plan hands off to the portal,
  // where starting a subscription needs card capture + checkout.
  const freePlans = isFree
    ? s.tiers.filter(tier => tier.is_enabled && tier.tier_order > 0).sort((a, b) => a.tier_order - b.tier_order)
    : []

  // Guard the async resume so a double-press cannot fire two DELETEs mid-await.
  const busyRef = useRef(false)

  const u = s.usage
  const freeNudge = isFree ? getTranslations().subscription.paidModelsNeedASubscriptionStartOne : null

  const lowNudge =
    u?.status === 'low'
      ? getTranslations().subscription.lowBalanceWarning(
          String(u.total_spendable_display ?? getTranslations().subscription.under5)
        )
      : null

  const doManage = () => {
    if (s.portal_url) {
      void ctx.openManageLink()
    } else {
      ctx.sys(getTranslations().subscription.noPortalUrlAvailableManageYourSubscription)
    }

    return onClose()
  }

  const doResume = () => {
    if (busyRef.current) {
      return
    }

    busyRef.current = true
    void resumeAndRoute(ctx, onPatch)
  }

  const rows: MenuRowSpec[] = []

  if (canChange) {
    // When a change is already scheduled, undo is the most likely next intent —
    // promote it to the first, highlighted action.
    if (hasPendingChange) {
      rows.push({
        color: t.color.ok,
        label: getTranslations().subscription.keepPlan(String(currentName)),
        run: doResume
      })
      rows.push({
        label: getTranslations().subscription.changePlan,
        run: () => onPatch({ pending: null, screen: 'picker' })
      })
    } else {
      rows.push({
        label: getTranslations().subscription.changePlan,
        run: () => onPatch({ pending: null, screen: 'picker' })
      })
      rows.push({
        label: getTranslations().subscription.cancelSubscription,
        run: () => onPatch({ pending: { kind: 'cancellation', preview: null, targetTierId: null }, screen: 'confirm' })
      })
    }
  }

  for (const tier of freePlans) {
    // NAS sends a bare decimal string; tolerate pre-grouped ("1,000") too.
    const credits = Number((tier.monthly_credits ?? '').replace(/,/g, ''))

    const suffix =
      Number.isFinite(credits) && credits > 0
        ? getTranslations().subscription.monthlyCredits(String(credits.toLocaleString('en-US')))
        : ''

    rows.push({
      label: getTranslations().subscription.tierPrice(
        String(tier.name),
        String(tier.dollars_per_month_display),
        String(suffix)
      ),
      run: () => {
        if (busyRef.current) {
          return
        }

        busyRef.current = true
        void ctx.openManageLink(tier.tier_id)
        onClose()
      }
    })
  }

  // The inline plan rows are the subscribe path; only a catalog-less free state
  // still needs the generic portal row.
  if (!isFree || freePlans.length === 0) {
    rows.push({
      label: isFree ? getTranslations().subscription.startASubscription : getTranslations().subscription.manageOnPortal,
      run: doManage
    })
  }

  rows.push({ label: getTranslations().subscription.close, run: onClose })

  const sel = useMenu(rows, onClose)

  return (
    <Box flexDirection="column">
      {/* Lead with the scheduled change so it can't read as "nothing happened". */}
      {trans && (
        <Box flexDirection="column" marginBottom={1}>
          <Text bold color={t.color.warn}>
            {getTranslations().subscription.scheduledChange}
          </Text>
          <Box>
            <Text color={t.color.text}>{currentName} </Text>
            <Text color={t.color.warn}>──▶ </Text>
            <Text color={t.color.text}>{trans.to}</Text>
            <Text color={t.color.muted}> · {trans.when}</Text>
          </Box>
          <Text color={t.color.muted}>{getTranslations().subscription.keepUntil(currentName)}</Text>
        </Box>
      )}

      <Text bold color={t.color.accent}>
        {statusLine(s)}
      </Text>
      <UsageBars model={s.usage} t={t} />
      {freeNudge && (
        <Box marginTop={1}>
          <Text color={t.color.warn}>
            {'> '}
            {freeNudge}
          </Text>
        </Box>
      )}
      {lowNudge && (
        <Box marginTop={1}>
          <Text color={t.color.warn}>
            {'! '}
            {lowNudge}
          </Text>
        </Box>
      )}
      {s.org_name && (
        <Text color={t.color.muted}>
          {getTranslations().subscription.org}
          {s.org_name}
          {s.role ? ` · ${s.role}` : ''}
        </Text>
      )}

      <Text />
      {rows.map((row, i) => (
        <MenuRow active={sel === i} index={i + 1} key={row.label} label={row.label} t={t} />
      ))}

      <Text />
      {footer(getTranslations().subscription.selectEnterConfirmEscClose, t)}
    </Box>
  )
}

// ── Screen: Picker (choose a tier → preview → confirm) ───────────────

function PickerScreen({ onPatch, overlay, t }: ScreenProps) {
  const { ctx, state: s } = overlay
  const currentOrder = s.tiers.find(tier => tier.is_current)?.tier_order ?? 0

  // Selectable = enabled, not the current plan, and not the free/no-sub tier
  // (going to free is a cancellation, offered on the overview). Sorted by price.
  const choices: SubscriptionTierOption[] = s.tiers
    .filter(tier => tier.is_enabled && !tier.is_current && tier.tier_order > 0)
    .sort((a, b) => a.tier_order - b.tier_order)

  // Guard the async preview so a double-press cannot fire two quotes.
  const busyRef = useRef(false)

  const pick = (tier: SubscriptionTierOption) => {
    if (busyRef.current) {
      return
    }

    busyRef.current = true
    void previewAndRoute(ctx, tier.tier_id, onPatch)
  }

  const back = () => onPatch({ screen: 'overview' })

  const rows: MenuRowSpec[] = choices.map(tier => {
    const direction =
      tier.tier_order > currentOrder ? getTranslations().subscription.upgrade : getTranslations().subscription.downgrade

    return {
      label: getTranslations().subscription.tierDirection(
        String(tier.name),
        String(tier.dollars_per_month_display),
        String(direction)
      ),
      run: () => pick(tier)
    }
  })

  rows.push({ label: getTranslations().subscription.back, run: back })

  const sel = useMenu(rows, back)

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {getTranslations().subscription.changePlan}
      </Text>
      <Text color={t.color.muted}>
        {getTranslations().subscription.currentChoice(s.current?.tier_name ?? getTranslations().subscription.free)}
      </Text>
      <Text />
      {choices.length === 0 && (
        <Text color={t.color.muted}>{getTranslations().subscription.noOtherPlansAreAvailableToSwitch}</Text>
      )}
      {rows.map((row, i) => (
        <MenuRow active={sel === i} index={i + 1} key={row.label} label={row.label} t={t} />
      ))}
      <Text />
      {footer(getTranslations().subscription.selectEnterPreviewEscBack, t)}
    </Box>
  )
}

// ── Screen: Confirm (show the previewed effect, then apply) ──────────

function ConfirmScreen({ onClose, onPatch, overlay, t }: ScreenProps) {
  const { ctx, state: s } = overlay
  const pending: null | SubscriptionPendingChange = overlay.pending ?? null
  const preview = pending?.preview ?? null
  const isCancellation = pending?.kind === 'cancellation'
  // Cancellation is always a scheduled (chargeless) effect; otherwise trust the
  // quote (default to blocked so a missing quote never offers an apply).
  const effect = isCancellation ? 'scheduled' : (preview?.effect ?? 'blocked')

  const [submitting, setSubmitting] = useState(false)
  // Synchronous guard: two key events can both see submitting===false before
  // React commits, double-firing the mutation/charge.
  const submittingRef = useRef(false)

  const back = () => {
    // Don't navigate away while an apply is in flight: the screen hasn't changed
    // yet (applyPendingAndRoute patches only after the RPC resolves), so a fresh
    // re-mount would re-fire the mutation — a second charge on the upgrade path.
    if (submittingRef.current) {
      return
    }

    onPatch({ pending: null, screen: isCancellation ? 'overview' : 'picker' })
  }

  const apply = () => {
    if (submittingRef.current || !pending) {
      return
    }

    submittingRef.current = true
    setSubmitting(true)
    void applyPendingAndRoute(ctx, pending, onPatch)
  }

  const manage = () => {
    void ctx.openManageLink()

    return onClose()
  }

  // WHICH card the upgrade will charge (brand + last4) — best-effort via
  // billing.state, shown only when the resolver rung matches what a
  // subscription charge actually uses (subPin / customerDefault, mirroring
  // Stripe's own precedence). Anything else → the generic line stands.
  const [chargeCard, setChargeCard] = useState<null | string>(null)

  useEffect(() => {
    if (isCancellation || effect !== 'charge_now') {
      return
    }

    let cancelled = false

    void ctx.fetchCard().then(card => {
      if (!cancelled && card && (card.resolved_via === 'subPin' || card.resolved_via === 'customerDefault')) {
        setChargeCard(card.masked)
      }
    })

    return () => {
      cancelled = true
    }
  }, [ctx, effect, isCancellation])

  const amount = centsDisplay(preview?.amount_due_now_cents)

  const targetName = isCancellation
    ? null
    : (preview?.target_tier_name ?? getTranslations().subscription.theSelectedPlan)

  let primary: MenuRowSpec | null = null

  if (isCancellation) {
    primary = { color: t.color.warn, label: getTranslations().subscription.cancelSubscription, run: apply }
  } else if (effect === 'charge_now') {
    primary = {
      color: t.color.ok,
      label: amount
        ? getTranslations().subscription.payAndUpgrade(String(amount))
        : getTranslations().subscription.upgradeNowProratedCharge,
      run: apply
    }
  } else if (effect === 'scheduled') {
    primary = {
      color: t.color.ok,
      label: getTranslations().subscription.scheduleChange(String(targetName)),
      run: apply
    }
  } else if (effect === 'blocked') {
    primary = { label: getTranslations().subscription.manageOnPortal, run: manage }
  }

  const rows: MenuRowSpec[] = primary
    ? [primary, { label: getTranslations().subscription.back, run: back }]
    : [{ label: getTranslations().subscription.back, run: back }]

  const sel = useMenu(rows, back)

  // Chip contrasts an immediate charge vs a period-end schedule at a glance.
  const chip =
    effect === 'charge_now'
      ? { color: t.color.ok, label: getTranslations().subscription.chargedNow }
      : effect === 'scheduled'
        ? { color: t.color.warn, label: getTranslations().subscription.scheduledNotToday }
        : null

  return (
    <Box flexDirection="column">
      <Box>
        <Text bold color={t.color.accent}>
          {isCancellation
            ? getTranslations().subscription.confirmCancellation
            : getTranslations().subscription.confirmPlanChange}
        </Text>
        {chip && <Text color={chip.color}> · {chip.label}</Text>}
      </Box>
      {submitting && <Text color={t.color.muted}>{getTranslations().subscription.working}</Text>}

      {isCancellation && (
        <>
          <Text color={t.color.text}>
            {getTranslations().subscription.cancelUntil(
              s.current?.tier_name ?? getTranslations().subscription.yourPlan,
              shortDate(s.current?.cycle_ends_at)
            )}
          </Text>
          <Text color={t.color.muted}>{getTranslations().subscription.youKeepYourRemainingCreditsForThis}</Text>
        </>
      )}

      {effect === 'charge_now' && !isCancellation && (
        <>
          <Text color={t.color.text}>
            {getTranslations().subscription.upgradeTo(targetName ?? '')}{' '}
            {amount
              ? getTranslations().subscription.chargeAmount(String(amount))
              : getTranslations().subscription.youWillBeChargedTheProratedAmount}
          </Text>
          {preview?.monthly_credits_delta && (
            <Text color={t.color.muted}>
              {getTranslations().subscription.monthlyCreditsChange}
              {preview.monthly_credits_delta}.
            </Text>
          )}
          <Text color={t.color.muted}>
            {chargeCard
              ? getTranslations().subscription.chargeCard(String(chargeCard))
              : getTranslations().subscription.theCardOnYourSubscriptionWillBe}
          </Text>
        </>
      )}

      {effect === 'scheduled' && !isCancellation && (
        <>
          <Text color={t.color.text}>
            {getTranslations().subscription.changeTo(targetName ?? '', shortDate(preview?.effective_at))}
          </Text>
          {preview?.monthly_credits_delta && (
            <Text color={t.color.muted}>
              {getTranslations().subscription.monthlyCreditsChange}
              {preview.monthly_credits_delta}.
            </Text>
          )}
        </>
      )}

      {effect === 'no_op' && !isCancellation && (
        <Text color={t.color.muted}>{getTranslations().subscription.alreadyOn(targetName ?? '')}</Text>
      )}

      {effect === 'blocked' && !isCancellation && (
        <Text color={t.color.warn}>
          {preview?.reason ?? getTranslations().subscription.thatChangeCannotBeMadeHereManage}
        </Text>
      )}

      <Text />
      {rows.map((row, i) => (
        <ActionRow active={sel === i} color={row.color} key={row.label} label={row.label} t={t} />
      ))}
      <Text />
      {footer(getTranslations().subscription.selectEnterConfirmEscBack, t)}
    </Box>
  )
}

// ── Screen: Result (outcome + optional portal recovery) ──────────────

function ResultScreen({ onClose, overlay, t }: Omit<ScreenProps, 'onPatch'>) {
  const { ctx } = overlay
  const result = overlay.result ?? null
  const recoveryUrl = result?.recoveryUrl ?? null
  const pendingTierId = result?.pendingTierId ?? null

  const [applyState, setApplyState] = useState<'applying' | 'confirmed' | 'timed_out'>(
    pendingTierId ? 'applying' : 'confirmed'
  )

  useEffect(() => {
    if (!pendingTierId) {
      return
    }

    let attempts = 0
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    const scheduleOrFinish = () => {
      if (cancelled) {
        return
      }

      if (attempts >= UPGRADE_CONFIRM_ATTEMPTS) {
        setApplyState('timed_out')

        return
      }

      timer = setTimeout(tick, UPGRADE_CONFIRM_INTERVAL_MS)
    }

    const tick = () => {
      attempts += 1
      void ctx
        .refreshState()
        .then(fresh => {
          if (cancelled) {
            return
          }

          if (fresh?.current?.tier_id === pendingTierId) {
            setApplyState('confirmed')

            return
          }

          scheduleOrFinish()
        })
        .catch(scheduleOrFinish)
    }

    timer = setTimeout(tick, UPGRADE_CONFIRM_INTERVAL_MS)

    return () => {
      cancelled = true

      if (timer) {
        clearTimeout(timer)
      }
    }
  }, [ctx, pendingTierId])

  const applying = result?.ok && applyState === 'applying'
  const timedOut = result?.ok && applyState === 'timed_out'

  const message = timedOut
    ? getTranslations().subscription.yourUpgradeSucceededAndIsStillApplying
    : (result?.message ?? '')

  const openRecovery = () => {
    if (recoveryUrl) {
      ctx.openPortal(recoveryUrl)
    }

    return onClose()
  }

  const rows: MenuRowSpec[] = recoveryUrl
    ? [
        { color: t.color.accent, label: getTranslations().subscription.openThePortalToFinish, run: openRecovery },
        { label: getTranslations().subscription.close, run: onClose }
      ]
    : [{ label: getTranslations().subscription.close, run: onClose }]

  const sel = useMenu(rows, onClose)

  return (
    <Box flexDirection="column">
      <Text bold color={result?.ok ? t.color.ok : t.color.warn}>
        {applying
          ? getTranslations().subscription.applying
          : timedOut
            ? getTranslations().subscription.stillApplying
            : result?.ok
              ? getTranslations().subscription.done
              : getTranslations().subscription.couldNotComplete}
      </Text>
      <Text color={t.color.text}>{message}</Text>
      {result?.ok && !applying && !timedOut && (
        <Text color={t.color.muted}>{getTranslations().subscription.reRunSubscriptionAnytimeToReviewIt}</Text>
      )}
      <Text />
      {rows.map((row, i) => (
        <ActionRow active={sel === i} color={row.color} key={row.label} label={row.label} t={t} />
      ))}
      <Text />
      {footer(getTranslations().subscription.selectEnterEscClose, t)}
    </Box>
  )
}

// ── Screen: Step-up (allow remote spending inline, then replay) ───────

function StepUpScreen({ onPatch, overlay, t }: ScreenProps) {
  const { ctx } = overlay
  const retry: null | SubscriptionStepUpRetry = overlay.stepUpRetry ?? null
  const [phase, setPhase] = useState<'granted' | 'prompt' | 'resuming' | 'waiting'>('prompt')
  const startedRef = useRef(false)
  // Set when the user cancels while the browser grant is still in flight. The
  // grant's late `.then` MUST NOT fire the held change after a cancel — otherwise
  // a cancel-then-approve charges the card the user just declined.
  const abortedRef = useRef(false)
  // Guards the post-grant replay from double-firing (double-Enter on the default
  // 'Continue' row) — mirrors billingOverlay.resume()'s phase flip.
  const resumingRef = useRef(false)

  const enable = () => {
    if (startedRef.current) {
      return
    }

    startedRef.current = true
    setPhase('waiting')
    void ctx.requestRemoteSpending().then(res => {
      if (abortedRef.current) {
        return
      }

      if (res.granted) {
        // HOLD — do not auto-fire the held change. Require an explicit Continue so
        // a cancelled/late grant can never charge (mirrors billingOverlay's
        // 'granted' phase). The user already consented at confirm; this reconfirms.
        return setPhase('granted')
      }

      // Typed denial (session_revoked / remote_spending_revoked / rate_limited /
      // admin-approval) → the right recovery copy, not a flat "admin must allow".
      onPatch({ result: stepUpDenialResult(res), screen: 'result', stepUpRetry: null })
    })
  }

  const resume = () => {
    // Fire the held replay at most once. Without this, a double-Enter on the
    // default 'Continue' row sends two mutations (the upgrade dedups on the shared
    // idempotency key, but schedule/cancel/resume replays carry none).
    if (resumingRef.current || phase !== 'granted') {
      return
    }

    resumingRef.current = true
    setPhase('resuming')
    onPatch({ stepUpRetry: null })

    if (!retry) {
      return onPatch({ screen: 'overview' })
    }

    // allowStepUp=false: a repeat scope denial surfaces a result, never a frozen
    // re-entry into this (already-mounted) stepup screen.
    if (retry.kind === 'preview') {
      return void previewAndRoute(ctx, retry.tierId, onPatch, false)
    }

    if (retry.kind === 'resume') {
      return void resumeAndRoute(ctx, onPatch, false)
    }

    return void applyPendingAndRoute(ctx, overlay.pending ?? null, onPatch, false)
  }

  const back = () => {
    // Once a replay is firing, block abandon — the mutation/charge is in flight and
    // re-mounting confirm would let a second submit through.
    if (resumingRef.current) {
      return
    }

    // Abandon. If a grant is in flight, mark it aborted so its .then no-ops (no
    // un-consented charge); if already granted, just leave without replaying.
    abortedRef.current = true
    onPatch({ screen: retry?.kind === 'apply' ? 'confirm' : 'overview', stepUpRetry: null })
  }

  const rows: MenuRowSpec[] =
    phase === 'granted'
      ? [
          {
            color: t.color.ok,
            label:
              retry?.kind === 'apply'
                ? getTranslations().subscription.continueTheChange
                : getTranslations().subscription.continue,
            run: resume
          },
          { label: getTranslations().subscription.cancel, run: back }
        ]
      : phase === 'prompt'
        ? [
            { color: t.color.ok, label: getTranslations().subscription.allowRemoteSpending, run: enable },
            { label: getTranslations().subscription.cancel, run: back }
          ]
        : []

  const sel = useMenu(rows, back)

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {getTranslations().subscription.remoteSpending}
      </Text>
      {phase === 'prompt' && (
        <>
          <Text color={t.color.text}>{getTranslations().subscription.changingYourPlanNeedsRemoteSpendingAllowed}</Text>
          <Text color={t.color.muted}>{getTranslations().subscription.someoneWithBillingPermissionsOwnerAdminOr}</Text>
        </>
      )}
      {phase === 'waiting' && (
        <Text color={t.color.muted}>{getTranslations().subscription.openingYourBrowserToApproveFinishThere}</Text>
      )}
      {phase === 'granted' && (
        <Text color={t.color.ok}>{getTranslations().subscription.remoteSpendingAllowedContinueToFinishYour}</Text>
      )}
      {phase === 'resuming' && <Text color={t.color.muted}>{getTranslations().subscription.applyingYourChange}</Text>}
      <Text />
      {rows.map((row, i) => (
        <ActionRow active={sel === i} color={row.color} key={row.label} label={row.label} t={t} />
      ))}
      <Text />
      {footer(
        phase === 'waiting'
          ? getTranslations().subscription.waitingForApprovalEscToCancel
          : phase === 'resuming'
            ? getTranslations().subscription.working
            : getTranslations().subscription.selectEnterEscBack,
        t
      )}
    </Box>
  )
}

// ── Screen: Team context (no tier picker — teams use shared credits) ──

interface TeamContextScreenProps {
  onClose: () => void
  s: SubscriptionStateResponse
  t: Theme
}

function TeamContextScreen({ onClose, s, t }: TeamContextScreenProps) {
  useInput((_ch, key) => {
    if (key.escape || key.return) {
      return onClose()
    }
  })

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {getTranslations().subscription.teamSubscription}
      </Text>
      {s.org_name && (
        <Text color={t.color.muted}>
          {getTranslations().subscription.org}
          {s.org_name}
          {s.role ? ` · ${s.role}` : ''}
        </Text>
      )}
      <Text />
      <Text color={t.color.text}>
        {getTranslations().subscription.teamBalance(s.org_name ?? getTranslations().subscription.aTeamOrg)}
      </Text>
      <Text color={t.color.muted}>{getTranslations().subscription.personalSubscriptionsLiveOnYourPersonalAccount}</Text>

      <Text />
      {footer(getTranslations().subscription.enterEscClose, t)}
    </Box>
  )
}
