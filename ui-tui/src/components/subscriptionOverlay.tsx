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
import { type I18nApi, type Locale, useI18n } from '../i18n/index.js'
import type { Theme } from '../theme.js'

import { ActionRow, footer, MenuRow, type MenuRowSpec, UsageBars, useMenu } from './overlayPrimitives.js'

const UPGRADE_CONFIRM_INTERVAL_MS = 2000
const UPGRADE_CONFIRM_ATTEMPTS = 15
type Translator = I18nApi['t']

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
 * when a mutation needs terminal billing. Downgrades / cancellations / resume are
 * chargeless; an upgrade charges the card on the subscription, and an SCA/decline
 * is handed off to the portal. Starting a NEW subscription still deep-links (needs
 * a fresh card). All RPCs live in subscription.ts, reached via `overlay.ctx`.
 */
export function SubscriptionOverlay({ onClose, onPatch, overlay, t }: SubscriptionOverlayProps) {
  const { screen, state: s } = overlay
  const { locale, t: tr } = useI18n()

  // Teams have no personal subscription — dead-end to /topup, no picker.
  if (s.context === 'team') {
    return (
      <Box borderColor={t.color.accent} borderStyle="round" flexDirection="column" paddingX={1}>
        <TeamContextScreen onClose={onClose} s={s} t={t} tr={tr} />
      </Box>
    )
  }

  return (
    <Box borderColor={t.color.accent} borderStyle="round" flexDirection="column" paddingX={1}>
      {screen === 'picker' && (
        <PickerScreen locale={locale} onClose={onClose} onPatch={onPatch} overlay={overlay} t={t} tr={tr} />
      )}
      {screen === 'confirm' && (
        <ConfirmScreen locale={locale} onClose={onClose} onPatch={onPatch} overlay={overlay} t={t} tr={tr} />
      )}
      {screen === 'result' && <ResultScreen locale={locale} onClose={onClose} overlay={overlay} t={t} tr={tr} />}
      {screen === 'stepup' && (
        <StepUpScreen locale={locale} onClose={onClose} onPatch={onPatch} overlay={overlay} t={t} tr={tr} />
      )}
      {screen === 'overview' && (
        <OverviewScreen locale={locale} onClose={onClose} onPatch={onPatch} overlay={overlay} t={t} tr={tr} />
      )}
    </Box>
  )
}

// ── Shared helpers ───────────────────────────────────────────────────

interface ScreenProps {
  locale: Locale
  onClose: () => void
  onPatch: (next: Partial<SubscriptionOverlayState>) => void
  overlay: SubscriptionOverlayState
  t: Theme
  tr: Translator
}

/** ISO datetime → YYYY-MM-DD for display, or a soft fallback. */
function shortDate(tr: Translator, iso?: null | string): string {
  return iso && iso.length >= 10 ? iso.slice(0, 10) : tr('subscription.periodEnd')
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
function errorResult(
  tr: Translator,
  r: { error?: string; message?: string; portal_url?: null | string } | null
): SubscriptionResult {
  const messageByCode: Partial<Record<string, string>> = {
    cli_billing_disabled: tr('billing.error.cliBillingDisabled'),
    consent_required: tr('billing.error.consentRequired'),
    org_access_denied: tr('billing.error.orgAccessDenied'),
    rate_limited: tr('subscription.stepUp.rateLimited'),
    remote_spending_disabled: tr('billing.error.cliBillingDisabled'),
    remote_spending_revoked: tr('subscription.stepUp.spendingRevoked'),
    role_required: tr('billing.error.roleRequired'),
    session_revoked: tr('subscription.stepUp.sessionRevoked'),
    upgrade_cap_exceeded: tr('billing.error.upgradeCapExceeded')
  }

  return {
    message: (r?.error && messageByCode[r.error]) || tr('subscription.error.generic'),
    ok: false,
    recoveryUrl: r?.portal_url ?? null
  }
}

/** Map a chargeless pending-change mutation (schedule / cancel / resume). */
function mutationResult(
  tr: Translator,
  r: null | { message?: string; ok?: boolean },
  okMessage: string
): SubscriptionResult {
  return r?.ok ? { message: okMessage, ok: true } : errorResult(tr, r)
}

/** Map an upgrade response, routing SCA / decline to a portal recovery. */
function upgradeResult(
  tr: Translator,
  r: null | SubscriptionUpgradeResponse,
  pendingTierId?: null | string
): SubscriptionResult {
  if (!r) {
    // null = a transport failure (WS drop / request timeout) on the CHARGING
    // route — NAS may have already prorated + charged. Report it as ambiguous and
    // steer to a safe re-check, never a blind retry (which #2's dedup can't cover
    // once the key is lost).
    return {
      message: tr('subscription.result.upgradeUnconfirmed'),
      ok: false
    }
  }

  if (r.reason === 'authentication_required' || r.reason === 'subscription_payment_intent_requires_action') {
    return {
      message: tr('subscription.result.verifyCard'),
      ok: false,
      recoveryUrl: r.recovery_url ?? null
    }
  }

  if (r.reason === 'card_declined') {
    return {
      message: tr('subscription.result.cardDeclinedPortal'),
      ok: false,
      recoveryUrl: r.recovery_url ?? null
    }
  }

  if (r.ok && r.status === 'already_on_tier') {
    return {
      message: tr('subscription.result.alreadyOnTier', {
        tier: r.target_tier_name ?? tr('subscription.thisPlan')
      }),
      ok: true
    }
  }

  if (r.ok && r.status === 'upgraded') {
    return {
      message: tr('subscription.result.upgraded', {
        tier: r.target_tier_name ?? tr('subscription.yourNewPlan')
      }),
      ok: true,
      pendingTierId: pendingTierId ?? null
    }
  }

  if (r.status === 'requires_action') {
    return {
      message: tr('subscription.result.requiresAction'),
      ok: false,
      recoveryUrl: r.recovery_url ?? null
    }
  }

  if (r.status === 'payment_failed') {
    return {
      message: tr('subscription.result.paymentFailed'),
      ok: false,
      recoveryUrl: r.recovery_url ?? null
    }
  }

  return errorResult(tr, r)
}

/** Map a failed terminal-billing step-up to the right recovery copy (typed). */
function stepUpDenialResult(tr: Translator, res: { error?: string; message?: string }): SubscriptionResult {
  if (res.error === 'session_revoked') {
    return { message: tr('subscription.stepUp.sessionRevoked'), ok: false }
  }

  if (res.error === 'remote_spending_revoked') {
    return { message: tr('subscription.stepUp.spendingRevoked'), ok: false }
  }

  if (res.error === 'rate_limited') {
    return { message: tr('subscription.stepUp.rateLimited'), ok: false }
  }

  return {
    message: tr('subscription.stepUp.notEnabled'),
    ok: false
  }
}

// ── Scope-aware routing (shared by the picker, confirm, overview + step-up) ──

// A REPEAT scope denial during a post-grant replay must NOT route back to the
// stepup screen: we're already mounted there (in the 'resuming' phase), so an
// onPatch({screen:'stepup'}) is a no-op that never remounts → the screen freezes.
// Post-grant replays pass allowStepUp=false and surface this instead (mirrors the
// CLI's allow_stepup=False cap).
const scopeStillDeniedResult = (tr: Translator): SubscriptionResult => ({
  message: tr('subscription.stepUp.stillDenied'),
  ok: false
})

/** Preview a tier and route: confirm (ok), stepup (scope), or result (other error). */
function previewAndRoute(
  ctx: SubscriptionOverlayState['ctx'],
  tierId: string,
  onPatch: ScreenProps['onPatch'],
  tr: Translator,
  allowStepUp = true
): Promise<void> {
  return ctx.preview(tierId).then(p => {
    if (!p) {
      return onPatch({ result: { message: tr('subscription.error.previewFailed'), ok: false }, screen: 'result' })
    }

    if (!p.ok) {
      if (isScopeDenial(p)) {
        return allowStepUp
          ? onPatch({ screen: 'stepup', stepUpRetry: { kind: 'preview', tierId } })
          : onPatch({ result: scopeStillDeniedResult(tr), screen: 'result' })
      }

      return onPatch({ result: errorResult(tr, p), screen: 'result' })
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
  tr: Translator,
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
      : onPatch({ result: scopeStillDeniedResult(tr), screen: 'result' })

  const finish = (result: SubscriptionResult) => onPatch({ result, screen: 'result' })

  if (pending.kind === 'cancellation') {
    return ctx
      .scheduleCancellation()
      .then(r =>
        isScopeDenial(r) ? toStepUp() : finish(mutationResult(tr, r, tr('subscription.result.cancellationScheduled')))
      )
  }

  if (pending.kind === 'upgrade') {
    return ctx
      .upgrade(pending.targetTierId ?? '', pending.idempotencyKey)
      .then(r => (isScopeDenial(r) ? toStepUp() : finish(upgradeResult(tr, r, pending.targetTierId))))
  }

  return ctx
    .scheduleChange(pending.targetTierId ?? '')
    .then(r =>
      isScopeDenial(r) ? toStepUp() : finish(mutationResult(tr, r, tr('subscription.result.changeScheduled')))
    )
}

/** Resume (undo the pending change) and route: result (ok/err) or stepup (scope). */
function resumeAndRoute(
  ctx: SubscriptionOverlayState['ctx'],
  onPatch: ScreenProps['onPatch'],
  tr: Translator,
  allowStepUp = true
): Promise<void> {
  return ctx.resume().then(r => {
    if (isScopeDenial(r)) {
      return allowStepUp
        ? onPatch({ screen: 'stepup', stepUpRetry: { kind: 'resume' } })
        : onPatch({ result: scopeStillDeniedResult(tr), screen: 'result' })
    }

    return onPatch({ result: mutationResult(tr, r, tr('subscription.result.changeUndone')), screen: 'result' })
  })
}

// ── The pending scheduled change (drives the banner + status echo) ──

interface PendingTransition {
  to: string
  when: string
}

/** The scheduled downgrade/cancel as a from→to transition, or null. */
function pendingTransition(c: SubscriptionStateResponse['current'], tr: Translator): null | PendingTransition {
  if (!c) {
    return null
  }

  if (c.cancel_at_period_end) {
    return {
      to: tr('subscription.cancels'),
      when: c.cancellation_effective_display ?? shortDate(tr, c.cancellation_effective_at)
    }
  }

  if (c.pending_downgrade_tier_name) {
    return {
      to: c.pending_downgrade_tier_name,
      when: c.pending_downgrade_display ?? shortDate(tr, c.pending_downgrade_at)
    }
  }

  return null
}

// ── Screen: Overview (plan + usage + entry to the change flow) ────────

/** Status line — dollars-only, and echoes a pending "Ultra → Plus" transition. */
function statusLine(s: SubscriptionStateResponse, tr: Translator): string {
  const u = s.usage
  const c = s.current
  const plan = c?.tier_name ?? u?.plan_name ?? null
  const trans = pendingTransition(c, tr)
  const flip = plan && trans ? ` → ${trans.to}` : ''
  const renewsRaw = u?.renews_display ?? null
  const renews = renewsRaw ? ` · ${tr('usage.renews', { date: renewsRaw })}` : ''
  const viewOnly = !s.can_change_plan

  if (!plan) {
    return tr('subscription.status.free')
  }

  if (u?.status === 'low' && u.total_spendable_display) {
    return tr('subscription.status.low', { balance: u.total_spendable_display, plan: `${plan}${flip}` })
  }

  const left = u?.total_spendable_display
    ? ` · ${tr('subscription.status.left', { balance: u.total_spendable_display })}`
    : ''

  return tr('subscription.status.plan', {
    detail: viewOnly ? ` · ${tr('subscription.status.viewOnly')}` : renews,
    left,
    plan: `${plan}${flip}`
  })
}

function OverviewScreen({ locale, onClose, onPatch, overlay, t, tr }: ScreenProps) {
  const { ctx, state: s } = overlay
  const c = s.current
  const isFree = !c?.tier_id
  const currentName = c?.tier_name ?? tr('subscription.yourPlan')
  const trans = pendingTransition(c, tr)
  const hasPendingChange = !!trans
  // Admin/owner on a personal paid plan can change it in-terminal; otherwise the
  // portal enforces who can act (members) / starting a new sub needs a card.
  const canChange = s.can_change_plan && !isFree

  // Guard the async resume so a double-press cannot fire two DELETEs mid-await.
  const busyRef = useRef(false)

  const u = s.usage
  const freeNudge = isFree ? tr('subscription.overview.freeNudge') : null

  const lowNudge =
    u?.status === 'low'
      ? tr('subscription.overview.lowNudge', {
          balance: u.total_spendable_display ?? tr('usage.underFive')
        })
      : null

  const doManage = () => {
    if (s.portal_url) {
      void ctx.openManageLink()
    } else {
      ctx.sys(tr('subscription.error.noPortalUrl'))
    }

    return onClose()
  }

  const doResume = () => {
    if (busyRef.current) {
      return
    }

    busyRef.current = true
    void resumeAndRoute(ctx, onPatch, tr)
  }

  const rows: MenuRowSpec[] = []

  if (canChange) {
    // When a change is already scheduled, undo is the most likely next intent —
    // promote it to the first, highlighted action.
    if (hasPendingChange) {
      rows.push({
        color: t.color.ok,
        label: tr('subscription.action.keepPlan', { plan: currentName }),
        run: doResume
      })
      rows.push({
        label: tr('subscription.action.changePlan'),
        run: () => onPatch({ pending: null, screen: 'picker' })
      })
    } else {
      rows.push({
        label: tr('subscription.action.changePlan'),
        run: () => onPatch({ pending: null, screen: 'picker' })
      })
      rows.push({
        label: tr('subscription.action.cancelSubscription'),
        run: () => onPatch({ pending: { kind: 'cancellation', preview: null, targetTierId: null }, screen: 'confirm' })
      })
    }
  }

  rows.push({
    label: isFree ? tr('subscription.action.startSubscription') : tr('billing.action.managePortal'),
    run: doManage
  })
  rows.push({ label: tr('common.close'), run: onClose })

  const sel = useMenu(rows, onClose)

  return (
    <Box flexDirection="column">
      {/* Lead with the scheduled change so it can't read as "nothing happened". */}
      {trans && (
        <Box flexDirection="column" marginBottom={1}>
          <Text bold color={t.color.warn}>
            {tr('subscription.overview.scheduledChange')}
          </Text>
          <Box>
            <Text color={t.color.text}>{currentName} </Text>
            <Text color={t.color.warn}>──▶ </Text>
            <Text color={t.color.text}>{trans.to}</Text>
            <Text color={t.color.muted}> · {trans.when}</Text>
          </Box>
          <Text color={t.color.muted}>{tr('subscription.overview.keepUntilThen', { plan: currentName })}</Text>
        </Box>
      )}

      <Text bold color={t.color.accent}>
        {statusLine(s, tr)}
      </Text>
      <UsageBars locale={locale} model={s.usage} t={t} />
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
          {tr('billing.orgLine', { org: s.org_name })}
          {s.role ? ` · ${s.role}` : ''}
        </Text>
      )}

      <Text />
      {rows.map((row, i) => (
        <MenuRow active={sel === i} index={i + 1} key={row.label} label={row.label} t={t} />
      ))}

      <Text />
      {footer(tr('subscription.footer.selectClose'), t)}
    </Box>
  )
}

// ── Screen: Picker (choose a tier → preview → confirm) ───────────────

function PickerScreen({ onPatch, overlay, t, tr }: ScreenProps) {
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
    void previewAndRoute(ctx, tier.tier_id, onPatch, tr)
  }

  const back = () => onPatch({ screen: 'overview' })

  const rows: MenuRowSpec[] = choices.map(tier => {
    const direction = tr(
      tier.tier_order > currentOrder ? 'subscription.picker.upgrade' : 'subscription.picker.downgrade'
    )

    return {
      label: tr('subscription.picker.tierLine', {
        direction,
        price: tier.dollars_per_month_display,
        tier: tier.name
      }),
      run: () => pick(tier)
    }
  })

  rows.push({ label: tr('common.back'), run: back })

  const sel = useMenu(rows, back)

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {tr('subscription.action.changePlan')}
      </Text>
      <Text color={t.color.muted}>
        {tr('subscription.picker.current', { tier: s.current?.tier_name ?? tr('subscription.free') })}
      </Text>
      <Text />
      {choices.length === 0 && <Text color={t.color.muted}>{tr('subscription.picker.noPlans')}</Text>}
      {rows.map((row, i) => (
        <MenuRow active={sel === i} index={i + 1} key={row.label} label={row.label} t={t} />
      ))}
      <Text />
      {footer(tr('subscription.footer.previewBack'), t)}
    </Box>
  )
}

// ── Screen: Confirm (show the previewed effect, then apply) ──────────

function ConfirmScreen({ onClose, onPatch, overlay, t, tr }: ScreenProps) {
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
    void applyPendingAndRoute(ctx, pending, onPatch, tr)
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
  const targetName = isCancellation ? null : (preview?.target_tier_name ?? tr('subscription.selectedPlan'))

  const blockedReason =
    preview?.reason === 'Retract the cancellation before upgrading.'
      ? tr('subscription.confirm.retractCancellation')
      : tr('subscription.confirm.blocked')

  let primary: MenuRowSpec | null = null

  if (isCancellation) {
    primary = { color: t.color.warn, label: tr('subscription.action.cancelSubscription'), run: apply }
  } else if (effect === 'charge_now') {
    primary = {
      color: t.color.ok,
      label: amount ? tr('subscription.confirm.payUpgrade', { amount }) : tr('subscription.confirm.upgradeProrated'),
      run: apply
    }
  } else if (effect === 'scheduled') {
    primary = {
      color: t.color.ok,
      label: tr('subscription.confirm.scheduleChange', { tier: targetName ?? tr('subscription.selectedPlan') }),
      run: apply
    }
  } else if (effect === 'blocked') {
    primary = { label: tr('billing.action.managePortal'), run: manage }
  }

  const rows: MenuRowSpec[] = primary
    ? [primary, { label: tr('common.back'), run: back }]
    : [{ label: tr('common.back'), run: back }]

  const sel = useMenu(rows, back)

  // Chip contrasts an immediate charge vs a period-end schedule at a glance.
  const chip =
    effect === 'charge_now'
      ? { color: t.color.ok, label: tr('subscription.confirm.chargedNow') }
      : effect === 'scheduled'
        ? { color: t.color.warn, label: tr('subscription.confirm.scheduledNotToday') }
        : null

  return (
    <Box flexDirection="column">
      <Box>
        <Text bold color={t.color.accent}>
          {isCancellation ? tr('subscription.confirm.cancellationTitle') : tr('subscription.confirm.changeTitle')}
        </Text>
        {chip && <Text color={chip.color}> · {chip.label}</Text>}
      </Box>
      {submitting && <Text color={t.color.muted}>{tr('common.working')}</Text>}

      {isCancellation && (
        <>
          <Text color={t.color.text}>
            {tr('subscription.confirm.cancelDetail', {
              date: shortDate(tr, s.current?.cycle_ends_at),
              plan: s.current?.tier_name ?? tr('subscription.yourPlan')
            })}
          </Text>
          <Text color={t.color.muted}>{tr('subscription.confirm.cancelCredits')}</Text>
        </>
      )}

      {effect === 'charge_now' && !isCancellation && (
        <>
          <Text color={t.color.text}>
            {amount
              ? tr('subscription.confirm.upgradeDetailAmount', { amount, tier: targetName ?? '' })
              : tr('subscription.confirm.upgradeDetailProrated', { tier: targetName ?? '' })}
          </Text>
          {preview?.monthly_credits_delta && (
            <Text color={t.color.muted}>
              {tr('subscription.confirm.monthlyChange', { change: preview.monthly_credits_delta })}
            </Text>
          )}
          <Text color={t.color.muted}>
            {chargeCard
              ? tr('subscription.confirm.cardChargedNamed', { card: chargeCard })
              : tr('subscription.confirm.cardCharged')}
          </Text>
        </>
      )}

      {effect === 'scheduled' && !isCancellation && (
        <>
          <Text color={t.color.text}>
            {tr('subscription.confirm.scheduledDetail', {
              date: shortDate(tr, preview?.effective_at),
              tier: targetName ?? ''
            })}
          </Text>
          {preview?.monthly_credits_delta && (
            <Text color={t.color.muted}>
              {tr('subscription.confirm.monthlyChange', { change: preview.monthly_credits_delta })}
            </Text>
          )}
        </>
      )}

      {effect === 'no_op' && !isCancellation && (
        <Text color={t.color.muted}>{tr('subscription.confirm.noOp', { tier: targetName ?? '' })}</Text>
      )}

      {effect === 'blocked' && !isCancellation && <Text color={t.color.warn}>{blockedReason}</Text>}

      <Text />
      {rows.map((row, i) => (
        <ActionRow active={sel === i} color={row.color} key={row.label} label={row.label} t={t} />
      ))}
      <Text />
      {footer(tr('subscription.footer.confirmBack'), t)}
    </Box>
  )
}

// ── Screen: Result (outcome + optional portal recovery) ──────────────

function ResultScreen({ onClose, overlay, t, tr }: Omit<ScreenProps, 'onPatch'>) {
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
  const message = timedOut ? tr('subscription.result.stillApplyingMessage') : (result?.message ?? '')

  const openRecovery = () => {
    if (recoveryUrl) {
      ctx.openPortal(recoveryUrl)
    }

    return onClose()
  }

  const rows: MenuRowSpec[] = recoveryUrl
    ? [
        { color: t.color.accent, label: tr('subscription.action.openPortalFinish'), run: openRecovery },
        { label: tr('common.close'), run: onClose }
      ]
    : [{ label: tr('common.close'), run: onClose }]

  const sel = useMenu(rows, onClose)

  return (
    <Box flexDirection="column">
      <Text bold color={result?.ok ? t.color.ok : t.color.warn}>
        {applying
          ? tr('subscription.result.applying')
          : timedOut
            ? tr('subscription.result.stillApplying')
            : result?.ok
              ? tr('common.done')
              : tr('subscription.result.couldNotComplete')}
      </Text>
      <Text color={t.color.text}>{message}</Text>
      {result?.ok && !applying && !timedOut && (
        <Text color={t.color.muted}>{tr('subscription.result.reviewAgain')}</Text>
      )}
      <Text />
      {rows.map((row, i) => (
        <ActionRow active={sel === i} color={row.color} key={row.label} label={row.label} t={t} />
      ))}
      <Text />
      {footer(tr('subscription.footer.selectClose'), t)}
    </Box>
  )
}

// ── Screen: Step-up (grant terminal billing inline, then replay) ──────

function StepUpScreen({ onPatch, overlay, t, tr }: ScreenProps) {
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
      onPatch({ result: stepUpDenialResult(tr, res), screen: 'result', stepUpRetry: null })
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
      return void previewAndRoute(ctx, retry.tierId, onPatch, tr, false)
    }

    if (retry.kind === 'resume') {
      return void resumeAndRoute(ctx, onPatch, tr, false)
    }

    return void applyPendingAndRoute(ctx, overlay.pending ?? null, onPatch, tr, false)
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
            label: retry?.kind === 'apply' ? tr('subscription.stepUp.continueChange') : tr('common.continue'),
            run: resume
          },
          { label: tr('common.cancel'), run: back }
        ]
      : phase === 'prompt'
        ? [
            { color: t.color.ok, label: tr('billing.stepUp.enableAction'), run: enable },
            { label: tr('common.cancel'), run: back }
          ]
        : []

  const sel = useMenu(rows, back)

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {tr('subscription.stepUp.title')}
      </Text>
      {phase === 'prompt' && (
        <>
          <Text color={t.color.text}>{tr('subscription.stepUp.prompt')}</Text>
          <Text color={t.color.muted}>{tr('subscription.stepUp.approver')}</Text>
        </>
      )}
      {phase === 'waiting' && <Text color={t.color.muted}>{tr('subscription.stepUp.waiting')}</Text>}
      {phase === 'granted' && <Text color={t.color.ok}>{tr('subscription.stepUp.granted')}</Text>}
      {phase === 'resuming' && <Text color={t.color.muted}>{tr('subscription.stepUp.applying')}</Text>}
      <Text />
      {rows.map((row, i) => (
        <ActionRow active={sel === i} color={row.color} key={row.label} label={row.label} t={t} />
      ))}
      <Text />
      {footer(
        phase === 'waiting'
          ? tr('subscription.footer.waitingCancel')
          : phase === 'resuming'
            ? tr('common.working')
            : tr('subscription.footer.selectBack'),
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
  tr: Translator
}

function TeamContextScreen({ onClose, s, t, tr }: TeamContextScreenProps) {
  useInput((_ch, key) => {
    if (key.escape || key.return) {
      return onClose()
    }
  })

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {tr('subscription.team.title')}
      </Text>
      {s.org_name && (
        <Text color={t.color.muted}>
          {tr('billing.orgLine', { org: s.org_name })}
          {s.role ? ` · ${s.role}` : ''}
        </Text>
      )}
      <Text />
      <Text color={t.color.text}>
        {tr('subscription.team.sharedBalance', { org: s.org_name ?? tr('subscription.team.defaultOrg') })}
      </Text>
      <Text color={t.color.muted}>{tr('subscription.team.personalAccount')}</Text>

      <Text />
      {footer(tr('subscription.footer.enterClose'), t)}
    </Box>
  )
}
