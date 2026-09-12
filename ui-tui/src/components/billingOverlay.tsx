import { randomUUID } from 'node:crypto'

import { Box, Text, useInput } from '@hermes/ink'
import { useRef, useState } from 'react'

import type { BillingOverlayState } from '../app/interfaces.js'
import type { BillingStateResponse } from '../gatewayTypes.js'
import { getTranslations, useTranslations } from '../i18n/index.js'
import type { Theme } from '../theme.js'

import { ActionRow, footer, MenuRow, type MenuRowSpec, UsageBars, useMenu } from './overlayPrimitives.js'
import { TextInput } from './textInput.js'

interface BillingOverlayProps {
  /** Replace the overlay slot (screen transitions + pending data). */
  onPatch: (next: Partial<BillingOverlayState>) => void
  /** Close the overlay entirely. */
  onClose: () => void
  overlay: BillingOverlayState
  t: Theme
}

function autoReloadLine(s: BillingStateResponse): null | string {
  if (!s.auto_reload) {
    return null
  }

  return s.auto_reload.enabled
    ? getTranslations().billing.autoReloadStatus(
        String(s.auto_reload.threshold_display),
        String(s.auto_reload.reload_to_display)
      )
    : getTranslations().billing.autoReloadOff
}

/**
 * The /billing modal.  A self-contained state machine:
 *   overview → buy | autoreload | limit  (and buy → confirm).
 * Esc from a sub-screen returns to overview; Esc from overview closes.
 * All RPCs + error mapping live in billing.ts and are reached through
 * `overlay.ctx` — this component only renders + routes keys.
 */
export function BillingOverlay({ onClose, onPatch, overlay, t }: BillingOverlayProps) {
  useTranslations()
  const { ctx, screen, state: s } = overlay

  return (
    <Box borderColor={t.color.accent} borderStyle="round" flexDirection="column" paddingX={1}>
      {screen === 'overview' && <OverviewScreen ctx={ctx} onClose={onClose} onPatch={onPatch} s={s} t={t} />}
      {screen === 'buy' && <BuyScreen ctx={ctx} onClose={onClose} onPatch={onPatch} s={s} t={t} />}
      {screen === 'confirm' && (
        <ConfirmScreen
          amount={overlay.pendingCharge?.amount ?? ''}
          ctx={ctx}
          idempotencyKey={overlay.pendingCharge?.idempotencyKey}
          onBack={() => onPatch({ pendingCharge: null, screen: 'buy' })}
          onClose={onClose}
          onPatch={onPatch}
          s={s}
          t={t}
        />
      )}
      {screen === 'autoreload' && <AutoReloadScreen ctx={ctx} onClose={onClose} onPatch={onPatch} s={s} t={t} />}
      {screen === 'limit' && <LimitScreen ctx={ctx} onClose={onClose} onPatch={onPatch} s={s} t={t} />}
      {screen === 'stepup' && (
        <StepUpScreen
          amount={overlay.pendingCharge?.amount ?? ''}
          ctx={ctx}
          idempotencyKey={overlay.pendingCharge?.idempotencyKey}
          onClose={onClose}
          t={t}
        />
      )}
    </Box>
  )
}

// ── Screen 1: Overview ────────────────────────────────────────────────

interface ScreenProps {
  ctx: BillingOverlayState['ctx']
  onClose: () => void
  onPatch: (next: Partial<BillingOverlayState>) => void
  s: BillingStateResponse
  t: Theme
}

function OverviewScreen({ ctx, onClose, onPatch, s, t }: ScreenProps) {
  // Full charge menu only for an admin with the org kill-switch on; otherwise it
  // collapses to Manage-on-portal / Close + a one-line note. NOTE: this is the
  // ORG-level gate (cli_billing_enabled), NOT the per-terminal remote spending scope —
  // that's discovered reactively at pay time (a charge 403s insufficient_scope
  // and the confirm screen routes into the resumable step-up). We deliberately
  // do NOT preflight the scope here.
  const full = s.is_admin && s.cli_billing_enabled

  const note = !s.is_admin
    ? getTranslations().billing.billingActionsNeedSomeoneWithBillingPermissions
    : !s.cli_billing_enabled
      ? getTranslations().billing.remoteSpendingIsOffForThisOrg
      : null

  // Always show the full billing menu for an admin/billing-on org — a missing
  // card does NOT mean nothing can be done (the org may already have balance,
  // auto-reload, a limit). The card only matters at CHARGE time: with no card
  // on file, "Add funds" opens the guided add-card path (portal + check-again)
  // instead of an amount picker that would 403 no_payment_method.
  const items = full
    ? [
        getTranslations().billing.addFunds,
        getTranslations().billing.autoReload,
        getTranslations().billing.monthlyLimit,
        getTranslations().billing.manageOnPortal,
        getTranslations().billing.cancel
      ]
    : [getTranslations().billing.manageOnPortal, getTranslations().billing.cancel]

  const choose = (i: number) => {
    if (full) {
      if (i === 0) {
        onPatch({ screen: 'buy' })
      } else if (i === 1) {
        onPatch({ screen: 'autoreload' })
      } else if (i === 2) {
        onPatch({ screen: 'limit' })
      } else {
        if (i === 3 && s.portal_url) {
          ctx.openPortal(s.portal_url)
        }

        onClose()
      }

      return
    }

    if (i === 0 && s.portal_url) {
      ctx.openPortal(s.portal_url)
    }

    onClose()
  }

  const rows: MenuRowSpec[] = items.map((label, i) => ({ label, run: () => choose(i) }))
  const sel = useMenu(rows, onClose)

  const auto = autoReloadLine(s)
  // Balance leads, in the title — the first thing seen (review feedback).
  const title = getTranslations().billing.topUpBalance(String(s.balance_display))

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {title}
      </Text>
      {s.org_name && (
        <Text color={t.color.muted}>
          {getTranslations().billing.org}
          {s.org_name}
          {s.role ? ` · ${s.role}` : ''}
        </Text>
      )}
      {/* The shared two-bar dollar usage (plan + top-up), same as /usage and
          /subscription. Renders nothing when no usage model is available. */}
      <UsageBars model={s.usage} t={t} />
      {auto && <Text color={t.color.muted}>{auto}</Text>}
      {/* Card presence at a glance: which card a charge would use (with why —
          "the card on your subscription"), or that none is saved. Only for the
          full menu — members/billing-off get the portal note instead. */}
      {full && (
        <Text color={t.color.muted}>
          {s.card
            ? getTranslations().billing.cardLabel(String(s.card.display ?? s.card.masked))
            : getTranslations().billing.noSavedCardOnFileAddFunds}
        </Text>
      )}
      {note && (
        <Box marginTop={1}>
          <Text color={t.color.warn}>{note}</Text>
        </Box>
      )}

      <Text />
      {items.map((label, i) => (
        <MenuRow active={sel === i} index={i + 1} key={label} label={label} t={t} />
      ))}

      <Text />
      {footer(getTranslations().billing.menuCloseHint(String(items.length)), t)}
    </Box>
  )
}

// ── Screen 2: Buy credits ─────────────────────────────────────────────

function BuyScreen({ ctx, onPatch, s, t }: ScreenProps) {
  const presets = s.charge_presets_display
  const rawPresets = s.charge_presets
  // No card on file → the buy screen becomes the ADD-CARD path: cards are added
  // on the portal (never in-terminal), and "check again" re-fetches state so the
  // flow continues right here once the card is saved. Card present → the normal
  // preset menu. (The card display is best-effort server-side, so "check again"
  // also recovers a transient miss.)
  const noCard = !s.card

  const rows = noCard
    ? [
        getTranslations().billing.addACardOnThePortal,
        getTranslations().billing.iVeAddedItCheckAgain,
        getTranslations().billing.back
      ]
    : [...presets, getTranslations().billing.customAmount, getTranslations().billing.cancel]

  const customIdx = presets.length

  const [sel, setSel] = useState(0)
  const [typing, setTyping] = useState(false)
  const [custom, setCustom] = useState('')
  const [error, setError] = useState<null | string>(null)
  const [checking, setChecking] = useState(false)
  // Synchronous guard: double-Enter on "check again" must not stack re-fetches.
  const checkingRef = useRef(false)

  const recheck = () => {
    if (checkingRef.current) {
      return
    }

    checkingRef.current = true
    setChecking(true)
    void ctx.refreshState().then(fresh => {
      checkingRef.current = false
      setChecking(false)

      if (!fresh) {
        return setError(getTranslations().billing.couldNotRefreshBillingStateTryAgain)
      }

      setError(null)
      // Re-render with the fresh state: if the card is now on file, this same
      // screen flips into the preset menu and the purchase continues here.
      onPatch({ state: fresh })

      if (fresh.card) {
        ctx.sys(getTranslations().billing.cardFound(String(fresh.card.display ?? fresh.card.masked)))
      } else {
        ctx.sys(getTranslations().billing.stillNoCardOnFileFinishAdding)
      }
    })
  }

  const toConfirm = (amount: string) => {
    // Mint the idempotency key here (purchase identity = this amount). It rides
    // pendingCharge into Confirm AND the step-up replay, so a retried charge
    // dedups server-side; a fresh amount selection gets a fresh key.
    onPatch({ pendingCharge: { amount, idempotencyKey: randomUUID() }, screen: 'confirm' })
  }

  const pickPreset = (i: number) => {
    // Prefer the raw (numeric) preset for the amount; fall back to stripping $.
    const raw = (rawPresets[i] ?? presets[i] ?? '').replace(/^\$/, '').trim()
    const v = ctx.validate(raw)

    if (v.error || !v.amount) {
      setError(v.error ?? getTranslations().billing.invalidPreset)

      return
    }

    toConfirm(v.amount)
  }

  const submitCustom = (raw: string) => {
    const v = ctx.validate(raw)

    if (v.error || !v.amount) {
      setError(v.error ?? getTranslations().billing.invalidAmount)

      return
    }

    toConfirm(v.amount)
  }

  const choose = (i: number) => {
    if (noCard) {
      if (i === 0) {
        if (s.portal_url) {
          ctx.openPortal(s.portal_url)
          ctx.sys(getTranslations().billing.addACardOnTheBillingPage)
        } else {
          setError(getTranslations().billing.couldNotBuildThePortalLinkIs)
        }

        return
      }

      if (i === 1) {
        return recheck()
      }

      return onPatch({ screen: 'overview' })
    }

    if (i < presets.length) {
      pickPreset(i)
    } else if (i === customIdx) {
      setError(null)
      setTyping(true)
    } else {
      onPatch({ screen: 'overview' })
    }
  }

  useInput((ch, key) => {
    if (key.escape) {
      return typing ? (setTyping(false), setError(null)) : onPatch({ screen: 'overview' })
    }

    if (typing) {
      return
    }

    if (key.upArrow && sel > 0) {
      setSel(v => v - 1)
    }

    if (key.downArrow && sel < rows.length - 1) {
      setSel(v => v + 1)
    }

    if (key.return) {
      return choose(Math.min(sel, rows.length - 1))
    }

    const n = parseInt(ch, 10)

    if (n >= 1 && n <= rows.length) {
      return choose(n - 1)
    }
  })

  // sel can go stale when a refresh flips the row set (3 add-card rows ↔ N
  // preset rows) — clamp for render + Enter.
  const cSel = Math.min(sel, rows.length - 1)

  const payLine = s.card
    ? getTranslations().billing.paymentMethod(String(s.card.display ?? s.card.masked))
    : getTranslations().billing.noSavedCardOnFile

  if (typing) {
    return (
      <Box flexDirection="column">
        <Text bold color={t.color.accent}>
          {getTranslations().billing.addFunds}
        </Text>
        <Text color={t.color.muted}>{payLine}</Text>
        <Text />
        <Text color={t.color.label}>{getTranslations().billing.enterACustomAmount}</Text>
        <Box>
          <Text color={t.color.label}>{'$'}</Text>
          <TextInput color={t.color.text} columns={20} onChange={setCustom} onSubmit={submitCustom} value={custom} />
        </Box>
        {error && <Text color={t.color.error}>{error}</Text>}
        <Text />
        {footer(getTranslations().billing.enterConfirmEscBack, t)}
      </Box>
    )
  }

  if (noCard) {
    return (
      <Box flexDirection="column">
        <Text bold color={t.color.accent}>
          {getTranslations().billing.addFunds}
        </Text>
        <Text color={t.color.text}>{getTranslations().billing.noSavedCardOnFile2}</Text>
        <Text color={t.color.muted}>{getTranslations().billing.addACardOnceOnThePortal}</Text>
        <Text />
        {rows.map((label, i) => (
          <MenuRow active={cSel === i} index={i + 1} key={label} label={label} t={t} />
        ))}
        {error && <Text color={t.color.error}>{error}</Text>}
        <Text />
        {footer(
          checking
            ? getTranslations().billing.checkingForACard
            : getTranslations().billing.menuBackHint(String(rows.length)),
          t
        )}
      </Box>
    )
  }

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {getTranslations().billing.addFunds}
      </Text>
      <Text color={t.color.muted}>{payLine}</Text>
      <Text />
      {rows.map((label, i) => (
        <MenuRow active={cSel === i} index={i + 1} key={label} label={label} t={t} />
      ))}
      {error && <Text color={t.color.error}>{error}</Text>}
      <Text />
      {footer(getTranslations().billing.menuBackHint(String(rows.length)), t)}
    </Box>
  )
}

// ── Screen 3: Confirm purchase ────────────────────────────────────────

function ConfirmScreen({
  amount,
  ctx,
  idempotencyKey,
  onBack,
  onClose,
  onPatch,
  s,
  t
}: {
  amount: string
  ctx: BillingOverlayState['ctx']
  idempotencyKey?: string
  onBack: () => void
  onClose: () => void
  onPatch: (next: Partial<BillingOverlayState>) => void
  s: BillingStateResponse
  t: Theme
}) {
  // rows: Pay $X now / Cancel
  const [sel, setSel] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  // Synchronous guard: two key events can both observe `submitting === false`
  // before React commits the state update, double-firing the charge (and the
  // gateway mints a fresh idempotency key per call → two charges).
  const submittingRef = useRef(false)

  const pay = () => {
    if (submittingRef.current || submitting) {
      return
    }

    submittingRef.current = true
    setSubmitting(true)
    void ctx.charge(amount, idempotencyKey).then(outcome => {
      if (outcome === 'needs_remote_spending') {
        // Resumable step-up: keep the modal MOUNTED, switch to the stepup
        // screen (which holds pendingCharge.amount for the post-grant replay).
        onPatch({ screen: 'stepup' })

        return
      }

      // submitted (settlement reported via transcript) or error (already
      // surfaced) → close the overlay. The transcript carries the outcome.
      onClose()
    })
  }

  const back = () => onBack()

  useInput((ch, key) => {
    if (key.escape) {
      return back()
    }

    const lower = ch.toLowerCase()

    if (lower === 'y') {
      return pay()
    }

    if (lower === 'n') {
      return back()
    }

    if (key.upArrow) {
      setSel(0)
    }

    if (key.downArrow) {
      setSel(1)
    }

    if (key.return) {
      return sel === 0 ? pay() : back()
    }
  })

  const payLine = s.card
    ? getTranslations().billing.paymentMethod(String(s.card.display ?? s.card.masked))
    : getTranslations().billing.noSavedCardOnFile

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {getTranslations().billing.confirmPurchase}
      </Text>
      <Text color={t.color.text}>
        {getTranslations().billing.total}
        {amount}
      </Text>
      <Text color={t.color.muted}>{payLine}</Text>
      {/* Provenance-less payloads (older NAS) keep the generic line; when the
          resolver says WHY this card, payLine already carries it. */}
      {s.card && !s.card.resolved_via && (
        <Text color={t.color.muted}>{getTranslations().billing.yourCardSavedOnThePortalWill}</Text>
      )}
      <Text color={t.color.muted}>{getTranslations().billing.byConfirmingYouAllowNousResearchTo}</Text>
      <Text />
      <ActionRow active={sel === 0} color={t.color.ok} label={getTranslations().billing.payNow(String(amount))} t={t} />
      <ActionRow active={sel === 1} label={getTranslations().billing.cancel} t={t} />
      <Text />
      {footer(getTranslations().billing.selectEnterConfirmYNQuickEsc, t)}
    </Box>
  )
}

// ── Screen: Step-up (resumable "Allow Remote Spending") ─────────────
// Reached ONLY when a charge returns insufficient_scope — there is no preflight
// or scope check anywhere; the buy path discovers it reactively. The modal stays
// MOUNTED through the browser device-flow:
//   prompt (heads-up) → waiting (browser authorize) → granted (press Enter to
//   resume) → replay the held charge (pendingCharge.amount) → settle → close.
// Never leaks the raw billing:manage scope — the user-facing concept is
// "Remote Spending".

function StepUpScreen({
  amount,
  ctx,
  idempotencyKey,
  onClose,
  t
}: {
  amount: string
  ctx: BillingOverlayState['ctx']
  idempotencyKey?: string
  onClose: () => void
  t: Theme
}) {
  const [sel, setSel] = useState(0)
  const [phase, setPhase] = useState<'granted' | 'prompt' | 'resuming' | 'waiting'>('prompt')

  const allow = () => {
    if (phase !== 'prompt') {
      return
    }

    setPhase('waiting')
    ctx.sys(getTranslations().billing.openingYourBrowserToAllowRemoteSpending)

    void ctx.requestRemoteSpending().then(granted => {
      if (!granted) {
        ctx.sys(getTranslations().billing.couldnTAllowRemoteSpendingSomeoneWith)
        onClose()

        return
      }

      // Granted → hold here and wait for an explicit Enter to resume the held
      // purchase (the reassuring "you're back, press Enter" beat).
      setPhase('granted')
    })
  }

  const resume = () => {
    if (phase !== 'granted') {
      return
    }

    setPhase('resuming')
    ctx.sys(getTranslations().billing.remoteSpendingAllowedResumingYourPurchase)
    void ctx.charge(amount, idempotencyKey).then(outcome => {
      // If the replay STILL can't spend (grant raced/expired or downscoped),
      // say so — don't close on a reassuring line with no charge made.
      if (outcome === 'needs_remote_spending') {
        ctx.sys(getTranslations().billing.remoteSpendingStillNeedsApprovalRunTopup)
      }

      onClose()
    })
  }

  const decline = () => {
    ctx.sys(getTranslations().billing.noChargeMadeRunTopupWhenYou)
    onClose()
  }

  useInput((ch, key) => {
    if (phase === 'waiting' || phase === 'resuming') {
      // While the device flow / replay runs, only Esc (give up) is live.
      if (key.escape) {
        onClose()
      }

      return
    }

    if (phase === 'granted') {
      // Back from the browser — Enter resumes, Esc abandons.
      if (key.escape) {
        return onClose()
      }

      if (key.return) {
        return resume()
      }

      return
    }

    // phase === 'prompt'
    if (key.escape) {
      return decline()
    }

    const lower = ch.toLowerCase()

    if (lower === 'y') {
      return allow()
    }

    if (lower === 'n') {
      return decline()
    }

    if (key.upArrow) {
      setSel(0)
    }

    if (key.downArrow) {
      setSel(1)
    }

    if (key.return) {
      return sel === 0 ? allow() : decline()
    }
  })

  if (phase === 'waiting') {
    return (
      <Box flexDirection="column">
        <Text bold color={t.color.accent}>
          {getTranslations().billing.allowRemoteSpending}
        </Text>
        <Text color={t.color.warn}>{getTranslations().billing.waitingForYourBrowser}</Text>
        <Text color={t.color.muted}>{getTranslations().billing.approveInThePageThatJustOpened}</Text>
        <Text color={t.color.muted}>{getTranslations().billing.held(amount)}</Text>
        <Text />
        {footer(getTranslations().billing.escCancel, t)}
      </Box>
    )
  }

  if (phase === 'granted') {
    return (
      <Box flexDirection="column">
        <Text bold color={t.color.ok}>
          {getTranslations().billing.remoteSpendingAllowed}
        </Text>
        <Text color={t.color.text}>{getTranslations().billing.ready(amount)}</Text>
        <Text />
        <ActionRow active color={t.color.ok} label={getTranslations().billing.pressEnterToResume} t={t} />
        <Text />
        {footer(getTranslations().billing.enterResumeEscCancel, t)}
      </Box>
    )
  }

  if (phase === 'resuming') {
    return (
      <Box flexDirection="column">
        <Text bold color={t.color.accent}>
          {getTranslations().billing.allowRemoteSpending}
        </Text>
        <Text color={t.color.muted}>{getTranslations().billing.resuming(amount)}</Text>
        <Text />
        {footer(getTranslations().billing.escCancel, t)}
      </Box>
    )
  }

  // phase === 'prompt' — the one heads-up, triggered only by the 403.
  return (
    <Box flexDirection="column">
      <Text bold color={t.color.warn}>
        {getTranslations().billing.oneTimeSetup}
      </Text>
      <Text color={t.color.text}>{getTranslations().billing.toChargeFromThisTerminalAllowRemote}</Text>
      <Text color={t.color.muted}>{getTranslations().billing.authorization(amount)}</Text>
      <Text />
      <ActionRow active={sel === 0} color={t.color.ok} label={getTranslations().billing.allowRemoteSpending} t={t} />
      <ActionRow active={sel === 1} label={getTranslations().billing.notNow} t={t} />
      <Text />
      {footer(getTranslations().billing.selectEnterConfirmYNQuickEsc2, t)}
    </Box>
  )
}

// ── Screen 4: Auto-reload (the 2-field form) ──────────────────────────

function AutoReloadScreen({ ctx, onClose, onPatch, s, t }: ScreenProps) {
  const ar = s.auto_reload
  const enabled = Boolean(ar?.enabled)
  const distinctCard = ar?.card?.kind === 'distinct' ? ar.card : null

  const distinctCardName = distinctCard
    ? [distinctCard.brand, distinctCard.last4 ? `••${distinctCard.last4}` : null].filter(Boolean).join(' ') ||
      getTranslations().billing.aDifferentCard
    : null

  const manageCardLabel = getTranslations().billing.useYourCardOnFileManageOn

  // Prefill from state (strip the $ from the *_usd raw fields if present).
  const prefill = (raw?: null | string) => (raw == null ? '' : String(raw).replace(/^\$/, '').trim())
  const [threshold, setThreshold] = useState(prefill(ar?.threshold_usd))
  const [reloadTo, setReloadTo] = useState(prefill(ar?.reload_to_usd))
  const [field, setField] = useState<'reloadTo' | 'threshold'>('threshold')
  const [error, setError] = useState<null | string>(null)
  // focusRow: 0=threshold field, 1=reloadTo field, 2=Agree, 3=Turn off (if enabled), last=Cancel
  const manageCardRows = distinctCard && s.portal_url ? [manageCardLabel] : []

  const actionRows = enabled
    ? [
        getTranslations().billing.agreeAndTurnOn,
        getTranslations().billing.turnOff,
        ...manageCardRows,
        getTranslations().billing.cancel
      ]
    : [getTranslations().billing.agreeAndTurnOn, ...manageCardRows, getTranslations().billing.cancel]

  const actionColors: Record<string, string> = {
    'Agree and turn on': t.color.ok,
    'Turn off': t.color.warn,
    [manageCardLabel]: t.color.accent
  }

  const FIELD_ROWS = 2
  const [row, setRow] = useState(0)

  const noCard = !s.card

  const validatePair = (): null | { reloadTo: string; threshold: string } => {
    const tv = ctx.validate(threshold)

    if (tv.error || !tv.amount) {
      setError(getTranslations().billing.thresholdError(String(tv.error ?? getTranslations().billing.invalid)))

      return null
    }

    const rv = ctx.validate(reloadTo)

    if (rv.error || !rv.amount) {
      setError(getTranslations().billing.reloadTargetError(String(rv.error ?? getTranslations().billing.invalid)))

      return null
    }

    if (Number(rv.amount) <= Number(tv.amount)) {
      setError(getTranslations().billing.reloadToAmountMustBeGreaterThan)

      return null
    }

    setError(null)

    return { reloadTo: rv.amount, threshold: tv.amount }
  }

  const turnOn = () => {
    if (noCard) {
      ctx.sys(getTranslations().billing.noSavedCardManageBillingOnThe)

      if (s.portal_url) {
        ctx.openPortal(s.portal_url)
      }

      onClose()

      return
    }

    const pair = validatePair()

    if (!pair) {
      return
    }

    void ctx.applyAutoReload(true, Number(pair.threshold), Number(pair.reloadTo)).then(ok => {
      if (ok) {
        ctx.sys(getTranslations().billing.reloadEnabled(String(pair.threshold), String(pair.reloadTo)))
      }
    })
    onClose()
  }

  const turnOff = () => {
    // The PATCH requires threshold/top_up_amount even when disabling (parity
    // with the CLI's _billing_auto_reload_disable) — echo the current values,
    // else the gateway rejects with invalid_request and auto-reload stays ON.
    const thr = Number(prefill(ar?.threshold_usd)) || 0
    const rel = Number(prefill(ar?.reload_to_usd)) || 0
    void ctx.applyAutoReload(false, thr, rel).then(ok => {
      if (ok) {
        ctx.sys(getTranslations().billing.autoReloadTurnedOff)
      }
    })
    onClose()
  }

  const onAction = (label: string) => {
    if (label === getTranslations().billing.agreeAndTurnOn) {
      turnOn()
    } else if (label === getTranslations().billing.turnOff) {
      turnOff()
    } else if (label === manageCardLabel) {
      if (s.portal_url) {
        ctx.openPortal(s.portal_url)
      }

      onClose()
    } else {
      onPatch({ screen: 'overview' })
    }
  }

  const editingField = row < FIELD_ROWS

  useInput((ch, key) => {
    if (key.escape) {
      return onPatch({ screen: 'overview' })
    }

    if (key.upArrow && row > 0) {
      setRow(v => v - 1)
      setField(row - 1 === 0 ? 'threshold' : 'reloadTo')
    }

    if (key.downArrow && row < FIELD_ROWS + actionRows.length - 1) {
      setRow(v => v + 1)
      setField(row + 1 === 0 ? 'threshold' : 'reloadTo')
    }

    // Tab cycles between the two fields when focused on a field.
    if (key.tab && editingField) {
      const next = field === 'threshold' ? 'reloadTo' : 'threshold'
      setField(next)
      setRow(next === 'threshold' ? 0 : 1)
    }

    if (key.return && !editingField) {
      const idx = row - FIELD_ROWS

      return onAction(actionRows[idx] ?? getTranslations().billing.cancel)
    }

    // a number quick-picks an action row (1..actionRows.length)
    if (!editingField) {
      const n = parseInt(ch, 10)

      if (n >= 1 && n <= actionRows.length) {
        return onAction(actionRows[n - 1]!)
      }
    }
  })

  const cardLine = s.card
    ? getTranslations().billing.savedCard(String(s.card.masked))
    : getTranslations().billing.noSavedCardOnFile

  const chargeCardName = distinctCardName ?? (s.card ? s.card.masked : getTranslations().billing.yourCard)

  const fieldBox = (label: string, value: string, onChange: (v: string) => void, focused: boolean, key: string) => (
    <Box flexDirection="column" key={key}>
      <Text color={focused ? t.color.label : t.color.muted}>{label}</Text>
      <Box borderColor={focused ? t.color.accent : t.color.border} borderStyle="round" paddingX={1}>
        <Text color={t.color.label}>{'$'}</Text>
        <TextInput
          color={t.color.text}
          columns={16}
          focus={focused}
          onChange={onChange}
          onSubmit={() => {
            // Enter inside the threshold field jumps to reload-to; inside
            // reload-to jumps to the Agree action.
            if (key === 'threshold') {
              setField('reloadTo')
              setRow(1)
            } else {
              setRow(FIELD_ROWS)
            }
          }}
          value={value}
        />
      </Box>
    </Box>
  )

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {getTranslations().billing.autoReload}
      </Text>
      <Text color={t.color.muted}>{getTranslations().billing.automaticallyAddFundsWhenYourBalanceIs}</Text>
      <Text color={t.color.muted}>{cardLine}</Text>
      {distinctCardName && (
        <Text color={t.color.warn}>
          {getTranslations().billing.autoRefillIsCharging}
          {distinctCardName}
          {getTranslations().billing.notYourCardOnFile}
        </Text>
      )}
      <Text />
      {fieldBox(getTranslations().billing.whenBalanceFallsBelow, threshold, setThreshold, row === 0, 'threshold')}
      {fieldBox(getTranslations().billing.reloadBalanceTo, reloadTo, setReloadTo, row === 1, 'reloadTo')}
      <Text />
      <Text color={t.color.muted}>
        {getTranslations().billing.byConfirmingYouAuthorizeNousResearchTo}
        {chargeCardName}
        {getTranslations().billing.wheneverYourBalanceFallsBelowTheThreshold}
      </Text>
      {error && <Text color={t.color.error}>{error}</Text>}
      <Text />
      {actionRows.map((label, i) => (
        <ActionRow
          active={!editingField && row - FIELD_ROWS === i}
          color={actionColors[label] ?? t.color.text}
          key={label}
          label={label}
          t={t}
        />
      ))}
      <Text />
      {footer(getTranslations().billing.moveTabSwitchFieldEnterNextConfirm, t)}
    </Box>
  )
}

// ── Screen 5: Monthly spend limit (read-only) ─────────────────────────

function LimitScreen({ ctx, onClose, onPatch, s, t }: ScreenProps) {
  const labels = [getTranslations().billing.manageOnPortal, getTranslations().billing.cancel]

  const choose = (i: number) => {
    if (i === 0 && s.portal_url) {
      ctx.openPortal(s.portal_url)

      return onClose()
    }

    onPatch({ screen: 'overview' })
  }

  const rows: MenuRowSpec[] = labels.map((label, i) => ({ label, run: () => choose(i) }))
  const sel = useMenu(rows, () => onPatch({ screen: 'overview' }))

  const cap = s.monthly_cap

  const usageLine =
    cap && cap.limit_usd != null
      ? getTranslations().billing.monthlyUsage(
          String(cap.spent_display),
          String(cap.limit_display),
          String(cap.is_default_ceiling ? getTranslations().billing.defaultCeiling : '')
        )
      : getTranslations().billing.noMonthlyCapVisibleManagedOnThe

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {getTranslations().billing.monthlySpendLimit}
      </Text>
      <Text color={t.color.text}>{usageLine}</Text>
      <Text color={t.color.muted}>{getTranslations().billing.theMonthlyLimitIsSetOnThe}</Text>
      <Text />
      {labels.map((label, i) => (
        <MenuRow active={sel === i} index={i + 1} key={label} label={label} t={t} />
      ))}
      <Text />
      {footer(getTranslations().billing.menuBackHint(String(labels.length)), t)}
    </Box>
  )
}
