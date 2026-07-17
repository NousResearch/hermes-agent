import {
  driveChargeSettlement,
  type SettlementOutcome
} from '@hermes/shared/charge-settlement'

import type {
  BillingChargeResponse,
  BillingChargeStatusResponse,
  BillingErrorPayload,
  BillingMutationResponse,
  BillingStateResponse
} from '../../../gatewayTypes.js'
import { openExternalUrl } from '../../../lib/openExternalUrl.js'
import type { BillingChargeOutcome, BillingOverlayCtx } from '../../interfaces.js'
import { patchOverlayState } from '../../overlayStore.js'
import type { SlashCommand, SlashRunCtx } from '../types.js'

const UNCONFIRMED_CHARGE_MESSAGE =
  '🟡 Your last charge’s outcome is unconfirmed — check your balance/history before retrying.'

type Sys = (text: string) => void

/** Map a typed billing error envelope to user-facing copy + portal funnel. */
const renderBillingError = (
  sys: Sys,
  ctx: SlashRunCtx,
  env: {
    actor?: string
    code?: string
    error?: string
    message?: string
    payload?: BillingErrorPayload
    portal_url?: string | null
    recovery?: string
    retry_after?: number | null
  }
): void => {
  const portal = env.portal_url

  switch (env.error) {
    case 'insufficient_scope':
      // Reached by non-charge mutations (e.g. auto-reload config) that need
      // terminal billing enabled. The resumable step-up lives on the buy/charge
      // path; point the user there rather than leaking the raw scope name.
      sys('This needs terminal billing enabled. Start a top-up to enable it, then retry.')

      break
    case 'remote_spending_revoked': {
      // CF-4: this terminal's spend was revoked. Kill the spend UI NOW (don't
      // wait for the token refresh ~15 min away) and tell the user who did it.
      patchOverlayState({ billing: null })

      const who =
        env.actor === 'admin'
          ? 'An admin turned off terminal billing for this terminal.'
          : 'You turned off terminal billing for this terminal.'

      sys(`${who} Reconnect to restore — run /portal to re-authorize this terminal.`)

      return
    }

    case 'session_revoked':
      // Stronger than a spend-revoke: the whole session is gone → full re-login.
      patchOverlayState({ billing: null })
      sys('Your session was logged out. Run /portal to log in again.')

      return

    case 'cli_billing_disabled':

    case 'remote_spending_disabled':
      // Account-wide switch is OFF (dual-emitted error/code). An admin must flip
      // it on the portal; this is NOT a per-terminal revoke.
      sys('Terminal billing is off for this account — an admin must enable it on the portal.')

      break

    case 'role_required':
      sys(
        'Adding funds needs someone with billing permissions (owner, admin, or finance admin), or manage this on the portal.'
      )

      break

    case 'consent_required':
      sys('This action needs a one-time card confirmation and consent step on the portal before it can proceed.')

      break

    case 'org_access_denied':
      sys("This token isn't bound to an org you can manage. Sign in with the right org, or manage this on the portal.")

      break

    case 'upgrade_cap_exceeded':
      sys('🔴 Daily plan-change limit reached (5 per org) — try again tomorrow, or manage this on the portal.')

      break

    case 'auto_top_up_disabled_failures':
      sys(
        'Auto-reload was turned off after repeated charge failures. Fix the card issue, then re-enable it from /topup → Auto-reload.'
      )

      break

    case 'idempotency_conflict':
      sys('🔴 That charge key was already used for a different amount. Start a fresh top-up.')

      break

    case 'no_payment_method':
      sys(
        '💳 No saved card for terminal charges yet. Set one up on the portal ' +
          "(one-time credit buys don't save a reusable card)."
      )

      break
    case 'monthly_cap_exceeded': {
      // Surface the remaining headroom the server attaches (parity with the CLI).
      const remaining = env.payload?.remainingUsd
      sys(
        remaining != null
          ? `🔴 Monthly spend cap reached — $${remaining} headroom left.`
          : '🔴 Monthly spend cap reached.'
      )

      break
    }

    case 'rate_limited':
    case 'temporarily_unavailable': {
      // 429 throttle OR 503 gate-fail-closed: NOT a payment failure, NOT a
      // revoke. Back off and tell the user to retry.
      const mins = env.retry_after ? ` (try again in ~${Math.max(1, Math.round(env.retry_after / 60))} min)` : ''
      sys(`🟡 Too many charges right now${mins}. This isn't a payment failure.`)

      break
    }

    case 'stripe_unavailable': {
      const mins = env.retry_after ? ` (try again in ~${Math.max(1, Math.round(env.retry_after / 60))} min)` : ''
      sys(`🟡 Stripe is having trouble right now — try again shortly${mins}.`)

      break
    }

    default:
      sys(`🔴 ${env.message || env.error || 'Billing request failed.'}`)
  }

  if (portal) {
    sys(`Portal: ${portal}`)
  }
}

/**
 * Run the Remote-Spending device flow and resolve whether the grant landed.
 *
 * The browser opens via the gateway's out-of-band `billing.step_up.verification`
 * event (handled globally in createGatewayEventHandler), so this just kicks the
 * blocking `billing.step_up` RPC and awaits its result. A reject (the device
 * flow can outlive the RPC's timeout while the user is still authorizing) is
 * treated as "not yet granted" — non-fatal; the grant persists gateway-side.
 *
 * NOTE: never surface the raw `billing:manage` scope — the user-facing concept
 * is "Remote Spending".
 */
const requestRemoteSpending = (ctx: SlashRunCtx): Promise<boolean> =>
  ctx.gateway
    .rpc<BillingMutationResponse>('billing.step_up', { session_id: ctx.sid ?? undefined })
    .then(r => !!(r && r.ok && r.granted))
    .catch(() => false)

/** Poll a charge to a terminal state (settled/failed/timeout). Non-blocking. */
const pollCharge = (sys: Sys, ctx: SlashRunCtx, chargeId: string, portalUrl?: string | null): void => {
  const renderOutcome = (outcome: SettlementOutcome): void => {
    switch (outcome.kind) {
      case 'settled':
        sys(`✅ ${outcome.status.amount_usd ? `$${outcome.status.amount_usd}` : 'Credits'} added.`)

        return

      case 'failed':
        renderChargeFailed(sys, outcome.status.reason, portalUrl)

        return

      case 'refused':
        sys(`🔴 Could not check the charge: ${outcome.status.message || outcome.status.error || 'error'}`)

        return

      case 'ambiguous':
        if (outcome.status) {
          renderBillingError(sys, ctx, outcome.status)
          sys(UNCONFIRMED_CHARGE_MESSAGE)

          return
        }

        if ('cause' in outcome) {
          ctx.guardedErr(outcome.cause)
        }

        if (!ctx.stale()) {
          sys(UNCONFIRMED_CHARGE_MESSAGE)
        }

        return

      case 'timed_out':
        sys(
          '🟡 Still processing after 5 minutes — this is a timeout, not a failure. ' +
            'Check /topup or the portal shortly.'
        )

        if (portalUrl) {
          sys(`Portal: ${portalUrl}`)
        }

        return

      case 'cancelled':
        return
    }
  }

  void driveChargeSettlement({
    fetchStatus: async () => {
      const status = await ctx.gateway.rpc<BillingChargeStatusResponse>('billing.charge_status', {
        charge_id: chargeId
      })

      if (!status) {
        throw new Error('billing.charge_status returned no response')
      }

      return status
    },
    isCancelled: () => ctx.stale(),
    now: () => Date.now(),
    sleep: ms => new Promise(resolve => setTimeout(resolve, ms))
  }).then(outcome => {
    if (outcome.kind === 'ambiguous' && !outcome.status) {
      renderOutcome(outcome)

      return
    }

    ctx.guarded<SettlementOutcome>(renderOutcome)(outcome)
  })
}

const renderChargeFailed = (sys: Sys, reason?: string | null, portalUrl?: string | null): void => {
  switch ((reason || '').trim()) {
    case 'authentication_required':
      sys('🔴 Your bank requires verification (3DS). Complete it on the portal to finish this purchase.')

      break

    case 'payment_method_expired':
      sys('🔴 Your card has expired. Update it on the portal.')

      break

    case 'card_declined':
      sys('🔴 Your card was declined. Try another card on the portal.')

      break

    case 'processing_error':
      sys("🔴 The charge didn't go through (processing_error).")

      break

    default:
      sys(`🔴 The charge didn't go through (${reason || 'processing_error'}).`)
  }

  // Funnel to the portal after any failure (parity with cli.py _billing_portal_hint).
  if (portalUrl) {
    sys(`Portal: ${portalUrl}`)
  }
}

/** Validate a custom amount against state bounds + 2dp, mirroring the server. */
const validateAmount = (raw: string, s: BillingStateResponse): { amount?: string; error?: string } => {
  const cleaned = raw.trim().replace(/^\$/, '').trim()

  if (!cleaned || !/^\d+(\.\d{1,2})?$/.test(cleaned)) {
    return { error: 'Enter a dollar amount, e.g. 100 (max 2 decimal places).' }
  }

  const value = Number(cleaned)

  if (!(value > 0)) {
    return { error: 'Amount must be greater than $0.' }
  }

  if (s.min_usd != null && value < Number(s.min_usd)) {
    return { error: `Minimum is $${s.min_usd}.` }
  }

  if (s.max_usd != null && value > Number(s.max_usd)) {
    return { error: `Maximum is $${s.max_usd}.` }
  }

  return { amount: cleaned }
}

/**
 * Build the closure bundle the BillingOverlay needs to talk to the gateway
 * and emit transcript lines.  Keeps ALL RPC + error-mapping logic here
 * (single source of truth) — the overlay only renders + routes keys.
 */
const buildOverlayCtx = (ctx: SlashRunCtx, sys: Sys, s: BillingStateResponse): BillingOverlayCtx => ({
  applyAutoReload: (enabled, threshold, topUp) =>
    ctx.gateway
      .rpc<BillingMutationResponse>('billing.auto_reload', {
        enabled,
        ...(threshold != null ? { threshold } : {}),
        ...(topUp != null ? { top_up_amount: topUp } : {})
      })
      .then(r => {
        if (r && r.ok) {
          return true
        }

        if (r) {
          renderBillingError(sys, ctx, r)
        }

        return false
      })
      .catch(e => {
        ctx.guardedErr(e)

        return false
      }),
  charge: (amount: string, idempotencyKey?: string): Promise<BillingChargeOutcome> => {
    sys('💳 Charge submitted — confirming settlement…')

    return ctx.gateway
      .rpc<BillingChargeResponse>('billing.charge', {
        amount_usd: amount,
        ...(idempotencyKey ? { idempotency_key: idempotencyKey } : {})
      })
      .then((r): BillingChargeOutcome => {
        if (!r) {
          return 'error'
        }

        if (r.ok && r.charge_id) {
          pollCharge(sys, ctx, r.charge_id, s.portal_url)

          return 'submitted'
        }

        // insufficient_scope → the overlay routes to the resumable step-up
        // (no error line here; the stepup screen owns that UX).
        if (r.error === 'insufficient_scope') {
          return 'needs_remote_spending'
        }

        renderBillingError(sys, ctx, r)

        return 'error'
      })
      .catch((e): BillingChargeOutcome => {
        ctx.guardedErr(e)

        return 'error'
      })
  },
  requestRemoteSpending: () => requestRemoteSpending(ctx),
  openPortal: (url: string) => {
    openExternalUrl(url)
    sys(`Opening portal: ${url}`)
  },
  refreshState: () =>
    ctx.gateway
      .rpc<BillingStateResponse>('billing.state', {})
      .then(r => (r?.ok ? r : null))
      .catch(() => null),
  sys,
  validate: (raw: string) => validateAmount(raw, s)
})

export const topupCommands: SlashCommand[] = [
  {
    help: 'Show your balance and manage billing — add funds, auto-reload, limits',
    name: 'topup',
    // ZERO sub-commands (plan §0.4): any arg is ignored. Bare `/topup`
    // fetches state and opens the interactive overlay (CLI/TUI parity).
    run: (_arg, ctx) => {
      const sys: Sys = ctx.transcript.sys

      ctx.gateway
        .rpc<BillingStateResponse>('billing.state', {})
        .then(
          ctx.guarded<BillingStateResponse>(s => {
            if (!s.logged_in) {
              sys('💳 Not logged into Nous Portal — run /portal to log in, then /topup.')

              return
            }

            patchOverlayState({
              billing: {
                ctx: buildOverlayCtx(ctx, sys, s),
                pendingCharge: null,
                screen: 'overview',
                state: s
              }
            })
          })
        )
        .catch(ctx.guardedErr)
    }
  }
]
