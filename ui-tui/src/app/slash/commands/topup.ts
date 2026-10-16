import { driveChargeSettlement, type SettlementOutcome } from '@hermes/shared/charge-settlement'

import type {
  BillingChargeResponse,
  BillingChargeStatusResponse,
  BillingErrorPayload,
  BillingMutationResponse,
  BillingStateResponse
} from '../../../gatewayTypes.js'
import { t } from '../../../i18n/runtime.js'
import { openExternalUrl } from '../../../lib/openExternalUrl.js'
import type { BillingChargeOutcome, BillingOverlayCtx } from '../../interfaces.js'
import { patchOverlayState } from '../../overlayStore.js'
import type { SlashCommand, SlashRunCtx } from '../types.js'

type Sys = (text: string) => void

/** " (try again in ~N min)" for throttled/unavailable errors, or '' when no retry_after. */
const retrySuffix = (retryAfter?: number | null): string =>
  retryAfter ? t('slashCmd.topup.error.retryIn', String(Math.max(1, Math.round(retryAfter / 60)))) : ''

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
      // Remote Spending allowed. The resumable step-up lives on the buy/charge
      // path; point the user there rather than leaking the raw scope name.
      sys(t('slashCmd.topup.error.insufficientScope'))

      break
    case 'remote_spending_revoked': {
      // CF-4: this terminal's spend was revoked. Kill the spend UI NOW (don't
      // wait for the token refresh ~15 min away) and tell the user who did it.
      patchOverlayState({ billing: null })

      const who =
        env.actor === 'admin' ? t('slashCmd.topup.error.revokedByAdmin') : t('slashCmd.topup.error.revokedByYou')

      sys(t('slashCmd.topup.error.revokedReconnect', who))

      return
    }

    case 'session_revoked':
      // Stronger than a spend-revoke: the whole session is gone → full re-login.
      patchOverlayState({ billing: null })
      sys(t('slashCmd.topup.error.sessionRevoked'))

      return

    case 'cli_billing_disabled':

    case 'remote_spending_disabled':
      // Account-wide switch is OFF (dual-emitted error/code). A billing admin can
      // turn it on from the portal's Hermes Agent page; this is NOT a per-terminal stop.
      sys(t('slashCmd.topup.error.remoteSpendingDisabled'))

      break

    case 'role_required':
      sys(t('slashCmd.topup.error.roleRequired'))

      break

    case 'consent_required':
      sys(t('slashCmd.topup.error.consentRequired'))

      break

    case 'org_access_denied':
      sys(t('slashCmd.topup.error.orgAccessDenied'))

      break

    case 'upgrade_cap_exceeded':
      sys(t('slashCmd.topup.error.upgradeCapExceeded'))

      break

    case 'auto_top_up_disabled_failures':
      sys(t('slashCmd.topup.error.autoTopUpDisabledFailures'))

      break

    case 'idempotency_conflict':
      sys(t('slashCmd.topup.error.idempotencyConflict'))

      break

    case 'no_payment_method':
      sys(t('slashCmd.topup.error.noPaymentMethod'))

      break
    case 'monthly_cap_exceeded': {
      // Surface the remaining headroom the server attaches (parity with the CLI).
      const remaining = env.payload?.remainingUsd
      sys(
        remaining != null
          ? t('slashCmd.topup.error.monthlyCapExceededRemaining', String(remaining))
          : t('slashCmd.topup.error.monthlyCapExceeded')
      )

      break
    }

    case 'rate_limited':
    case 'temporarily_unavailable': {
      // 429 throttle OR 503 gate-fail-closed: NOT a payment failure, NOT a
      // revoke. Back off and tell the user to retry.
      sys(t('slashCmd.topup.error.rateLimited', retrySuffix(env.retry_after)))

      break
    }

    case 'stripe_unavailable': {
      sys(t('slashCmd.topup.error.stripeUnavailable', retrySuffix(env.retry_after)))

      break
    }

    default:
      sys(t('slashCmd.topup.error.generic', env.message || env.error || t('slashCmd.topup.error.genericFallback')))
  }

  if (portal) {
    sys(t('slashCmd.topup.error.portal', portal))
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
        sys(
          t(
            'slashCmd.topup.charge.settled',
            outcome.status.amount_usd ? `$${outcome.status.amount_usd}` : t('slashCmd.topup.charge.creditsFallback')
          )
        )

        return

      case 'failed':
        renderChargeFailed(sys, outcome.status.reason, portalUrl)

        return

      case 'refused':
        sys(
          t(
            'slashCmd.topup.charge.couldNotCheck',
            outcome.status.message || outcome.status.error || t('slashCmd.topup.charge.couldNotCheckFallback')
          )
        )

        return

      case 'ambiguous':
        if (outcome.status) {
          renderBillingError(sys, ctx, outcome.status)
          sys(t('slashCmd.topup.charge.unconfirmed'))

          return
        }

        if ('cause' in outcome) {
          ctx.guardedErr(outcome.cause)
        }

        if (!ctx.stale()) {
          sys(t('slashCmd.topup.charge.unconfirmed'))
        }

        return

      case 'timed_out':
        sys(t('slashCmd.topup.charge.timedOut'))

        if (portalUrl) {
          sys(t('slashCmd.topup.error.portal', portalUrl))
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
      sys(t('slashCmd.topup.charge.authenticationRequired'))

      break

    case 'payment_method_expired':
      sys(t('slashCmd.topup.charge.paymentMethodExpired'))

      break

    case 'card_declined':
      sys(t('slashCmd.topup.charge.cardDeclined'))

      break

    case 'processing_error':
      sys(t('slashCmd.topup.charge.failedReason', 'processing_error'))

      break

    default:
      sys(t('slashCmd.topup.charge.failedReason', reason || 'processing_error'))
  }

  // Funnel to the portal after any failure (parity with cli.py _billing_portal_hint).
  if (portalUrl) {
    sys(t('slashCmd.topup.error.portal', portalUrl))
  }
}

/** Validate a custom amount against state bounds + 2dp, mirroring the server. */
const validateAmount = (raw: string, s: BillingStateResponse): { amount?: string; error?: string } => {
  const cleaned = raw.trim().replace(/^\$/, '').trim()

  if (!cleaned || !/^\d+(\.\d{1,2})?$/.test(cleaned)) {
    return { error: t('slashCmd.topup.validate.invalid') }
  }

  const value = Number(cleaned)

  if (!(value > 0)) {
    return { error: t('slashCmd.topup.validate.notPositive') }
  }

  if (s.min_usd != null && value < Number(s.min_usd)) {
    return { error: t('slashCmd.topup.validate.minimum', String(s.min_usd)) }
  }

  if (s.max_usd != null && value > Number(s.max_usd)) {
    return { error: t('slashCmd.topup.validate.maximum', String(s.max_usd)) }
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
    sys(t('slashCmd.topup.charge.submitted'))

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
    sys(t('slashCmd.topup.openingPortal', url))
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
              sys(t('slashCmd.topup.notLoggedIn'))

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
