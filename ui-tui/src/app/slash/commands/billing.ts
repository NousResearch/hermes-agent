import type {
  BillingChargeResponse,
  BillingChargeStatusResponse,
  BillingErrorPayload,
  BillingMutationResponse,
  BillingStateResponse
} from '../../../gatewayTypes.js'
import { translate } from '../../../i18n/index.js'
import { openExternalUrl } from '../../../lib/openExternalUrl.js'
import type { BillingOverlayCtx } from '../../interfaces.js'
import { patchOverlayState } from '../../overlayStore.js'
import type { SlashCommand, SlashRunCtx } from '../types.js'

// Poll cadence (plan §5, frozen): 2s interval, 5-minute cap.
const POLL_INTERVAL_MS = 2000
const POLL_CAP_MS = 5 * 60 * 1000

type Sys = (text: string) => void
type Translator = (key: Parameters<typeof translate>[1], vars?: Record<string, string | number>) => string

/** Map a typed billing error envelope to user-facing copy + portal funnel. */
const renderBillingError = (
  sys: Sys,
  ctx: SlashRunCtx,
  tr: Translator,
  env: {
    error?: string
    message?: string
    payload?: BillingErrorPayload
    portal_url?: string | null
    retry_after?: number | null
  }
): void => {
  const portal = env.portal_url

  switch (env.error) {
    case 'insufficient_scope':
      armStepUp(sys, ctx, tr)

      return

    case 'no_payment_method':
      sys(tr('billing.error.noPaymentMethod'))

      break

    case 'cli_billing_disabled':
      sys(tr('billing.error.cliBillingDisabled'))

      break
    case 'monthly_cap_exceeded': {
      // Surface the remaining headroom the server attaches (parity with the CLI).
      const remaining = env.payload?.remainingUsd
      sys(
        remaining != null
          ? tr('billing.error.monthlyCapExceededRemaining', { remaining })
          : tr('billing.error.monthlyCapExceeded')
      )

      break
    }

    case 'rate_limited': {
      const mins = env.retry_after ? Math.max(1, Math.round(env.retry_after / 60)) : null
      sys(mins ? tr('billing.error.rateLimitedWithRetry', { minutes: mins }) : tr('billing.error.rateLimited'))

      break
    }

    default:
      sys(tr('billing.error.generic', { message: env.message || env.error || tr('billing.error.requestFailed') }))
  }

  if (portal) {
    sys(tr('billing.portalLine', { url: portal }))
  }
}

/** 403 insufficient_scope → arm a ConfirmReq that runs the lazy step-up. */
const armStepUp = (sys: Sys, ctx: SlashRunCtx, tr: Translator): void => {
  sys(tr('billing.stepUp.needsPermission'))
  patchOverlayState({
    confirm: {
      cancelLabel: tr('billing.stepUp.cancel'),
      confirmLabel: tr('billing.stepUp.confirm'),
      detail: tr('billing.stepUp.detail'),
      onConfirm: () => {
        // session_id lets the gateway route the billing.step_up.verification
        // event (the verification link) back to this session — the device flow
        // runs headless in the gateway, so the link can't be printed there.
        ctx.gateway
          .rpc<BillingMutationResponse>('billing.step_up', { session_id: ctx.sid ?? undefined })
          .then(
            ctx.guarded<BillingMutationResponse>(r => {
              if (r.ok && r.granted) {
                // Step-up only grants the billing:manage TOKEN scope — the ORG
                // kill-switch (cli_billing_enabled) is a separate gate. Re-fetch
                // /state so we don't over-promise "enabled" when a charge would
                // still hit cli_billing_disabled.
                sys(tr('billing.stepUp.granted'))
                ctx.gateway
                  .rpc<BillingStateResponse>('billing.state', {})
                  .then(
                    ctx.guarded<BillingStateResponse>(s => {
                      if (s.cli_billing_enabled) {
                        sys(tr('billing.runAgain'))
                      } else {
                        sys(tr('billing.stepUp.grantedButDisabled'))

                        if (s.portal_url) {
                          sys(tr('billing.portalLine', { url: s.portal_url }))
                        }
                      }
                    })
                  )
                  .catch(() => {
                    sys(tr('billing.runAgain'))
                  })
              } else {
                sys(tr('billing.stepUp.notGranted'))
              }
            })
          )
          .catch(() => {
            // The device flow can outlive the RPC's 120s timeout while the user
            // is still authorizing in the browser. A reject here is NOT a hard
            // failure — the grant (if it lands) is persisted gateway-side; tell
            // the user to re-run /billing rather than reporting an error.
            sys(tr('billing.stepUp.stillWaiting'))
          })
      },
      title: tr('billing.stepUp.title')
    }
  })
}

/** Poll a charge to a terminal state (settled/failed/timeout). Non-blocking. */
const pollCharge = (sys: Sys, ctx: SlashRunCtx, tr: Translator, chargeId: string, portalUrl?: string | null): void => {
  const start = Date.now()

  const tick = (): void => {
    if (ctx.stale()) {
      return
    }

    ctx.gateway
      .rpc<BillingChargeStatusResponse>('billing.charge_status', { charge_id: chargeId })
      .then(
        ctx.guarded<BillingChargeStatusResponse>(r => {
          if (!r.ok) {
            // 429/503 while polling = retry-after, NOT a failure. Back off + continue.
            if (r.error === 'rate_limited') {
              const wait = (r.retry_after ?? 5) * 1000
              setTimeout(tick, Math.min(wait, 30000))

              return
            }

            sys(
              tr('billing.charge.couldNotCheck', { message: r.message || r.error || tr('billing.error.genericError') })
            )

            return
          }

          if (r.status === 'settled') {
            sys(r.amount_usd ? tr('billing.charge.addedAmount', { amount: r.amount_usd }) : tr('billing.charge.added'))

            return
          }

          if (r.status === 'failed') {
            renderChargeFailed(sys, tr, r.reason, portalUrl)

            return
          }

          // pending → keep polling until the 5-min cap, then call it a timeout.
          if (Date.now() - start >= POLL_CAP_MS) {
            sys(tr('billing.charge.timeout'))

            if (portalUrl) {
              sys(tr('billing.portalLine', { url: portalUrl }))
            }

            return
          }

          setTimeout(tick, POLL_INTERVAL_MS)
        })
      )
      .catch(ctx.guardedErr)
  }

  tick()
}

const renderChargeFailed = (sys: Sys, tr: Translator, reason?: string | null, portalUrl?: string | null): void => {
  switch ((reason || '').trim()) {
    case 'authentication_required':
      sys(tr('billing.charge.authRequired'))

      break

    case 'payment_method_expired':
      sys(tr('billing.charge.cardExpired'))

      break

    case 'card_declined':
      sys(tr('billing.charge.cardDeclined'))

      break

    default:
      sys(tr('billing.charge.failed', { reason: reason || 'processing_error' }))
  }

  // Funnel to the portal after any failure (parity with cli.py _billing_portal_hint).
  if (portalUrl) {
    sys(tr('billing.portalLine', { url: portalUrl }))
  }
}

/** Validate a custom amount against state bounds + 2dp, mirroring the server. */
const validateAmount = (raw: string, s: BillingStateResponse, tr: Translator): { amount?: string; error?: string } => {
  const cleaned = raw.trim().replace(/^\$/, '').trim()

  if (!cleaned || !/^\d+(\.\d{1,2})?$/.test(cleaned)) {
    return { error: tr('billing.validate.amountFormat') }
  }

  const value = Number(cleaned)

  if (!(value > 0)) {
    return { error: tr('billing.validate.amountPositive') }
  }

  if (s.min_usd != null && value < Number(s.min_usd)) {
    return { error: tr('billing.validate.minimum', { amount: s.min_usd }) }
  }

  if (s.max_usd != null && value > Number(s.max_usd)) {
    return { error: tr('billing.validate.maximum', { amount: s.max_usd }) }
  }

  return { amount: cleaned }
}

/**
 * Build the closure bundle the BillingOverlay needs to talk to the gateway
 * and emit transcript lines.  Keeps ALL RPC + error-mapping logic here
 * (single source of truth) — the overlay only renders + routes keys.
 */
const buildOverlayCtx = (ctx: SlashRunCtx, sys: Sys, s: BillingStateResponse, tr: Translator): BillingOverlayCtx => ({
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
          renderBillingError(sys, ctx, tr, r)
        }

        return false
      })
      .catch(e => {
        ctx.guardedErr(e)

        return false
      }),
  charge: (amount: string) => {
    sys(tr('billing.charge.submitted'))
    ctx.gateway
      .rpc<BillingChargeResponse>('billing.charge', { amount_usd: amount })
      .then(
        ctx.guarded<BillingChargeResponse>(r => {
          if (r.ok && r.charge_id) {
            pollCharge(sys, ctx, tr, r.charge_id, s.portal_url)
          } else {
            renderBillingError(sys, ctx, tr, r)
          }
        })
      )
      .catch(ctx.guardedErr)
  },
  openPortal: (url: string) => {
    openExternalUrl(url)
    sys(tr('billing.openingPortal', { url }))
  },
  sys,
  validate: (raw: string) => validateAmount(raw, s, tr)
})

export const billingCommands: SlashCommand[] = [
  {
    help: 'Manage Nous terminal billing — buy credits, auto-reload, limits',
    name: 'billing',
    // ZERO sub-commands (plan §0.4): any arg is ignored. Bare `/billing`
    // fetches state and opens the interactive overlay (CLI/TUI parity).
    run: (_arg, ctx) => {
      const sys: Sys = ctx.transcript.sys
      const tr: Translator = (key, vars) => translate(ctx.ui.locale, key, vars)

      ctx.gateway
        .rpc<BillingStateResponse>('billing.state', {})
        .then(
          ctx.guarded<BillingStateResponse>(s => {
            if (!s.logged_in) {
              sys(tr('billing.notLoggedIn'))

              return
            }

            patchOverlayState({
              billing: {
                ctx: buildOverlayCtx(ctx, sys, s, tr),
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
