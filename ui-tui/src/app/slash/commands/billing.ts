import type {
  BillingChargeResponse,
  BillingChargeStatusResponse,
  BillingErrorPayload,
  BillingMutationResponse,
  BillingStateResponse
} from '../../../gatewayTypes.js'
import { openExternalUrl } from '../../../lib/openExternalUrl.js'
import type { BillingOverlayCtx } from '../../interfaces.js'
import { patchOverlayState } from '../../overlayStore.js'
import type { SlashCommand, SlashRunCtx } from '../types.js'

// Poll cadence (plan §5, frozen): 2s interval, 5-minute cap.
const POLL_INTERVAL_MS = 2000
const POLL_CAP_MS = 5 * 60 * 1000

type Sys = (text: string) => void

/** Map a typed billing error envelope to user-facing copy + portal funnel. */
const renderBillingError = (
  sys: Sys,
  ctx: SlashRunCtx,
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
      armStepUp(sys, ctx)

      return

    case 'no_payment_method':
      sys(
        '💳 No saved card for terminal charges yet. Set one up on the portal ' +
          "(one-time credit buys don't save a reusable card)."
      )

      break

    case 'cli_billing_disabled':
      sys('🔴 Terminal billing is turned off for this org — an admin must enable it on the portal.')

      break

    case 'monthly_cap_exceeded': {
      // Surface the remaining headroom the server attaches (parity with the CLI).
      const remaining = env.payload?.remainingUsd
      sys(remaining != null ? `🔴 Monthly spend cap reached — $${remaining} headroom left.` : '🔴 Monthly spend cap reached.')

      break
    }
    case 'rate_limited': {
      const mins = env.retry_after ? ` (try again in ~${Math.max(1, Math.round(env.retry_after / 60))} min)` : ''
      sys(`🟡 Too many charges right now${mins}. This isn't a payment failure.`)

      break
    }

    default:
      sys(`🔴 ${env.message || env.error || 'Billing request failed.'}`)
  }

  if (portal) {
    sys(`Portal: ${portal}`)
  }
}

/** 403 insufficient_scope → arm a ConfirmReq that runs the lazy step-up. */
const armStepUp = (sys: Sys, ctx: SlashRunCtx): void => {
  sys('💳 Terminal billing needs an extra permission (billing:manage).')
  patchOverlayState({
    confirm: {
      cancelLabel: 'Not now',
      confirmLabel: 'Re-authorize',
      detail: 'An org admin/owner must tick "Allow terminal billing" in the portal.',
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
                sys('✅ Billing permission granted.')
                ctx.gateway
                  .rpc<BillingStateResponse>('billing.state', {})
                  .then(
                    ctx.guarded<BillingStateResponse>(s => {
                      if (s.cli_billing_enabled) {
                        sys('Run /billing again to continue.')
                      } else {
                        sys(
                          '🟡 Permission granted, but terminal billing is still turned off ' +
                            'for this org. Enable it in the portal, then run /billing again.'
                        )
                        if (s.portal_url) {
                          sys(`Portal: ${s.portal_url}`)
                        }
                      }
                    })
                  )
                  .catch(() => {
                    sys('Run /billing again to continue.')
                  })
              } else {
                sys('🟡 Terminal billing was not granted (an admin must tick the box).')
              }
            })
          )
          .catch(() => {
            // The device flow can outlive the RPC's 120s timeout while the user
            // is still authorizing in the browser. A reject here is NOT a hard
            // failure — the grant (if it lands) is persisted gateway-side; tell
            // the user to re-run /billing rather than reporting an error.
            sys('🟡 Still waiting on approval — finish in the browser, then run /billing again.')
          })
      },
      title: 'Grant terminal billing access?'
    }
  })
}

/** Poll a charge to a terminal state (settled/failed/timeout). Non-blocking. */
const pollCharge = (sys: Sys, ctx: SlashRunCtx, chargeId: string, portalUrl?: string | null): void => {
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

            sys(`🔴 Could not check the charge: ${r.message || r.error || 'error'}`)

            return
          }

          if (r.status === 'settled') {
            sys(`✅ ${r.amount_usd ? `$${r.amount_usd}` : 'Credits'} added.`)

            return
          }

          if (r.status === 'failed') {
            renderChargeFailed(sys, r.reason, portalUrl)

            return
          }

          // pending → keep polling until the 5-min cap, then call it a timeout.
          if (Date.now() - start >= POLL_CAP_MS) {
            sys(
              '🟡 Still processing after 5 minutes — this is a timeout, not a failure. ' +
                'Check /billing or the portal shortly.'
            )
            if (portalUrl) {
              sys(`Portal: ${portalUrl}`)
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
  charge: (amount: string) => {
    sys('💳 Charge submitted — confirming settlement…')
    ctx.gateway
      .rpc<BillingChargeResponse>('billing.charge', { amount_usd: amount })
      .then(
        ctx.guarded<BillingChargeResponse>(r => {
          if (r.ok && r.charge_id) {
            pollCharge(sys, ctx, r.charge_id, s.portal_url)
          } else {
            renderBillingError(sys, ctx, r)
          }
        })
      )
      .catch(ctx.guardedErr)
  },
  openPortal: (url: string) => {
    openExternalUrl(url)
    sys(`Opening portal: ${url}`)
  },
  sys,
  validate: (raw: string) => validateAmount(raw, s)
})

export const billingCommands: SlashCommand[] = [
  {
    help: 'Manage Nous terminal billing — buy credits, auto-reload, limits',
    name: 'billing',
    // ZERO sub-commands (plan §0.4): any arg is ignored. Bare `/billing`
    // fetches state and opens the interactive overlay (CLI/TUI parity).
    run: (_arg, ctx) => {
      const sys: Sys = ctx.transcript.sys

      ctx.gateway
        .rpc<BillingStateResponse>('billing.state', {})
        .then(
          ctx.guarded<BillingStateResponse>(s => {
            if (!s.logged_in) {
              sys('💳 Not logged into Nous Portal — run /portal to log in, then /billing.')

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
