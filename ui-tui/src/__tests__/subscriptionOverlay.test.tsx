import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it, vi } from 'vitest'

const inputHarness = vi.hoisted(() => ({
  handler: undefined as undefined | ((input: string, key: Record<string, boolean>) => void)
}))

// Stub useInput so the overlay doesn't try to enter raw mode under renderSync
// (PassThrough stdin doesn't support it). Box/Text pass through to real Ink.
vi.mock('@hermes/ink', async importOriginal => {
  const mod = await importOriginal()

  return {
    ...mod,
    useInput: (handler: (input: string, key: Record<string, boolean>) => void) => {
      inputHarness.handler = handler
    }
  }
})

import type { SubscriptionOverlayState } from '../app/interfaces.js'
import { SubscriptionOverlay } from '../components/subscriptionOverlay.js'
import type { SubscriptionStateResponse } from '../gatewayTypes.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const t = DEFAULT_THEME

/** Mount a SubscriptionOverlay via renderSync + PassThrough. */
function mount(
  overlay: SubscriptionOverlayState,
  onPatch: (next: Partial<SubscriptionOverlayState>) => void = () => {}
) {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  let output = ''

  Object.assign(stdout, { columns: 100, isTTY: false, rows: 40 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  inputHarness.handler = undefined
  const element = React.createElement(SubscriptionOverlay, { onClose: () => {}, onPatch, overlay, t })

  const instance = renderSync(element, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  return {
    cleanup: () => {
      instance.unmount()
      instance.cleanup()
    },
    output: () => stripAnsi(output),
    rerender: () => instance.rerender(element)
  }
}

/** Render a SubscriptionOverlay to a string via renderSync + PassThrough. */
function render(overlay: SubscriptionOverlayState): string {
  const mounted = mount(overlay)
  const output = mounted.output()
  mounted.cleanup()

  return output
}

const TIERS = [
  {
    tier_id: 'free',
    name: 'Free',
    tier_order: 0,
    dollars_per_month_display: '$0',
    monthly_credits: '0',
    is_current: false,
    is_enabled: true
  },
  {
    tier_id: 'plus',
    name: 'Plus',
    tier_order: 1,
    dollars_per_month_display: '$20',
    monthly_credits: '1000',
    is_current: true,
    is_enabled: true
  },
  {
    tier_id: 'ultra',
    name: 'Ultra',
    tier_order: 2,
    dollars_per_month_display: '$40',
    monthly_credits: '3000',
    is_current: false,
    is_enabled: true
  }
]

const state = (overrides: Partial<SubscriptionStateResponse> = {}): SubscriptionStateResponse => ({
  ok: true,
  logged_in: true,
  is_admin: true,
  can_change_plan: true,
  org_name: 'Acme',
  org_id: 'org_acme',
  role: 'OWNER',
  current: null,
  tiers: [],
  portal_url: 'https://portal.nousresearch.com/billing',
  ...overrides
})

const ctx = {
  fetchCard: vi.fn(() => Promise.resolve(null)),
  openManageLink: vi.fn(() => Promise.resolve(true)),
  openPortal: vi.fn(),
  preview: vi.fn(() => Promise.resolve(null)),
  refreshState: vi.fn(() => Promise.resolve(null)),
  requestRemoteSpending: vi.fn(() => Promise.resolve({ granted: true })),
  resume: vi.fn(() => Promise.resolve(null)),
  scheduleCancellation: vi.fn(() => Promise.resolve(null)),
  scheduleChange: vi.fn(() => Promise.resolve(null)),
  sys: vi.fn(),
  upgrade: vi.fn(() => Promise.resolve(null))
}

const overlay = (s: SubscriptionStateResponse): SubscriptionOverlayState => ({ ctx, screen: 'overview', state: s })

// Overview: the entry screen across every account state (plan + usage + the
// actions that enter the in-terminal change flow).
describe('SubscriptionOverlay — overview', () => {
  it('free: upsell + "Start a subscription", no tier list, no "credits"', () => {
    const out = render(overlay(state({ current: null, usage: { available: true, status: 'free', plan_name: null } })))

    expect(out).toContain('Plan: Free · free models only')
    expect(out).toContain('Paid models need a subscription')
    expect(out).toContain('Start a subscription')
    expect(out).not.toContain('$20/mo')
    expect(out.toLowerCase()).not.toContain('credits')
  })

  it('free with catalog: plans render inline; the generic portal row disappears', () => {
    const out = render(overlay(freeWithCatalog()))

    expect(out).toContain('Plus · $20/mo · $1,000 credits/mo')
    expect(out).toContain('Ultra · $40/mo · $3,000 credits/mo')
    expect(out).not.toContain('upgrade') // a start, not a move
    expect(out).not.toContain('$0/mo') // free tier is not an option
    expect(out).not.toContain('Choose a plan')
    expect(out).not.toContain('Start a subscription')
  })

  it('free with catalog: picking a plan opens the portal once, even on double-Enter', async () => {
    const openManageLink = vi.fn(() => Promise.resolve(true))
    const preview = vi.fn(() => Promise.resolve(null))
    const sys = vi.fn()

    const mounted = mount({
      ctx: { ...ctx, openManageLink, preview, sys } as SubscriptionOverlayState['ctx'],
      screen: 'overview',
      state: freeWithCatalog()
    })

    inputHarness.handler?.('', { return: true }) // first row = Plus
    inputHarness.handler?.('', { return: true })
    await vi.waitFor(() => expect(openManageLink).toHaveBeenCalled())
    mounted.cleanup()

    expect(openManageLink).toHaveBeenCalledTimes(1)
    expect(openManageLink).toHaveBeenCalledWith('plus')
    expect(preview).not.toHaveBeenCalled()
    // openManageLink narrates the handoff itself.
    expect(sys).not.toHaveBeenCalled()
  })

  it('subscriber: status line + plan bar + top-up bar, no "credits"', () => {
    const out = render(
      overlay(
        state({
          current: {
            tier_id: 'pro',
            tier_name: 'Pro',
            monthly_credits: '1000',
            credits_remaining: '700',
            cycle_ends_at: '2026-07-01',
            pending_downgrade_tier_name: null,
            pending_downgrade_at: null
          },
          usage: {
            available: true,
            status: 'healthy',
            plan_name: 'Pro',
            renews_display: 'Jul 1, 2026',
            total_spendable_display: '$26.00',
            has_topup: true,
            plan_bar: {
              kind: 'plan',
              remaining_display: '$14.00',
              total_display: '$20.00',
              spent_display: '$6.00',
              pct_used: 30,
              fill_fraction: 0.7
            },
            topup_bar: {
              kind: 'topup',
              remaining_display: '$12.00',
              total_display: '$12.00',
              spent_display: '$0.00',
              pct_used: null,
              fill_fraction: 1
            }
          }
        })
      )
    )

    expect(out).toContain('Plan: Pro')
    expect(out).toContain('$14.00 left of $20.00')
    expect(out).toContain('30% used')
    expect(out).toContain('top-up')
    expect(out).toContain('never expires')
    expect(out.toLowerCase()).not.toContain('credits')
  })

  it('low balance: shows alert nudge', () => {
    const out = render(
      overlay(
        state({
          current: {
            tier_id: 'pro',
            tier_name: 'Pro',
            monthly_credits: '1000',
            credits_remaining: '170',
            cycle_ends_at: '2026-07-01',
            pending_downgrade_tier_name: null,
            pending_downgrade_at: null
          },
          usage: {
            available: true,
            status: 'low',
            plan_name: 'Pro',
            total_spendable_display: '$3.40',
            plan_bar: {
              kind: 'plan',
              remaining_display: '$3.40',
              total_display: '$20.00',
              spent_display: '$16.60',
              pct_used: 83,
              fill_fraction: 0.17
            }
          }
        })
      )
    )

    expect(out).toContain('Plan: Pro · $3.40 left')
    expect(out).toContain('Low balance')
  })

  it('not-admin: shows read-only note', () => {
    const out = render(
      overlay(
        state({
          is_admin: false,
          can_change_plan: false,
          role: 'MEMBER',
          current: {
            tier_id: 'pro',
            tier_name: 'Pro',
            monthly_credits: '1000',
            credits_remaining: '500',
            cycle_ends_at: '2026-07-01',
            pending_downgrade_tier_name: null,
            pending_downgrade_at: null
          },
          usage: { available: true, status: 'healthy', plan_name: 'Pro' }
        })
      )
    )

    expect(out).toContain('view only')
    expect(out).toContain('Manage on portal')
  })

  it('downgrade-pending: leads with a Pro ──▶ Free banner + status echo', () => {
    const out = render(
      overlay(
        state({
          current: {
            tier_id: 'pro',
            tier_name: 'Pro',
            monthly_credits: '1000',
            credits_remaining: '500',
            cycle_ends_at: '2026-07-01',
            pending_downgrade_tier_name: 'Free',
            pending_downgrade_at: '2026-07-15',
            pending_downgrade_display: 'Jul 15, 2026'
          },
          usage: { available: true, status: 'healthy', plan_name: 'Pro' }
        })
      )
    )

    expect(out).toContain('Scheduled change')
    expect(out).toContain('──▶')
    expect(out).toContain('Free')
    expect(out).toContain('Jul 15, 2026')
    // the status line itself echoes the transition
    expect(out).toContain('Plan: Pro → Free')
  })

  it('team context: redirects to /topup, no tier picker', () => {
    const out = render(overlay(state({ context: 'team', current: null })))

    expect(out).toContain('shared balance')
    expect(out).toContain('/topup')
  })
})

// In-terminal change flow (V3): picker → confirm → result. useInput is mocked
// (no key simulation), so these assert each screen's rendered content.

const subscriber = (overrides: Partial<SubscriptionStateResponse> = {}): SubscriptionStateResponse =>
  state({
    current: {
      tier_id: 'plus',
      tier_name: 'Plus',
      monthly_credits: '1000',
      credits_remaining: '500',
      cycle_ends_at: '2026-07-01',
      pending_downgrade_tier_name: null,
      pending_downgrade_at: null
    },
    tiers: TIERS,
    usage: { available: true, status: 'healthy', plan_name: 'Plus' },
    ...overrides
  })

const at = (
  screen: SubscriptionOverlayState['screen'],
  s: SubscriptionStateResponse,
  extra: Partial<SubscriptionOverlayState> = {}
): SubscriptionOverlayState => ({ ctx, screen, state: s, ...extra })

// Free account (no current sub) where NAS still returns the tier catalog.
const freeWithCatalog = (): SubscriptionStateResponse =>
  state({
    current: null,
    tiers: TIERS.map(tier => ({ ...tier, is_current: false })),
    usage: { available: true, plan_name: null, status: 'free' }
  })

describe('SubscriptionOverlay — overview actions', () => {
  it('admin subscriber: offers Change plan + Cancel subscription', () => {
    const out = render(overlay(subscriber()))

    expect(out).toContain('Change plan')
    expect(out).toContain('Cancel subscription')
  })

  it('pending change: offers undo instead of cancel', () => {
    const out = render(
      overlay(
        subscriber({
          current: {
            tier_id: 'plus',
            tier_name: 'Plus',
            monthly_credits: '1000',
            credits_remaining: '500',
            cycle_ends_at: '2026-07-01',
            cancel_at_period_end: true,
            cancellation_effective_at: '2026-07-01',
            pending_downgrade_tier_name: null,
            pending_downgrade_at: null
          }
        })
      )
    )

    // undo is promoted to the first action; the banner shows the pending cancel
    expect(out).toContain('Keep Plus (undo this change)')
    expect(out).toContain('cancels')
    expect(out).not.toContain('Cancel subscription')
  })
})

describe('SubscriptionOverlay — step-up', () => {
  it('prompts to allow Remote Spending (never leaks the raw scope)', () => {
    const out = render(at('stepup', subscriber(), { stepUpRetry: { kind: 'preview', tierId: 'ultra' } }))

    expect(out).toContain('Remote Spending')
    expect(out).toContain('Allow Remote Spending')
    expect(out).not.toContain('billing:manage')
  })
})

describe('SubscriptionOverlay — picker', () => {
  it('lists other paid tiers with direction hints; hides current + free', () => {
    const out = render(at('picker', subscriber()))

    expect(out).toContain('Ultra')
    expect(out).toContain('$40/mo')
    expect(out).toContain('upgrade') // ultra (order 2) > plus (order 1)
    expect(out).not.toContain('Plus · $20/mo') // current tier is not selectable
    expect(out).not.toContain('$0/mo') // free tier excluded — use Cancel instead
  })
})

describe('SubscriptionOverlay — confirm', () => {
  it('charge_now: shows the prorated charge + upgrade copy', () => {
    const out = render(
      at('confirm', subscriber(), {
        pending: {
          kind: 'upgrade',
          targetTierId: 'ultra',
          preview: {
            ok: true,
            effect: 'charge_now',
            target_tier_name: 'Ultra',
            amount_due_now_cents: 1234,
            monthly_credits_delta: '2000'
          }
        }
      })
    )

    expect(out).toContain('Pay $12.34 & upgrade now')
    expect(out).toContain('Upgrade to Ultra')
  })

  it('scheduled: shows effective date + no charge now', () => {
    const out = render(
      at('confirm', subscriber(), {
        pending: {
          kind: 'tier_change',
          targetTierId: 'plus',
          preview: {
            ok: true,
            effect: 'scheduled',
            target_tier_name: 'Plus',
            effective_at: '2026-08-01T00:00:00Z',
            amount_due_now_cents: null
          }
        }
      })
    )

    expect(out).toContain('Schedule change to Plus')
    expect(out).toContain('2026-08-01')
    expect(out).toContain('No charge now')
  })

  it('cancellation: shows cancel-at-period-end copy', () => {
    const out = render(
      at('confirm', subscriber(), { pending: { kind: 'cancellation', targetTierId: null, preview: null } })
    )

    expect(out).toContain('Confirm cancellation')
    expect(out).toContain('will not renew')
  })

  it('blocked: shows the reason + Manage on portal', () => {
    const out = render(
      at('confirm', subscriber(), {
        pending: {
          kind: 'tier_change',
          targetTierId: 'ultra',
          preview: { ok: true, effect: 'blocked', reason: 'Retract the cancellation before upgrading.' }
        }
      })
    )

    expect(out).toContain('Retract the cancellation')
    expect(out).toContain('Manage on portal')
  })
})

describe('SubscriptionOverlay — result', () => {
  it('ok: shows Done + the re-run hint', () => {
    const out = render(at('result', subscriber(), { result: { ok: true, message: 'Upgraded to Ultra.' } }))

    expect(out).toContain('Done')
    expect(out).toContain('Upgraded to Ultra.')
    expect(out).toContain('Re-run /subscription')
  })

  it('error with recovery: shows the message + Open the portal', () => {
    const out = render(
      at('result', subscriber(), {
        result: {
          ok: false,
          message: 'This upgrade needs extra verification (3DS).',
          recoveryUrl: 'https://portal.example/x'
        }
      })
    )

    expect(out).toContain('Could not complete')
    expect(out).toContain('3DS')
    expect(out).toContain('Open the portal to finish')
  })
})

describe('SubscriptionOverlay — upgrade response mapping', () => {
  const applyUpgrade = async (response: unknown) => {
    const onPatch = vi.fn()
    const upgrade = vi.fn(() => Promise.resolve(response))

    const mounted = mount(
      at('confirm', subscriber(), {
        ctx: { ...ctx, upgrade } as SubscriptionOverlayState['ctx'],
        pending: {
          idempotencyKey: 'upgrade-key',
          kind: 'upgrade',
          preview: { ok: true, effect: 'charge_now', target_tier_name: 'Ultra', amount_due_now_cents: 1234 },
          targetTierId: 'ultra'
        }
      }),
      onPatch
    )

    inputHarness.handler?.('', { return: true })
    await vi.waitFor(() => expect(onPatch).toHaveBeenCalled())
    mounted.cleanup()

    return onPatch.mock.calls.at(-1)?.[0] as Partial<SubscriptionOverlayState>
  }

  it.each([
    ['authentication_required', 'upgraded'],
    ['subscription_payment_intent_requires_action', 'payment_failed']
  ])('reason %s routes to card verification regardless of status %s', async (reason, status) => {
    const patch = await applyUpgrade({
      ok: status === 'upgraded',
      reason,
      recovery_url: 'https://portal.example/verify',
      status,
      target_tier_name: 'Ultra'
    })

    expect(patch.screen).toBe('result')
    expect(patch.result?.ok).toBe(false)
    expect(patch.result?.message).toContain('verify your card in the portal')
    expect(patch.result?.recoveryUrl).toBe('https://portal.example/verify')
  })

  it('card_declined reason routes to a different-card recovery', async () => {
    const patch = await applyUpgrade({
      ok: false,
      reason: 'card_declined',
      recovery_url: 'https://portal.example/card',
      status: 'requires_action'
    })

    expect(patch.result?.message).toContain('try a different card on the portal')
    expect(patch.result?.recoveryUrl).toBe('https://portal.example/card')
  })

  it('already_on_tier remains an immediate success', async () => {
    const patch = await applyUpgrade({ ok: true, status: 'already_on_tier', target_tier_name: 'Ultra' })

    expect(patch.result).toMatchObject({ message: 'You are already on Ultra.', ok: true })
    expect(patch.result).not.toHaveProperty('pendingTierId')
  })

  it('upgraded marks the result as applying to the target tier', async () => {
    const patch = await applyUpgrade({ ok: true, status: 'upgraded', target_tier_name: 'Ultra' })

    expect(patch.result).toMatchObject({ ok: true, pendingTierId: 'ultra' })
  })

  it('shows Applying, then Done once refreshed state reaches the upgraded tier', async () => {
    vi.useFakeTimers()

    try {
      const refreshState = vi.fn(() =>
        Promise.resolve(
          subscriber({
            current: {
              tier_id: 'ultra',
              tier_name: 'Ultra',
              monthly_credits: '3000',
              credits_remaining: '3000',
              cycle_ends_at: '2026-08-01',
              pending_downgrade_tier_name: null,
              pending_downgrade_at: null
            }
          })
        )
      )

      const result = { message: 'Upgraded to Ultra.', ok: true, pendingTierId: 'ultra' }
      const mounted = mount(at('result', subscriber(), { ctx: { ...ctx, refreshState }, result }))

      expect(mounted.output()).toContain('Applying…')
      await vi.advanceTimersByTimeAsync(2000)
      mounted.rerender()
      expect(refreshState).toHaveBeenCalledTimes(1)
      expect(mounted.output()).toContain('Done')
      expect(mounted.output()).toContain('Upgraded to Ultra.')
      mounted.cleanup()
    } finally {
      vi.useRealTimers()
    }
  })

  it('keeps a successful upgrade soft-pending after the bounded confirmation window', async () => {
    vi.useFakeTimers()

    try {
      const refreshState = vi.fn(() => Promise.resolve(subscriber()))
      const result = { message: 'Upgraded to Ultra.', ok: true, pendingTierId: 'ultra' }
      const mounted = mount(at('result', subscriber(), { ctx: { ...ctx, refreshState }, result }))

      await vi.advanceTimersByTimeAsync(30_000)
      mounted.rerender()
      expect(refreshState).toHaveBeenCalledTimes(15)
      expect(mounted.output()).toContain('Still applying')
      expect(mounted.output()).toContain('refresh in a moment')
      expect(mounted.output()).not.toContain('Could not complete')
      mounted.cleanup()
    } finally {
      vi.useRealTimers()
    }
  })
})
