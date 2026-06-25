import { beforeEach, describe, expect, it, vi } from 'vitest'

import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { billingCommands } from '../app/slash/commands/billing.js'
import type { BillingStateResponse } from '../gatewayTypes.js'

vi.mock('../lib/openExternalUrl.js', () => ({
  openExternalUrl: vi.fn(() => true)
}))

const billingCommand = billingCommands.find(cmd => cmd.name === 'billing')!

const ownerState = (overrides: Partial<BillingStateResponse> = {}): BillingStateResponse => ({
  auto_reload: {
    enabled: false,
    reload_to_display: '—',
    reload_to_usd: null,
    threshold_display: '—',
    threshold_usd: null
  },
  balance_display: '$142.50',
  balance_usd: '142.5',
  can_charge: true,
  card: { brand: 'visa', last4: '4242', masked: 'visa ····4242' },
  charge_presets: ['25', '50', '100'],
  charge_presets_display: ['$25', '$50', '$100'],
  cli_billing_enabled: true,
  is_admin: true,
  logged_in: true,
  max_usd: '10000',
  min_usd: '10',
  monthly_cap: {
    is_default_ceiling: true,
    limit_display: '$1000',
    limit_usd: '1000',
    spent_display: '$180',
    spent_this_month_usd: '180'
  },
  ok: true,
  org_name: 'Acme',
  portal_url: 'https://portal/billing?topup=open',
  role: 'OWNER',
  ...overrides
})

const guarded =
  <T>(fn: (r: T) => void) =>
  (r: null | T) => {
    if (r) {
      fn(r)
    }
  }

/** Build a ctx whose rpc routes by method name to a supplied map of results. */
const buildCtx = (results: Record<string, unknown>) => {
  const sys = vi.fn()
  const calls: Array<{ method: string; params: unknown }> = []

  const rpc = vi.fn((method: string, params: unknown) => {
    calls.push({ method, params })

    return Promise.resolve(results[method])
  })

  const ctx = {
    gateway: { rpc },
    guarded,
    guardedErr: vi.fn(),
    sid: 'sid-1',
    stale: () => false,
    transcript: { page: vi.fn(), panel: vi.fn(), sys }
  }

  const run = async (arg: string) => {
    billingCommand.run(arg, ctx as any, 'billing')
    await rpc.mock.results[0]?.value
    await Promise.resolve()
    await Promise.resolve()
  }

  return { calls, ctx, rpc, run, sys }
}

const printed = (sys: ReturnType<typeof vi.fn>) => sys.mock.calls.map(c => c[0]).join('\n')

describe('/billing slash command (overlay-driven)', () => {
  beforeEach(() => {
    resetOverlayState()
  })

  it('not logged in → prompts to log in, no overlay', async () => {
    const { run, sys } = buildCtx({ 'billing.state': { ...ownerState(), logged_in: false, ok: true } })
    await run('')
    expect(printed(sys)).toContain('Not logged into Nous Portal')
    expect(getOverlayState().billing).toBeNull()
  })

  it('bare /billing opens the overlay on the overview screen with state', async () => {
    const { run, rpc } = buildCtx({ 'billing.state': ownerState() })
    await run('')
    expect(rpc).toHaveBeenCalledWith('billing.state', {})
    const billing = getOverlayState().billing
    expect(billing).toBeTruthy()
    expect(billing?.screen).toBe('overview')
    expect(billing?.state.balance_display).toBe('$142.50')
    expect(billing?.state.charge_presets_display).toEqual(['$25', '$50', '$100'])
  })

  it('any sub-command arg is ignored — still opens the overview overlay', async () => {
    const { run } = buildCtx({ 'billing.state': ownerState() })
    await run('buy 100')
    const billing = getOverlayState().billing
    expect(billing?.screen).toBe('overview')
    // No confirm overlay armed directly by the command anymore.
    expect(getOverlayState().confirm).toBeNull()
  })

  it('member overview carries the non-admin state for component-side gating', async () => {
    const { run } = buildCtx({
      'billing.state': ownerState({
        is_admin: false,
        can_charge: false,
        role: 'MEMBER',
        card: null,
        monthly_cap: null,
        auto_reload: null
      })
    })

    await run('')
    const billing = getOverlayState().billing
    expect(billing?.state.is_admin).toBe(false)
    expect(billing?.screen).toBe('overview')
  })

  // ── Overlay ctx behaviors (RPC + error mapping live in billing.ts) ──

  it('ctx.validate rejects out-of-bounds and sub-cent amounts, accepts valid', async () => {
    const { run } = buildCtx({ 'billing.state': ownerState() })
    await run('')
    const ctx = getOverlayState().billing!.ctx
    expect(ctx.validate('5').error).toContain('Minimum is $10')
    expect(ctx.validate('10.005').error).toContain('2 decimal places')
    expect(ctx.validate('100').amount).toBe('100')
    expect(ctx.validate('$50').amount).toBe('50')
  })

  it('ctx.charge → poll → settled', async () => {
    vi.useFakeTimers()

    try {
      const { run, sys } = buildCtx({
        'billing.state': ownerState(),
        'billing.charge': { ok: true, charge_id: 'ch_1', idempotency_key: 'k' },
        'billing.charge_status': { ok: true, status: 'settled', amount_usd: '100' }
      })

      await run('')
      const ctx = getOverlayState().billing!.ctx
      ctx.charge('100')
      await vi.runAllTimersAsync()
      const out = printed(sys)
      expect(out).toContain('Charge submitted')
      expect(out).toContain('✅ $100 added.')
    } finally {
      vi.useRealTimers()
    }
  })

  it('ctx.charge → poll → failed adds the portal funnel line', async () => {
    vi.useFakeTimers()

    try {
      const { run, sys } = buildCtx({
        'billing.state': ownerState(),
        'billing.charge': { ok: true, charge_id: 'ch_1', idempotency_key: 'k' },
        'billing.charge_status': { ok: true, status: 'failed', reason: 'card_declined' }
      })

      await run('')
      getOverlayState().billing!.ctx.charge('100')
      await vi.runAllTimersAsync()
      const out = printed(sys)
      expect(out).toContain('Your card was declined')
      // Parity with the CLI: a failed poll funnels to the portal (from state.portal_url).
      expect(out).toContain('Portal: https://portal/billing?topup=open')
    } finally {
      vi.useRealTimers()
    }
  })

  it('ctx.charge monthly_cap_exceeded surfaces remaining headroom', async () => {
    const { run, sys } = buildCtx({
      'billing.state': ownerState(),
      'billing.charge': {
        ok: false,
        error: 'monthly_cap_exceeded',
        message: 'Monthly spend cap reached.',
        payload: { remainingUsd: '42.50' },
        portal_url: '/billing?topup=open',
        idempotency_key: 'k'
      }
    })

    await run('')
    getOverlayState().billing!.ctx.charge('100')
    await Promise.resolve()
    await Promise.resolve()
    const out = printed(sys)
    expect(out).toContain('Monthly spend cap reached — $42.50 headroom left.')
    expect(out).toContain('Portal: /billing?topup=open')
  })

  it('ctx.charge no_payment_method → portal funnel copy', async () => {
    const { run, sys } = buildCtx({
      'billing.state': ownerState(),
      'billing.charge': {
        ok: false,
        error: 'no_payment_method',
        portal_url: '/billing?topup=open',
        idempotency_key: 'k'
      }
    })

    await run('')
    getOverlayState().billing!.ctx.charge('100')
    await Promise.resolve()
    await Promise.resolve()
    const out = printed(sys)
    expect(out).toContain('No saved card for terminal charges')
    expect(out).toContain('Portal: /billing?topup=open')
  })

  it('ctx.charge insufficient_scope → arms step-up confirm', async () => {
    const { run } = buildCtx({
      'billing.state': ownerState(),
      'billing.charge': { ok: false, error: 'insufficient_scope', idempotency_key: 'k' }
    })

    await run('')
    getOverlayState().billing!.ctx.charge('100')
    await Promise.resolve()
    await Promise.resolve()
    // The charge failed with insufficient_scope → a NEW confirm (step-up) is armed.
    const stepUp = getOverlayState().confirm
    expect(stepUp?.title).toBe('Grant terminal billing access?')
  })

  it('ctx.applyAutoReload(true, …) → billing.auto_reload RPC, resolves true', async () => {
    const { run, calls } = buildCtx({
      'billing.state': ownerState(),
      'billing.auto_reload': { ok: true }
    })

    await run('')
    const ok = await getOverlayState().billing!.ctx.applyAutoReload(true, 20, 100)
    expect(ok).toBe(true)
    const ar = calls.find(c => c.method === 'billing.auto_reload')
    expect(ar?.params).toEqual({ enabled: true, threshold: 20, top_up_amount: 100 })
  })

  it('ctx.applyAutoReload(false) → disables (enabled:false, no amounts)', async () => {
    const { run, calls } = buildCtx({
      'billing.state': ownerState({
        auto_reload: {
          enabled: true,
          reload_to_display: '$100',
          reload_to_usd: '100',
          threshold_display: '$20',
          threshold_usd: '20'
        }
      }),
      'billing.auto_reload': { ok: true }
    })

    await run('')
    const ok = await getOverlayState().billing!.ctx.applyAutoReload(false)
    expect(ok).toBe(true)
    const ar = calls.find(c => c.method === 'billing.auto_reload')
    expect(ar?.params).toEqual({ enabled: false })
  })

  it('ctx.applyAutoReload error → resolves false + maps the error', async () => {
    const { run, sys } = buildCtx({
      'billing.state': ownerState(),
      'billing.auto_reload': { ok: false, error: 'monthly_cap_exceeded', message: 'Monthly spend cap reached.' }
    })

    await run('')
    const ok = await getOverlayState().billing!.ctx.applyAutoReload(true, 20, 100)
    expect(ok).toBe(false)
    expect(printed(sys)).toContain('Monthly spend cap reached.')
  })

  it('ctx.openPortal opens the URL + echoes a transcript line', async () => {
    const { run, sys } = buildCtx({ 'billing.state': ownerState() })
    await run('')
    getOverlayState().billing!.ctx.openPortal('https://portal/x')
    expect(printed(sys)).toContain('Opening portal: https://portal/x')
  })
})
