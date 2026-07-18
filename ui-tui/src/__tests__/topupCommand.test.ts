import { beforeEach, describe, expect, it, vi } from 'vitest'

import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { topupCommands } from '../app/slash/commands/topup.js'
import type { BillingStateResponse } from '../gatewayTypes.js'

vi.mock('../lib/openExternalUrl.js', () => ({
  openExternalUrl: vi.fn(() => true)
}))

const topupCommand = topupCommands.find(cmd => cmd.name === 'topup')!

const ownerState = (overrides: Partial<BillingStateResponse> = {}): BillingStateResponse => ({
  auto_reload: {
    card: { kind: 'canonical' },
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
    topupCommand.run(arg, ctx as any, 'topup')
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

  it('ctx.charge consent_required → one-time portal confirmation copy + portal funnel', async () => {
    const { run, sys } = buildCtx({
      'billing.state': ownerState(),
      'billing.charge': {
        ok: false,
        error: 'consent_required',
        portal_url: '/billing/consent',
        idempotency_key: 'k'
      }
    })

    await run('')
    await getOverlayState().billing!.ctx.charge('100')
    const out = printed(sys)
    expect(out).toContain('one-time card confirmation')
    expect(out).toContain('Portal: /billing/consent')
  })

  it.each([
    ['org_access_denied', "This token isn't bound to an org you can manage"],
    ['upgrade_cap_exceeded', 'Daily plan-change limit reached'],
    ['auto_top_up_disabled_failures', 'Auto-reload was turned off after repeated charge failures']
  ])('ctx.charge %s → typed recovery copy', async (error, copy) => {
    const { run, sys } = buildCtx({
      'billing.state': ownerState(),
      'billing.charge': { ok: false, error, idempotency_key: 'k' }
    })

    await run('')
    await getOverlayState().billing!.ctx.charge('100')
    expect(printed(sys)).toContain(copy)
  })

  it.each([
    [undefined, 'Stripe is having trouble right now — try again shortly.'],
    [120, 'Stripe is having trouble right now — try again shortly (try again in ~2 min).']
  ])('ctx.charge stripe_unavailable (retry_after=%s) → transient Stripe copy', async (retryAfter, copy) => {
    const { run, sys } = buildCtx({
      'billing.state': ownerState(),
      'billing.charge': {
        ok: false,
        error: 'stripe_unavailable',
        idempotency_key: 'k',
        ...(retryAfter == null ? {} : { retry_after: retryAfter })
      }
    })

    await run('')
    await getOverlayState().billing!.ctx.charge('100')
    const out = printed(sys)
    expect(out).toContain(copy)
    expect(out).not.toContain('Too many charges')
  })

  it('ctx.charge insufficient_scope → resolves needs_remote_spending (overlay routes to stepup)', async () => {
    const { run } = buildCtx({
      'billing.state': ownerState(),
      'billing.charge': { ok: false, error: 'insufficient_scope', idempotency_key: 'k' }
    })

    await run('')
    const outcome = await getOverlayState().billing!.ctx.charge('100')
    // No separate confirm overlay is armed anymore — the overlay's stepup
    // screen owns the UX; the ctx just reports the outcome.
    expect(outcome).toBe('needs_remote_spending')
    expect(getOverlayState().confirm).toBeNull()
  })

  it.each([[true], [false]])('ctx.requestRemoteSpending → billing.step_up resolves %s', async granted => {
    const { run, calls } = buildCtx({ 'billing.state': ownerState(), 'billing.step_up': { ok: true, granted } })

    await run('')
    expect(await getOverlayState().billing!.ctx.requestRemoteSpending()).toBe(granted)
    expect(calls.find(c => c.method === 'billing.step_up')).toBeTruthy()
  })

  // ── CF-4: revoked-terminal UX (kill the "15-minute zombie button") ──

  it.each([
    ['admin', 'An admin turned off terminal billing for this terminal'],
    ['self', 'You turned off terminal billing for this terminal']
  ])(
    'ctx.charge remote_spending_revoked (%s) → clears the overlay (no zombie button) + actor copy',
    async (actor, copy) => {
      const { run, sys } = buildCtx({
        'billing.state': ownerState(),
        'billing.charge': {
          ok: false,
          error: 'remote_spending_revoked',
          actor,
          recovery: 'reconnect',
          idempotency_key: 'k'
        }
      })

      await run('')
      getOverlayState().billing!.ctx.charge('100')
      await Promise.resolve()
      await Promise.resolve()
      expect(printed(sys)).toContain(copy)
      expect(getOverlayState().billing).toBeNull()
    }
  )

  it('ctx.charge session_revoked → clears overlay + re-login (not reconnect) copy', async () => {
    const { run, sys } = buildCtx({
      'billing.state': ownerState(),
      'billing.charge': { ok: false, error: 'session_revoked', recovery: 'login', idempotency_key: 'k' }
    })

    await run('')
    getOverlayState().billing!.ctx.charge('100')
    await Promise.resolve()
    await Promise.resolve()
    expect(printed(sys)).toContain('Your session was logged out')
    expect(getOverlayState().billing).toBeNull()
  })

  it('ctx.charge → poll transport loss reports an unconfirmed outcome', async () => {
    const { ctx, rpc, run, sys } = buildCtx({
      'billing.state': ownerState(),
      'billing.charge': { ok: true, charge_id: 'ch_1', idempotency_key: 'k' }
    })

    await run('')
    rpc.mockImplementation((method: string) => {
      if (method === 'billing.charge_status') {
        return Promise.reject(new Error('socket closed'))
      }

      return Promise.resolve(method === 'billing.charge' ? { ok: true, charge_id: 'ch_1', idempotency_key: 'k' } : null)
    })
    await getOverlayState().billing!.ctx.charge('100')
    await vi.waitFor(() => expect(printed(sys)).toContain('outcome is unconfirmed'))
    expect(ctx.guardedErr).toHaveBeenCalled()
  })

  it('ctx.charge cli_billing_disabled / remote_spending_disabled → account-toggle copy', async () => {
    const { run, sys } = buildCtx({
      'billing.state': ownerState(),
      'billing.charge': {
        ok: false,
        error: 'cli_billing_disabled',
        code: 'remote_spending_disabled',
        recovery: 'enable_account_toggle',
        portal_url: '/billing',
        idempotency_key: 'k'
      }
    })

    await run('')
    getOverlayState().billing!.ctx.charge('100')
    await Promise.resolve()
    await Promise.resolve()
    const out = printed(sys)
    expect(out).toContain('Terminal billing is off for this account')
    // Account-wide switch is NOT a per-terminal revoke — overlay stays open.
    expect(getOverlayState().billing).toBeTruthy()
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
          card: { kind: 'canonical' },
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

  it('ctx.charge → poll → processing_error has intentional failure copy', async () => {
    vi.useFakeTimers()

    try {
      const { run, sys } = buildCtx({
        'billing.state': ownerState(),
        'billing.charge': { ok: true, charge_id: 'ch_1', idempotency_key: 'k' },
        'billing.charge_status': { ok: true, status: 'failed', reason: 'processing_error' }
      })

      await run('')
      getOverlayState().billing!.ctx.charge('100')
      await vi.runAllTimersAsync()
      expect(printed(sys)).toContain("The charge didn't go through (processing_error).")
    } finally {
      vi.useRealTimers()
    }
  })

  it('ctx.openPortal opens the URL + echoes a transcript line', async () => {
    const { run, sys } = buildCtx({ 'billing.state': ownerState() })
    await run('')
    getOverlayState().billing!.ctx.openPortal('https://portal/x')
    expect(printed(sys)).toContain('Opening portal: https://portal/x')
  })
})
