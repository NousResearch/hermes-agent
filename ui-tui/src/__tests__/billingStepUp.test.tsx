import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it, vi } from 'vitest'

// Stub useInput so the overlay doesn't enter raw mode under renderSync.
vi.mock('@hermes/ink', async importOriginal => {
  const mod = await importOriginal()

  return { ...mod, useInput: () => {} }
})

import type { BillingOverlayState } from '../app/interfaces.js'
import { BillingOverlay } from '../components/billingOverlay.js'
import type { BillingStateResponse } from '../gatewayTypes.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const t = DEFAULT_THEME

function render(overlay: BillingOverlayState): string {
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

  const instance = renderSync(
    React.createElement(BillingOverlay, {
      onClose: () => {},
      onPatch: () => {},
      overlay,
      t
    }),
    {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    }
  )

  instance.unmount()
  instance.cleanup()

  return stripAnsi(output)
}

const billState = (overrides: Partial<BillingStateResponse> = {}): BillingStateResponse =>
  ({
    auto_reload: null,
    balance_display: '$12.00',
    balance_usd: '12',
    can_charge: true,
    card: { brand: 'visa', last4: '4242', masked: 'visa ····4242' },
    charge_presets: ['25', '50'],
    charge_presets_display: ['$25', '$50'],
    cli_billing_enabled: true,
    is_admin: true,
    logged_in: true,
    max_usd: '1000',
    min_usd: '10',
    monthly_cap: null,
    ok: true,
    org_name: 'Acme',
    portal_url: 'https://portal/billing',
    role: 'OWNER',
    ...overrides
  }) as BillingStateResponse

const ctx = {
  applyAutoReload: vi.fn(() => Promise.resolve(true)),
  charge: vi.fn(() => Promise.resolve('submitted' as const)),
  openPortal: vi.fn(),
  refreshState: vi.fn(() => Promise.resolve(null)),
  requestRemoteSpending: vi.fn(() => Promise.resolve(true)),
  sys: vi.fn(),
  validate: vi.fn((raw: string) => ({ amount: raw }))
}

const overlay = (screen: BillingOverlayState['screen']): BillingOverlayState => ({
  ctx,
  pendingCharge: { amount: '100' },
  screen,
  state: billState()
})

describe('BillingOverlay — step-up screen (Enable terminal billing)', () => {
  it('renders the one-time-setup prompt with the held amount, never leaking the raw scope', () => {
    const out = render(overlay('stepup'))
    expect(out).toContain('One-time setup')
    expect(out).toContain('Enable terminal billing')
    expect(out).toContain('$100') // resumes the held purchase
    expect(out).toContain('Not now')
    expect(out).not.toContain('billing:manage')
  })
})

describe('BillingOverlay — overview (reordered, dollars)', () => {
  it('leads with balance in the title, Add funds first, no "credits"', () => {
    const out = render(overlay('overview'))
    expect(out).toContain('Top up · balance $12.00') // balance in the title
    expect(out).toContain('Add funds') // buy action, renamed
    expect(out).toContain('Auto-reload')
    expect(out).toContain('Manage on portal')
    expect(out.toLowerCase()).not.toContain('credits') // dollars only
    // No standalone "Enable terminal billing" item — discovered at pay time.
    expect(out).not.toContain('Enable terminal billing')
  })

  it('renders the two-bar dollar usage when a usage model is present', () => {
    const withUsage: BillingOverlayState = {
      ...overlay('overview'),
      state: {
        ...billState(),
        usage: {
          available: true,
          status: 'healthy',
          plan_name: 'Plus',
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
      }
    }

    const out = render(withUsage)
    expect(out).toContain('$14.00 left of $20.00')
    expect(out).toContain('30% used')
    expect(out).toContain('never expires')
  })
})

describe('BillingOverlay — auto-reload card divergence', () => {
  const autoReload = (card: NonNullable<BillingStateResponse['auto_reload']>['card']) => ({
    card,
    enabled: true,
    reload_to_display: '$100',
    reload_to_usd: '100',
    threshold_display: '$20',
    threshold_usd: '20'
  })

  it('warns when auto-reload charges a distinct card and offers the portal hand-off', () => {
    const out = render({
      ...overlay('autoreload'),
      state: billState({
        auto_reload: autoReload({ kind: 'distinct', payment_method_id: 'pm_other', brand: 'Visa', last4: '9999' })
      })
    })

    expect(out).toContain('Auto-refill is charging Visa ••9999 — not your card on file')
    expect(out).toContain('authorize Nous Research to charge Visa ••9999')
    expect(out).toContain('Use your card on file — manage on portal')
  })

  it('uses generic distinct-card copy when metadata is unresolved', () => {
    const out = render({
      ...overlay('autoreload'),
      state: billState({
        auto_reload: autoReload({ kind: 'distinct', payment_method_id: 'pm_other', brand: null, last4: null })
      })
    })

    expect(out).toContain('Auto-refill is charging a different card — not your card on file')
  })

  it.each(['canonical', 'none'] as const)('does not warn for a %s auto-reload card', kind => {
    const out = render({
      ...overlay('autoreload'),
      state: billState({ auto_reload: autoReload({ kind }) })
    })

    expect(out).not.toContain('not your card on file')
    expect(out).not.toContain('Use your card on file — manage on portal')
  })
})
