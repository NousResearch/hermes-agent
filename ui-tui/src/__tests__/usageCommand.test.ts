import { beforeEach, describe, expect, it, vi } from 'vitest'

import { sessionCommands } from '../app/slash/commands/session.js'
import type { SessionUsageResponse } from '../gatewayTypes.js'

const usageCommand = sessionCommands.find(cmd => cmd.name === 'usage')!

const USAGE_CTA = 'Run /subscription to change plan · /topup to add to your balance'

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
  const panel = vi.fn()

  const rpc = vi.fn((method: string, _params: unknown) => Promise.resolve(results[method]))

  const ctx = {
    gateway: { rpc },
    guarded,
    guardedErr: vi.fn(),
    sid: 'sid-1',
    stale: () => false,
    transcript: { page: vi.fn(), panel, sys }
  }

  const run = async (arg: string) => {
    usageCommand.run(arg, ctx as any, 'usage')
    await rpc.mock.results[0]?.value
    await Promise.resolve()
    await Promise.resolve()
  }

  return { ctx, panel, run, sys }
}

const baseUsage = (overrides: Partial<SessionUsageResponse> = {}): SessionUsageResponse =>
  ({ calls: 0, input: 0, output: 0, total: 0, ...overrides }) as SessionUsageResponse

const printed = (sys: ReturnType<typeof vi.fn>) => sys.mock.calls.map(c => c[0]).join('\n')

const balancePanel = (panel: ReturnType<typeof vi.fn>) => {
  const sections = panel.mock.calls.find(c => c[0] === 'Balance')?.[1] as { text?: string }[] | undefined

  return (sections ?? []).map(s => s.text ?? '').join('\n')
}

describe('/usage slash command', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('always shows the CTA; "no API calls yet" only when there is no balance', async () => {
    const empty = buildCtx({ 'session.usage': baseUsage({ calls: 0, credits_lines: [] }) })
    await empty.run('')
    expect(printed(empty.sys)).toContain('no API calls yet')
    expect(printed(empty.sys)).toContain(USAGE_CTA)

    const withBalance = buildCtx({ 'session.usage': baseUsage({ calls: 0, credits_lines: ['$50.00 remaining'] }) })
    await withBalance.run('')
    expect(printed(withBalance.sys)).not.toContain('no API calls yet')
    expect(printed(withBalance.sys)).toContain(USAGE_CTA)
  })

  it('renders the dollar two-bar model (no "credits" wording) when available', async () => {
    const { panel, run } = buildCtx({
      'session.usage': baseUsage({
        usage: {
          available: true,
          status: 'healthy',
          plan_name: 'Plus',
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
    })

    await run('')

    const body = balancePanel(panel)
    expect(body).toContain('Plus')
    expect(body).toContain('$14.00 left of $20.00')
    expect(body).toContain('30% used')
    expect(body).toContain('top-up')
    expect(body).toContain('$12.00')
    expect(body.toLowerCase()).not.toContain('credits')
  })

  it('shows the free-models upsell for a free account', async () => {
    const { panel, run } = buildCtx({
      'session.usage': baseUsage({ usage: { available: true, status: 'free', plan_name: null } })
    })

    await run('')

    const body = balancePanel(panel)
    expect(body).toContain('free models only')
    expect(body).toContain('/subscription')
  })
})
