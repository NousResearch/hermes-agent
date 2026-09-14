import type { SessionUsageResult } from '@hermes/shared/gateway-events'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { sessionCommands } from '../app/slash/commands/session.js'
import { ZERO } from '../domain/usage.js'
import type { PanelSection } from '../types.js'

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

const baseUsage = (overrides: Partial<SessionUsageResult> = {}): SessionUsageResult => ({
  ...ZERO,
  credits_lines: null,
  ...overrides
})

const printed = (sys: ReturnType<typeof vi.fn>) => sys.mock.calls.map(c => c[0]).join('\n')

const balancePanel = (panel: ReturnType<typeof vi.fn>) => {
  // SAFETY: the spy records this command's own panel(title, PanelSection[]) call.
  const sections = panel.mock.calls.find(c => c[0] === 'Nous balance')?.[1] as PanelSection[] | undefined

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

  it('renders the Nous balance lines the gateway sends with the counters', async () => {
    const { panel, run } = buildCtx({
      'session.usage': baseUsage({ calls: 3, credits_lines: ['$26.00 remaining', 'renews Jul 1, 2026'] })
    })

    await run('')

    const body = balancePanel(panel)
    expect(body).toContain('$26.00 remaining')
    expect(body).toContain('renews Jul 1, 2026')
  })
})
