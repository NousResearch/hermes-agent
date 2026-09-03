import { beforeEach, describe, expect, it, vi } from 'vitest'

import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { subscriptionCommands } from '../app/slash/commands/subscription.js'
import { findSlashCommand } from '../app/slash/registry.js'
import type { SubscriptionStateResponse } from '../gatewayTypes.js'

vi.mock('../lib/openExternalUrl.js', () => ({
  openExternalUrl: vi.fn(() => true)
}))

const subscriptionCommand = subscriptionCommands.find(cmd => cmd.name === 'subscription')!

const loggedInState = (overrides: Partial<SubscriptionStateResponse> = {}): SubscriptionStateResponse => ({
  ok: true,
  logged_in: true,
  is_admin: true,
  can_change_plan: true,
  org_name: 'Acme',
  role: 'OWNER',
  current: null,
  portal_url: 'https://portal.nousresearch.com/billing',
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
    subscriptionCommand.run(arg, ctx as any, 'subscription')
    await rpc.mock.results[0]?.value
    await Promise.resolve()
    await Promise.resolve()
  }

  return { calls, ctx, rpc, run, sys }
}

const printed = (sys: ReturnType<typeof vi.fn>) => sys.mock.calls.map(c => c[0]).join('\n')

describe('/subscription slash command', () => {
  beforeEach(() => {
    resetOverlayState()
  })

  it('fetches subscription.state and opens the overlay', async () => {
    const { run } = buildCtx({
      'subscription.state': loggedInState()
    })

    await run('')

    const overlay = getOverlayState().subscription

    expect(overlay).not.toBeNull()
    expect(overlay?.screen).toBe('overview')
  })

  it('shows portal-login sys line when not logged in', async () => {
    const { run, sys } = buildCtx({
      'subscription.state': loggedInState({ logged_in: false })
    })

    await run('')

    expect(printed(sys)).toContain('Not logged into Nous Portal')
    expect(getOverlayState().subscription).toBeNull()
  })

  it('/upgrade alias resolves to the same command', () => {
    expect(findSlashCommand('upgrade')).toBe(subscriptionCommand)
  })

  it('/subscription resolves to the same command', () => {
    expect(findSlashCommand('subscription')).toBe(subscriptionCommand)
  })
})
