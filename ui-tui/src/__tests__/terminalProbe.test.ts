import { describe, expect, it, vi } from 'vitest'

import { probeTerminalCapabilities } from '../lib/terminalProbe.js'

type MockQuerier = {
  flush: () => Promise<void>
  send: <T>(query: { request: string; match: (r: unknown) => r is T }) => Promise<T | undefined>
}

describe('probeTerminalCapabilities', () => {
  it('returns xtversionName when response arrives', async () => {
    const querier: MockQuerier = {
      flush: vi.fn().mockResolvedValue(undefined),
      send: vi.fn().mockResolvedValue({ name: 'WezTerm' })
    }

    const result = await probeTerminalCapabilities(querier)

    expect(result.xtversionName).toBe('WezTerm')
  })

  it('returns empty result on timeout', async () => {
    const querier: MockQuerier = {
      flush: vi.fn().mockResolvedValue(undefined),
      send: vi.fn().mockImplementation(() => new Promise(resolve => setTimeout(() => resolve(undefined), 500)))
    }

    const result = await probeTerminalCapabilities(querier, { timeoutMs: 50 })

    expect(result.xtversionName).toBeUndefined()
    expect(result.bracketedPaste).toBeUndefined()
  })
})
