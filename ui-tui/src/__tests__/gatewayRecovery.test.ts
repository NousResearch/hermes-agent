import { describe, expect, it } from 'vitest'

import { evalRecovery, GATEWAY_RECOVERY_LIMIT, GATEWAY_RECOVERY_WINDOW_MS } from '../app/gatewayRecovery.js'

describe('evalRecovery', () => {
  it('allows up to the limit within the window, then blocks', () => {
    let attempts: number[] = []
    const t0 = 1_000_000

    for (let i = 0; i < GATEWAY_RECOVERY_LIMIT; i++) {
      const { allowed, recent } = evalRecovery(attempts, t0 + i)

      expect(allowed).toBe(true)
      attempts = [...recent, t0 + i]
    }

    // Budget exhausted: the next death within the window falls back to inert.
    expect(evalRecovery(attempts, t0 + GATEWAY_RECOVERY_LIMIT).allowed).toBe(false)
  })

  it('prunes attempts older than the window so recovery is allowed again', () => {
    const old = Array.from({ length: GATEWAY_RECOVERY_LIMIT }, (_, i) => i)
    const now = GATEWAY_RECOVERY_WINDOW_MS + 100

    const { allowed, recent } = evalRecovery(old, now)

    expect(recent).toEqual([])
    expect(allowed).toBe(true)
  })
})
