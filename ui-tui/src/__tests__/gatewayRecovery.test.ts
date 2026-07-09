import { describe, expect, it } from 'vitest'

import { GATEWAY_RECOVERY_LIMIT, GATEWAY_RECOVERY_WINDOW_MS, planGatewayRecovery } from '../app/gatewayRecovery.js'

describe('planGatewayRecovery', () => {
  it('recovers the live session and records the attempt', () => {
    const plan = planGatewayRecovery('sess-1', null, [], 1000)

    expect(plan).toEqual({ attempts: [1000], recover: true, sid: 'sess-1' })
  })

  it('does not recover when there is no session to resume', () => {
    expect(planGatewayRecovery(null, null, [], 1000)).toEqual({ attempts: [], recover: false, sid: null })
  })

  it('keeps retrying the recovery target through a startup crash-loop, bounded by the budget', () => {
    // First exit: live sid present.
    let attempts: number[] = []
    let plan = planGatewayRecovery('sess-1', null, attempts, 0)

    expect(plan.recover).toBe(true)
    expect(plan.sid).toBe('sess-1')
    attempts = plan.attempts

    // Respawn crash-loops before gateway.ready: live sid is now null, but the
    // recovery target carries it forward so we keep trying up to the budget.
    for (let i = 1; i < GATEWAY_RECOVERY_LIMIT; i++) {
      plan = planGatewayRecovery(null, 'sess-1', attempts, i)
      expect(plan.recover).toBe(true)
      expect(plan.sid).toBe('sess-1')
      attempts = plan.attempts
    }

    // Budget exhausted: fall back to the inert state instead of spawn-storming.
    plan = planGatewayRecovery(null, 'sess-1', attempts, GATEWAY_RECOVERY_LIMIT)
    expect(plan.recover).toBe(false)
    expect(plan.sid).toBe('sess-1')
  })

  it('prunes attempts older than the window so recovery re-arms', () => {
    const old = Array.from({ length: GATEWAY_RECOVERY_LIMIT }, (_, i) => i)
    const plan = planGatewayRecovery('sess-1', null, old, GATEWAY_RECOVERY_WINDOW_MS + 100)

    expect(plan.attempts).toEqual([GATEWAY_RECOVERY_WINDOW_MS + 100])
    expect(plan.recover).toBe(true)
  })
})
