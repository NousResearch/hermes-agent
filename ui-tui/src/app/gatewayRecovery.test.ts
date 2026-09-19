import { describe, expect, it, beforeEach } from 'vitest'
import { planGatewayRecovery, GATEWAY_RECOVERY_LIMIT } from '../app/gatewayRecovery.js'

describe('planGatewayRecovery', () => {
  it('recovers on first attempt with live sid', () => {
    const plan = planGatewayRecovery('sid-1', null, [], Date.now())
    expect(plan.recover).toBe(true)
    expect(plan.sid).toBe('sid-1')
    expect(plan.attempts).toHaveLength(1)
  })

  it('uses recoverSid when liveSid is null', () => {
    const plan = planGatewayRecovery(null, 'sid-pending', [], Date.now())
    expect(plan.recover).toBe(true)
    expect(plan.sid).toBe('sid-pending')
  })

  it('does not recover when both sids are null', () => {
    const plan = planGatewayRecovery(null, null, [], Date.now())
    expect(plan.recover).toBe(false)
  })

  it('stops recovering after limit reached', () => {
    const now = Date.now()
    const attempts = Array.from({ length: GATEWAY_RECOVERY_LIMIT }, (_, i) => now - i * 1000)
    const plan = planGatewayRecovery('sid-1', null, attempts, now)
    expect(plan.recover).toBe(false)
  })

  it('prunes old attempts outside the window', () => {
    const now = Date.now()
    const oldAttempts = [now - 120000, now - 130000, now - 140000] // all outside 60s window
    const plan = planGatewayRecovery('sid-1', null, oldAttempts, now)
    expect(plan.recover).toBe(true) // old attempts pruned, budget available
  })
})
