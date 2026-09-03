import { describe, expect, it } from 'vitest'

import { mergeUsageStable, usageChanged } from '../app/createGatewayEventHandler.js'
import type { Usage } from '../types.js'

const baseUsage: Usage = {
  calls: 3,
  input: 1200,
  output: 400,
  total: 1600,
  context_max: 200000,
  context_percent: 12,
  context_used: 24000
}

describe('mergeUsageStable (#41480 status-bar flicker)', () => {
  it('returns the PRIOR reference when a patch changes nothing', () => {
    // The load-bearing behavior: an unchanged-value patch must NOT mint a new
    // object, or every $uiState subscriber re-renders per streaming delta.
    const patch = { calls: 3, total: 1600 }
    expect(mergeUsageStable(baseUsage, patch)).toBe(baseUsage)
  })

  it('returns the prior reference for an undefined patch', () => {
    expect(mergeUsageStable(baseUsage, undefined)).toBe(baseUsage)
  })

  it('returns a new merged object when a value actually changes', () => {
    const merged = mergeUsageStable(baseUsage, { total: 1700 })
    expect(merged).not.toBe(baseUsage)
    expect(merged.total).toBe(1700)
    expect(merged.calls).toBe(3)
  })

  it('detects an active_subagents-only update (field the original PR missed)', () => {
    // usageChanged iterates the key union generically, so optional fields the
    // status rule consumes (active_subagents drives the ⛓ segment and the
    // resume hint) can never be silently dropped from the comparison.
    const withSubagents = mergeUsageStable(baseUsage, { active_subagents: 2 })
    expect(withSubagents).not.toBe(baseUsage)
    expect(withSubagents.active_subagents).toBe(2)

    // And clearing it back down is also a change.
    const cleared = mergeUsageStable(withSubagents, { active_subagents: 0 })
    expect(cleared).not.toBe(withSubagents)
    expect(cleared.active_subagents).toBe(0)
  })

  it('treats a key present on only one side as a change', () => {
    expect(usageChanged(baseUsage, { ...baseUsage, cost_usd: 0.01 })).toBe(true)
    expect(usageChanged({ ...baseUsage, cost_usd: 0.01 }, baseUsage)).toBe(true)
  })

  it('reports no change for deep-equal usages', () => {
    expect(usageChanged(baseUsage, { ...baseUsage })).toBe(false)
  })
})
