import type { Usage } from '@hermes/shared/gateway-events'
import { describe, expect, it } from 'vitest'

import { mergeUsageStable, usageChanged } from '../app/createGatewayEventHandler.js'

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

  it('drops the previous window’s occupancy when a patch states a window without a reading', () => {
    // 1M -> 500k: a committed model switch publishes the new window alone (the compressor's reading
    // was priced against the old model and `update_model` zeroed it), so the merged bar must not
    // pair the old figures with the new window — it shows `—` until the new window is measured.
    const switched = mergeUsageStable({ ...baseUsage, context_source: 'provider_usage' }, { context_max: 500_000 })
    expect(switched.context_max).toBe(500_000)
    expect(switched.calls).toBe(3)

    for (const key of ['context_estimated', 'context_percent', 'context_source', 'context_used'] as const) {
      expect(switched).not.toHaveProperty(key)
    }

    // The same window restated without a reading is still not a reading (a same-size switch).
    expect(mergeUsageStable(baseUsage, { context_max: 200_000 })).not.toHaveProperty('context_used')

    // A patch that carries its own reading keeps it; one that says nothing about the window changes nothing.
    expect(mergeUsageStable(baseUsage, { context_max: 200_000, context_percent: 3, context_used: 5_000 })).toMatchObject({
      context_max: 200_000,
      context_percent: 3,
      context_used: 5_000
    })
    expect(mergeUsageStable(baseUsage, { total: 1700 })).toMatchObject({ context_used: 24_000 })
  })

  it('treats a key present on only one side as a change', () => {
    expect(usageChanged(baseUsage, { ...baseUsage, cost_usd: 0.01 })).toBe(true)
    expect(usageChanged({ ...baseUsage, cost_usd: 0.01 }, baseUsage)).toBe(true)
  })

  it('reports no change for deep-equal usages', () => {
    expect(usageChanged(baseUsage, { ...baseUsage })).toBe(false)
  })
})
