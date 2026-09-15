import { describe, expect, it } from 'vitest'

import { replaceUsageStable, usageChanged, ZERO } from '../domain/usage.js'
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

describe('replaceUsageStable (#41480 status-bar flicker / live-token authoritative snapshots)', () => {
  it('returns the PRIOR reference when an identical full snapshot arrives', () => {
    // The load-bearing behavior: a same-value snapshot must NOT mint a new
    // object, or every $uiState subscriber re-renders per streaming event.
    const identical: Usage = { ...baseUsage }
    expect(replaceUsageStable(baseUsage, identical)).toBe(baseUsage)
  })

  it('returns the prior reference when no snapshot arrives', () => {
    expect(replaceUsageStable(baseUsage, undefined)).toBe(baseUsage)
  })

  it('clears fields the new snapshot omits instead of keeping the stale prior values', () => {
    // Every usage producer (session.info, session.usage, message.complete,
    // manual /compress) sends a COMPLETE snapshot. A compressed session that
    // no longer reports context_* must actually lose the gauge, not keep
    // showing the pre-compression number forever.
    const merged = replaceUsageStable(baseUsage, { calls: 3, input: 1200, output: 400, total: 1600 })
    expect(merged).not.toBe(baseUsage)
    expect(merged.context_max).toBeUndefined()
    expect(merged.context_percent).toBeUndefined()
    expect(merged.context_used).toBeUndefined()
  })

  it('replaces prior data wholesale when a value actually changes', () => {
    const merged = replaceUsageStable(baseUsage, { ...baseUsage, total: 1700 })
    expect(merged).not.toBe(baseUsage)
    expect(merged.total).toBe(1700)
    expect(merged.calls).toBe(3)
  })

  it('floors fields missing from the snapshot to the ZERO defaults, never to the prior snapshot', () => {
    // Deliberately a sparse snapshot (a real server payload could legitimately
    // omit calls/input/output, e.g. a compress summary) -- cast, not padded,
    // so the assertions below document ZERO-flooring rather than masking it.
    const merged = replaceUsageStable(baseUsage, { total: 1700 } as Usage)
    expect(merged.total).toBe(1700)
    expect(merged.calls).toBe(ZERO.calls)
    expect(merged.input).toBe(ZERO.input)
    expect(merged.context_max).toBeUndefined()
  })

  it('detects an active_subagents-only change against an otherwise-identical snapshot', () => {
    // usageChanged iterates the key union generically, so optional fields the
    // status rule consumes (active_subagents drives the ⛓ segment and the
    // resume hint) can never be silently dropped from the comparison.
    const withSubagents = replaceUsageStable(baseUsage, { ...baseUsage, active_subagents: 2 })
    expect(withSubagents).not.toBe(baseUsage)
    expect(withSubagents.active_subagents).toBe(2)

    // And clearing it back down is also a change.
    const cleared = replaceUsageStable(withSubagents, { ...baseUsage, active_subagents: 0 })
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

  it('treats equal-size snapshots with different undefined own keys as changed', () => {
    const prev: Usage = { ...baseUsage, cost_usd: undefined }
    const next: Usage = { ...baseUsage, cost_status: undefined }

    expect(usageChanged(prev, next)).toBe(true)

    const replaced = replaceUsageStable(prev, baseUsage)
    expect(replaced).not.toBe(prev)
    expect(Object.prototype.hasOwnProperty.call(replaced, 'cost_usd')).toBe(false)
  })
})
