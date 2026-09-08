import { describe, expect, it } from 'vitest'

import {
  addStreamedText,
  adoptAuthoritativeUsage,
  createLiveContextState,
  liveContextPatch,
  resetLiveContext
} from '../app/liveContextEstimate.js'
import { estimateTokensRough } from '../lib/text.js'
import type { Usage } from '../types.js'

const usageWithWindow = (used: number, max = 150016): Usage => ({
  calls: 1,
  context_max: max,
  context_percent: Math.round((used / max) * 100),
  context_used: used,
  input: 0,
  output: 0,
  total: 0
})

describe('liveContextEstimate', () => {
  it('starts with no base and no streamed estimate', () => {
    const state = createLiveContextState()
    expect(state).toEqual({ base: null, streamed: 0 })
  })

  it('adopts an authoritative reading that carries context_used', () => {
    const state = createLiveContextState()
    addStreamedText(state, 'some in-flight thinking')
    adoptAuthoritativeUsage(state, usageWithWindow(96000))
    expect(state.base).toBe(96000)
    expect(state.streamed).toBe(0)
  })

  it('resets base and streamed on an explicit reset (session switch)', () => {
    // session.info is the only point where a fresh window is guaranteed: the
    // previous session's base must not survive into the new one.
    const state = createLiveContextState()
    adoptAuthoritativeUsage(state, usageWithWindow(140000))
    addStreamedText(state, 'x'.repeat(4000))
    resetLiveContext(state)
    expect(state).toEqual({ base: null, streamed: 0 })
  })

  it('does NOT reset the estimate when context_used is unchanged (ticker re-emit)', () => {
    // The 1 Hz ticker re-emits when other fields (calls, active_subagents)
    // move while context_used is frozen. Resetting on every such frame would
    // kill the running estimate and re-freeze the gauge.
    const state = createLiveContextState()
    adoptAuthoritativeUsage(state, usageWithWindow(96000))
    addStreamedText(state, 'x'.repeat(4000)) // ~1000 tokens of thinking
    const before = state.streamed
    adoptAuthoritativeUsage(state, { ...usageWithWindow(96000), calls: 2 })
    expect(state.base).toBe(96000)
    expect(state.streamed).toBe(before) // untouched
  })

  it('re-anchors when context_used changes', () => {
    const state = createLiveContextState()
    adoptAuthoritativeUsage(state, usageWithWindow(96000))
    addStreamedText(state, 'x'.repeat(4000))
    adoptAuthoritativeUsage(state, usageWithWindow(98000))
    expect(state.base).toBe(98000)
    expect(state.streamed).toBe(0)
  })

  it('ignores snapshots without context_used (external engines, #50421)', () => {
    const state = createLiveContextState()
    addStreamedText(state, 'in-flight')
    adoptAuthoritativeUsage(state, { calls: 1, input: 0, output: 0, total: 0 })
    // No base to anchor on — the running estimate survives untouched.
    expect(state.base).toBeNull()
    expect(state.streamed).toBeGreaterThan(0)
  })

  it('ignores undefined usage', () => {
    const state = createLiveContextState()
    adoptAuthoritativeUsage(state, undefined)
    expect(state).toEqual({ base: null, streamed: 0 })
  })

  it('accumulates streamed text with the rough estimator', () => {
    const state = createLiveContextState()
    addStreamedText(state, 'a'.repeat(400))
    addStreamedText(state, 'b'.repeat(400))
    expect(state.streamed).toBe(estimateTokensRough('a'.repeat(400)) + estimateTokensRough('b'.repeat(400)))
  })

  it('ignores empty/undefined streamed text', () => {
    const state = createLiveContextState()
    addStreamedText(state, '')
    addStreamedText(state, undefined)
    expect(state.streamed).toBe(0)
  })

  it('returns no patch until there is a base, a window, and progress', () => {
    const state = createLiveContextState()
    expect(liveContextPatch(state, usageWithWindow(96000))).toBeNull() // no streamed yet
    addStreamedText(state, 'x'.repeat(400))
    // base still null — nothing to anchor on
    expect(liveContextPatch(state, usageWithWindow(96000))).toBeNull()
    adoptAuthoritativeUsage(state, usageWithWindow(96000))
    // streamed reset by the authoritative reading
    expect(liveContextPatch(state, usageWithWindow(96000))).toBeNull()
  })

  it('projects base + streamed onto the gauge and clamps the percent', () => {
    const state = createLiveContextState()
    adoptAuthoritativeUsage(state, usageWithWindow(149000, 150016))
    addStreamedText(state, 'x'.repeat(8000)) // ~2000 tokens over the window
    const patch = liveContextPatch(state, usageWithWindow(149000, 150016))
    expect(patch).not.toBeNull()
    expect(patch!.context_used).toBe(149000 + estimateTokensRough('x'.repeat(8000)))
    expect(patch!.context_percent).toBe(100) // clamped, never >100
  })

  it('keeps the percent in range for small estimates', () => {
    const state = createLiveContextState()
    adoptAuthoritativeUsage(state, usageWithWindow(1000, 150016))
    addStreamedText(state, 'hello')
    const patch = liveContextPatch(state, usageWithWindow(1000, 150016))
    expect(patch!.context_percent).toBeGreaterThanOrEqual(0)
    expect(patch!.context_percent).toBeLessThanOrEqual(100)
  })
})
