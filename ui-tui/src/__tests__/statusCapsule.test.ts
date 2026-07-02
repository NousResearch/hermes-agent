import { describe, expect, it } from 'vitest'

import {
  buildStatusCapsule,
  foldStatusCapsuleEvent,
  formatStatusCapsule,
  initialStatusCapsule
} from '../domain/statusCapsule.js'

const usage = { calls: 0, input: 0, output: 0, total: 0 }

describe('StatusCapsule schema and accuracy boundaries', () => {
  it('creates an Auto Session capsule without pretending token or elapsed accuracy is exact', () => {
    const capsule = buildStatusCapsule({
      autoSession: true,
      busy: false,
      lastTurnEndedAt: null,
      sessionStartedAt: null,
      status: 'forging session…',
      turnStartedAt: null,
      usage
    })

    expect(capsule.schemaVersion).toBe(1)
    expect(capsule.route).toBe('Auto Session')
    expect(capsule.state).toBe('session_starting')
    expect(capsule.precision.tokens).toBe('unknown')
    expect(capsule.precision.elapsed).toBe('unknown')
    expect(capsule.observed.opencode).toBe('unknown')
    expect(capsule.observed.deep).toBe('unknown')
    expect(formatStatusCapsule(capsule)).toContain('Auto Session')
    expect(formatStatusCapsule(capsule)).not.toContain('exact')
  })

  it('marks token and elapsed numbers as observed rather than exact', () => {
    const now = Date.now()

    const capsule = buildStatusCapsule({
      busy: true,
      lastTurnEndedAt: null,
      sessionStartedAt: now - 90_000,
      status: 'running…',
      turnStartedAt: now - 12_000,
      usage: { calls: 2, context_max: 200_000, context_used: 50_000, input: 100, output: 40, total: 140 }
    })

    expect(capsule.route).toBe('Hermes')
    expect(capsule.state).toBe('running')
    expect(capsule.precision.tokens).toBe('observed')
    expect(capsule.precision.elapsed).toBe('observed')
    expect(capsule.tokenSource).toBe('provider_usage')
    expect(capsule.elapsedSource).toBe('local_wall_clock')
  })
  it('marks context-window occupancy as observed without calling it provider usage', () => {
    const capsule = buildStatusCapsule({
      busy: false,
      lastTurnEndedAt: null,
      sessionStartedAt: null,
      status: 'ready',
      turnStartedAt: null,
      usage: { calls: 0, context_max: 200_000, context_used: 50_000, input: 0, output: 0, total: 0 }
    })

    expect(capsule.precision.tokens).toBe('observed')
    expect(capsule.tokenSource).toBe('context_window')
  })
})

describe('StatusCapsule runtime event folding', () => {
  it('folds tools, MoA, and delegation as observed while leaving OpenCode/Deep unknown without direct evidence', () => {
    let capsule = initialStatusCapsule()

    capsule = foldStatusCapsuleEvent(capsule, { type: 'message.start' })
    expect(capsule.state).toBe('running')

    capsule = foldStatusCapsuleEvent(capsule, { payload: { name: 'terminal' }, type: 'tool.start' })
    expect(capsule.state).toBe('tooling')
    expect(capsule.tools).toEqual(['terminal'])
    expect(capsule.observed.opencode).toBe('unknown')
    expect(capsule.observed.deep).toBe('unknown')

    capsule = foldStatusCapsuleEvent(capsule, { payload: { subagent_id: 'sa-1' }, type: 'subagent.start' })
    expect(capsule.observed.delegate).toBe('observed')

    capsule = foldStatusCapsuleEvent(capsule, { payload: { label: 'openrouter:openai/gpt-5.5' }, type: 'moa.reference' })
    expect(capsule.route).toBe('MoA')
    expect(capsule.observed.moa).toBe('observed')
    expect(capsule.observed.opencode).toBe('unknown')

    capsule = foldStatusCapsuleEvent(capsule, {
      payload: { usage: { calls: 1, input: 10, output: 5, total: 15 } },
      type: 'message.complete'
    })
    expect(capsule.state).toBe('idle')
    expect(capsule.precision.tokens).toBe('observed')
  })

  it('does not leak MoA or delegate observations into the next plain Hermes turn', () => {
    let capsule = initialStatusCapsule()

    capsule = foldStatusCapsuleEvent(capsule, { type: 'message.start' })
    capsule = foldStatusCapsuleEvent(capsule, { payload: { subagent_id: 'sa-1' }, type: 'subagent.start' })
    capsule = foldStatusCapsuleEvent(capsule, { payload: { label: 'reference' }, type: 'moa.reference' })
    capsule = foldStatusCapsuleEvent(capsule, { payload: {}, type: 'message.complete' })

    capsule = foldStatusCapsuleEvent(capsule, { type: 'message.start' })

    expect(capsule.route).toBe('Hermes')
    expect(capsule.observed.delegate).toBe('unknown')
    expect(capsule.observed.moa).toBe('unknown')
    expect(capsule.observed.opencode).toBe('unknown')
    expect(formatStatusCapsule(capsule)).not.toContain('MoA')
    expect(formatStatusCapsule(capsule)).not.toContain('delegate')
  })

  it('preserves folded live states when the render path rebuilds from usage and timing', () => {
    let capsule = initialStatusCapsule()

    capsule = foldStatusCapsuleEvent(capsule, { type: 'message.start' })
    capsule = foldStatusCapsuleEvent(capsule, { payload: { name: 'terminal' }, type: 'tool.start' })

    const rebuilt = buildStatusCapsule({
      busy: true,
      lastTurnEndedAt: null,
      prior: capsule,
      sessionStartedAt: Date.now() - 30_000,
      status: 'running…',
      turnStartedAt: Date.now() - 5_000,
      usage
    })

    expect(rebuilt.state).toBe('tooling')
    expect(rebuilt.tools).toEqual(['terminal'])
  })
})
