import { describe, expect, it } from 'vitest'

import { parseSteerCommand, resolveSteerTargetId } from '../lib/subagentSteer.js'
import type { SubagentProgress } from '../types.js'

const sub = (over: Partial<SubagentProgress> = {}): SubagentProgress => ({
  depth: 1,
  goal: 'patch token-bucket refill race',
  id: 'b7c2',
  index: 0,
  notes: [],
  status: 'running',
  taskCount: 0,
  thinking: [],
  toolCount: 0,
  tools: [],
  ...over
})

describe('parseSteerCommand', () => {
  it('parses "@id text" into token + body', () => {
    expect(parseSteerCommand('@b7c2 switch approach')).toEqual({ body: 'switch approach', token: 'b7c2' })
  })

  it('keeps multi-line steer bodies', () => {
    const cmd = parseSteerCommand('@b7c2 line one\nline two')
    expect(cmd?.token).toBe('b7c2')
    expect(cmd?.body).toBe('line one\nline two')
  })

  it('returns null for a bare @token with no message', () => {
    expect(parseSteerCommand('@b7c2')).toBeNull()
    expect(parseSteerCommand('@b7c2   ')).toBeNull()
  })

  it('returns null for ordinary text that is not a steer', () => {
    expect(parseSteerCommand('email @john the report')).toBeNull()
    expect(parseSteerCommand('just a normal prompt')).toBeNull()
  })

  it('tolerates leading whitespace before the @', () => {
    expect(parseSteerCommand('  @b7c2 go')).toEqual({ body: 'go', token: 'b7c2' })
  })

  it('trims trailing whitespace from the body', () => {
    expect(parseSteerCommand('@b7c2   go now   ')).toEqual({ body: 'go now', token: 'b7c2' })
  })

  it('returns null for a lone @', () => {
    expect(parseSteerCommand('@')).toBeNull()
    expect(parseSteerCommand('@ hi')).toBeNull()
  })
})

describe('resolveSteerTargetId', () => {
  it('resolves an exact live subagent id', () => {
    expect(resolveSteerTargetId('b7c2', [sub({ id: 'b7c2' })])).toBe('b7c2')
  })

  it('resolves a unique id prefix', () => {
    expect(resolveSteerTargetId('b7', [sub({ id: 'b7c2' }), sub({ id: 'a11a' })])).toBe('b7c2')
  })

  it('refuses an ambiguous prefix (never steer a guess)', () => {
    expect(resolveSteerTargetId('b', [sub({ id: 'b7c2' }), sub({ id: 'b999' })])).toBeNull()
  })

  it('ignores finished subagents — only running/queued are addressable', () => {
    expect(resolveSteerTargetId('b7c2', [sub({ id: 'b7c2', status: 'completed' })])).toBeNull()
  })

  it('returns null when nothing matches (falls back to a normal turn)', () => {
    expect(resolveSteerTargetId('nope', [sub({ id: 'b7c2' })])).toBeNull()
  })
})
