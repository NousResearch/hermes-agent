import { describe, expect, it } from 'vitest'

import { blockRenders, hasLeadGap, messageGroup, prevRenderedMsg } from '../domain/blockLayout.js'
import type { Msg } from '../types.js'

const m = (over: Partial<Msg>): Msg => ({ role: 'assistant', text: '', ...over })

describe('messageGroup', () => {
  it('classifies each block kind into its visual band', () => {
    expect(messageGroup(m({ role: 'assistant' }))).toBe('model')
    expect(messageGroup(m({ role: 'assistant', kind: 'diff' }))).toBe('diff')
    expect(messageGroup(m({ role: 'system', kind: 'trail' }))).toBe('trail')
    expect(messageGroup(m({ role: 'system' }))).toBe('note')
    expect(messageGroup(m({ role: 'user' }))).toBe('user')
    expect(messageGroup(m({ role: 'user', kind: 'slash' }))).toBe('slash')
    expect(messageGroup(m({ role: 'system', kind: 'intro' }))).toBe('intro')
    expect(messageGroup(m({ role: 'system', kind: 'panel' }))).toBe('intro')
  })
})

describe('hasLeadGap', () => {
  const trail = m({ role: 'system', kind: 'trail' })
  const model = m({ role: 'assistant' })
  const note = m({ role: 'system' })
  const user = m({ role: 'user' })
  const diff = m({ role: 'assistant', kind: 'diff' })
  const slash = m({ role: 'user', kind: 'slash' })

  it('opens a gap only at a boundary between working-area groups', () => {
    expect(hasLeadGap(trail, model)).toBe(true)
    expect(hasLeadGap(model, trail)).toBe(true)
    expect(hasLeadGap(model, note)).toBe(true)
    expect(hasLeadGap(note, model)).toBe(true)
  })

  it('keeps same-group neighbours flush (the grouping)', () => {
    expect(hasLeadGap(trail, trail)).toBe(false)
    expect(hasLeadGap(model, model)).toBe(false)
    expect(hasLeadGap(note, note)).toBe(false)
  })

  it('never gaps the first block (no predecessor)', () => {
    expect(hasLeadGap(undefined, model)).toBe(false)
    expect(hasLeadGap(undefined, trail)).toBe(false)
  })

  it('suppresses the gap after blocks that already paint a trailing line', () => {
    // user and diff carry their own marginBottom — the following block must
    // not add a second blank line on top of it.
    expect(hasLeadGap(user, trail)).toBe(false)
    expect(hasLeadGap(user, model)).toBe(false)
    expect(hasLeadGap(diff, model)).toBe(false)
  })

  it('still gaps after a slash echo (it has no trailing margin)', () => {
    expect(hasLeadGap(slash, model)).toBe(true)
    expect(hasLeadGap(slash, trail)).toBe(true)
  })

  it('lets user / slash / diff own their spacing (never managed here)', () => {
    expect(hasLeadGap(model, user)).toBe(false)
    expect(hasLeadGap(model, slash)).toBe(false)
    expect(hasLeadGap(model, diff)).toBe(false)
  })
})

describe('blockRenders', () => {
  const trail: Msg = { role: 'system', kind: 'trail', text: '', tools: ['Edit foo.ts'] }
  const model: Msg = { role: 'assistant', text: 'hi' }
  const todos: Msg = { role: 'system', kind: 'trail', text: '', todos: [{ content: 'a', id: '1', status: 'pending' }] }

  it('always renders non-trail blocks', () => {
    expect(blockRenders(model, { detailsMode: 'hidden', commandOverride: true })).toBe(true)
  })

  it('renders a content-bearing trail unless every section is hidden', () => {
    expect(blockRenders(trail, { detailsMode: 'collapsed' })).toBe(true)
    expect(blockRenders(trail, { detailsMode: 'expanded' })).toBe(true)
    // /details hidden routes through commandOverride, which hides every section.
    expect(blockRenders(trail, { detailsMode: 'hidden', commandOverride: true })).toBe(false)
  })

  it('does not render a content-less trail (e.g. finalDetails with only a token tally)', () => {
    const tally: Msg = { role: 'system', kind: 'trail', text: '', toolTokens: 40 }

    expect(blockRenders(tally, { detailsMode: 'expanded' })).toBe(false)
  })

  it('keeps todo trails visible even when details are hidden', () => {
    expect(blockRenders(todos, { detailsMode: 'hidden', commandOverride: true })).toBe(true)
  })
})

describe('prevRenderedMsg', () => {
  const hiddenCtx = { commandOverride: true, detailsMode: 'hidden' as const }
  const shownCtx = { detailsMode: 'collapsed' as const }

  const rows: Msg[] = [
    { role: 'user', text: 'q' }, // 0
    { role: 'system', kind: 'trail', text: '', tools: ['Edit foo.ts'] }, // 1
    { role: 'assistant', text: 'first' }, // 2
    { role: 'system', kind: 'trail', text: '', tools: ['Edit bar.ts'] }, // 3
    { role: 'assistant', text: 'second' } // 4
  ]
  const at = (i: number) => rows[i]

  it('returns the literal predecessor when everything renders', () => {
    expect(prevRenderedMsg(at, 2, shownCtx)).toBe(rows[1])
    expect(prevRenderedMsg(at, 4, shownCtx)).toBe(rows[3])
  })

  it('skips hidden trails so grouping sees the nearest visible block', () => {
    // With trails hidden, the prose at index 2 groups against the user (not the
    // invisible trail) and the prose at index 4 groups against the prose at 2.
    expect(prevRenderedMsg(at, 2, hiddenCtx)).toBe(rows[0])
    expect(prevRenderedMsg(at, 4, hiddenCtx)).toBe(rows[2])
  })

  it('returns undefined at the top of the transcript', () => {
    expect(prevRenderedMsg(at, 0, shownCtx)).toBeUndefined()
  })
})
