import { describe, expect, it } from 'vitest'

import { shouldShowResponseSeparator } from '../components/messageLine.js'
import { sectionMode } from '../domain/details.js'
import { DEFAULT_TOOL_TRAIL_POSITION, nextToolTrailPosition, parseToolTrailPosition } from '../domain/toolTrail.js'
import { estimatedMsgHeight } from '../lib/virtualHeights.js'
import type { Msg } from '../types.js'

describe('parseToolTrailPosition', () => {
  it('reads both positions, ignoring surrounding whitespace and casing', () => {
    expect(parseToolTrailPosition('above')).toBe('above')
    expect(parseToolTrailPosition('below')).toBe('below')
    expect(parseToolTrailPosition('  BELOW ')).toBe('below')
  })

  it('falls back to above for an unset, invalid, or non-string value', () => {
    expect(parseToolTrailPosition(undefined)).toBe('above')
    expect(parseToolTrailPosition(null)).toBe('above')
    expect(parseToolTrailPosition('')).toBe('above')
    expect(parseToolTrailPosition('under')).toBe('above')
    expect(parseToolTrailPosition(true)).toBe('above')
    expect(parseToolTrailPosition(3)).toBe('above')
    expect(parseToolTrailPosition({ position: 'below' })).toBe('above')
  })

  it('defaults to the existing rendering so an unset config changes nothing', () => {
    expect(DEFAULT_TOOL_TRAIL_POSITION).toBe('above')
    expect(parseToolTrailPosition(undefined)).toBe(DEFAULT_TOOL_TRAIL_POSITION)
  })

  it('flips between the two positions', () => {
    expect(nextToolTrailPosition('above')).toBe('below')
    expect(nextToolTrailPosition('below')).toBe('above')
  })
})

describe('tool trail position is orthogonal to section visibility', () => {
  it('leaves hidden/collapsed/expanded resolution untouched', () => {
    // Position is not an input to sectionMode at all — a trail hidden by
    // /details stays hidden wherever it would otherwise have been drawn.
    expect(sectionMode('tools', 'collapsed', { tools: 'hidden' })).toBe('hidden')
    expect(sectionMode('tools', 'collapsed', {})).toBe('expanded')
    expect(sectionMode('thinking', 'hidden', { thinking: 'collapsed' })).toBe('collapsed')
  })
})

describe('shouldShowResponseSeparator under tool_trail_position', () => {
  const msg: Msg = { role: 'assistant', text: 'final', thinking: 'plan' }

  it('keeps the separator in the default position', () => {
    expect(shouldShowResponseSeparator(msg, true)).toBe(true)
    expect(shouldShowResponseSeparator(msg, true, 'above')).toBe(true)
  })

  it('drops the separator when the trail moved below the answer', () => {
    // With nothing above it, a "Response" rule would announce an answer that
    // already started; the trail below carries ToolTrail's own headers.
    expect(shouldShowResponseSeparator(msg, true, 'below')).toBe(false)
  })

  it('still never draws a separator without visible details', () => {
    expect(shouldShowResponseSeparator({ role: 'assistant', text: 'final' }, false, 'below')).toBe(false)
    expect(shouldShowResponseSeparator({ role: 'assistant', text: 'final' }, false, 'above')).toBe(false)
  })
})

describe('estimatedMsgHeight under tool_trail_position', () => {
  const detailed: Msg = { role: 'assistant', text: 'ok', thinking: 'plan', tools: ['Tool A', 'Tool B'] }
  const opts = { compact: false, details: true } as const

  it('estimates the default position exactly as before the option existed', () => {
    expect(estimatedMsgHeight(detailed, 80, { ...opts, toolTrailPosition: 'above' })).toBe(
      estimatedMsgHeight(detailed, 80, opts)
    )
  })

  it('drops exactly the separator rows when the trail renders below', () => {
    // MessageLine trades the separator (1 row + 1 margin) for the trail's top
    // margin, which replaces its bottom margin one-for-one — so `below` is
    // shorter by precisely 2 rows, never by the trail itself.
    expect(estimatedMsgHeight(detailed, 80, { ...opts, toolTrailPosition: 'below' })).toBe(
      estimatedMsgHeight(detailed, 80, { ...opts, toolTrailPosition: 'above' }) - 2
    )
  })

  it('still counts every trail row in both positions', () => {
    const bare: Msg = { role: 'assistant', text: 'ok' }

    for (const toolTrailPosition of ['above', 'below'] as const) {
      expect(estimatedMsgHeight(detailed, 80, { ...opts, toolTrailPosition })).toBeGreaterThan(
        estimatedMsgHeight(bare, 80, { compact: false, details: false })
      )
    }
  })

  it('does not change rows for a message that draws no separator either way', () => {
    // No body text → no separator in `above` either, so there is nothing for
    // `below` to reclaim and the two estimates must agree.
    const trailOnly: Msg = { kind: 'trail', role: 'system', text: '', tools: ['Tool A'] }

    expect(estimatedMsgHeight(trailOnly, 80, { ...opts, toolTrailPosition: 'below' })).toBe(
      estimatedMsgHeight(trailOnly, 80, { ...opts, toolTrailPosition: 'above' })
    )
  })

  it('leaves user rows and their inter-turn separator alone', () => {
    const user: Msg = { role: 'user', text: 'prompt' }

    expect(estimatedMsgHeight(user, 80, { ...opts, toolTrailPosition: 'below', withSeparator: true })).toBe(
      estimatedMsgHeight(user, 80, { ...opts, toolTrailPosition: 'above', withSeparator: true })
    )
  })
})
