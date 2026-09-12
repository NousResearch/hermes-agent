import { describe, expect, it } from 'vitest'

import {
  activePromptIndex,
  promptOffsetAtRailRow,
  promptRowIndexes,
  promptTicks,
  steppedPromptOffset,
  stickyPromptFromViewport
} from '../domain/viewport.js'

describe('stickyPromptFromViewport', () => {
  it('hides the sticky prompt when a newer user message is already visible', () => {
    const messages = [
      { role: 'user' as const, text: 'older prompt' },
      { role: 'assistant' as const, text: 'older answer' },
      { role: 'user' as const, text: 'current prompt' },
      { role: 'assistant' as const, text: 'current answer' }
    ]

    const offsets = [0, 2, 10, 12, 20]

    expect(stickyPromptFromViewport(messages, offsets, 8, 16, false)).toBe('')
  })

  it('shows the latest user message above the viewport when no user message is visible', () => {
    const messages = [
      { role: 'user' as const, text: 'older prompt' },
      { role: 'assistant' as const, text: 'older answer' },
      { role: 'user' as const, text: 'current prompt' },
      { role: 'assistant' as const, text: 'current answer' }
    ]

    const offsets = [0, 2, 10, 12, 20]

    expect(stickyPromptFromViewport(messages, offsets, 16, 20, false)).toBe('current prompt')
  })

  it('shows the last prompt once the viewport starts after the history tail', () => {
    const messages = [
      { role: 'user' as const, text: 'current prompt' },
      { role: 'assistant' as const, text: 'completed answer' }
    ]

    expect(stickyPromptFromViewport(messages, [0, 2, 5], 8, 14, false)).toBe('current prompt')
  })

  it('shows a prompt as soon as its full row is above the viewport', () => {
    const messages = [
      { role: 'user' as const, text: 'current prompt' },
      { role: 'assistant' as const, text: 'current answer' }
    ]

    expect(stickyPromptFromViewport(messages, [0, 2, 10], 2, 8, false)).toBe('current prompt')
  })

  it('hides the sticky prompt at the bottom', () => {
    const messages = [
      { role: 'user' as const, text: 'current prompt' },
      { role: 'assistant' as const, text: 'current answer' }
    ]

    expect(stickyPromptFromViewport(messages, [0, 2, 10], 8, 10, true)).toBe('')
  })
})

// A six-message session: system intro, then three prompt/answer turns. Row
// indexes are message indexes, so the prompts sit on rows 1, 3 and 5 and
// `offsets` carries the top of each row plus the tail (length n+1).
const TURNS = [
  { role: 'system' as const, text: '' },
  { role: 'user' as const, text: 'first' },
  { role: 'assistant' as const, text: 'answer one' },
  { role: 'user' as const, text: 'second' },
  { role: 'assistant' as const, text: 'answer two' },
  { role: 'user' as const, text: 'third' }
]

const ROWS = [1, 3, 5]
const OFFSETS = [0, 4, 10, 40, 60, 100, 120]

describe('prompt navigation', () => {
  it('lists the prompts you actually wrote, in order', () => {
    expect(promptRowIndexes(TURNS)).toEqual(ROWS)

    // Injected turns arrive as user messages and blank rows carry no prompt:
    // neither earns a tick, or the rail answers "where is my prompt?" wrongly.
    const injected = [
      { role: 'user' as const, text: '[IMPORTANT: Background process proc_1 completed normally.' },
      { role: 'user' as const, text: '[System: Your previous response was truncated' },
      { role: 'user' as const, text: '   ' },
      { role: 'user' as const, text: 'real one' }
    ]

    expect(promptRowIndexes(injected)).toEqual([3])
    expect(promptRowIndexes([])).toEqual([])
  })

  it('steps onto the neighbouring prompt row, and stops at both ends', () => {
    // The step resolves to the NEIGHBOUR'S OWN ROW — a relationship between two
    // pieces of data, not a number frozen here. Rows past the measured span are
    // height-estimated, so a row delta would drift the longer the run goes.
    expect(steppedPromptOffset(OFFSETS, ROWS, 50, false, 1)).toBe(OFFSETS[ROWS[2]!])
    expect(steppedPromptOffset(OFFSETS, ROWS, 120, true, -1)).toBe(OFFSETS[ROWS[1]!])
    // Ends do not wrap, and an empty session has nowhere to go.
    expect(steppedPromptOffset(OFFSETS, ROWS, 0, false, -1)).toBeNull()
    expect(steppedPromptOffset(OFFSETS, ROWS, 100, false, 1)).toBeNull()
    expect(steppedPromptOffset(OFFSETS, [], 0, false, 1)).toBeNull()
    // Above the first prompt — the normal shape, an intro row above the first
    // user message — the viewport is not ON a prompt yet, so the forward step
    // lands on the FIRST one rather than stepping over it.
    expect(steppedPromptOffset(OFFSETS, ROWS, 0, false, 1)).toBe(OFFSETS[ROWS[0]!])
    // The prompt you are "on" is the last one at or above the viewport top, the
    // newest while the view is following the tail, and none of them while the
    // top is still above them all.
    expect(activePromptIndex(OFFSETS, ROWS, 50, false)).toBe(1)
    expect(activePromptIndex(OFFSETS, ROWS, 100, false)).toBe(2)
    expect(activePromptIndex(OFFSETS, ROWS, 0, false)).toBe(-1)
    expect(activePromptIndex(OFFSETS, ROWS, -999, true)).toBe(2)
    expect(activePromptIndex(OFFSETS, [], 0, false)).toBe(-1)
  })

  it('gives every prompt a tick in order, and keeps the active one when they collide', () => {
    const ticks = promptTicks(OFFSETS, ROWS, 120, 11, 2)

    // One tick per prompt, in prompt order, inside the rail. Exactly which row
    // each lands on is the proportional scale's business — asserting today's
    // numbers would fail the day someone tunes the scale, so this pins the
    // relationship instead.
    expect(ticks.map(t => t.index)).toEqual([0, 1, 2])
    expect(ticks.every(t => t.row >= 0 && t.row < 11)).toBe(true)
    expect(ticks.map(t => t.row)).toEqual([...ticks.map(t => t.row)].sort((a, b) => a - b))
    // Two prompts scaling onto the same rail row collapse to ONE tick, and the
    // prompt you are on wins — so the marker for where you are can never be
    // absorbed by a neighbour in a dense session.
    const dense = [0, 4, 5, 6, 7, 8, 200]

    expect(promptTicks(dense, [1, 3], 200, 11, 1)).toEqual([{ index: 1, row: 0 }])
    expect(promptTicks(dense, [1, 3], 200, 11, 0)).toEqual([{ index: 0, row: 0 }])
    // Nothing to draw without prompts, or without a viewport to draw in.
    expect(promptTicks(OFFSETS, [], 120, 11, -1)).toEqual([])
    expect(promptTicks(OFFSETS, ROWS, 120, 0, 0)).toEqual([])
  })

  it('resolves a tick row to that prompt exactly, and other rows to no tick', () => {
    // Whichever row a tick lands on, clicking it resolves to THAT prompt's own
    // offset — the property that makes a click land exactly rather than near.
    const ticks = promptTicks(OFFSETS, ROWS, 120, 11, 2)

    for (const tick of ticks) {
      expect(promptOffsetAtRailRow(OFFSETS, ROWS, 120, 11, 2, tick.row)).toBe(OFFSETS[ROWS[tick.index]!])
    }

    // A rail row with no tick keeps the plain scrub path.
    const bare = [...Array(11).keys()].find(row => !ticks.some(tick => tick.row === row))

    expect(bare).toBeDefined()
    expect(promptOffsetAtRailRow(OFFSETS, ROWS, 120, 11, 2, bare!)).toBeNull()
  })
})
