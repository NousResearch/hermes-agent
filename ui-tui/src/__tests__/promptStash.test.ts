import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import type { ComposerToken } from '../app/interfaces.js'
import { buildPreview, formatStashAge, MAX_STASH_ITEMS, PromptStash, resolveCtrlS } from '../domain/promptStash.js'

// The same file the classic CLI's suite reads (tests/hermes_cli/test_prompt_stash_vectors.py).
// A change to either implementation fails the other side here rather than drifting quietly.
const VECTORS = JSON.parse(
  readFileSync(join(process.cwd(), '..', 'tests', 'fixtures', 'prompt_stash_vectors.json'), 'utf8')
) as {
  build_preview: { expected: string; text: string; width?: number }[]
  cap: { expected_len: number; expected_newest: string; push: number }
  placeholder_hint: { entries: string[]; expected: string }[]
  resolve: { action: string; buffer: string; entries: string[]; panel_open: boolean; tokens: number }[]
}

const token = (i: number): ComposerToken => ({ kind: 'paste', label: `[[ Paste ${i} ]]`, text: `payload ${i}` })

const seed = (entries: string[], panelOpen = false): PromptStash => {
  const stash = new PromptStash()

  for (const text of entries) {
    expect(stash.stash(text)).toBe(true)
  }

  if (panelOpen) {
    expect(stash.openPanel()).toBe(true)
  }

  return stash
}

describe('buildPreview — same shaping as hermes_cli.prompt_stash.build_preview', () => {
  it.each(VECTORS.build_preview)('$text at width $width → $expected', ({ expected, text, width }) => {
    expect(buildPreview(text, width ?? 60)).toBe(expected)
  })
})

describe('resolveCtrlS — same decision table as hermes_cli.prompt_stash.resolve_ctrl_s', () => {
  it.each(VECTORS.resolve)('$buffer with $entries stashed → $action', ({ action, buffer, entries, panel_open, tokens }) => {
    const stash = seed(entries, panel_open)
    const chips = Array.from({ length: tokens }, (_, i) => token(i))

    expect(resolveCtrlS(stash, buffer, chips).action).toBe(action)
  })

  it('hands back the exact draft and chips on restore', () => {
    const stash = new PromptStash()

    stash.stash('parked text', [token(1)])

    const { action, payload } = resolveCtrlS(stash, '')

    expect(action).toBe('restored')
    expect(payload?.text).toBe('parked text')
    expect(payload?.tokens).toEqual([token(1)])
  })

  it('parks nothing when the buffer is only whitespace and no chip is attached', () => {
    const stash = new PromptStash()

    expect(resolveCtrlS(stash, '  \n\t ').action).toBe('noop')
    expect(stash.length).toBe(0)
  })

  it('returns the newest draft first, so a second Ctrl+S never loses the first', () => {
    const stash = new PromptStash()

    resolveCtrlS(stash, 'first')
    resolveCtrlS(stash, 'second')

    expect(stash.length).toBe(2)
    expect(stash.peek(0)?.text).toBe('second')
    expect(resolveCtrlS(stash, '').action).toBe('open_panel')

    expect(stash.restoreAtCursor()?.text).toBe('second')
    expect(stash.peek(0)?.text).toBe('first')
  })

  it('closes an open panel instead of parking or restoring', () => {
    const stash = seed(['a', 'b'], true)

    const { action } = resolveCtrlS(stash, 'draft typed while browsing')

    expect(action).toBe('close_panel')
    expect(stash.length).toBe(2)
  })
})

describe('placeholder and indicator — same words as the classic CLI', () => {
  it.each(VECTORS.placeholder_hint)('$entries → $expected', ({ entries, expected }) => {
    expect(seed(entries).placeholderHint()).toBe(expected)
  })

  it('shows the count, and the browse marker only while the panel is open', () => {
    const stash = seed(['a', 'b'])

    expect(stash.indicator()).toBe('📌 2')
    stash.openPanel()
    expect(stash.indicator()).toBe('📌 2 ▲')

    const empty = new PromptStash()

    expect(empty.indicator()).toBe('')
  })
})

describe('stack cap and panel editing', () => {
  it('keeps the newest entries and drops the oldest past the cap', () => {
    const { expected_len, expected_newest, push } = VECTORS.cap
    const stash = new PromptStash()

    for (let i = 1; i <= push; i++) {
      stash.stash(`draft ${i}`)
    }

    expect(stash.length).toBe(expected_len)
    expect(MAX_STASH_ITEMS).toBe(expected_len)
    expect(stash.peek(0)?.text).toBe(expected_newest)
  })

  it('clamps the panel cursor and closes the panel when the last draft is deleted', () => {
    const stash = seed(['oldest', 'newest'], true)

    expect(stash.moveCursor(-5)).toBe(0)
    expect(stash.moveCursor(9)).toBe(1)
    expect(stash.deleteAtCursor()).toBe(true)
    expect(stash.length).toBe(1)

    expect(stash.deleteAtCursor()).toBe(true)
    expect(stash.length).toBe(0)
    expect(stash.panelOpen).toBe(false)
    expect(stash.deleteAtCursor()).toBe(false)
  })

  it('restores nothing from an empty stash', () => {
    expect(new PromptStash().restoreAtCursor()).toBeNull()
    expect(new PromptStash().pop()).toBeNull()
  })

  it('renders "(images only)" for a chip-only draft', () => {
    const stash = new PromptStash()

    stash.stash('', [token(1)])

    expect(stash.peek(0)?.preview).toBe('(images only)')
  })
})

describe('formatStashAge — mirrors _fmt_stash_age', () => {
  const at = 1_000_000

  it.each([
    [3, 'just now'],
    [9, 'just now'],
    [10, '10s ago'],
    [89, '89s ago'],
    [90, '1 min ago'],
    [3599, '59 min ago'],
    [3600, '1h ago'],
    [7200, '2h ago']
  ])('%is elapsed → %s', (secs, expected) => {
    expect(formatStashAge(at, at + secs * 1000)).toBe(expected)
  })
})
