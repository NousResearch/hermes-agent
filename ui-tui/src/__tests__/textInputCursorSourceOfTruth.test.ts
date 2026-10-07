import { describe, expect, it } from 'vitest'

import { fastAppendEffect, fastBackspaceEffect, resolveCursorLayout } from '../components/textInput.js'
import { cursorLayout } from '../lib/inputMetrics.js'

// Closes Copilot follow-up on PR #26717: the original cursor-drift
// fix bumped Ink's displayCursor / cursorDeclaration on fast-echo, but
// if TextInput itself re-renders before the deferred 16ms `setCur`
// flushes (parent state change, status-bar tick, spinner) the layout
// effect inside `useDeclaredCursor` re-publishes a declaration
// computed from the STALE React `cur` state and clobbers the Ink-level
// bump. The fix is structural: read `curRef.current` (always
// up-to-date) when computing the layout, not the `cur` state.
//
// These tests exercise the real, exported `resolveCursorLayout`,
// `fastBackspaceEffect`, and `fastAppendEffect` helpers that
// `textInput.tsx` calls at its render site and fast-echo call sites —
// no source-text regex, no readFileSync.
describe('resolveCursorLayout', () => {
  it('uses curRefCurrent (the fresh ref value), not the stale cur state', () => {
    // Simulate the exact bug scenario: `cur` (React state) is stale —
    // it still reflects the value before a fast-echo append — while
    // `curRef.current` has already advanced past it.
    const display = 'hello world'
    const staleCur = 5
    const freshCurRefCurrent = 11
    const columns = 80

    const result = resolveCursorLayout(display, staleCur, freshCurRefCurrent, columns)
    const expected = cursorLayout(display, freshCurRefCurrent, columns)

    expect(result).toEqual(expected)
  })
})

describe('fastBackspaceEffect', () => {
  it('removes the last character, moves the cursor back one, and pairs the write with the advance delta', () => {
    const effect = fastBackspaceEffect('hello', 5)

    expect(effect.newValue).toBe('hell')
    expect(effect.newCursor).toBe(4)
    expect(effect.removed).toBe('o')
    // Both the stdout write and the noteCursorAdvance delta live on the
    // same returned object — a caller cannot apply `write` without also
    // having `advanceDelta` in hand, so the pairing can't silently drift.
    expect(effect.write).toBe('\b \b')
    expect(effect.advanceDelta).toBe(-1)
  })
})

describe('fastAppendEffect', () => {
  it('appends the text, advances the cursor by the inserted length, and pairs the write with the advance delta', () => {
    const effect = fastAppendEffect('hello', 5, ' world')

    expect(effect.newValue).toBe('hello world')
    expect(effect.newCursor).toBe(11)
    // The stdout write is exactly the inserted text, and the
    // noteCursorAdvance delta is bundled into the same object.
    expect(effect.write).toBe(' world')
    expect(effect.advanceDelta).toBe(' world'.length)
  })
})
