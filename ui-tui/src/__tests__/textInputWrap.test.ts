import { wrapAnsi } from '@hermes/ink'
import { describe, expect, it } from 'vitest'

import { offsetFromPosition } from '../components/textInput.js'
import { composerPromptWidth, cursorLayout, inputVisualHeight, stableComposerColumns } from '../lib/inputMetrics.js'

// Helper: compute the "end of text" position that wrap-ansi would render
// the input to. This is what Ink's <Text wrap="wrap"> uses, so cursorLayout
// MUST agree. Disagreement is the cursor-drift bug.
function wrapAnsiEndPosition(text: string, cols: number): { line: number; column: number } {
  const wrapped = wrapAnsi(text, cols, { hard: true, trim: false })
  const lines = wrapped.split('\n')
  const last = lines[lines.length - 1] ?? ''

  return { line: lines.length - 1, column: last.length }
}

describe('cursorLayout — word-wrap parity with wrap-ansi', () => {
  it('places cursor mid-line at its column', () => {
    expect(cursorLayout('hello world', 6, 40)).toEqual({ column: 6, line: 0 })
  })

  it('places cursor at end of a non-full line', () => {
    expect(cursorLayout('hi', 2, 10)).toEqual({ column: 2, line: 0 })
  })

  it('does not push exact-fill text onto a phantom next line', () => {
    // Regression: the previous hand-rolled wrap algorithm forced the cursor
    // onto (line+1, 0) when the text exactly filled the row. wrap-ansi keeps
    // it on the same row (no soft-wrap), so the cursor must too — otherwise
    // useDeclaredCursor parks the hardware cursor below the last char and
    // the user sees several blank cells between text and cursor block
    // (#cursor-drift-multiline).
    expect(cursorLayout('abcdefgh', 8, 8)).toEqual({ column: 8, line: 0 })
    expect(cursorLayout('abcdefgh', 8, 8)).toEqual(wrapAnsiEndPosition('abcdefgh', 8))
  })

  it('keeps short words on the current line when they fit (no phantom wrap)', () => {
    // wrap-ansi: "hello wo" at cols=8 stays as one line "hello wo".
    // The old cursorLayout incorrectly pushed to (1,0) because column=8 hit
    // the column>=width check, but that disagreed with what Ink actually
    // rendered.
    expect(cursorLayout('hello wo', 8, 8)).toEqual({ column: 8, line: 0 })
    expect(cursorLayout('hello wo', 8, 8)).toEqual(wrapAnsiEndPosition('hello wo', 8))
  })

  it('moves words across wrap boundaries instead of splitting them', () => {
    // "hello wor" at cols=8: wrap-ansi breaks at the space, "hello \nwor".
    expect(cursorLayout('hello wor', 9, 8)).toEqual({ column: 3, line: 1 })
    expect(cursorLayout('hello worl', 10, 8)).toEqual({ column: 4, line: 1 })
    expect(cursorLayout('hello world', 11, 8)).toEqual({ column: 5, line: 1 })

    // Each must match what wrap-ansi would actually render.
    expect(cursorLayout('hello wor', 9, 8)).toEqual(wrapAnsiEndPosition('hello wor', 8))
    expect(cursorLayout('hello worl', 10, 8)).toEqual(wrapAnsiEndPosition('hello worl', 8))
    expect(cursorLayout('hello world', 11, 8)).toEqual(wrapAnsiEndPosition('hello world', 8))
  })

  it('wraps the next word instead of splitting it at the right edge', () => {
    const text = 'hello world baby chickens are so cool its really rainy outside but wish'

    expect(cursorLayout(text, text.length, 70)).toEqual({ column: 4, line: 1 })
    expect(inputVisualHeight(text, 70)).toBe(2)
  })

  it('honours explicit newlines', () => {
    expect(cursorLayout('one\ntwo', 5, 40)).toEqual({ column: 1, line: 1 })
    expect(cursorLayout('one\ntwo', 4, 40)).toEqual({ column: 0, line: 1 })
  })

  it('does not wrap when cursor is before the right edge', () => {
    expect(cursorLayout('abcdefg', 7, 8)).toEqual({ column: 7, line: 0 })
  })

  it('matches wrap-ansi end-position for typing-style incremental input', () => {
    // Pins the actual fix: type a long message char-by-char at a narrow
    // width and assert the cursor follows wrap-ansi every step of the way.
    // Before the fix, ~5 boundary positions per pass disagreed and Ink
    // parked the cursor several cells past the last rendered character.
    const MSG = 'on a new bb branch investigate and fix the cursor drift bug here'

    for (const cols of [10, 14, 20, 30, 50, 80]) {
      let acc = ''

      for (const ch of MSG) {
        acc += ch
        expect(cursorLayout(acc, acc.length, cols)).toEqual(wrapAnsiEndPosition(acc, cols))
      }
    }
  })
})

describe('input metrics helpers', () => {
  it('computes visual height matching wrap-ansi line count', () => {
    // Exact-fill text stays on one line in wrap-ansi (no phantom wrap), so
    // visual height is 1. The previous implementation reported 2 here.
    expect(inputVisualHeight('abcdefgh', 8)).toBe(1)
    expect(inputVisualHeight('one\ntwo', 40)).toBe(2)
    // Multi-line wrap case sanity
    expect(inputVisualHeight('hello world', 8)).toBe(2)
  })

  it('counts the prompt gap as its own cell', () => {
    expect(composerPromptWidth('>')).toBe(2)
    expect(composerPromptWidth('❯')).toBe(2)
    expect(composerPromptWidth('Ψ >')).toBe(4)
  })

  it('reserves gutters on wide panes without starving narrow composer width', () => {
    expect(stableComposerColumns(100, 3)).toBe(93)
    expect(stableComposerColumns(100, 5)).toBe(91)
    expect(stableComposerColumns(10, 3)).toBe(5)
    expect(stableComposerColumns(6, 3)).toBe(1)
  })
})

describe('offsetFromPosition — word-wrap inverse of cursorLayout', () => {
  it('returns 0 for empty input', () => {
    expect(offsetFromPosition('', 0, 0, 10)).toBe(0)
  })

  it('maps clicks within a single line', () => {
    expect(offsetFromPosition('hello', 0, 3, 40)).toBe(3)
  })

  it('maps clicks past end to value length', () => {
    expect(offsetFromPosition('hi', 0, 10, 40)).toBe(2)
  })

  it('maps clicks on a wrapped second row at cols boundary', () => {
    // Long words still hard-wrap when there is no word boundary.
    expect(offsetFromPosition('abcdefghij', 1, 0, 8)).toBe(8)
  })

  it('maps clicks on a word-wrapped second row', () => {
    // "hello world" at cols=8 wraps to "hello \nworld".
    expect(offsetFromPosition('hello world', 1, 0, 8)).toBe(6)
    expect(offsetFromPosition('hello world', 1, 3, 8)).toBe(9)
  })

  it('maps clicks on the moved final word', () => {
    const text = 'hello world baby chickens are so cool its really rainy outside but wish'

    expect(offsetFromPosition(text, 1, 0, 70)).toBe(text.indexOf('wish'))
    expect(offsetFromPosition(text, 1, 3, 70)).toBe(text.indexOf('wish') + 3)
  })

  it('maps clicks past a \\n into the target line', () => {
    expect(offsetFromPosition('one\ntwo', 1, 2, 40)).toBe(6)
  })
})
