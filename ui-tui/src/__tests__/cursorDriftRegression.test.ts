/**
 * Pinned regression for the multi-line composer cursor-drift bug.
 *
 * Symptom: in `hermes --tui`, typing into the composer until the input
 * wraps across multiple visual rows would leave several blank cells
 * between the last typed character and the (hardware) cursor block.
 * Worse on narrow terminals (the Cursor IDE built-in terminal in
 * particular).
 *
 * Root cause: the composer's `cursorLayout` (used by `useDeclaredCursor`
 * to place the hardware cursor) ran a hand-rolled word-wrap algorithm,
 * while Ink's `<Text wrap="wrap">` renders via `wrap-ansi`. The two
 * disagreed on many real inputs — wrap-ansi would keep "branch
 * investigate" on one row while cursorLayout claimed it had wrapped,
 * etc. — so the declared cursor position drifted from where the text
 * was actually rendered. The fix sources cursorLayout's line breaks
 * directly from wrap-ansi, guaranteeing agreement.
 *
 * This test pins the contract: for every char that would be typed into
 * the composer, the cursor position reported by cursorLayout MUST equal
 * the end-of-text position that wrap-ansi would render. Any future
 * regression that lets the two diverge re-introduces the drift.
 */
import { wrapAnsi } from '@hermes/ink'
import { describe, expect, it } from 'vitest'

import { cursorLayout, inputVisualHeight } from '../lib/inputMetrics.js'

function wrapAnsiEnd(text: string, cols: number): { line: number; column: number } {
  const wrapped = wrapAnsi(text, cols, { hard: true, trim: false })
  const lines = wrapped.split('\n')
  const last = lines[lines.length - 1] ?? ''

  return { line: lines.length - 1, column: last.length }
}

const USER_REPORT_MESSAGE =
  // Paraphrase of the user's actual bug report, included verbatim so the
  // test is grounded in a realistic typing pattern (long single line,
  // mixed-length words, punctuation, no hard newlines).
  'im in cursor terminal using hermes --tui and as i type multiline my caret at the end will often ' +
  'go.. randomly.. like multiple spaces away lol and idk why. theres no rhyme/reason really but ' +
  'there should literally never be a non-user added space at the end of my composer input right? ' +
  'i dont think it happens on new sessions but only existing ones. there have been a few prs to ' +
  'try to fix this and all not working. ok it just happened, to me, nowso attaching screenshot ' +
  'and you can see its multiline, new session. on a new bb/<xxx> branch investigate'

describe('cursor-drift regression — composer cursorLayout matches Ink rendering', () => {
  it('agrees with wrap-ansi at every typing-prefix of the user-reported message', () => {
    // Walks the message char-by-char (mirroring what the TUI sees when a
    // user types). At every prefix, cursorLayout must place the cursor
    // exactly where wrap-ansi would render the end of the text.
    //
    // Pre-fix: this failed on most narrow widths because the hand-rolled
    // wrap algorithm broke at slightly different points than wrap-ansi.
    for (const cols of [40, 50, 55, 60, 65, 70, 80]) {
      let acc = ''

      for (const ch of USER_REPORT_MESSAGE) {
        acc += ch
        const layout = cursorLayout(acc, acc.length, cols)
        const expected = wrapAnsiEnd(acc, cols)

        expect(
          layout,
          `mismatch at cols=${cols}, len=${acc.length}, last-char=${JSON.stringify(ch)}, ` +
            `tail=${JSON.stringify(acc.slice(-30))}`
        ).toEqual(expected)
      }
    }
  })

  it('keeps cursor on the same row when text exactly fills the terminal width', () => {
    // wrap-ansi does NOT push exact-fill text onto a phantom next line.
    // The previous algorithm did — that's what produced the visible
    // "cursor parked one row below the last char" symptom on narrow
    // terminals at certain message lengths.
    for (const cols of [8, 12, 18, 24]) {
      const text = 'a'.repeat(cols)
      const layout = cursorLayout(text, text.length, cols)
      const inkLines = wrapAnsi(text, cols, { hard: true, trim: false }).split('\n')

      expect(layout.line).toBe(0)
      expect(layout.column).toBe(cols)
      expect(inkLines).toHaveLength(1)
      expect(inputVisualHeight(text, cols)).toBe(1)
    }
  })

  it('does not stuff a trailing whitespace word onto a phantom line', () => {
    // "branch investigate" at cols=20 fits on one row in wrap-ansi. The
    // bug claimed otherwise, parking the cursor at (line=1, col=?) and
    // leaving the user's "branch investigate" rendered alone on row 0
    // with the cursor block several cells past it.
    const text = 'branch investigate'
    const cols = 20

    expect(cursorLayout(text, text.length, cols)).toEqual({ column: text.length, line: 0 })
    expect(cursorLayout(text, text.length, cols)).toEqual(wrapAnsiEnd(text, cols))
  })

  it('agrees with wrap-ansi for word-wrap that pushes a word onto the next line', () => {
    // "hello world" at cols=8 wraps to ["hello ", "world"] in wrap-ansi.
    // The cursor at end-of-text must land at line=1, col=5 — where Ink
    // actually renders the last 'd'. The previous algorithm reported
    // (line=2, col=0) here (phantom extra wrap), which parked the
    // cursor on a row Ink never painted.
    const text = 'hello world'
    const cols = 8

    expect(cursorLayout(text, text.length, cols)).toEqual({ column: 5, line: 1 })
    expect(cursorLayout(text, text.length, cols)).toEqual(wrapAnsiEnd(text, cols))
  })
})
