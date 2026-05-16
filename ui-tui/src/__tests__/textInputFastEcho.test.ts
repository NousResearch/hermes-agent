import { describe, expect, it } from 'vitest'

import { canFastAppendShape, canFastBackspaceShape } from '../components/textInput.js'

// The fast-echo path bypasses Ink and writes characters directly to stdout
// for the common case of typing plain English at the end of the line. These
// tests pin the shape preconditions that make that bypass safe.
//
// Regression intent: any non-ASCII text — Vietnamese precomposed letters
// (one grapheme, `text.length === 1`, `stringWidth === 1`, but produced
// via IME composition across multiple keystrokes), combining marks
// (zero width), CJK (double width), emoji (variable width), or anything
// that could be produced by an in-flight IME composition — must NOT
// take the bypass. Closes:
//   - "TUI is experiencing font errors when using Unicode to type Vietnamese"
//   - #5221  TUI input box renders incorrectly for CJK / East-Asian wide
//   - #7443  CLI TUI renders and deletes Chinese characters incorrectly
//   - #17602 / #17603  Chinese text scattering / ghosting

describe('canFastAppendShape', () => {
  const COLS = 40

  it('accepts plain ASCII appended at end of single-line input', () => {
    expect(canFastAppendShape('hello', 5, 'x', COLS, 5)).toBe(true)
    expect(canFastAppendShape('hello', 5, ' world', COLS, 5)).toBe(true)
  })

  it('rejects when cursor is not at end of line', () => {
    expect(canFastAppendShape('hello', 3, 'x', COLS, 5)).toBe(false)
  })

  it('rejects when current is empty (placeholder render path needed)', () => {
    expect(canFastAppendShape('', 0, 'x', COLS, 0)).toBe(false)
  })

  it('rejects when current contains a newline (multi-line layout)', () => {
    expect(canFastAppendShape('hi\nthere', 8, 'x', COLS, 5)).toBe(false)
  })

  it('rejects when appending would hit the wrap column', () => {
    // Reaching cols on append must trigger a wrap, which the bypass
    // cannot draw. Stay strictly below cols.
    expect(canFastAppendShape('hello', 5, 'x', 6, 5)).toBe(false)
  })

  // -- Regression coverage: Vietnamese / combining marks / IME --

  it('rejects Vietnamese precomposed letter ề (U+1EC1) — IME composition path', () => {
    // 'ề' is one grapheme, length 1, width 1, but Vietnamese Telex/IME
    // produces it via a multi-key composition. Fast-echo would commit the
    // intermediate state to stdout and desync once the final commit
    // arrives.
    expect(canFastAppendShape('hello', 5, 'ề', COLS, 5)).toBe(false)
  })

  it('rejects Vietnamese tone marks ă, ơ, ư (Latin-Extended-A/B)', () => {
    for (const ch of ['ă', 'ắ', 'ơ', 'ờ', 'ư', 'ự']) {
      expect(canFastAppendShape('hello', 5, ch, COLS, 5)).toBe(false)
    }
  })

  it('rejects NFD combining marks (U+0300 grave, U+0301 acute, U+0302 circumflex)', () => {
    // Decomposed Vietnamese: 'e' + combining circumflex + combining grave
    // = 'ề'. Each combining mark is zero-width but length 1; without the
    // ASCII guard the second/third keypress would be fast-echoed and
    // desync the cell column.
    expect(canFastAppendShape('hello', 5, '\u0300', COLS, 5)).toBe(false)
    expect(canFastAppendShape('hello', 5, '\u0301', COLS, 5)).toBe(false)
    expect(canFastAppendShape('hello', 5, '\u0302', COLS, 5)).toBe(false)
  })

  it('rejects CJK (East-Asian wide) characters', () => {
    expect(canFastAppendShape('hello', 5, '你', COLS, 5)).toBe(false)
    expect(canFastAppendShape('hello', 5, '日本', COLS, 5)).toBe(false)
  })

  it('rejects emoji', () => {
    expect(canFastAppendShape('hello', 5, '🙂', COLS, 5)).toBe(false)
  })

  it('rejects ANSI-bearing or control text', () => {
    expect(canFastAppendShape('hello', 5, '\x1b[31m', COLS, 5)).toBe(false)
    expect(canFastAppendShape('hello', 5, '\t', COLS, 5)).toBe(false)
    expect(canFastAppendShape('hello', 5, '\x7f', COLS, 5)).toBe(false)
  })

  it('rejects NBSP and Latin-1 letters that would change the line shape', () => {
    expect(canFastAppendShape('hello', 5, '\u00a0', COLS, 5)).toBe(false)
    expect(canFastAppendShape('hello', 5, 'é', COLS, 5)).toBe(false)
    expect(canFastAppendShape('hello', 5, 'ñ', COLS, 5)).toBe(false)
  })
})

describe('canFastBackspaceShape', () => {
  it('accepts deleting the last ASCII char', () => {
    expect(canFastBackspaceShape('hello', 5)).toBe(true)
  })

  it('rejects when cursor is not at end', () => {
    expect(canFastBackspaceShape('hello', 3)).toBe(false)
  })

  it('rejects when there is nothing to delete', () => {
    expect(canFastBackspaceShape('', 0)).toBe(false)
    expect(canFastBackspaceShape('hello', 0)).toBe(false)
  })

  it('rejects when value contains a newline', () => {
    expect(canFastBackspaceShape('hi\nthere', 8)).toBe(false)
  })

  it('rejects deleting Vietnamese precomposed letter ề', () => {
    // The "\b \b" shortcut clears one terminal cell; that's fine for a
    // 1-cell ASCII char but if the previous grapheme is a Vietnamese
    // letter that the IME may still be holding open, we want Ink to
    // re-render so composition state stays consistent.
    expect(canFastBackspaceShape('helloề', 'helloề'.length)).toBe(false)
  })

  it('rejects deleting a CJK character (2 cells)', () => {
    expect(canFastBackspaceShape('hi你', 'hi你'.length)).toBe(false)
  })

  it('rejects deleting a NFD-composed grapheme with combining marks', () => {
    // 'e' + U+0302 (circumflex) + U+0300 (grave) — final grapheme is one
    // cluster but the previous-grapheme slice is multi-codepoint. Width
    // is 1 but the bypass would be unsafe because the rendered cell
    // already contained the combined glyph.
    const s = 'hello' + 'e\u0302\u0300'
    expect(canFastBackspaceShape(s, s.length)).toBe(false)
  })

  it('rejects deleting an emoji', () => {
    expect(canFastBackspaceShape('hi🙂', 'hi🙂'.length)).toBe(false)
  })

  // Closes Copilot PR #26717 round 3: the "\b \b" sequence cannot move
  // the terminal cursor onto the previous visual row across a
  // soft-wrap boundary. When the caret sits at visual column 0 of a
  // wrapped row (column == 0 in the computed cursor layout), backspace
  // would leave the physical cursor in place while the logical caret
  // moves up to the end of the previous visual line — desyncing both
  // Ink's displayCursor model and the user-visible position. The fast
  // path must fall through in that case so the normal Ink render path
  // can lay out the correct cursor position.
  it('rejects fast-backspace at a soft-wrap boundary when columns is known', () => {
    // value width 6 in a column of 6 → cursorLayout produces (line 1, col 0)
    // i.e. the caret has overflowed onto the next visual line.
    const value = 'hello '
    expect(canFastBackspaceShape(value, value.length, 6)).toBe(false)
  })

  it('rejects fast-backspace at an exact multiple of columns (wide wrap)', () => {
    // 12 chars at width 6 → two full visual rows, caret at (line 2, col 0).
    const value = 'abcdefghijkl'
    expect(canFastBackspaceShape(value, value.length, 6)).toBe(false)
  })

  it('still accepts fast-backspace inside a wrapped line', () => {
    // Caret mid-visual-line — "\b \b" can move the cursor one cell left
    // without crossing a wrap boundary.
    expect(canFastBackspaceShape('hello world', 'hello world'.length, 20)).toBe(true)
    expect(canFastBackspaceShape('abcdefghi', 9, 6)).toBe(true) // visual line 1, col 3 → ok
  })

  it('skips the wrap-boundary check when columns is omitted (legacy contract)', () => {
    // Callers that don't pass `columns` fall back to the pre-wrap-aware
    // behavior — the function does NOT magically reject anything that
    // could be a wrap boundary without the width. Production callers
    // must always pass `columns`; this case is for unit tests of the
    // pre-wrap shape contract.
    expect(canFastBackspaceShape('hello ', 'hello '.length)).toBe(true)
  })
})
