import { describe, expect, it } from 'vitest'

import { visualRowNav } from '../components/textInput.js'

describe('visualRowNav', () => {
  // ── Hard-newline cases (parity with lineNav) ──────────────────────────

  it('returns null for single-line input that fits in one row (up)', () => {
    expect(visualRowNav('hello', 3, -1, 80)).toBeNull()
  })

  it('returns null for single-line input that fits in one row (down)', () => {
    expect(visualRowNav('hello', 3, 1, 80)).toBeNull()
  })

  it('moves up across a hard newline preserving column', () => {
    // "hello\nworld", cols=80 → two visual rows, both under 80 chars
    // cursor at offset 9 = col 3 of "world" → col 3 of "hello" = offset 3
    expect(visualRowNav('hello\nworld', 9, -1, 80)).toBe(3)
  })

  it('moves down across a hard newline preserving column', () => {
    // cursor at offset 2 = col 2 of "hello" → col 2 of "world" = offset 8
    expect(visualRowNav('hello\nworld', 2, 1, 80)).toBe(8)
  })

  it('clamps to end of shorter destination line on up', () => {
    // "abc\nlong long text" — cursor at col 10 of line 1 → clamp to end of "abc"
    expect(visualRowNav('abc\nlong long text', 14, -1, 80)).toBe(3)
  })

  it('clamps to end of shorter destination line on down', () => {
    // "long long text\nabc" — cursor at col 10 → clamp to end of "abc"
    expect(visualRowNav('long long text\nabc', 10, 1, 80)).toBe(18)
  })

  it('returns null when on first line going up', () => {
    expect(visualRowNav('one\ntwo\nthree', 2, -1, 80)).toBeNull()
  })

  it('returns null when on last line going down', () => {
    expect(visualRowNav('one\ntwo\nthree', 10, 1, 80)).toBeNull()
  })

  // ── Soft-wrap cases (the new behavior) ────────────────────────────────

  it('navigates up within a soft-wrapped line', () => {
    // "abcdefghij" with cols=5 → visual rows: "abcde" (row 0), "fghij" (row 1)
    // cursor at offset 7 = 'h' → visual col 2 on row 1 → same col on row 0 = offset 2
    expect(visualRowNav('abcdefghij', 7, -1, 5)).toBe(2)
  })

  it('navigates down within a soft-wrapped line', () => {
    // "abcdefghij" with cols=5 → visual rows: "abcde" (row 0), "fghij" (row 1)
    // cursor at offset 2 = 'c' → visual col 2 on row 0 → same col on row 1 = offset 7
    expect(visualRowNav('abcdefghij', 2, 1, 5)).toBe(7)
  })

  it('returns null going up from first visual row of a soft-wrapped line', () => {
    // "abcdefghij" with cols=5 → cursor on row 0 → up returns null (history)
    expect(visualRowNav('abcdefghij', 2, -1, 5)).toBeNull()
  })

  it('returns null going down from last visual row of a soft-wrapped line', () => {
    // "abcdefgh" with cols=5 → rows: "abcde" (row 0), "fgh" (row 1)
    // cursor at offset 6 on row 1 → down returns null (no row 2)
    expect(visualRowNav('abcdefgh', 6, 1, 5)).toBeNull()
  })

  it('clamps column when soft-wrapped destination row is shorter', () => {
    // "abcdefgh" with cols=5 → rows: "abcde" (row 0, 5 chars), "fgh" (row 1, 3 chars)
    // cursor at offset 4 = 'e' → visual col 4 on row 0 → row 1 only has cols 0-2
    // should clamp to end of row 1 = offset 8 (end of string)
    expect(visualRowNav('abcdefgh', 4, 1, 5)).toBe(8)
  })

  // ── Mixed: hard newlines + soft wraps ─────────────────────────────────

  it('navigates down from a soft-wrapped row into a hard-newline row', () => {
    // "abcdefghij\nxy" with cols=5
    // visual rows: "abcde" (row 0), "fghij" (row 1), "xy" (row 2)
    // cursor at offset 7 = 'h' → visual col 2 on row 1 → row 2 col 2 = offset 13
    expect(visualRowNav('abcdefghij\nxy', 7, 1, 5)).toBe(13)
  })

  it('navigates up from a hard-newline row into a soft-wrapped row', () => {
    // "abcdefghij\nxy" with cols=5
    // visual rows: "abcde" (row 0), "fghij" (row 1), "xy" (row 2)
    // cursor at offset 11 = 'x' → visual col 0 on row 2 → row 1 col 0 = offset 5
    expect(visualRowNav('abcdefghij\nxy', 11, -1, 5)).toBe(5)
  })

  it('navigates across multiple soft-wrapped rows step by step', () => {
    // "abcdefghijklmno" with cols=5
    // visual rows: "abcde" (row 0), "fghij" (row 1), "klmno" (row 2)
    expect(visualRowNav('abcdefghijklmno', 12, -1, 5)).toBe(7)
    expect(visualRowNav('abcdefghijklmno', 7, -1, 5)).toBe(2)
    expect(visualRowNav('abcdefghijklmno', 2, -1, 5)).toBeNull()
  })

  // ── Edge cases ────────────────────────────────────────────────────────

  it('handles empty string', () => {
    expect(visualRowNav('', 0, -1, 80)).toBeNull()
    expect(visualRowNav('', 0, 1, 80)).toBeNull()
  })

  it('handles cols=1 (every character is its own visual row)', () => {
    // "abc" with cols=1 → rows: "a" (row 0), "b" (row 1), "c" (row 2)
    // cursor at offset 1 = 'b' → up → offset 0
    expect(visualRowNav('abc', 1, -1, 1)).toBe(0)
    // cursor at offset 1 = 'b' → down → offset 2
    expect(visualRowNav('abc', 1, 1, 1)).toBe(2)
  })

  it('handles cursor at end of string on a soft-wrap boundary', () => {
    // "abcde" with cols=5 fills exactly one visual row; cursor at offset 5
    // (end) stays on that row under wrap-ansi-aligned cursorLayout.
    expect(visualRowNav('abcde', 5, 1, 5)).toBeNull()
    expect(visualRowNav('abcde', 5, -1, 5)).toBeNull()
  })
})
