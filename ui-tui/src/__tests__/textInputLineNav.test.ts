import { describe, expect, it } from 'vitest'

import {
  homeEndTarget,
  lineNav,
  logicalLineEnd,
  logicalLineStart,
  smartEndTarget,
  smartHomeTarget
} from '../components/textInput.js'

describe('logicalLineStart', () => {
  it('returns the buffer start for single-line input', () => {
    expect(logicalLineStart('hello world', 6)).toBe(0)
  })

  it('returns the start of the current logical line', () => {
    expect(logicalLineStart('one\ntwo\nthree', 7)).toBe(4)
  })

  it('keeps a cursor already at a line start on that line', () => {
    expect(logicalLineStart('one\ntwo', 4)).toBe(4)
  })

  it('treats a cursor on a newline as the end of the preceding line', () => {
    expect(logicalLineStart('one\ntwo', 3)).toBe(0)
  })

  it('keeps the start of an empty first line at zero', () => {
    expect(logicalLineStart('\nnext', 0)).toBe(0)
  })

  it('handles empty lines and Unicode text', () => {
    expect(logicalLineStart('one\n\nthree', 4)).toBe(4)
    expect(logicalLineStart('🙂a\nβγ', 6)).toBe(4)
  })
})

describe('logicalLineEnd', () => {
  it('returns the buffer end for single-line input', () => {
    expect(logicalLineEnd('hello world', 6)).toBe(11)
  })

  it('returns the end of the current logical line', () => {
    expect(logicalLineEnd('one\ntwo\nthree', 5)).toBe(7)
  })

  it('keeps a cursor already at a line end before the newline', () => {
    expect(logicalLineEnd('one\ntwo', 3)).toBe(3)
  })

  it('keeps the end of an empty first line at zero', () => {
    expect(logicalLineEnd('\nnext', 0)).toBe(0)
  })

  it('handles empty lines and Unicode text', () => {
    expect(logicalLineEnd('one\n\nthree', 4)).toBe(4)
    expect(logicalLineEnd('🙂a\nβγ', 4)).toBe(6)
  })

  it('returns the final empty line boundary', () => {
    expect(logicalLineEnd('one\n', 4)).toBe(4)
  })
})

describe('smartHomeTarget', () => {
  it('moves to the current line start before crossing a line boundary', () => {
    expect(smartHomeTarget('one\ntwo\nthree', 11)).toBe(8)
  })

  it('moves to the previous line start when already at a line start', () => {
    expect(smartHomeTarget('one\ntwo\nthree', 8)).toBe(4)
  })

  it('walks backward across logical lines on repeated calls', () => {
    const value = 'one\ntwo\nthree'
    let cursor = 11

    cursor = smartHomeTarget(value, cursor)
    expect(cursor).toBe(8)
    cursor = smartHomeTarget(value, cursor)
    expect(cursor).toBe(4)
    cursor = smartHomeTarget(value, cursor)
    expect(cursor).toBe(0)
    expect(smartHomeTarget(value, cursor)).toBe(0)
  })

  it('stops at an intervening empty line', () => {
    expect(smartHomeTarget('one\n\nthree', 5)).toBe(4)
  })
})

describe('smartEndTarget', () => {
  it('moves to the current line end before crossing a line boundary', () => {
    expect(smartEndTarget('one\ntwo\nthree', 1)).toBe(3)
  })

  it('moves to the next line end when already at a line end', () => {
    expect(smartEndTarget('one\ntwo\nthree', 3)).toBe(7)
  })

  it('walks forward across logical lines on repeated calls', () => {
    const value = 'one\ntwo\nthree'
    let cursor = 1

    cursor = smartEndTarget(value, cursor)
    expect(cursor).toBe(3)
    cursor = smartEndTarget(value, cursor)
    expect(cursor).toBe(7)
    cursor = smartEndTarget(value, cursor)
    expect(cursor).toBe(value.length)
    expect(smartEndTarget(value, cursor)).toBe(value.length)
  })

  it('stops at an intervening empty line', () => {
    expect(smartEndTarget('one\n\nthree', 3)).toBe(4)
  })
})

describe('homeEndTarget', () => {
  const key = (overrides: Partial<{ ctrl: boolean; end: boolean; home: boolean }>) => ({
    ctrl: false,
    end: false,
    home: false,
    ...overrides
  })

  it('moves Ctrl+Home and Ctrl+End across the whole buffer', () => {
    const value = 'one\ntwo\nthree'

    expect(homeEndTarget(value, 6, key({ ctrl: true, home: true }))).toBe(0)
    expect(homeEndTarget(value, 6, key({ ctrl: true, end: true }))).toBe(value.length)
  })

  it('keeps unmodified Home and End on smart logical-line navigation', () => {
    const value = 'one\ntwo\nthree'

    expect(homeEndTarget(value, 6, key({ home: true }))).toBe(4)
    expect(homeEndTarget(value, 6, key({ end: true }))).toBe(7)
  })

  it('ignores unrelated keys', () => {
    expect(homeEndTarget('one\ntwo', 5, key({ ctrl: true }))).toBeNull()
  })
})

describe('lineNav', () => {
  it('returns null for single-line input (up)', () => {
    expect(lineNav('hello world', 6, -1)).toBeNull()
  })

  it('returns null for single-line input (down)', () => {
    expect(lineNav('hello world', 6, 1)).toBeNull()
  })

  it('returns null when cursor already on first line of a multiline block', () => {
    expect(lineNav('one\ntwo\nthree', 2, -1)).toBeNull()
  })

  it('returns null when cursor on last line of a multiline block', () => {
    expect(lineNav('one\ntwo\nthree', 10, 1)).toBeNull()
  })

  it('moves cursor up one line preserving column', () => {
    // "hello\nworld" — cursor at col 3 of line 1 ('l' in world) → col 3 of line 0 ('l' in hello)
    expect(lineNav('hello\nworld', 9, -1)).toBe(3)
  })

  it('moves cursor down one line preserving column', () => {
    // cursor at col 2 of line 0 → col 2 of line 1
    expect(lineNav('hello\nworld', 2, 1)).toBe(8)
  })

  it('clamps to end of shorter destination line on up', () => {
    // col 10 on long line → clamp to end of short line "abc"
    const s = 'abc\nlong long text'
    const from = 14

    expect(lineNav(s, from, -1)).toBe(3)
  })

  it('clamps to end of shorter destination line on down', () => {
    // col 10 on line 0 → clamp to end of "abc" on line 1
    const s = 'long long text\nabc'

    expect(lineNav(s, 10, 1)).toBe(18)
  })

  it('handles empty lines correctly', () => {
    // "a\n\nb" — cursor at line 2 (b) → up to empty line 1
    expect(lineNav('a\n\nb', 3, -1)).toBe(2)
  })

  it('handles leading newline without crashing', () => {
    expect(lineNav('\nfoo', 2, -1)).toBe(0)
  })
})
