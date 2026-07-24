import { describe, expect, it } from 'vitest'

import { offsetFromPosition } from './inputMetrics.js'

// offsetFromPosition maps a click at a visual (row, col) back to a character
// offset in the composer value. It backs textInput's click-to-position and
// drag-select (offsetAt → offsetFromPosition), so these cases pin the
// coordinate mapping across single-line, multi-line, and soft-wrapped input
// (issue #30536). Columns/rows are 0-indexed visual cells.

describe('offsetFromPosition', () => {
  it('returns 0 for an empty value regardless of position', () => {
    expect(offsetFromPosition('', 0, 0, 80)).toBe(0)
    expect(offsetFromPosition('', 3, 7, 80)).toBe(0)
  })

  describe('single line (no wrap)', () => {
    const v = 'hello world'

    it('maps the first column to offset 0', () => {
      expect(offsetFromPosition(v, 0, 0, 80)).toBe(0)
    })

    it('maps a column to the character under it', () => {
      expect(offsetFromPosition(v, 0, 6, 80)).toBe(6) // 'w'
    })

    it('clamps a click past the end of the line to the line end', () => {
      expect(offsetFromPosition(v, 0, 100, 80)).toBe(v.length)
    })
  })

  describe('multi-line (hard newlines)', () => {
    const v = 'ab\ncd' // line 0: "ab" [0,2), '\n' at 2, line 1: "cd" [3,5)

    it('maps a click on the second line to the correct offset', () => {
      expect(offsetFromPosition(v, 1, 0, 80)).toBe(3) // 'c'
      expect(offsetFromPosition(v, 1, 1, 80)).toBe(4) // 'd'
    })

    it('maps a click on the first line to the correct offset', () => {
      expect(offsetFromPosition(v, 0, 1, 80)).toBe(1) // 'b'
    })

    it('clamps a row past the last line to the last line', () => {
      expect(offsetFromPosition(v, 9, 0, 80)).toBe(3) // start of "cd"
    })
  })

  describe('soft-wrapped line', () => {
    const v = 'abcdef' // at cols=3 wraps to "abc" / "def"

    it('maps a click on the first wrapped row', () => {
      expect(offsetFromPosition(v, 0, 2, 3)).toBe(2) // 'c'
    })

    it('maps a click on the second wrapped row past the wrap boundary', () => {
      expect(offsetFromPosition(v, 1, 0, 3)).toBe(3) // 'd'
      expect(offsetFromPosition(v, 1, 2, 3)).toBe(5) // 'f'
    })
  })

  it('floors fractional row/column inputs', () => {
    expect(offsetFromPosition('hello world', 0.9, 6.4, 80)).toBe(6)
  })
})
