import { describe, expect, it } from 'vitest'

import { composerLocalPoint, offsetFromPosition } from './inputMetrics.js'

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

// Presses that land on the composer PARENT row (prompt gutter, the gap right
// of the input) or on the spacer above it bubble past TextInput's own handler,
// so appLayout has to translate them itself. This used to call
// startAtBeginning() and drop the caret at offset 0 no matter where the user
// clicked — the parent-row half of #30536. composerLocalPoint is that mapping;
// composed with offsetFromPosition it must land on the clicked cell.

describe('composerLocalPoint', () => {
  const promptWidth = 3 // e.g. "> " plus the gap column

  describe("parent composer row ('shared' origin)", () => {
    it('backs the prompt gutter out of the column and keeps the row', () => {
      expect(composerLocalPoint({ localCol: 9, localRow: 1 }, promptWidth, 'shared')).toEqual({ col: 6, row: 1 })
    })

    it('maps a press on the parent row to the clicked offset, not offset 0', () => {
      // "ab\ncd": clicking row 1, terminal column 4 → local column 1 → 'd'.
      const at = composerLocalPoint({ localCol: promptWidth + 1, localRow: 1 }, promptWidth, 'shared')

      expect(offsetFromPosition('ab\ncd', at.row, at.col, 80)).toBe(4)
    })

    it('resolves a press on the prompt cell to column 0 of that row, not the buffer start', () => {
      // Negative local column (inside the gutter) clamps to the row start —
      // row 1 of "ab\ncd" begins at offset 3, so the caret must NOT go to 0.
      const at = composerLocalPoint({ localCol: 0, localRow: 1 }, promptWidth, 'shared')

      expect(at.col).toBe(-promptWidth)
      expect(offsetFromPosition('ab\ncd', at.row, at.col, 80)).toBe(3)
    })

    it('honors soft-wrapped rows', () => {
      // "abcdef" at cols=3 wraps to "abc" / "def"; row 1, local col 2 → 'f'.
      const at = composerLocalPoint({ localCol: promptWidth + 2, localRow: 1 }, promptWidth, 'shared')

      expect(offsetFromPosition('abcdef', at.row, at.col, 3)).toBe(5)
    })
  })

  describe("spacer row ('row-zero' origin)", () => {
    it('pins the row to 0 so a vertical offset cannot pick the wrong wrapped line', () => {
      expect(composerLocalPoint({ localCol: 9, localRow: 4 }, promptWidth, 'row-zero')).toEqual({ col: 6, row: 0 })
    })

    it('still maps the column onto the first visual row', () => {
      const at = composerLocalPoint({ localCol: promptWidth + 1, localRow: 2 }, promptWidth, 'row-zero')

      expect(offsetFromPosition('ab\ncd', at.row, at.col, 80)).toBe(1) // 'b'
    })
  })

  it('treats missing coordinates as the origin', () => {
    expect(composerLocalPoint({}, promptWidth, 'shared')).toEqual({ col: -promptWidth, row: 0 })
  })
})
