import { stringWidth } from '@hermes/ink'
import { describe, expect, it } from 'vitest'

import {
  cursorLayout,
  inputVisualHeight,
  offsetFromPosition
} from '../lib/inputMetrics.js'

describe('inputMetrics — CJK / emoji / multi-line edge cases', () => {
  describe('cursorLayout', () => {
    it('places cursor on row 0 col 0 for empty input', () => {
      expect(cursorLayout('', 0, 80)).toEqual({ column: 0, line: 0 })
    })

    it('clamps cursor below 0 to start', () => {
      expect(cursorLayout('hello', -5, 80)).toEqual({ column: 0, line: 0 })
    })

    it('clamps cursor past end to last position', () => {
      expect(cursorLayout('hi', 999, 80)).toEqual({ column: 2, line: 0 })
    })

    it('treats cols<=0 as width 1', () => {
      const result = cursorLayout('abc', 3, 0)
      expect(result.line).toBeGreaterThanOrEqual(2)
    })

    it('counts CJK chars as width 2', () => {
      expect(cursorLayout('你好', 2, 10)).toEqual({ column: 4, line: 0 })
    })

    it('wraps when CJK char would overflow', () => {
      const result = cursorLayout('你好', 2, 3)
      expect(result.line).toBeGreaterThanOrEqual(1)
    })

    it('treats multi-codepoint emoji ZWJ sequence as a single grapheme', () => {
      const family = '👨‍👩‍👧'
      const layout = cursorLayout(family, family.length, 80)
      // If the ZWJ sequence were segmented into multiple graphemes, the
      // cumulative column would exceed the grapheme's stringWidth. Assert
      // the exact column matches stringWidth(family) so any regression in
      // grapheme segmentation fails this test.
      expect(layout).toEqual({ column: stringWidth(family), line: 0 })
    })

    it('handles \\n with row increment', () => {
      const v = 'a\nb\nc'
      expect(cursorLayout(v, 0, 80)).toEqual({ column: 0, line: 0 })
      expect(cursorLayout(v, 2, 80)).toEqual({ column: 0, line: 1 })
      expect(cursorLayout(v, 4, 80)).toEqual({ column: 0, line: 2 })
    })

    it('cursor at end of first line lands at trailing column', () => {
      expect(cursorLayout('abc\ndef', 3, 80)).toEqual({ column: 3, line: 0 })
    })

    it('overflows trailing cursor to next visual row at exact wrap column', () => {
      const layout = cursorLayout('abcde', 5, 5)
      expect(layout).toEqual({ column: 0, line: 1 })
    })
  })

  describe('offsetFromPosition', () => {
    it('returns 0 for empty input', () => {
      expect(offsetFromPosition('', 0, 0, 80)).toBe(0)
    })

    it('clamps negative row/col', () => {
      expect(offsetFromPosition('hello', -1, -1, 80)).toBe(0)
    })

    it('returns end-of-line for col past the line width', () => {
      expect(offsetFromPosition('abc', 0, 999, 80)).toBe(3)
    })

    it('maps CJK columns back to grapheme offsets', () => {
      expect(offsetFromPosition('你好', 0, 0, 10)).toBe(0)
      expect(offsetFromPosition('你好', 0, 2, 10)).toBe(1)
    })

    it('snaps to grapheme start when col lands inside a wide grapheme', () => {
      expect(offsetFromPosition('你好', 0, 1, 10)).toBe(0)
    })
  })

  describe('inputVisualHeight', () => {
    it('returns 1 for empty input', () => {
      expect(inputVisualHeight('', 80)).toBe(1)
    })

    it('returns 1 for single short line', () => {
      expect(inputVisualHeight('hello', 80)).toBe(1)
    })

    it('counts \\n-separated logical lines', () => {
      expect(inputVisualHeight('a\nb\nc', 80)).toBe(3)
    })

    it('counts wrapped visual lines for long input', () => {
      const long = 'a'.repeat(200)
      expect(inputVisualHeight(long, 50)).toBeGreaterThanOrEqual(4)
    })

    it('counts CJK width when computing wrap', () => {
      const cjk = '你'.repeat(50)
      expect(inputVisualHeight(cjk, 20)).toBeGreaterThanOrEqual(5)
    })
  })
})
