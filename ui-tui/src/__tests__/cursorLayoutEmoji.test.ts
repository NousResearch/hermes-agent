import { describe, expect, it } from 'vitest'

import { cursorLayout } from '../lib/inputMetrics.js'

describe('cursorLayout — grapheme boundary snapping', () => {
  it('snaps mid-surrogate cursor in single emoji back to grapheme start', () => {
    expect(cursorLayout('👍', 1, 80)).toEqual({ column: 0, line: 0 })
  })

  it('reports full width at end of single emoji', () => {
    expect(cursorLayout('👍', 2, 80)).toEqual({ column: 2, line: 0 })
  })

  it('snaps mid-flag (regional indicator pair) cursor back to flag start', () => {
    expect(cursorLayout('🇧🇷', 2, 80)).toEqual({ column: 0, line: 0 })
  })

  it('reports full width at end of regional flag', () => {
    expect(cursorLayout('🇧🇷', 4, 80)).toEqual({ column: 2, line: 0 })
  })

  it('positions cursor before emoji embedded in ascii', () => {
    expect(cursorLayout('a👍b', 1, 80)).toEqual({ column: 1, line: 0 })
  })

  it('snaps mid-surrogate cursor inside embedded emoji back to emoji start', () => {
    expect(cursorLayout('a👍b', 2, 80)).toEqual({ column: 1, line: 0 })
  })

  it('positions cursor at end of embedded emoji', () => {
    expect(cursorLayout('a👍b', 3, 80)).toEqual({ column: 3, line: 0 })
  })

  it('handles ZWJ family sequences as a single grapheme', () => {
    const family = '👨‍👩‍👧'
    expect(cursorLayout(family, 1, 80)).toEqual({ column: 0, line: 0 })
    expect(cursorLayout(family, family.length, 80)).toEqual({
      column: 2,
      line: 0
    })
  })

  it('preserves ascii cursor positions unchanged', () => {
    expect(cursorLayout('hello', 0, 80)).toEqual({ column: 0, line: 0 })
    expect(cursorLayout('hello', 3, 80)).toEqual({ column: 3, line: 0 })
    expect(cursorLayout('hello', 5, 80)).toEqual({ column: 5, line: 0 })
  })

  it('clamps negative cursor to start', () => {
    expect(cursorLayout('👍', -5, 80)).toEqual({ column: 0, line: 0 })
  })

  it('clamps overflow cursor to end of value', () => {
    expect(cursorLayout('👍', 99, 80)).toEqual({ column: 2, line: 0 })
  })
})
