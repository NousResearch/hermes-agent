import { describe, expect, it } from 'vitest'

import { applyPrintableInsert, normalizeComposerInputState, shouldRouteMultiCharInputAsPaste } from '../components/textInput.js'

describe('applyPrintableInsert', () => {
  it('applies non-bracketed multi-character bursts immediately', () => {
    const burst = applyPrintableInsert('abc', 3, 'xxxxx')

    const repeated = [...'xxxxx'].reduce(
      (state, ch) => applyPrintableInsert(state.value, state.cursor, ch)!,
      { cursor: 3, value: 'abc' }
    )

    expect(burst).toEqual({ cursor: 8, value: 'abcxxxxx' })
    expect(burst).toEqual(repeated)
  })

  it('replaces the selected range for burst input', () => {
    expect(applyPrintableInsert('abZZef', 4, 'cd', { end: 4, start: 2 })).toEqual({
      cursor: 4,
      value: 'abcdef'
    })
  })

  it('rejects control or escape-bearing input', () => {
    expect(applyPrintableInsert('abc', 3, '\x1b[200~pasted')).toBeNull()
    expect(applyPrintableInsert('abc', 3, '\t')).toBeNull()
  })
})

describe('normalizeComposerInputState', () => {
  it('normalizes decomposed Hangul jamo to NFC and preserves cursor position', () => {
    const decomposed = '\u1112\u1161\u11AB\u1100\u1173\u11AF'
    const result = normalizeComposerInputState(decomposed, decomposed.length)

    expect(result).toEqual({ cursor: 2, value: '한글' })
  })

  it('keeps non-Hangul combining marks untouched', () => {
    const value = 'Cafe\u0301'

    expect(normalizeComposerInputState(value, value.length)).toEqual({ cursor: value.length, value })
  })

  it('updates cursor from the normalized prefix', () => {
    const value = 'A\u1112\u1161B\u1102\u1161'

    expect(normalizeComposerInputState(value, 3)).toEqual({ cursor: 2, value: 'A하B나' })
  })
})

describe('shouldRouteMultiCharInputAsPaste', () => {
  it('keeps newline-bearing chunks on the paste path', () => {
    expect(shouldRouteMultiCharInputAsPaste('hello\nworld')).toBe(true)
    expect(shouldRouteMultiCharInputAsPaste('hello\r\nworld'.replace(/\r\n/g, '\n'))).toBe(true)
  })

  it('treats repeated printable key bursts as immediate input', () => {
    expect(shouldRouteMultiCharInputAsPaste('xxxxx')).toBe(false)
  })
})
