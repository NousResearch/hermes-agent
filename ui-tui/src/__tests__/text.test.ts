import { describe, expect, it } from 'vitest'

import {
  compactPreview,
  estimateRows,
  fmtK,
  hasAnsi,
  hasInterpolation,
  pick,
  stripAnsi,
  userDisplay
} from '../lib/text.js'


describe('stripAnsi / hasAnsi', () => {

  it('strips ANSI codes', () => {
    expect(stripAnsi('\x1b[31mred\x1b[0m')).toBe('red')
  })

  it('passes plain text through', () => {
    expect(stripAnsi('hello')).toBe('hello')
  })

  it('detects ANSI', () => {
    expect(hasAnsi('\x1b[1mbold\x1b[0m')).toBe(true)
    expect(hasAnsi('plain')).toBe(false)
  })
})


describe('compactPreview', () => {

  it('truncates with ellipsis', () => {
    expect(compactPreview('a'.repeat(100), 20)).toHaveLength(20)
    expect(compactPreview('a'.repeat(100), 20).at(-1)).toBe('…')
  })

  it('returns short strings as-is', () => {
    expect(compactPreview('hello', 20)).toBe('hello')
  })

  it('collapses whitespace', () => {
    expect(compactPreview('  a   b  ', 20)).toBe('a b')
  })

  it('returns empty for whitespace-only', () => {
    expect(compactPreview('   ', 20)).toBe('')
  })
})


describe('estimateRows', () => {

  it('single line', () => expect(estimateRows('hello', 80)).toBe(1))

  it('wraps long lines', () => expect(estimateRows('a'.repeat(160), 80)).toBe(2))

  it('counts newlines', () => expect(estimateRows('a\nb\nc', 80)).toBe(3))

  it('skips table separators', () => {
    expect(estimateRows('| a | b |\n|---|---|\n| 1 | 2 |', 80)).toBe(2)
  })

  it('handles code blocks', () => {
    expect(estimateRows('```python\nprint("hi")\n```', 80)).toBeGreaterThanOrEqual(2)
  })

  it('compact mode skips empty lines', () => {
    expect(estimateRows('a\n\nb', 80, true)).toBe(2)
    expect(estimateRows('a\n\nb', 80, false)).toBe(3)
  })
})


describe('fmtK', () => {

  it('formats thousands', () => expect(fmtK(1500)).toBe('1.5k'))

  it('keeps small numbers', () => expect(fmtK(42)).toBe('42'))

  it('boundary', () => {
    expect(fmtK(1000)).toBe('1.0k')
    expect(fmtK(999)).toBe('999')
  })
})


describe('hasInterpolation', () => {

  it('detects {!cmd}', () => expect(hasInterpolation('echo {!date}')).toBe(true))

  it('rejects plain text', () => expect(hasInterpolation('plain')).toBe(false))
})


describe('pick', () => {

  it('returns element from array', () => {
    expect([1, 2, 3]).toContain(pick([1, 2, 3]))
  })
})


describe('userDisplay', () => {

  it('returns short messages as-is', () => expect(userDisplay('hello')).toBe('hello'))

  it('truncates long messages', () => {
    expect(userDisplay('word '.repeat(100))).toContain('[long message]')
  })
})
