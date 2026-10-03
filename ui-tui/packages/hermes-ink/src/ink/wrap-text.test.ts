import { describe, expect, it } from 'vitest'

import wrapText, { wrapTextWithTrim } from './wrap-text.js'

describe('wrapText wrap-trim', () => {
  it('removes a single soft-wrap boundary space', () => {
    expect(wrapText('Let me', 5, 'wrap-trim')).toBe('Let\nme')
  })

  it('preserves extra original spacing at soft-wrap boundaries', () => {
    expect(wrapText('foo  bar', 5, 'wrap-trim')).toBe('foo \nbar')
  })

  it('preserves leading whitespace on unwrapped source lines', () => {
    expect(wrapText('  indented', 20, 'wrap-trim')).toBe('  indented')
  })
})

describe('wrapTextWithTrim boundary flags', () => {
  it('flags a word-wrap boundary that dropped the separator space', () => {
    expect(wrapTextWithTrim('Let me', 5, 'wrap-trim')).toEqual({ text: 'Let\nme', trimmed: [true] })
  })

  it('still flags when one of several spaces survives on screen', () => {
    expect(wrapTextWithTrim('foo  bar', 5, 'wrap-trim')).toEqual({ text: 'foo \nbar', trimmed: [true] })
  })

  it('never flags a hard mid-word split (copier must keep it glued)', () => {
    const entry = wrapTextWithTrim('abcdefghij', 7, 'wrap-trim')

    expect(entry.trimmed).toEqual([false])
    expect(entry.text.split('\n').join('')).toBe('abcdefghij')
  })

  it('never flags plain wrap mode boundaries', () => {
    expect(wrapTextWithTrim('Let me', 5, 'wrap').trimmed).toEqual([false])
  })

  it('marks hard source newlines as untrimmed', () => {
    const entry = wrapTextWithTrim('Let me\nab cde', 5, 'wrap-trim')

    expect(entry.text).toBe('Let\nme\nab\ncde')
    expect(entry.trimmed).toEqual([true, false, true])
  })
})
