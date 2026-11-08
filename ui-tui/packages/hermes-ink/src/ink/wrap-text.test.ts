import { describe, expect, it } from 'vitest'

import wrapText from './wrap-text.js'

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
