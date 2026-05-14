import { describe, expect, it } from 'vitest'

import { returnKeyAction } from '../components/textInput.js'

const key = (overrides: Record<string, unknown> = {}) =>
  ({ ctrl: false, meta: false, shift: false, super: false, return: true, ...overrides }) as any

describe('returnKeyAction', () => {
  it('uses chat-style macOS composer behavior', () => {
    expect(returnKeyAction(key(), true)).toBe('newline')
    expect(returnKeyAction(key({ shift: true }), true)).toBe('newline')
    expect(returnKeyAction(key({ ctrl: true }), true)).toBe('newline')
    expect(returnKeyAction(key({ meta: true }), true)).toBe('newline')
    expect(returnKeyAction(key({ super: true }), true)).toBe('submit')
    expect(returnKeyAction(key({ ctrl: true, super: true }), true)).toBe('submit')
    expect(returnKeyAction(key({ shift: true, super: true }), true)).toBe('newline')
  })

  it('preserves the existing non-macOS submit behavior', () => {
    expect(returnKeyAction(key(), false)).toBe('submit')
    expect(returnKeyAction(key({ shift: true }), false)).toBe('newline')
    expect(returnKeyAction(key({ ctrl: true }), false)).toBe('newline')
    expect(returnKeyAction(key({ meta: true }), false)).toBe('newline')
  })
})
