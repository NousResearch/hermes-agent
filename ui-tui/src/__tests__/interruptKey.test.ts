import { describe, expect, it } from 'vitest'

import {
  DEFAULT_INTERRUPT_KEY,
  formatInterruptKey,
  isInterruptKey,
  parseInterruptKey
} from '../lib/platform.js'

const key = (overrides: Record<string, unknown> = {}) =>
  ({ ctrl: false, meta: false, alt: false, shift: false, escape: false, ...overrides }) as any

describe('parseInterruptKey', () => {
  it('returns default (escape) for empty/null/undefined input', () => {
    expect(parseInterruptKey('')).toEqual(DEFAULT_INTERRUPT_KEY)
    expect(parseInterruptKey(null)).toEqual(DEFAULT_INTERRUPT_KEY)
    expect(parseInterruptKey(undefined)).toEqual(DEFAULT_INTERRUPT_KEY)
  })

  it('normalizes "esc" and "escape" to the default escape key', () => {
    expect(parseInterruptKey('escape')).toEqual(DEFAULT_INTERRUPT_KEY)
    expect(parseInterruptKey('Esc')).toEqual(DEFAULT_INTERRUPT_KEY)
    expect(parseInterruptKey('ESCAPE')).toEqual(DEFAULT_INTERRUPT_KEY)
    expect(parseInterruptKey('  esc  ')).toEqual(DEFAULT_INTERRUPT_KEY)
  })

  it('parses modifier combos like ctrl+g', () => {
    const parsed = parseInterruptKey('ctrl+g')

    expect(parsed.ch).toBe('g')
    expect(parsed.mod).toBe('ctrl')
    expect(parsed.raw).toBe('ctrl+g')
  })

  it('parses alt+i', () => {
    const parsed = parseInterruptKey('alt+i')

    expect(parsed.ch).toBe('i')
    expect(parsed.mod).toBe('alt')
    expect(parsed.raw).toBe('alt+i')
  })

  it('falls back to default for invalid input', () => {
    expect(parseInterruptKey('not+a+valid+combo')).toEqual(DEFAULT_INTERRUPT_KEY)
    expect(parseInterruptKey(123)).toEqual(DEFAULT_INTERRUPT_KEY)
  })

  it('accepts ctrl+l (not reserved for interrupt unlike voice)', () => {
    const parsed = parseInterruptKey('ctrl+l')

    expect(parsed.ch).toBe('l')
    expect(parsed.mod).toBe('ctrl')
    expect(parsed.raw).toBe('ctrl+l')
  })

  it('still rejects ctrl+c (SIGINT)', () => {
    expect(parseInterruptKey('ctrl+c')).toEqual(DEFAULT_INTERRUPT_KEY)
  })
})

describe('isInterruptKey', () => {
  it('matches bare Escape for the default config', () => {
    expect(isInterruptKey(key({ escape: true }), '', DEFAULT_INTERRUPT_KEY)).toBe(true)
  })

  it('does not match Escape with modifiers held for default config', () => {
    expect(isInterruptKey(key({ escape: true, ctrl: true }), '', DEFAULT_INTERRUPT_KEY)).toBe(false)
    expect(isInterruptKey(key({ escape: true, alt: true }), '', DEFAULT_INTERRUPT_KEY)).toBe(false)
  })

  it('matches ctrl+g when configured', () => {
    const cfg = parseInterruptKey('ctrl+g')

    expect(isInterruptKey(key({ ctrl: true }), 'g', cfg)).toBe(true)
  })

  it('does not match bare g when ctrl+g is configured', () => {
    const cfg = parseInterruptKey('ctrl+g')

    expect(isInterruptKey(key(), 'g', cfg)).toBe(false)
  })

  it('does not match ctrl+g when escape is configured', () => {
    expect(isInterruptKey(key({ ctrl: true }), 'g', DEFAULT_INTERRUPT_KEY)).toBe(false)
  })

  it('does not match Cmd+B (super) for ctrl+b configured interrupt key', () => {
    const cfg = parseInterruptKey('ctrl+b')

    expect(isInterruptKey(key({ super: true }), 'b', cfg)).toBe(false)
  })
})

describe('formatInterruptKey', () => {
  it('formats default as Esc', () => {
    expect(formatInterruptKey(DEFAULT_INTERRUPT_KEY)).toBe('Esc')
  })

  it('formats ctrl+g as Ctrl+G', () => {
    expect(formatInterruptKey(parseInterruptKey('ctrl+g'))).toBe('Ctrl+G')
  })
})
