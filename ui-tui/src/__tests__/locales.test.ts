import { describe, expect, test } from 'vitest'

import { en } from '../locales/en.js'
import { getLocale, setLocale, t, zh } from '../locales/index.js'

describe('locales', () => {
  test('default locale is english', () => {
    const locale = t()
    expect(locale.messages.welcome).toBe('Welcome to Hermes')
  })

  test('can switch to chinese', () => {
    setLocale('zh')
    const locale = t()
    expect(locale.messages.welcome).toBe('欢迎使用 Hermes')
  })

  test('falls back to english for unknown locale', () => {
    setLocale('fr' as any)
    const locale = t()
    expect(locale.messages.welcome).toBe('Welcome to Hermes')
  })

  test('setLocale/getLocale round trip', () => {
    setLocale('zh')
    expect(getLocale()).toBe('zh')
    expect(t().messages.welcome).toBe('欢迎使用 Hermes')
    setLocale('en')
    expect(getLocale()).toBe('en')
    expect(t().messages.welcome).toBe('Welcome to Hermes')
  })

  test('getLocale returns current locale', () => {
    setLocale('en')
    expect(getLocale()).toBe('en')
    setLocale('zh')
    expect(getLocale()).toBe('zh')
  })

  test('en locale has all required keys', () => {
    setLocale('en')
    const locale = t()
    expect(locale.common.loading).toBeDefined()
    expect(locale.menu.help).toBeDefined()
    expect(locale.statusBar.online).toBeDefined()
    expect(locale.messages.welcome).toBeDefined()
    expect(locale.errors.connectionFailed).toBeDefined()
  })

  test('zh locale has all required keys', () => {
    setLocale('zh')
    const locale = t()
    expect(locale.common.loading).toBeDefined()
    expect(locale.menu.help).toBeDefined()
    expect(locale.statusBar.online).toBeDefined()
    expect(locale.messages.welcome).toBeDefined()
    expect(locale.errors.connectionFailed).toBeDefined()
  })
})
