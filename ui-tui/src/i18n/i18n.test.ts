import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import { en } from './en.js'
import {
  $tuiLocale,
  $tuiText,
  clearTuiCatalogCache,
  normalizeTuiLocale,
  resolveTuiCatalog,
  setTuiLocale
} from './index.js'
import { mergeStrings, readUserOverrides } from './overrides.js'

function writeTuiOverride(lang: string, obj: unknown): string {
  const base = mkdtempSync(join(tmpdir(), 'hermes-tui-ov-'))
  const dir = join(base, 'tui')
  mkdirSync(dir, { recursive: true })
  writeFileSync(join(dir, `${lang}.json`), JSON.stringify(obj), 'utf8')
  return base
}

describe('normalizeTuiLocale', () => {
  it('maps supported codes and aliases to a locale', () => {
    expect(normalizeTuiLocale('ja')).toBe('ja')
    expect(normalizeTuiLocale('JA')).toBe('ja')
    expect(normalizeTuiLocale('ja-JP')).toBe('ja')
    expect(normalizeTuiLocale('japanese')).toBe('ja')
    expect(normalizeTuiLocale('jp')).toBe('ja')
    expect(normalizeTuiLocale('en')).toBe('en')
  })

  it('falls back to English for unknown or non-string values', () => {
    expect(normalizeTuiLocale('klingon')).toBe('en')
    expect(normalizeTuiLocale('')).toBe('en')
    expect(normalizeTuiLocale(undefined)).toBe('en')
    expect(normalizeTuiLocale(42)).toBe('en')
  })
})

describe('mergeStrings', () => {
  it('replaces string leaves and recurses, ignoring unknown keys', () => {
    const base = { a: 'x', nested: { b: 'y' } }
    const merged = mergeStrings(base, { a: 'X', nested: { b: 'Y', extra: 'no' }, unknown: 'no' })
    expect(merged).toEqual({ a: 'X', nested: { b: 'Y' } })
  })

  it('returns the same reference on a no-op merge', () => {
    const base = { a: 'x' }
    expect(mergeStrings(base, {})).toBe(base)
    expect(mergeStrings(base, { a: 'x' })).toBe(base)
    expect(mergeStrings(base, undefined)).toBe(base)
  })

  it('ignores a type mismatch (object over a string leaf)', () => {
    const base = { a: 'x' }
    expect(mergeStrings(base, { a: { nope: 1 } })).toBe(base)
  })
})

describe('resolveTuiCatalog', () => {
  afterEach(() => {
    delete process.env.HERMES_LOCALE_OVERRIDES
    clearTuiCatalogCache()
  })

  it('returns English for the en locale', () => {
    expect(resolveTuiCatalog('en').branding.gateway.disabled).toBe(en.branding.gateway.disabled)
  })

  it('layers bundled Japanese on top of English', () => {
    const cat = resolveTuiCatalog('ja')
    expect(cat.branding.gateway.disabled).toBe('無効')
    expect(cat.branding.noSystemPrompt).toBe('システムプロンプトは読み込まれていません。')
  })

  it('layers user overrides on top, surviving updates', () => {
    process.env.HERMES_LOCALE_OVERRIDES = writeTuiOverride('ja', {
      branding: { gateway: { disabled: '停止中（独自）' } }
    })
    clearTuiCatalogCache()
    const cat = resolveTuiCatalog('ja')
    expect(cat.branding.gateway.disabled).toBe('停止中（独自）')
    // a key not in the override keeps the bundled Japanese
    expect(cat.branding.gateway.failed).toBe('失敗')
  })
})

describe('readUserOverrides', () => {
  afterEach(() => {
    delete process.env.HERMES_LOCALE_OVERRIDES
  })

  it('returns null for an absent file and rejects bad language tokens', () => {
    process.env.HERMES_LOCALE_OVERRIDES = mkdtempSync(join(tmpdir(), 'hermes-tui-empty-'))
    expect(readUserOverrides('ja')).toBeNull()
    expect(readUserOverrides('../../etc/passwd')).toBeNull()
  })
})

describe('store', () => {
  afterEach(() => {
    setTuiLocale('en')
    clearTuiCatalogCache()
  })

  it('updates the derived catalog when the locale changes', () => {
    setTuiLocale('ja')
    expect($tuiLocale.get()).toBe('ja')
    expect($tuiText.get().branding.gateway.connecting).toBe('接続中')
    setTuiLocale('en')
    expect($tuiText.get().branding.gateway.connecting).toBe('connecting')
  })
})
