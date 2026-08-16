import { describe, expect, it } from 'vitest'
import { af } from './af'
import { ar } from './ar'
import { LOCALE_META } from './context'
import { de } from './de'
import { en } from './en'
import { es } from './es'
import { fr } from './fr'
import { ga } from './ga'
import { hu } from './hu'
import { it as itCatalog } from './it'
import { ja } from './ja'
import { ko } from './ko'
import { pl } from './pl'
import { pt } from './pt'
import { ru } from './ru'
import { tr } from './tr'
import type { Translations } from './types'
import { uk } from './uk'
import { zhHant } from './zh-hant'
import { zh } from './zh'

const catalogs: ReadonlyArray<readonly [string, Translations]> = [
  ['en', en],
  ['pl', pl],
  ['zh', zh],
  ['zh-hant', zhHant],
  ['ja', ja],
  ['de', de],
  ['es', es],
  ['fr', fr],
  ['tr', tr],
  ['uk', uk],
  ['af', af],
  ['ko', ko],
  ['it', itCatalog],
  ['ga', ga],
  ['pt', pt],
  ['ru', ru],
  ['hu', hu],
  ['ar', ar]
]

interface CatalogLeaf {
  kind: string
  value: unknown
}

function placeholders(value: string): string[] {
  return [...value.matchAll(/\{([A-Za-z_][A-Za-z0-9_]*)\}/g)].map(match => match[1]).sort()
}

function ownLeaves(node: unknown, prefix = ''): Map<string, CatalogLeaf> {
  if (
    typeof node === 'function' ||
    typeof node === 'string' ||
    typeof node === 'number' ||
    typeof node === 'boolean' ||
    node === null
  ) {
    return new Map([[prefix, { kind: node === null ? 'null' : typeof node, value: node }]])
  }

  if (Array.isArray(node)) {
    return new Map(node.flatMap((value, index) => [...ownLeaves(value, `${prefix}[${index}]`)]))
  }

  if (!node || typeof node !== 'object') {
    return new Map([[prefix, { kind: typeof node, value: node }]])
  }

  return new Map(
    Object.keys(node).flatMap(key => [
      ...ownLeaves((node as Record<string, unknown>)[key], prefix ? `${prefix}.${key}` : key)
    ])
  )
}

const argumentPairs: ReadonlyArray<readonly [unknown, unknown]> = [
  ['alpha', 'beta'],
  [1, 2],
  [false, true],
  [null, 'value'],
  [undefined, 'value'],
  ['skills', 'tools'],
  ['linux', 'windows']
]

function observesArgument(fn: (...args: never[]) => unknown, arity: number, index: number): boolean {
  for (const fill of ['value', 1, false, null, undefined]) {
    for (const [left, right] of argumentPairs) {
      const leftArgs: unknown[] = Array.from({ length: arity }, () => fill)
      const rightArgs = [...leftArgs]
      leftArgs[index] = left
      rightArgs[index] = right

      try {
        if (fn(...(leftArgs as never[])) !== fn(...(rightArgs as never[]))) return true
      } catch {
        // This vector does not match the callback's runtime input shape.
      }
    }
  }
  return false
}

describe('Polish dashboard localization', () => {
  it('registers Polish in the language picker', () => {
    expect(LOCALE_META.pl).toEqual({ name: 'Polski' })
  })

  it('has exactly the same own translation paths as English', () => {
    const englishPaths = [...ownLeaves(en).keys()]
    const polishPaths = [...ownLeaves(pl).keys()]

    expect(polishPaths).toEqual(expect.arrayContaining(englishPaths))
    expect(englishPaths).toEqual(expect.arrayContaining(polishPaths))
    expect(polishPaths).toHaveLength(englishPaths.length)
  })

  it('preserves runtime leaf kinds, callback arity, placeholders, and argument flow', () => {
    const englishLeaves = ownLeaves(en)
    const polishLeaves = ownLeaves(pl)

    expect([...polishLeaves.keys()].sort()).toEqual([...englishLeaves.keys()].sort())
    for (const [path, englishLeaf] of englishLeaves) {
      const polishLeaf = polishLeaves.get(path)
      expect(polishLeaf?.kind, path).toBe(englishLeaf.kind)
      if (englishLeaf.kind === 'string' && polishLeaf?.kind === 'string') {
        expect(placeholders(polishLeaf.value as string), `${path} placeholders`).toEqual(
          placeholders(englishLeaf.value as string)
        )
      }
      if (englishLeaf.kind !== 'function' || polishLeaf?.kind !== 'function') continue

      const englishFn = englishLeaf.value as (...args: never[]) => unknown
      const polishFn = polishLeaf.value as (...args: never[]) => unknown
      expect(polishFn.length, `${path} arity`).toBe(englishFn.length)
      for (let index = 0; index < englishFn.length; index += 1) {
        expect(observesArgument(polishFn, polishFn.length, index), `${path} argument ${index}`).toBe(
          observesArgument(englishFn, englishFn.length, index)
        )
      }
    }
  })

  it('translates representative visible interface and confirmation copy', () => {
    expect(pl.common.save).toBe('Zapisz')
    expect(pl.common.gateway).toBe('Brama')
    expect(pl.common.tools).toBe('narz.')
    expect(pl.app.nav.sessions).toBe('Sesje')
    expect(pl.app.diskFreeSpace?.(120)).toBe('(pozostało 120 MB)')
    expect(pl.models.toolCalls).toBe('wyw. narzędzi')
    expect(pl.config.resetDefaults).toBe('Przywróć domyślne')
    expect(pl.sessions.confirmDeleteMessage).toContain('trwałe usunięcie')
    expect(pl.sessions.confirmDeleteMessage).toContain('nie można cofnąć')
  })

  it('renders count-dependent labels as complete locale-owned messages', () => {
    expect(pl.skills.skillCount(1)).toBe('Liczba umiejętności: 1')
    expect(pl.skills.resultCount(22)).toBe('Liczba wyników: 22')
    expect(pl.config.fields(3)).toBe('3 poz.')
    expect(pl.env.keysCount(5)).toBe('Liczba kluczy: 5')
    expect(pl.env.customConfigured(2)).toBe('Liczba skonfigurowanych własnych kluczy: 2')
    expect(pl.env.configuredCount(3)).toBe('Skonfigurowano: 3')
    expect(pl.env.configuredSummary(1, 2)).toBe('Skonfigurowano: 1/2')

    expect(en.skills.skillCount(1)).toBe('1 skill')
    expect(en.skills.skillCount(2)).toBe('2 skills')
    expect(en.config.fields(1)).toBe('1 field')
    expect(en.config.fields(3)).toBe('3 fields')
    expect(en.env.keysCount(2)).toBe('2 keys')
    expect(en.env.customConfigured(1)).toBe('1 custom key set')
    expect(en.env.configuredCount(3)).toBe('3 configured')
    expect(en.env.configuredSummary(1, 2)).toBe('1 of 2 configured')
  })

  it('keeps every dashboard count callback executable and locale-owned', () => {
    for (const [locale, catalog] of catalogs) {
      const oneArgumentCallbacks = [
        catalog.skills.skillCount,
        catalog.skills.resultCount,
        catalog.config.fields,
        catalog.env.keysCount,
        catalog.env.customConfigured,
        catalog.env.configuredCount
      ]

      for (const callback of oneArgumentCallbacks) {
        expect(callback.length, `${locale} callback arity`).toBe(1)
        expect(callback(1), `${locale} callback count=1`).toContain('1')
        expect(callback(2), `${locale} callback count=2`).toContain('2')
        expect(callback(2), `${locale} unresolved grammar token`).not.toMatch(/\{(?:count|s)\}|\(s\)/)
      }

      expect(catalog.env.configuredSummary.length, `${locale} summary arity`).toBe(2)
      expect(catalog.env.configuredSummary(1, 2), `${locale} summary configured`).toContain('1')
      expect(catalog.env.configuredSummary(1, 2), `${locale} summary total`).toContain('2')
    }

    expect(af.skills.skillCount(2)).toBe('2 vaardighede')
    expect(ar.skills.skillCount(2)).toBe('المهارات: 2')
    expect(itCatalog.config.fields(2)).toBe('2 campi')
    expect(ja.env.keysCount(2)).toBe('2 キー')
    expect(ru.skills.resultCount(2)).toBe('Результаты: 2')
    expect(tr.config.fields(2)).toBe('2 alan')
  })
})
