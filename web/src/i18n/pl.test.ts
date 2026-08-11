import { describe, expect, it } from 'vitest'
import { LOCALE_META } from './context'
import { en } from './en'
import { pl } from './pl'

interface CatalogLeaf {
  kind: string
  value: unknown
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

  it('preserves runtime leaf kinds, callback arity, and argument flow', () => {
    const englishLeaves = ownLeaves(en)
    const polishLeaves = ownLeaves(pl)

    expect([...polishLeaves.keys()].sort()).toEqual([...englishLeaves.keys()].sort())
    for (const [path, englishLeaf] of englishLeaves) {
      const polishLeaf = polishLeaves.get(path)
      expect(polishLeaf?.kind, path).toBe(englishLeaf.kind)
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
    expect(pl.models.toolCalls).toBe('wyw. narzędzi')
    expect(pl.config.resetDefaults).toBe('Przywróć domyślne')
    expect(pl.sessions.confirmDeleteMessage).toContain('trwałe usunięcie')
    expect(pl.sessions.confirmDeleteMessage).toContain('nie można cofnąć')
  })
})
