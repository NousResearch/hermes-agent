import { describe, expect, it, vi } from 'vitest'

import { en } from './en'
import { sv } from './sv'

// English fallback must not mask a missing Swedish translation in this check.
vi.mock('./define-locale', () => ({
  defineLocale: (overrides: unknown) => overrides
}))

function leaves(node: unknown, prefix = ''): Array<[string, unknown]> {
  if (typeof node !== 'object' || node === null) return [[prefix, node]]
  return Object.entries(node).flatMap(([key, value]) => leaves(value, prefix ? `${prefix}.${key}` : key))
}

function placeholders(value: string): string[] {
  // Callers supply an English plural suffix. Swedish uses invariant nouns or
  // count labels instead, so {s} is intentionally absent from these messages.
  return [...value.matchAll(/\{(\w+)\}/g)]
    .map(match => match[1])
    .filter(name => name !== 's')
    .sort()
}

describe('Swedish dashboard translations', () => {
  it('covers the English message tree without fallback', () => {
    const source = Object.fromEntries(leaves(en))
    const translated = Object.fromEntries(leaves(sv))

    expect(Object.keys(translated).sort()).toEqual(Object.keys(source).sort())
    for (const [path, value] of Object.entries(source)) {
      expect(typeof translated[path], path).toBe(typeof value)
      if (typeof value === 'string') {
        expect(placeholders(translated[path] as string), path).toEqual(placeholders(value))
      }
    }
  })

  it.each([0, 1, 2, 21])('renders counts without English plural suffixes: %i', count => {
    const labels = [sv.skills.skillCount, sv.skills.resultCount, sv.env.keysCount, sv.env.customConfigured]

    for (const label of labels) {
      expect(label).not.toContain('{s}')
      const rendered = label.replace('{count}', String(count)).replace('{s}', count === 1 ? '' : 's')
      expect(rendered).toContain(String(count))
      expect(rendered).not.toMatch(/\{\w+\}/)
    }
    expect(sv.config.fields).toBe('fält')
  })
})
