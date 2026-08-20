import { describe, expect, it } from 'vitest'

import { CharCache } from './char-cache.js'

type TestChar = { value: string }

const characters = (value: string): TestChar[] => [{ value }]

describe('CharCache', () => {
  it('evicts the least-recently-used entries to stay within its byte budget', () => {
    // Each one-character key/value entry weighs 132 estimated bytes:
    // 64 entry overhead + 2 key bytes + 64 cluster overhead + 2 value bytes.
    const cache = new CharCache<TestChar[]>(264, 100)

    cache.set('a', characters('a'))
    cache.set('b', characters('b'))
    cache.set('c', characters('c'))

    expect(cache.estimatedBytes).toBeLessThanOrEqual(264)
    expect(cache.get('a')).toBeUndefined()
    expect(cache.get('b')).toEqual(characters('b'))
    expect(cache.get('c')).toEqual(characters('c'))
  })

  it('keeps a recently read entry when later writes apply pressure', () => {
    const cache = new CharCache<TestChar[]>(264, 100)

    cache.set('a', characters('a'))
    cache.set('b', characters('b'))
    expect(cache.get('a')).toEqual(characters('a'))
    cache.set('c', characters('c'))

    expect(cache.get('a')).toEqual(characters('a'))
    expect(cache.get('b')).toBeUndefined()
    expect(cache.get('c')).toEqual(characters('c'))
  })

  it('treats replacing an entry as recent use', () => {
    const cache = new CharCache<TestChar[]>(264, 100)

    cache.set('a', characters('a'))
    cache.set('b', characters('b'))
    cache.set('a', characters('a'))
    cache.set('c', characters('c'))

    expect(cache.get('a')).toEqual(characters('a'))
    expect(cache.get('b')).toBeUndefined()
    expect(cache.get('c')).toEqual(characters('c'))
  })

  it('uses estimated bytes rather than entry count as the primary bound', () => {
    const cache = new CharCache<TestChar[]>(400, 100)

    cache.set('a', characters('x'.repeat(100)))
    cache.set('b', characters('x'.repeat(100)))

    expect(cache.size).toBe(1)
    expect(cache.get('a')).toBeUndefined()
    expect(cache.get('b')).toEqual(characters('x'.repeat(100)))
    expect(cache.estimatedBytes).toBe(330)
  })

  it('retains the entry-count bound for unusually small values', () => {
    const cache = new CharCache<TestChar[]>(10000, 2)

    cache.set('a', characters('a'))
    cache.set('b', characters('b'))
    cache.set('c', characters('c'))

    expect(cache.size).toBe(2)
    expect(cache.get('a')).toBeUndefined()
    expect(cache.get('b')).toEqual(characters('b'))
    expect(cache.get('c')).toEqual(characters('c'))
  })

  it('replaces entries without leaking weight and clears all accounting', () => {
    const cache = new CharCache<TestChar[]>(10000, 100)

    cache.set('same', characters('a'))
    cache.set('same', characters('界'))

    expect(cache.size).toBe(1)
    expect(cache.estimatedBytes).toBe(138)
    expect(cache.get('same')).toEqual(characters('界'))

    cache.clear()

    expect(cache.size).toBe(0)
    expect(cache.estimatedBytes).toBe(0)
    expect(cache.get('same')).toBeUndefined()
  })

  it('accounts for UTF-16, grapheme clusters, and ANSI source text', () => {
    const cache = new CharCache<TestChar[]>(10000, 100)
    const styledFamily = '\u001B[31m👨‍👩‍👧‍👦\u001B[39m'

    cache.set(styledFamily, characters('👨‍👩‍👧‍👦'))

    expect(cache.get(styledFamily)).toEqual(characters('👨‍👩‍👧‍👦'))
    expect(cache.estimatedBytes).toBe(64 + styledFamily.length * 2 + 64 + '👨‍👩‍👧‍👦'.length * 2)
  })

  it('accounts for every grapheme cluster in a cached line', () => {
    const cache = new CharCache<TestChar[]>(10000, 100)
    const clusters = [{ value: 'a' }, { value: '界' }, { value: '👨‍👩‍👧‍👦' }]

    cache.set('a界👨‍👩‍👧‍👦', clusters)

    expect(cache.get('a界👨‍👩‍👧‍👦')).toBe(clusters)
    expect(cache.estimatedBytes).toBe(64 + 'a界👨‍👩‍👧‍👦'.length * 2 + 3 * 64 + (1 + 1 + 11) * 2)
  })

  it('does not admit an entry larger than the entire budget', () => {
    const cache = new CharCache<TestChar[]>(264, 100)

    cache.set('hot', characters('x'))
    cache.set('oversized', characters('x'.repeat(100)))

    expect(cache.get('hot')).toEqual(characters('x'))
    expect(cache.get('oversized')).toBeUndefined()
    expect(cache.estimatedBytes).toBe(136)
  })

  it('preserves an existing entry when an oversized replacement is rejected', () => {
    const cache = new CharCache<TestChar[]>(264, 100)
    const original = characters('x')

    cache.set('same', original)
    cache.set('same', characters('x'.repeat(100)))

    expect(cache.get('same')).toBe(original)
    expect(cache.estimatedBytes).toBe(138)
  })

  it('does not admit entries when either budget is zero', () => {
    const noBytes = new CharCache<TestChar[]>(0, 100)
    const noEntries = new CharCache<TestChar[]>(10000, 0)

    noBytes.set('a', characters('a'))
    noEntries.set('a', characters('a'))

    expect(noBytes.size).toBe(0)
    expect(noBytes.estimatedBytes).toBe(0)
    expect(noEntries.size).toBe(0)
    expect(noEntries.estimatedBytes).toBe(0)
  })
})
