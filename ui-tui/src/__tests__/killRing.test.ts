import { describe, expect, it } from 'vitest'

import {
  applyYank,
  isKillStroke,
  isLineKillModifier,
  KILL_RING_SIZE,
  KillRing,
  removedText,
  type YankState
} from '../domain/killRing.js'

const key = (patch: Partial<Parameters<typeof isKillStroke>[0]> = {}) => ({
  backspace: false,
  ctrl: false,
  delete: false,
  meta: false,
  ...patch
})

const noFallback = () => false

describe('KillRing', () => {
  it('keeps the newest cut first and ignores empty or repeat pushes', () => {
    const ring = new KillRing()

    expect(ring.push('one')).toBe(true)
    expect(ring.push('')).toBe(false)
    expect(ring.push('one')).toBe(false)
    expect(ring.latest()).toBe('one')

    ring.push('two')

    expect(ring.entries).toEqual(['two', 'one'])
  })

  it('caps the ring and drops the oldest piece', () => {
    const ring = new KillRing(3)

    for (const text of ['a', 'b', 'c', 'd']) {
      ring.push(text)
    }

    expect(ring.entries).toEqual(['d', 'c', 'b'])
    expect(ring.length).toBe(3)
    expect(ring.at(3)).toBe('')
  })

  it('ships a ten-entry ring, like the classic CLI', () => {
    expect(KILL_RING_SIZE).toBe(10)
    expect(new KillRing().entries).toEqual([])
  })
})

describe('removedText — what a kill chord actually took', () => {
  it('reads a backward kill from the two cursors', () => {
    // Ctrl+W at the end of "hello world" → caret lands on 6.
    expect(removedText('hello world', 'hello ', 11, 6)).toBe('world')
    expect(removedText('hello', 'hello', 5, 5)).toBe('')
    expect(removedText('', 'typed', 0, 5)).toBe('')
  })

  it('reads a forward kill, including the shared-character case a diff would get wrong', () => {
    // Ctrl+K on the middle line: the removed `two\n` starts and ends with characters the
    // surrounding text also has, which is exactly where a prefix/suffix diff lies.
    expect(removedText('one\ntwo\nthree', 'one\nthree', 4, 4)).toBe('two\n')
    expect(removedText('alpha BETA gamma', 'alpha  gamma', 6, 6)).toBe('BETA')
  })

  it('reads a replaced selection in either direction', () => {
    expect(removedText('keep DROP keep', 'keep keep', 5, 5)).toBe('DROP ')
    expect(removedText('keep DROP keep', 'keep keep', 10, 5)).toBe('DROP ')
  })
})

describe('applyYank', () => {
  const seeded = () => {
    const ring = new KillRing()

    ring.push('older')
    ring.push('newer')

    return ring
  }

  it('does nothing with an empty ring', () => {
    expect(applyYank({ cursor: 0, last: null, ring: new KillRing(), value: '' })).toBeNull()
  })

  it('inserts the newest cut at the caret and remembers the span', () => {
    const result = applyYank({ cursor: 6, last: null, ring: seeded(), value: 'hello ' })

    expect(result).toEqual({
      cursor: 11,
      next: { index: 0, span: { end: 11, start: 6 } },
      value: 'hello newer'
    })
  })

  it('replaces the previous yank with the older cut when nothing else moved', () => {
    const first = applyYank({ cursor: 0, last: null, ring: seeded(), value: '' })!
    const second = applyYank({ cursor: first.cursor, last: first.next, ring: seeded(), value: first.value })!

    expect(second.value).toBe('older')
    expect(second.cursor).toBe(5)
    expect(second.next.index).toBe(1)
  })

  it('starts a fresh insert when the caret moved away from the last yank', () => {
    const last: YankState = { index: 0, span: { end: 5, start: 0 } }
    const result = applyYank({ cursor: 9, last, ring: seeded(), value: 'newer and' })!

    expect(result.value).toBe('newer andnewer')
    expect(result.next).toEqual({ index: 0, span: { end: 14, start: 9 } })
  })

  it('starts a fresh insert when the yanked text was edited under the span', () => {
    const last: YankState = { index: 0, span: { end: 5, start: 0 } }
    const result = applyYank({ cursor: 5, last, ring: seeded(), value: 'NEWER' })!

    expect(result.value).toBe('NEWERnewer')
  })
})

describe('isKillStroke — which chords feed the ring', () => {
  const opts = { alt: false, delFwd: false, isActionFallback: noFallback, mod: true }

  it('counts readline kill chords', () => {
    expect(isKillStroke(key({ ctrl: true }), 'w', opts)).toBe(true)
    expect(isKillStroke(key({ ctrl: true }), 'u', opts)).toBe(true)
    expect(isKillStroke(key({ ctrl: true }), 'k', opts)).toBe(true)
    expect(isKillStroke(key({ meta: true }), 'd', { ...opts, alt: true })).toBe(true)
  })

  it('counts the word and line deletes that share Backspace and Delete', () => {
    expect(isKillStroke(key({ super: true, backspace: true }), '', opts)).toBe(true)
    expect(isKillStroke(key({ ctrl: true, backspace: true }), '', opts)).toBe(true)
    expect(isKillStroke(key({ super: true, delete: true }), '', { ...opts, delFwd: true })).toBe(true)
  })

  it('leaves plain editing out of the ring', () => {
    expect(isKillStroke(key({ backspace: true }), '', { ...opts, mod: false })).toBe(false)
    expect(isKillStroke(key(), 'x', opts)).toBe(false)
    expect(isKillStroke(key({ ctrl: true }), 'a', opts)).toBe(false)
    expect(isKillStroke(key({ ctrl: true }), 'z', opts)).toBe(false)
  })

  it('honours the Cmd-rewritten-to-Ctrl fallback', () => {
    const fallback = (_k: unknown, inp: string, letter: string) => letter === 'w' && inp === 'w'

    expect(isKillStroke(key(), 'w', { ...opts, isActionFallback: fallback, mod: false })).toBe(true)
  })

  it('keeps the line-kill modifier to the super bit alone', () => {
    expect(isLineKillModifier({ ctrl: true, meta: false, super: true })).toBe(true)
    expect(isLineKillModifier({ ctrl: true, meta: true })).toBe(false)
  })
})
