import { describe, expect, it } from 'vitest'

import { applyYank, isKillChord, KillRing, removedText, type YankState } from '../domain/killRing.js'

describe('KillRing', () => {
  it('keeps the newest cut first and ignores empty or repeat pushes', () => {
    const ring = new KillRing()

    ring.push('one')
    ring.push('')
    ring.push('one')
    ring.push('two')

    expect([ring.length, ring.at(0), ring.at(1)]).toEqual([2, 'two', 'one'])
  })

  it('keeps ten cuts, like the classic CLI, and drops the oldest', () => {
    const ring = new KillRing()

    for (let i = 0; i < 12; i++) {
      ring.push(`cut ${i}`)
    }

    expect([ring.length, ring.at(0), ring.at(9), ring.at(10)]).toEqual([10, 'cut 11', 'cut 2', ''])
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
    // The removed `two\n` starts and ends with characters its surroundings share.
    expect(removedText('one\ntwo\nthree', 'one\nthree', 4, 4)).toBe('two\n')
    expect(removedText('alpha BETA gamma', 'alpha  gamma', 6, 6)).toBe('BETA')
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
    expect(applyYank(new KillRing(), '', 0, null)).toBeNull()
  })

  it('inserts the newest cut at the caret and remembers where it landed', () => {
    expect(applyYank(seeded(), 'hello ', 6, null)).toEqual({
      cursor: 11,
      next: { end: 11, index: 0, start: 6 },
      value: 'hello newer'
    })
  })

  it('swaps the previous yank for the next older cut, wrapping at the end', () => {
    const ring = seeded()
    const first = applyYank(ring, '', 0, null)!
    const second = applyYank(ring, first.value, first.cursor, first.next)!
    const third = applyYank(ring, second.value, second.cursor, second.next)!

    expect([second.value, second.cursor, third.value]).toEqual(['older', 5, 'newer'])
  })

  it('starts a fresh insert when the caret moved or the yanked text was edited', () => {
    const last: YankState = { end: 5, index: 0, start: 0 }

    expect(applyYank(seeded(), 'newer and', 9, last)!.value).toBe('newer andnewer')
    expect(applyYank(seeded(), 'NEWER', 5, last)!.value).toBe('NEWERnewer')
  })
})

describe('isKillChord — which keystrokes feed the ring', () => {
  const none = {
    deleteToStart: false,
    deleteWord: false,
    deleting: false,
    killToEnd: false,
    lineKill: false,
    wordMod: false
  }

  it('counts the readline kill chords and M-d', () => {
    expect(isKillChord('w', { ...none, deleteWord: true })).toBe(true)
    expect(isKillChord('u', { ...none, deleteToStart: true })).toBe(true)
    expect(isKillChord('k', { ...none, killToEnd: true })).toBe(true)
    expect(isKillChord('d', { ...none, wordMod: true })).toBe(true)
  })

  it('counts word and line deletes on Backspace / Delete, but not a plain one', () => {
    expect(isKillChord('', { ...none, deleting: true, wordMod: true })).toBe(true)
    expect(isKillChord('', { ...none, deleting: true, lineKill: true })).toBe(true)
    expect(isKillChord('', { ...none, deleting: true })).toBe(false)
    expect(isKillChord('x', none)).toBe(false)
  })
})
