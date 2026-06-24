import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  applyScrollState,
  persistScrollState,
  restoreScrollState,
  scrollStatePath,
  SCROLL_STATE_TTL_MS,
  type ScrollState
} from '../lib/scrollPersistence.js'

describe('scrollPersistence (Stage 1: recycle keeps scroll position)', () => {
  let dir: string

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'scrollpers-'))
  })
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true })
  })

  it('round-trips a mid-history scroll position', () => {
    expect(persistScrollState('abcd1234', { top: 420, atBottom: false }, dir)).toBe(true)
    const r = restoreScrollState('abcd1234', dir)
    expect(r).not.toBeNull()
    expect(r!.top).toBe(420)
    expect(r!.atBottom).toBe(false)
  })

  it('round-trips an at-bottom (sticky) state', () => {
    persistScrollState('sticky01', { top: 0, atBottom: true }, dir)
    const r = restoreScrollState('sticky01', dir)
    expect(r!.atBottom).toBe(true)
  })

  it('returns null for an absent sid (caller falls back to scrollToBottom)', () => {
    expect(restoreScrollState('does-not-exist', dir)).toBeNull()
  })

  it('returns null for a stale entry past the TTL', () => {
    persistScrollState('staleabc', { top: 100, atBottom: false }, dir)
    const future = Date.now() + SCROLL_STATE_TTL_MS + 1
    expect(restoreScrollState('staleabc', dir, future)).toBeNull()
  })

  it('returns null for a corrupt file (never worse than today)', () => {
    writeFileSync(scrollStatePath('corrupt1', dir), '{not json', 'utf8')
    expect(restoreScrollState('corrupt1', dir)).toBeNull()
  })

  it('returns null for a malformed (wrong-shape) entry', () => {
    writeFileSync(scrollStatePath('badshape', dir), JSON.stringify({ top: 'x' }), 'utf8')
    expect(restoreScrollState('badshape', dir)).toBeNull()
  })

  it('empty sid is a no-op on both persist and restore', () => {
    expect(persistScrollState('', { top: 5, atBottom: false }, dir)).toBe(false)
    expect(restoreScrollState('', dir)).toBeNull()
  })

  describe('applyScrollState', () => {
    function fakeTarget() {
      const calls: string[] = []
      return {
        calls,
        scrollTo: (y: number) => calls.push(`scrollTo(${y})`),
        scrollToBottom: () => calls.push('scrollToBottom')
      }
    }

    it('applies mid-history position via scrollTo', () => {
      const t = fakeTarget()
      const state: ScrollState = { top: 333, atBottom: false, savedAt: Date.now() }
      expect(applyScrollState(t, state)).toBe(true)
      expect(t.calls).toEqual(['scrollTo(333)'])
    })

    it('falls back to scrollToBottom when atBottom', () => {
      const t = fakeTarget()
      const state: ScrollState = { top: 999, atBottom: true, savedAt: Date.now() }
      expect(applyScrollState(t, state)).toBe(false)
      expect(t.calls).toEqual(['scrollToBottom'])
    })

    it('falls back to scrollToBottom when state is null', () => {
      const t = fakeTarget()
      expect(applyScrollState(t, null)).toBe(false)
      expect(t.calls).toEqual(['scrollToBottom'])
    })

    it('clamps a negative top to 0', () => {
      const t = fakeTarget()
      const state: ScrollState = { top: -50, atBottom: false, savedAt: Date.now() }
      applyScrollState(t, state)
      expect(t.calls).toEqual(['scrollTo(0)'])
    })
  })
})
