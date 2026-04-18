import { describe, expect, it } from 'vitest'

import { CONFIRM_WINDOW_MS, createDestructiveGate } from '../domain/destructive.js'

describe('createDestructiveGate', () => {
  it('uses a generous default window so real humans can retype (#4069)', () => {
    expect(CONFIRM_WINDOW_MS).toBeGreaterThanOrEqual(15_000)
  })

  it('first request is not confirmed — it arms the gate', () => {
    const g = createDestructiveGate()
    expect(g.request('clear', 0)).toBe(false)
  })

  it('second request within window with same key is confirmed', () => {
    const g = createDestructiveGate()
    g.request('clear', 0)
    expect(g.request('clear', CONFIRM_WINDOW_MS - 1)).toBe(true)
  })

  it('second request outside the window re-arms and is not confirmed', () => {
    const g = createDestructiveGate()
    g.request('clear', 0)
    expect(g.request('clear', CONFIRM_WINDOW_MS + 1)).toBe(false)
  })

  it('armed() reports the pending key while fresh, null otherwise', () => {
    const g = createDestructiveGate(100)
    expect(g.armed()).toBe(null)
    g.request('clear')
    expect(g.armed()).toBe('clear')
    g.reset()
    expect(g.armed()).toBe(null)
  })

  it('different key re-arms the gate, does not confirm', () => {
    const g = createDestructiveGate()
    g.request('clear', 0)
    expect(g.request('undo', 500)).toBe(false)
    expect(g.request('undo', 900)).toBe(true)
  })

  it('confirmation consumes the pending state so a third press re-arms', () => {
    const g = createDestructiveGate()
    g.request('clear', 0)
    g.request('clear', 500)
    expect(g.request('clear', 600)).toBe(false)
  })

  it('reset clears pending state', () => {
    const g = createDestructiveGate()
    g.request('clear', 0)
    g.reset()
    expect(g.request('clear', 500)).toBe(false)
  })

  it('respects a custom window', () => {
    const g = createDestructiveGate(100)
    g.request('clear', 0)
    expect(g.request('clear', 50)).toBe(true)

    g.request('clear', 0)
    expect(g.request('clear', 150)).toBe(false)
  })
})
