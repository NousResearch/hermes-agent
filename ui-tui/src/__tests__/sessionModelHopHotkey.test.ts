import { describe, expect, it } from 'vitest'

import { sessionModelHopHotkey } from '../app/useInputHandlers.js'

const key = (over: Record<string, boolean> = {}) =>
  ({ ctrl: false, meta: false, shift: false, super: false, ...over })

describe('sessionModelHopHotkey', () => {
  it('matches Alt+P (omp hop parity)', () => {
    expect(sessionModelHopHotkey(key({ meta: true }), 'p')).toBe(true)
    expect(sessionModelHopHotkey(key({ meta: true }), 'P')).toBe(true)
  })

  it('ignores Ctrl+P, Shift+P, and bare p', () => {
    expect(sessionModelHopHotkey(key({ ctrl: true }), 'p')).toBe(false)
    expect(sessionModelHopHotkey(key({ meta: true, shift: true }), 'p')).toBe(false)
    expect(sessionModelHopHotkey(key({ meta: true, ctrl: true }), 'p')).toBe(false)
    expect(sessionModelHopHotkey(key(), 'p')).toBe(false)
  })
})
