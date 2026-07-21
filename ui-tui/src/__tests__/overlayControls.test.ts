import { describe, expect, it } from 'vitest'

import { clampIndex, windowOffset } from '../components/overlayControls.js'

describe('clampIndex', () => {
  it('keeps an in-range index unchanged', () => {
    expect(clampIndex(0, 5)).toBe(0)
    expect(clampIndex(2, 5)).toBe(2)
    expect(clampIndex(4, 5)).toBe(4)
  })

  it('snaps a stale index to the last row when the list shrinks', () => {
    expect(clampIndex(5, 3)).toBe(2)
    expect(clampIndex(99, 1)).toBe(0)
  })

  it('returns 0 for an empty list or negative index', () => {
    expect(clampIndex(3, 0)).toBe(0)
    expect(clampIndex(-4, 5)).toBe(0)
  })
})

describe('windowOffset', () => {
  it('never produces a negative offset', () => {
    expect(windowOffset(3, 0, 12)).toBe(0)
    expect(windowOffset(0, 0, 12)).toBe(0)
  })

  it('keeps the selection roughly centered for long lists', () => {
    expect(windowOffset(100, 50, 12)).toBe(44)
  })
})
