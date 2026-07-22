import { describe, expect, it, vi } from 'vitest'

import { ensureVirtualItemHeight, MAX_MOUNTED } from '../hooks/useVirtualHistory.js'

describe('ensureVirtualItemHeight', () => {
  it('reuses cached heights without invoking the estimator', () => {
    const heights = new Map([['a', 7]])
    const estimateHeight = vi.fn(() => 99)

    expect(ensureVirtualItemHeight(heights, 'a', 0, 4, estimateHeight)).toBe(7)
    expect(estimateHeight).not.toHaveBeenCalled()
    expect(heights.get('a')).toBe(7)
  })

  it('lazily seeds missing heights from the estimator', () => {
    const heights = new Map<string, number>()
    const estimateHeight = vi.fn((index: number) => 10 + index)

    expect(ensureVirtualItemHeight(heights, 'b', 2, 4, estimateHeight)).toBe(12)
    expect(estimateHeight).toHaveBeenCalledTimes(1)
    expect(estimateHeight).toHaveBeenCalledWith(2, 'b')
    expect(heights.get('b')).toBe(12)
  })

  it('falls back to the default estimate when no estimator is provided', () => {
    const heights = new Map<string, number>()

    expect(ensureVirtualItemHeight(heights, 'c', 0, 4)).toBe(4)
    expect(heights.get('c')).toBe(4)
  })

  it('normalizes non-positive estimates to a minimum of one row', () => {
    const heights = new Map<string, number>()
    const estimateHeight = vi.fn(() => 0)

    expect(ensureVirtualItemHeight(heights, 'd', 0, 0, estimateHeight)).toBe(1)
    expect(heights.get('d')).toBe(1)
  })
})

// Issue #55594: long assistant responses scroll out of the mounted range
// and the clamp holds the viewport at the edge of mounted content while
// the user catches up.  Raising the default cap from 120 → 300 keeps
// longer responses reachable.  The constant is exported so callers can
// override per-instance via the `maxMounted` option.
describe('MAX_MOUNTED default', () => {
  it('is at least 300 to keep long responses reachable', () => {
    expect(MAX_MOUNTED).toBeGreaterThanOrEqual(300)
  })
})
