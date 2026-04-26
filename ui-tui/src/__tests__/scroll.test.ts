import { describe, expect, it, vi } from 'vitest'

import { scrollWithSelectionBy } from '../app/scroll.js'

function makeScroll(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    getPendingDelta: vi.fn(() => 0),
    getScrollHeight: vi.fn(() => 100),
    getScrollTop: vi.fn(() => 10),
    getViewportHeight: vi.fn(() => 20),
    getViewportTop: vi.fn(() => 0),
    scrollBy: vi.fn(),
    ...overrides
  }
}

describe('scrollWithSelectionBy', () => {
  it('clamps to the actual remaining scroll distance before calling scrollBy', () => {
    const s = makeScroll({
      getScrollHeight: vi.fn(() => 30),
      getScrollTop: vi.fn(() => 9),
      getViewportHeight: vi.fn(() => 20)
    })
    const selection = {
      captureScrolledRows: vi.fn(),
      getState: vi.fn(() => null),
      shiftAnchor: vi.fn(),
      shiftSelection: vi.fn()
    }

    scrollWithSelectionBy(10, { scrollRef: { current: s as never }, selection })

    expect(s.scrollBy).toHaveBeenCalledWith(1)
  })

  it('does nothing at the edge instead of queueing dead pending deltas', () => {
    const s = makeScroll({
      getScrollHeight: vi.fn(() => 30),
      getScrollTop: vi.fn(() => 10),
      getViewportHeight: vi.fn(() => 20)
    })
    const selection = {
      captureScrolledRows: vi.fn(),
      getState: vi.fn(() => null),
      shiftAnchor: vi.fn(),
      shiftSelection: vi.fn()
    }

    scrollWithSelectionBy(10, { scrollRef: { current: s as never }, selection })

    expect(s.scrollBy).not.toHaveBeenCalled()
  })
})
