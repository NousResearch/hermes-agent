import { describe, expect, it, vi } from 'vitest'

import { scrollWithSelectionBy } from '../app/scroll.js'

function makeScroll(overrides: Partial<Record<string, unknown>> = {}) {
  const getScrollHeight = (overrides.getScrollHeight as (() => number) | undefined) ?? vi.fn(() => 100)

  return {
    getFreshScrollHeight: vi.fn(() => getScrollHeight()),
    getPendingDelta: vi.fn(() => 0),
    getScrollHeight,
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

  it('uses fresh scroll height when cached height would swallow a down-scroll at a fake bottom', () => {
    const s = makeScroll({
      getFreshScrollHeight: vi.fn(() => 34),
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

    expect(s.scrollBy).toHaveBeenCalledWith(4)
  })

  it('uses fresh height when pending down-scroll reaches the cached fake bottom', () => {
    const s = makeScroll({
      getFreshScrollHeight: vi.fn(() => 38),
      getPendingDelta: vi.fn(() => 2),
      getScrollHeight: vi.fn(() => 32),
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

    expect(s.scrollBy).toHaveBeenCalledWith(6)
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
