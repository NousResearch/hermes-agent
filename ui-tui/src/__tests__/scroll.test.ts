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
    scrollTo: vi.fn(),
    ...overrides
  }
}

describe('scrollWithSelectionBy', () => {
  it('commits the clamped target directly instead of queueing a scroll delta', () => {
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

    expect(s.scrollTo).toHaveBeenCalledWith(10)
    expect(s.scrollBy).not.toHaveBeenCalled()
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

    expect(s.scrollTo).toHaveBeenCalledWith(14)
    expect(s.scrollBy).not.toHaveBeenCalled()
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

    expect(s.scrollTo).toHaveBeenCalledWith(18)
    expect(s.scrollBy).not.toHaveBeenCalled()
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

    expect(s.scrollTo).not.toHaveBeenCalled()
    expect(s.scrollBy).not.toHaveBeenCalled()
  })

  it('preserves selection capture and shifting on the direct path', () => {
    const s = makeScroll({
      getScrollTop: vi.fn(() => 10),
      getViewportHeight: vi.fn(() => 20),
      getViewportTop: vi.fn(() => 5)
    })

    const selection = {
      captureScrolledRows: vi.fn(),
      getState: vi.fn(() => ({ anchor: { row: 10 }, focus: { row: 12 } })),
      shiftAnchor: vi.fn(),
      shiftSelection: vi.fn()
    }

    scrollWithSelectionBy(3, { scrollRef: { current: s as never }, selection })

    expect(selection.captureScrolledRows).toHaveBeenCalledWith(5, 7, 'above')
    expect(selection.shiftSelection).toHaveBeenCalledWith(-3, 5, 24)
    expect(s.scrollTo).toHaveBeenCalledWith(13)
    expect(s.scrollBy).not.toHaveBeenCalled()
  })
})
