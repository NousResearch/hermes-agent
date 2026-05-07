import { describe, expect, it } from 'vitest'

import { getScrollbarSnapshot, getViewportSnapshot, scrollbarSnapshotKey, viewportSnapshotKey } from '../lib/viewportStore.js'

describe('viewportStore', () => {
  it('normalizes absent scroll handles', () => {
    expect(getViewportSnapshot(null)).toEqual({
      atBottom: true,
      bottom: 0,
      pending: 0,
      scrollHeight: 0,
      top: 0,
      viewportHeight: 0
    })
  })

  it('includes pending scroll delta in snapshot math and keying', () => {
    const handle = {
      getPendingDelta: () => 3,
      getScrollHeight: () => 40,
      getScrollTop: () => 10,
      getViewportHeight: () => 5,
      isSticky: () => false
    }

    const snap = getViewportSnapshot(handle as any)

    expect(snap).toMatchObject({
      atBottom: false,
      bottom: 18,
      pending: 3,
      scrollHeight: 40,
      top: 13,
      viewportHeight: 5
    })
    expect(viewportSnapshotKey(snap)).toBe('0:16:5:40:3')
  })

  it('uses fresh scroll height to clear stale non-bottom state', () => {
    const handle = {
      getFreshScrollHeight: () => 20,
      getPendingDelta: () => 0,
      getScrollHeight: () => 40,
      getScrollTop: () => 15,
      getViewportHeight: () => 5,
      isSticky: () => false
    }

    const snap = getViewportSnapshot(handle as any)

    expect(snap.atBottom).toBe(true)
    expect(snap.scrollHeight).toBe(20)
  })

  it('keeps scrollbar position tied to committed scrollTop, not pending target', () => {
    const handle = {
      getPendingDelta: () => 24,
      getScrollHeight: () => 100,
      getScrollTop: () => 10,
      getViewportHeight: () => 20,
      isSticky: () => false
    }

    const viewport = getViewportSnapshot(handle as any)
    const scrollbar = getScrollbarSnapshot(handle as any)

    expect(viewport.top).toBe(34)
    expect(scrollbar).toEqual({
      scrollHeight: 100,
      top: 10,
      viewportHeight: 20
    })
    expect(scrollbarSnapshotKey(scrollbar)).toBe('10:20:100')
  })

  it('clamps scrollbar position to committed scroll bounds', () => {
    const handle = {
      getScrollHeight: () => 30,
      getScrollTop: () => 50,
      getViewportHeight: () => 20
    }

    expect(getScrollbarSnapshot(handle as any).top).toBe(10)
  })
})
