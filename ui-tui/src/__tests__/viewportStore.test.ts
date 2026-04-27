import { describe, expect, it } from 'vitest'

import { getViewportSnapshot, viewportSnapshotKey } from '../lib/viewportStore.js'

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
})
