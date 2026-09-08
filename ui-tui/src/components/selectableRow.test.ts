import { describe, expect, it } from 'vitest'

import { decideRowClick, ROW_DOUBLE_CLICK_MS } from './selectableRow.js'

const NONE = { at: 0, index: -1 }

describe('decideRowClick', () => {
  it('selects an unhighlighted row on first click', () => {
    expect(decideRowClick(1000, NONE, 3, false)).toBe('select')
  })

  it('activates a row that is already highlighted', () => {
    expect(decideRowClick(1000, NONE, 3, true)).toBe('activate')
  })

  it('activates on a second click of the same row within the window', () => {
    const first = { at: 1000, index: 3 }
    expect(decideRowClick(1000 + ROW_DOUBLE_CLICK_MS - 1, first, 3, false)).toBe('activate')
  })

  it('selects again when the second click is too slow', () => {
    const first = { at: 1000, index: 3 }
    expect(decideRowClick(1000 + ROW_DOUBLE_CLICK_MS, first, 3, false)).toBe('select')
  })

  it('selects when a fast second click lands on a different row', () => {
    const first = { at: 1000, index: 3 }
    expect(decideRowClick(1100, first, 4, false)).toBe('select')
  })
})
