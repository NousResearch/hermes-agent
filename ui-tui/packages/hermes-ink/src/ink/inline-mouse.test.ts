import { describe, expect, it } from 'vitest'

import { resolveInlineMouseTarget } from './inline-mouse.js'

// A 24-row terminal with a 5-row bottom-anchored inline frame: the frame
// occupies terminal rows 19..23, so absRow 19 -> localRow 0, 23 -> localRow 4.
const GEOM = { terminalRows: 24, frameWidth: 80, frameHeight: 5 }

describe('resolveInlineMouseTarget', () => {
  it('maps the frame top row to local row 0', () => {
    expect(resolveInlineMouseTarget(10, 19, GEOM)).toEqual({ col: 10, row: 0 })
  })

  it('maps the frame bottom row to the last local row', () => {
    expect(resolveInlineMouseTarget(10, 23, GEOM)).toEqual({ col: 10, row: 4 })
  })

  it('preserves the column unchanged', () => {
    expect(resolveInlineMouseTarget(37, 21, GEOM)).toEqual({ col: 37, row: 2 })
  })

  it('rejects a click above the frame (committed scrollback)', () => {
    expect(resolveInlineMouseTarget(10, 18, GEOM)).toBeNull()
    expect(resolveInlineMouseTarget(10, 0, GEOM)).toBeNull()
  })

  it('rejects a click below the frame', () => {
    expect(resolveInlineMouseTarget(10, 24, GEOM)).toBeNull()
  })

  it('rejects a click past the right edge', () => {
    expect(resolveInlineMouseTarget(80, 21, GEOM)).toBeNull()
    expect(resolveInlineMouseTarget(-1, 21, GEOM)).toBeNull()
  })

  it('accepts the last valid column', () => {
    expect(resolveInlineMouseTarget(79, 21, GEOM)).toEqual({ col: 79, row: 2 })
  })

  it('returns null for a zero-height or zero-width frame', () => {
    expect(resolveInlineMouseTarget(10, 20, { terminalRows: 24, frameWidth: 80, frameHeight: 0 })).toBeNull()
    expect(resolveInlineMouseTarget(10, 20, { terminalRows: 24, frameWidth: 0, frameHeight: 5 })).toBeNull()
  })

  it('maps correctly when the frame overflows the viewport into scrollback', () => {
    // 30-row frame in a 24-row terminal: rows 0..5 are scrolled off the top,
    // so the visible top (absRow 0) is frame-local row 6, bottom is row 29.
    const tall = { terminalRows: 24, frameWidth: 80, frameHeight: 30 }
    expect(resolveInlineMouseTarget(4, 0, tall)).toEqual({ col: 4, row: 6 })
    expect(resolveInlineMouseTarget(4, 23, tall)).toEqual({ col: 4, row: 29 })
  })

  describe('clamp mode (drag / release)', () => {
    it('clamps a point above the frame to the top row', () => {
      expect(resolveInlineMouseTarget(10, 5, GEOM, true)).toEqual({ col: 10, row: 0 })
    })

    it('clamps a point below the frame to the bottom row', () => {
      expect(resolveInlineMouseTarget(10, 40, GEOM, true)).toEqual({ col: 10, row: 4 })
    })

    it('clamps columns to the frame width', () => {
      expect(resolveInlineMouseTarget(200, 21, GEOM, true)).toEqual({ col: 79, row: 2 })
      expect(resolveInlineMouseTarget(-5, 21, GEOM, true)).toEqual({ col: 0, row: 2 })
    })

    it('still rejects a degenerate frame even when clamping', () => {
      expect(resolveInlineMouseTarget(10, 20, { terminalRows: 24, frameWidth: 80, frameHeight: 0 }, true)).toBeNull()
    })
  })
})
