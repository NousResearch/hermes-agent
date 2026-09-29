import { describe, expect, it } from 'vitest'

import { agentsOverlayGeometry } from '../lib/agentsOverlayLayout.js'

describe('agentsOverlayGeometry', () => {
  it('uses full-screen fallback on a short terminal', () => {
    expect(agentsOverlayGeometry(24, 80)).toEqual({ canCompact: false, mode: 'full', rows: 24 })
  })

  it('uses full-screen fallback on a narrow terminal', () => {
    expect(agentsOverlayGeometry(40, 40)).toEqual({ canCompact: false, mode: 'full', rows: 40 })
  })

  it('uses a bounded compact overlay on a normal terminal', () => {
    const out = agentsOverlayGeometry(40, 80)

    expect(out.canCompact).toBe(true)
    expect(out.mode).toBe('compact')
    expect(out.rows).toBe(17)
    expect(40 - out.rows).toBeGreaterThanOrEqual(10)
  })

  it('caps a tall terminal instead of consuming half the screen forever', () => {
    expect(agentsOverlayGeometry(80, 120)).toEqual({ canCompact: true, mode: 'compact', rows: 22 })
  })

  it('allows an explicit full-height expansion on a compact-capable terminal', () => {
    expect(agentsOverlayGeometry(40, 80, true)).toEqual({ canCompact: true, mode: 'full', rows: 40 })
  })
})
