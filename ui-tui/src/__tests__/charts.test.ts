import { describe, expect, it } from 'vitest'

import { gauge, hbars, sparkline, sparkRows } from '../lib/charts.js'

describe('chart primitives', () => {
  it('sparkline spans the block ramp and respects width', () => {
    const line = sparkline([0, 1, 2, 3, 4, 5, 6, 7], 8)

    expect(line).toBe('▁▂▃▄▅▆▇█')
    expect(sparkline([1, 2, 3, 4], 2)).toHaveLength(2) // window = last N
  })

  it('is dimension-stable: short/empty series pad to exactly width', () => {
    // Warm-up must never resize the card — latest sample pins right.
    expect(sparkline([5], 6)).toHaveLength(6)
    expect(sparkline([5], 6).endsWith('▁')).toBe(true) // flat series → bottom block, right-pinned
    expect(sparkline([], 6)).toBe('      ')
    expect(sparkRows([7], 5, 2).every(row => row.length === 5)).toBe(true)
    expect(sparkRows([], 5, 2)).toEqual(['     ', '     '])
    expect(hbars([1, 4], 8).every(bar => bar.length === 8)).toBe(true)
  })

  it('sparkRows partitions each column across rows (top line first)', () => {
    const rows = sparkRows([0, 8, 4], 3, 2)

    expect(rows).toHaveLength(2)
    expect(rows.every(r => r.length === 3)).toBe(true)
    // Max value fills the top row cell; min value leaves it blank.
    expect(rows[0]![1]).toBe('█')
    expect(rows[0]![0]).toBe(' ')
  })

  it('gauge clamps and fills proportionally', () => {
    expect(gauge(0.5, 8)).toBe('████░░░░')
    expect(gauge(-1, 4)).toBe('░░░░')
    expect(gauge(9, 4)).toBe('████')
  })

  it('hbars scales to the max with eighth-block tips', () => {
    const [half, full] = hbars([4, 8], 8)

    expect(full).toBe('████████')
    expect(half).toBe('████    ')
    expect(hbars([3, 8], 8)[0]).toBe('███     ') // 3/8 of 8 cells, padded
    expect(hbars([1, 2], 3)[0]).toMatch(/^█?[▏▎▍▌▋▊▉█] *$/) // fractional tip, padded
  })
})
