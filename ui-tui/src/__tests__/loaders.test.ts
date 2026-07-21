import { describe, expect, it } from 'vitest'

import { renderToScreen } from '../../packages/hermes-ink/src/ink/render-to-screen.js'
import { cellAtIndex } from '../../packages/hermes-ink/src/ink/screen.js'
import { ShimmerRows, shimmerSegments } from '../components/loaders.js'

describe('ShimmerRows leniency (agent-authored calls)', () => {
  it('accepts a bare row COUNT and derives widths — the generated-code shape', async () => {
    const { createElement } = await import('react')

    const { screen, height } = renderToScreen(
      createElement(ShimmerRows, {
        rows: 3,
        width: 20,
        t: { color: { completionBg: '#1a1a2e', label: '#DAA520', muted: '#B8860B' } }
      }),
      30
    )

    expect(height).toBe(3)
    // Row 0 renders block cells, not a crash.
    expect(cellAtIndex(screen, 0).char).toBe('▁')
  })
})

describe('shimmerSegments', () => {
  it('always partitions the full width', () => {
    for (let phase = -40; phase < 80; phase++) {
      const [pre, band, post] = shimmerSegments(20, phase)

      expect(pre + band + post).toBe(20)
      expect(Math.min(pre, band, post)).toBeGreaterThanOrEqual(0)
    }
  })

  it('sweeps: enters from the left edge, exits off the right, then wraps', () => {
    const bandAt = (phase: number) => shimmerSegments(10, phase, 4)

    expect(bandAt(0)).toEqual([10, 0, 0]) // band fully off-left
    expect(bandAt(1)).toEqual([0, 1, 9]) // entering
    expect(bandAt(7)).toEqual([3, 4, 3]) // mid-sweep
    expect(bandAt(13)).toEqual([9, 1, 0]) // exiting
    expect(bandAt(14)).toEqual([10, 0, 0]) // gone → next cycle re-enters
    expect(bandAt(15)).toEqual([0, 1, 9])
  })

  it('negative phases (row stagger) wrap instead of vanishing', () => {
    const [pre, band, post] = shimmerSegments(10, -3, 4)

    expect(pre + band + post).toBe(10)
  })
})
