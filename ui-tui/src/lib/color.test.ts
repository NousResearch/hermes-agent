import { contrastRatio, relativeLuminance } from '@hermes/shared/color'
import { describe, expect, it } from 'vitest'

import { color } from './color.js'

describe('color() chain', () => {
  it('composes the ladder operations', () => {
    const out = color('#F1E6CF').mix('#101014', 0.35).mix('#DD4A3A', 0.18).ensureContrast('#101014', 2.8).hex()

    expect(out).toMatch(/^#[0-9a-f]{6}$/)
    expect(contrastRatio(out, '#101014')!).toBeGreaterThanOrEqual(2.8)
    expect(color('#808080').luminance()).toBeCloseTo(relativeLuminance('#808080')!, 10)
  })
})
