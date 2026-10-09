import { describe, expect, it } from 'vitest'

import { color, contrastRatio, ensureContrast, mix, parseColor, readableOn, relativeLuminance, toHex } from './color.js'

describe('parseColor', () => {
  it('parses hex6, hex3 and rgb() forms', () => {
    expect(parseColor('#1a2b3c')).toEqual([0x1a, 0x2b, 0x3c])
    expect(parseColor('1a2b3c')).toEqual([0x1a, 0x2b, 0x3c])
    expect(parseColor('#abc')).toEqual([0xaa, 0xbb, 0xcc])
    expect(parseColor('rgb(220,255,220)')).toEqual([220, 255, 220])
    expect(parseColor('rgba(10, 20, 30, 0.5)')).toEqual([10, 20, 30])
  })

  it('rejects garbage', () => {
    expect(parseColor('')).toBeNull()
    expect(parseColor('ansi256(245)')).toBeNull()
    expect(parseColor('#12345')).toBeNull()
  })
})

describe('mix', () => {
  it('lerps in sRGB and round-trips through toHex', () => {
    expect(mix('#000000', '#ffffff', 0.5)).toBe('#808080')
    expect(mix('#ff0000', '#0000ff', 0)).toBe('#ff0000')
    expect(mix('#ff0000', '#0000ff', 1)).toBe('#0000ff')
    expect(toHex(parseColor(mix('#123456', '#654321', 0.3))!)).toBe(mix('#123456', '#654321', 0.3))
  })

  it('passes unparseable inputs through unchanged', () => {
    expect(mix('ansi256(245)', '#ffffff', 0.5)).toBe('ansi256(245)')
    expect(mix('#ff0000', 'nope', 0.5)).toBe('#ff0000')
  })
})

describe('contrast', () => {
  it('measures WCAG ratios at the anchors', () => {
    expect(contrastRatio('#000000', '#ffffff')).toBeCloseTo(21, 0)
    expect(contrastRatio('#ffffff', '#ffffff')).toBeCloseTo(1, 5)
    expect(relativeLuminance('#ffffff')).toBeCloseTo(1, 5)
    expect(relativeLuminance('#000000')).toBeCloseTo(0, 5)
  })

  it('readableOn picks the ink pole', () => {
    expect(readableOn('#ffffff')).toBe('#000000')
    expect(readableOn('#101014')).toBe('#ffffff')
  })

  it('ensureContrast lifts failing colors monotonically and leaves passing ones alone', () => {
    const pale = '#FFF8DC'
    const fixed = ensureContrast(pale, '#ffffff', 3.9)

    expect(contrastRatio(fixed, '#ffffff')!).toBeGreaterThanOrEqual(3.9)
    expect(ensureContrast('#3D2F13', '#ffffff', 3.9)).toBe('#3D2F13')
    expect(ensureContrast('ansi256(245)', '#ffffff', 3.9)).toBe('ansi256(245)')
  })
})

describe('color() chain', () => {
  it('composes the ladder operations', () => {
    const out = color('#F1E6CF').mix('#101014', 0.35).mix('#DD4A3A', 0.18).ensureContrast('#101014', 2.8).hex()

    expect(out).toMatch(/^#[0-9a-f]{6}$/)
    expect(contrastRatio(out, '#101014')!).toBeGreaterThanOrEqual(2.8)
    expect(color('#808080').luminance()).toBeCloseTo(relativeLuminance('#808080')!, 10)
  })
})
