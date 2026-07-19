import { describe, expect, it } from 'vitest'

import { fmtCost, fmtPages, fmtTok, fmtUntil, gauge } from '../components/arcStatusBar.js'

describe('fmtPages', () => {
  it('anchors 400 tokens ≈ 1 page', () => {
    expect(fmtPages(400)).toBe('1p')
    expect(fmtPages(200_000)).toBe('500p')
  })

  it('handles sub-page and kilo-page magnitudes', () => {
    expect(fmtPages(0)).toBe('0p')
    expect(fmtPages(20)).toBe('0p') // < 0.1p
    expect(fmtPages(200)).toBe('0.5p')
    expect(fmtPages(500_000)).toBe('1.3Kp') // 1250p → 1.25K → 1.3Kp
  })
})

describe('fmtTok', () => {
  it('compacts thousands and millions', () => {
    expect(fmtTok(950)).toBe('950')
    expect(fmtTok(12_400)).toBe('12K')
    expect(fmtTok(2_500_000)).toBe('2.5M')
  })
})

describe('fmtUntil', () => {
  const now = 1_000_000_000_000 // fixed epoch ms

  it('formats epoch-seconds resets as H/M', () => {
    expect(fmtUntil(now / 1000 + 3900, now)).toBe('1H05M') // 65 min
    expect(fmtUntil(now / 1000 + 300, now)).toBe('5M')
  })

  it('accepts ISO strings', () => {
    const iso = new Date(now + 600_000).toISOString() // +10 min
    expect(fmtUntil(iso, now)).toBe('10M')
  })

  it('returns empty / zero for missing or elapsed resets', () => {
    expect(fmtUntil(undefined, now)).toBe('')
    expect(fmtUntil(0, now)).toBe('')
    expect(fmtUntil(now / 1000 - 60, now)).toBe('0M')
    expect(fmtUntil('not-a-date', now)).toBe('')
  })
})

describe('gauge', () => {
  it('fills proportionally and thresholds colors', () => {
    const low = gauge(20, 10)
    expect(low.filled).toBe('▓▓')
    expect(low.empty).toBe('░░░░░░░░')
    expect(low.warn).toBe(false)

    const mid = gauge(60, 10)
    expect(mid.warn).toBe(false)
    expect(mid.color).not.toBe(low.color) // amber ≠ cyan

    const high = gauge(95, 10)
    expect(high.warn).toBe(true)
    expect(high.pctText).toBe(' 95%')
  })

  it('clamps out-of-range percentages', () => {
    expect(gauge(-10, 10).filled).toBe('')
    expect(gauge(150, 10).filled).toBe('▓▓▓▓▓▓▓▓▓▓')
    expect(gauge(150, 10).pctText).toBe('100%')
  })
})

describe('fmtCost', () => {
  it('prefixes ~ for estimated, nothing for actual', () => {
    expect(fmtCost({ cost_usd: 1.234, cost_status: 'estimated' })).toBe('~$1.23')
    expect(fmtCost({ cost_usd: 1.234, cost_status: 'actual' })).toBe('$1.23')
  })

  it('shows incl for subscription and hides unknown', () => {
    expect(fmtCost({ cost_usd: 0, cost_status: 'included' })).toBe('incl')
    expect(fmtCost({ cost_status: 'unknown' })).toBeNull()
    expect(fmtCost({})).toBeNull()
  })

  it('shows a bare number when a cost is sent without a status', () => {
    expect(fmtCost({ cost_usd: 0.5 })).toBe('$0.50')
  })
})
