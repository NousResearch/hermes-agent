import { describe, expect, it } from 'vitest'

import { formatBytes } from '../lib/memory.js'

describe('formatBytes', () => {
  it('returns 0B for zero', () => {
    expect(formatBytes(0)).toBe('0B')
  })

  it('returns 0B for negative', () => {
    expect(formatBytes(-100)).toBe('0B')
  })

  it('returns 0B for NaN', () => {
    expect(formatBytes(Number.NaN)).toBe('0B')
  })

  it('returns 0B for Infinity', () => {
    expect(formatBytes(Number.POSITIVE_INFINITY)).toBe('0B')
  })

  it('formats small byte values with B unit', () => {
    expect(formatBytes(1)).toMatch(/B$/)
    expect(formatBytes(500)).toMatch(/B$/)
  })

  it('formats kilobyte-range values with KB unit', () => {
    expect(formatBytes(2048)).toMatch(/KB$/)
  })

  it('formats megabyte-range values with MB unit', () => {
    expect(formatBytes(2_000_000)).toMatch(/MB$/)
  })

  it('formats gigabyte-range values with GB unit', () => {
    expect(formatBytes(5_000_000_000)).toMatch(/GB$/)
  })

  it('formats terabyte-range values with TB unit', () => {
    expect(formatBytes(2_000_000_000_000)).toMatch(/TB$/)
  })

  it('caps unit at TB for petabyte-scale inputs', () => {
    expect(formatBytes(1e18)).toMatch(/TB$/)
  })

  it('uses one decimal place for values below 100 in their unit', () => {
    const out = formatBytes(2_500_000)
    expect(out).toMatch(/^\d+\.\d[A-Z]+$/)
  })

  it('uses zero decimal places for values >= 100 in their unit', () => {
    const out = formatBytes(150_000_000)
    expect(out).toMatch(/^\d+[A-Z]+$/)
    expect(out).not.toMatch(/\./)
  })

  it('produces a stable shape: digits then unit suffix', () => {
    for (const n of [1, 999, 1024, 10_000, 1_000_000, 1_000_000_000]) {
      expect(formatBytes(n)).toMatch(/^\d+(\.\d)?[A-Z]+$/)
    }
  })
})
