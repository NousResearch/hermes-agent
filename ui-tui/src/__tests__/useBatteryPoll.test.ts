import { describe, expect, it } from 'vitest'

import { toBatteryInfo } from '../app/useBatteryPoll.js'

describe('toBatteryInfo', () => {
  it('returns null for a null payload', () => {
    expect(toBatteryInfo(null)).toBeNull()
  })

  it('maps a full reading through faithfully', () => {
    expect(toBatteryInfo({ available: true, category: 'warn', percent: 44, plugged: false })).toEqual({
      available: true,
      category: 'warn',
      percent: 44,
      plugged: false
    })
  })

  it('clamps and rounds the percent into 0-100', () => {
    expect(toBatteryInfo({ available: true, category: 'good', percent: 142.7, plugged: true })?.percent).toBe(100)
    expect(toBatteryInfo({ available: true, category: 'critical', percent: -5, plugged: false })?.percent).toBe(0)
    expect(toBatteryInfo({ available: true, category: 'warn', percent: 43.4, plugged: false })?.percent).toBe(43)
  })

  it('coerces a missing/invalid percent to null', () => {
    expect(toBatteryInfo({ available: true, category: 'dim' })?.percent).toBeNull()
  })

  it('falls back to the dim category for an unknown value', () => {
    expect(toBatteryInfo({ available: true, category: 'purple', percent: 50, plugged: false })?.category).toBe('dim')
  })

  it('treats a non-boolean plugged as unknown (null)', () => {
    expect(toBatteryInfo({ available: false, category: 'dim', percent: null })?.plugged).toBeNull()
  })
})
