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

  it('keeps a null percent null', () => {
    expect(toBatteryInfo({ available: true, category: 'dim', percent: null, plugged: null })?.percent).toBeNull()
  })

  it('keeps an unknown plugged state null', () => {
    expect(toBatteryInfo({ available: false, category: 'dim', percent: null, plugged: null })?.plugged).toBeNull()
  })
})
