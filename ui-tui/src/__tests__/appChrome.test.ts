import { describe, expect, it } from 'vitest'

import { getStatusRuleLayout, nextSessionDurationDelay } from '../components/appChrome.js'

describe('getStatusRuleLayout', () => {
  it('never allocates more columns than the terminal width', () => {
    const layout = getStatusRuleLayout(150, '/Users/diegoveras/omega/DonTrade-DATABASE')

    expect(layout.leftWidth + layout.dividerWidth + layout.rightWidth).toBeLessThanOrEqual(150)
    expect(layout.leftWidth).toBeGreaterThan(0)
  })

  it('shrinks the cwd lane instead of letting the status row wrap on narrow terminals', () => {
    const layout = getStatusRuleLayout(32, '/Users/diegoveras/omega/DonTrade-DATABASE/very/deep/path')

    expect(layout.leftWidth + layout.dividerWidth + layout.rightWidth).toBeLessThanOrEqual(32)
    expect(layout.leftWidth).toBeGreaterThanOrEqual(12)
    expect(layout.rightWidth).toBeLessThanOrEqual(Math.floor(32 * 0.4))
  })

  it('drops the divider when there is no room for a cwd lane', () => {
    const layout = getStatusRuleLayout(10, '/tmp/demo')

    expect(layout.rightWidth).toBe(0)
    expect(layout.dividerWidth).toBe(0)
    expect(layout.leftWidth).toBe(10)
  })
})

describe('nextSessionDurationDelay', () => {
  it('updates every second while the status text includes seconds', () => {
    expect(nextSessionDurationDelay(5_250)).toBe(750)
    expect(nextSessionDurationDelay(125_999)).toBe(1)
  })

  it('updates on minute boundaries once the status text only shows hours and minutes', () => {
    expect(nextSessionDurationDelay(3_600_000)).toBe(60_000)
    expect(nextSessionDurationDelay(6_630_250)).toBe(29_750)
  })
})
