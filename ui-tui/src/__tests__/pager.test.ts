import { describe, expect, it } from 'vitest'

import { PAGER_CHROME_ROWS, pagerLineWidth, pagerPageSize, pagerPanelWidth } from '../lib/pager.js'

describe('pager geometry', () => {
  it('keeps a whole page inside the terminal', () => {
    for (const rows of [12, 20, 24, 45, 60]) {
      const page = pagerPageSize(rows)

      expect(page).toBeGreaterThan(0)
      // panel height (page + chrome) must never exceed the terminal
      expect(page + PAGER_CHROME_ROWS).toBeLessThanOrEqual(rows)
    }
  })

  it('spends the full height of a 45-row terminal', () => {
    expect(pagerPageSize(45)).toBe(36)
  })

  it('leaves one column of margin on each side', () => {
    expect(pagerPanelWidth(120)).toBe(118)
    expect(pagerLineWidth(120)).toBeLessThanOrEqual(pagerPanelWidth(120) - 6)
    expect(pagerLineWidth(120)).toBeGreaterThan(60)
  })
})
