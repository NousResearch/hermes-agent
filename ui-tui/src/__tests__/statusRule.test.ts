import { describe, expect, it } from 'vitest'

import { statusRuleWidths } from '../components/appChrome.js'

describe('statusRuleWidths', () => {
  it('keeps the status rule within the terminal width', () => {
    for (const cols of [8, 12, 20, 40, 100]) {
      const widths = statusRuleWidths(cols, '~/src/hermes-agent/main (some-long-branch-name)')

      expect(widths.leftWidth + widths.separatorWidth + widths.rightWidth).toBeLessThanOrEqual(cols)
      expect(widths.leftWidth).toBeGreaterThan(0)
    }
  })

  it('truncates the cwd segment before it can wrap in skinny terminals', () => {
    const widths = statusRuleWidths(24, '~/src/hermes-agent/main (bb/some-extremely-long-branch)')

    expect(widths.rightWidth).toBeLessThan('~/src/hermes-agent/main (bb/some-extremely-long-branch)'.length)
    expect(widths.leftWidth).toBeGreaterThanOrEqual(8)
  })

  it('omits the cwd segment when there is no room for it', () => {
    expect(statusRuleWidths(2, 'abcdef')).toEqual({ leftWidth: 2, rightWidth: 0, separatorWidth: 0 })
  })

  it('budgets the cwd segment by display width, not utf-16 length', () => {
    const widths = statusRuleWidths(30, '目录/分支')

    expect(widths.leftWidth + widths.separatorWidth + widths.rightWidth).toBeLessThanOrEqual(30)
    expect(widths.rightWidth).toBeGreaterThan('目录/分支'.length)
  })
})
