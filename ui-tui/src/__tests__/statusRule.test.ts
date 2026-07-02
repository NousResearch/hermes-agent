import { describe, expect, it } from 'vitest'

import { busyIndicatorWidth, statusBarSegments, statusRuleWidths } from '../components/appChrome.js'

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

  it('reserves the high-priority left content so the cwd/branch yields first', () => {
    const cwd = '~/src/hermes-agent/apps/desktop (bb/tui-statusbar-responsive)'

    const greedy = statusRuleWidths(70, cwd) // legacy behaviour: cwd hogs the row
    const reserved = statusRuleWidths(70, cwd, 40) // reserve indicator+model+ctx

    expect(reserved.leftWidth).toBeGreaterThanOrEqual(40)
    expect(reserved.leftWidth).toBeGreaterThan(greedy.leftWidth)
    expect(reserved.rightWidth).toBeLessThan(greedy.rightWidth)
    expect(reserved.leftWidth + reserved.separatorWidth + reserved.rightWidth).toBeLessThanOrEqual(70)
  })

  it('drops the cwd entirely when the essential left content needs the whole row', () => {
    expect(statusRuleWidths(40, '~/some/cwd (branch)', 60)).toEqual({
      leftWidth: 40,
      rightWidth: 0,
      separatorWidth: 0
    })
  })

  it('keeps the default (no reservation) behaviour identical for legacy callers', () => {
    const cwd = '~/src/hermes-agent/main (some-long-branch-name)'

    expect(statusRuleWidths(80, cwd, 0)).toEqual(statusRuleWidths(80, cwd))
  })
})

describe('statusBarSegments', () => {
  it('shows every segment on a wide terminal', () => {
    const s = statusBarSegments(120)

    expect(s).toEqual({
      compactCtx: false,
      bar: true,
      duration: true,
      compressions: true,
      voice: true,
      bg: true,
      subagents: true,
      cost: true
    })
  })

  it('collapses the context bar to a token count on narrow terminals', () => {
    const s = statusBarSegments(60)

    expect(s.compactCtx).toBe(true)
    expect(s.bar).toBe(false)
    expect(s.duration).toBe(false)
    expect(s.cost).toBe(false)
  })

  it('sheds tail segments in priority order as the terminal narrows', () => {
    // cost is the first to go, the context bar the last of the tail.
    const order: (keyof ReturnType<typeof statusBarSegments>)[] = [
      'bar',
      'duration',
      'compressions',
      'voice',
      'bg',
      'subagents',
      'cost'
    ]

    let prevCount = Infinity

    for (const cols of [120, 95, 87, 83, 79, 75, 71]) {
      const s = statusBarSegments(cols)
      const visible = order.filter(k => s[k]).length

      expect(visible).toBeLessThanOrEqual(prevCount)
      prevCount = visible
    }
  })
})

describe('busyIndicatorWidth', () => {
  it('reserves a bare spinner for the verb-less unicode style', () => {
    // unicode is a 1-col braille spinner with no verb; far slimmer than the
    // kaomoji face which carries a wide glyph + rotating verb.
    expect(busyIndicatorWidth('unicode', false)).toBeLessThan(busyIndicatorWidth('kaomoji', false))
    expect(busyIndicatorWidth('unicode', false)).toBe(1)
  })

  it('reserves room for the elapsed-time tail only when a turn is timed', () => {
    for (const style of ['kaomoji', 'emoji', 'ascii', 'unicode'] as const) {
      expect(busyIndicatorWidth(style, true)).toBeGreaterThan(busyIndicatorWidth(style, false))
    }
  })
})
