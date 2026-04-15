import { describe, expect, it } from 'vitest'

import { DEFAULT_THEME } from '../theme.js'
import type { WidgetSpec } from '../widgets.js'
import {
  bloombergTheme,
  buildWidgets,
  cityTime,
  livePoints,
  marquee,
  plotLineRows,
  sparkline,
  widgetsInRegion,
  wrapWindow
} from '../widgets.js'

const BASE_CTX = {
  bgCount: 0,
  busy: false,
  cols: 120,
  cwdLabel: '~/hermes-agent',
  durationLabel: '11s',
  model: 'claude',
  status: 'idle',
  t: DEFAULT_THEME,
  usage: { calls: 0, input: 0, output: 0, total: 0 },
  voiceLabel: 'voice off'
}

describe('sparkline', () => {
  it('respects requested width', () => {
    expect([...sparkline([1, 2, 3, 4, 5], 3)]).toHaveLength(3)
  })

  it('is stable for flat series', () => {
    const line = sparkline([7, 7, 7, 7], 4)

    expect([...line]).toHaveLength(4)
    expect(new Set([...line]).size).toBe(1)
  })
})

describe('widgetsInRegion', () => {
  it('filters and sorts by region + order', () => {
    const widgets: WidgetSpec[] = [
      { id: 'c', node: 'c', order: 20, region: 'dock' },
      { id: 'a', node: 'a', order: 5, region: 'dock' },
      { id: 'b', node: 'b', order: 1, region: 'overlay' }
    ]

    expect(widgetsInRegion(widgets, 'dock').map(w => w.id)).toEqual(['a', 'c'])
  })
})

describe('wrapWindow', () => {
  it('wraps around the array', () => {
    expect(wrapWindow([1, 2, 3], 2, 5)).toEqual([3, 1, 2, 3, 1])
  })
})

describe('marquee', () => {
  it('returns fixed-width slice', () => {
    expect(marquee('abc', 0, 5)).toHaveLength(5)
    expect(marquee('abc', 1, 5)).not.toEqual(marquee('abc', 0, 5))
  })
})

describe('plotLineRows', () => {
  it('returns the requested height', () => {
    expect(plotLineRows([1, 2, 3, 4], 4, 3)).toHaveLength(3)
  })

  it('each row has the requested width', () => {
    const rows = plotLineRows([1, 4, 2, 5], 20, 4)

    for (const row of rows) {
      expect([...row]).toHaveLength(20)
    }
  })

  it('draws visible braille for varied data', () => {
    expect(plotLineRows([1, 4, 2, 5], 8, 3).join('')).toMatch(/[^\u2800 ]/)
  })
})

describe('livePoints', () => {
  const asset = { label: 'TEST', series: [100, 102, 101, 103, 105] }

  it('extends the series by one', () => {
    expect(livePoints(asset, 0)).toHaveLength(asset.series.length + 1)
  })

  it('starts at series[0]', () => {
    expect(livePoints(asset, 0)[0]).toBe(100)
  })

  it('live point stays within ±2% of last value', () => {
    const last = asset.series.at(-1)!

    for (let t = 0; t < 50; t++) {
      const pts = livePoints(asset, t)
      const live = pts.at(-1)!

      expect(live).toBeGreaterThan(last * 0.98)
      expect(live).toBeLessThan(last * 1.02)
    }
  })
})

describe('cityTime', () => {
  it('returns HH:MM:SS format', () => {
    expect(cityTime('America/New_York')).toMatch(/^\d{2}:\d{2}:\d{2}$/)
  })

  it('works for all clock zones', () => {
    for (const tz of ['America/New_York', 'Europe/London', 'Asia/Tokyo', 'Australia/Sydney']) {
      expect(cityTime(tz)).toMatch(/^\d{2}:\d{2}:\d{2}$/)
    }
  })
})

describe('buildWidgets', () => {
  it('routes widgets into dock + sidebar regions', () => {
    const widgets = buildWidgets({ ...BASE_CTX, blocked: true })
    const byId = new Map(widgets.map(w => [w.id, w.region]))

    expect(byId.get('ticker')).toBe('dock')
    expect(byId.get('world-clock')).toBe('sidebar')
    expect(byId.get('weather')).toBe('sidebar')
    expect(byId.get('heartbeat')).toBe('sidebar')
  })

  it('filters by enabled map', () => {
    const enabled = { ticker: true, 'world-clock': false, weather: true, heartbeat: false }
    const widgets = buildWidgets(BASE_CTX, { enabled })
    const ids = widgets.map(w => w.id)

    expect(ids).toContain('ticker')
    expect(ids).toContain('weather')
    expect(ids).not.toContain('world-clock')
    expect(ids).not.toContain('heartbeat')
  })

  it('accepts widget params config', () => {
    const widgets = buildWidgets(BASE_CTX, {
      enabled: { ticker: true, 'world-clock': true, weather: true, heartbeat: true },
      params: { ticker: { asset: 'ETH' } }
    })

    expect(widgets.map(w => w.id)).toContain('ticker')
  })

  it('returns all when no enabled map given', () => {
    const widgets = buildWidgets({ ...BASE_CTX, blocked: false })

    expect(widgets.some(w => w.region === 'overlay')).toBe(false)
    expect(widgets.some(w => w.region === 'dock')).toBe(true)
  })

  it('includes all expected widget ids', () => {
    const ids = buildWidgets(BASE_CTX).map(w => w.id)

    expect(ids).toContain('ticker')
    expect(ids).toContain('weather')
    expect(ids).toContain('world-clock')
    expect(ids).toContain('heartbeat')
  })
})

describe('bloombergTheme', () => {
  it('overrides color keys while preserving brand', () => {
    const bt = bloombergTheme(DEFAULT_THEME)

    expect(bt.brand).toEqual(DEFAULT_THEME.brand)
    expect(bt.color.cornsilk).toBe('#FFFFFF')
    expect(bt.color.statusGood).toBe('#00EE00')
    expect(bt.color.statusBad).toBe('#FF2200')
  })
})
