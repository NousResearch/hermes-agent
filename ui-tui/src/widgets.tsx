import { Box, Text } from '@hermes/ink'
import { Fragment, type ReactNode, useEffect, useState } from 'react'

import type { Theme } from './theme.js'
import type { Usage } from './types.js'

// ── Region types ──────────────────────────────────────────────────────
export const WIDGET_REGIONS = [
  'transcript-header',
  'transcript-inline',
  'transcript-tail',
  'dock',
  'overlay',
  'sidebar'
] as const
export type WidgetRegion = (typeof WIDGET_REGIONS)[number]

export interface WidgetCtx {
  blocked?: boolean
  bgCount: number
  busy: boolean
  cols: number
  cwdLabel?: string
  durationLabel?: string
  model?: string
  status: string
  t: Theme
  // tick is intentionally NOT here — each widget calls useWidgetTicker() internally.
  // Passing tick via props caused useMemo in AppLayout to rebuild JSX on every second,
  // which created stale prop snapshots and broke animated text rendering.
  usage: Usage
  voiceLabel?: string
}

export interface WidgetSpec {
  id: string
  node: ReactNode
  order?: number
  region: WidgetRegion
  // Optional: theme transform applied to `t` before rendering. This lets
  // individual widgets opt into a different color palette (e.g. Bloomberg)
  // without touching the main app theme.
  themeOverride?: (base: Theme) => Theme
}

export interface WidgetRenderState {
  enabled?: Record<string, boolean>
  params?: Record<string, Record<string, string>>
}

// ── Theme overrides ───────────────────────────────────────────────────
// Bloomberg terminal palette: high-contrast orange/green/red on dark intent.
export function bloombergTheme(t: Theme): Theme {
  return {
    ...t,
    color: {
      ...t.color,
      gold: '#FFE000', // bright yellow titles
      amber: '#FF8C00', // orange for values
      bronze: '#FF6600', // orange borders
      cornsilk: '#FFFFFF', // white for primary text
      dim: '#777777', // gray secondary
      label: '#FFCC00', // amber labels
      statusGood: '#00EE00', // classic Bloomberg green
      statusBad: '#FF2200', // Bloomberg red
      statusWarn: '#FFAA00'
    }
  }
}

// ── Data ─────────────────────────────────────────────────────────────
interface Asset {
  label: string
  series: number[]
}

const BTC: number[] = [
  83900, 84210, 85140, 84680, 85990, 86540, 87310, 86820, 87940, 88600, 89200, 90100, 90560, 91840, 91210, 92680, 92100,
  91500, 92900, 93200
]

const ETH: number[] = [
  2920, 2960, 3015, 2990, 3070, 3050, 3110, 3080, 3160, 3200, 3225, 3260, 3290, 3250, 3310, 3330, 3385, 3360, 3400, 3420
]

const NVDA: number[] = [
  125, 128, 129, 127, 131, 130, 133, 132, 136, 137, 139, 138, 141, 140, 142, 143, 145, 144, 146, 148
]

const TSLA: number[] = [
  172, 176, 178, 175, 181, 180, 184, 182, 188, 187, 191, 189, 195, 193, 196, 198, 201, 199, 203, 205
]

const TEMP_DAY: number[] = [65, 66, 67, 69, 71, 73, 74, 75, 74, 73, 72, 71, 70, 69, 68]

const USD = new Intl.NumberFormat('en-US', { currency: 'USD', maximumFractionDigits: 0, style: 'currency' })
const BARS = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
const ORBS = ['◐', '◓', '◑', '◒']

const ASSETS: Asset[] = [
  { label: 'BTC', series: BTC },
  { label: 'ETH', series: ETH },
  { label: 'NVDA', series: NVDA },
  { label: 'TSLA', series: TSLA }
]

const CLOCKS = [
  { label: 'NYC', tz: 'America/New_York' },
  { label: 'LON', tz: 'Europe/London' },
  { label: 'TKY', tz: 'Asia/Tokyo' },
  { label: 'SYD', tz: 'Australia/Sydney' }
]

const SKY_DESC = ['Overcast', 'Partly cloudy', 'Clear']
const SKY_ICON = ['☁', '⛅', '☀']

// ── Ticker ────────────────────────────────────────────────────────────
export function useWidgetTicker(ms = 1000) {
  const [tick, setTick] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setTick(v => v + 1), ms)

    return () => clearInterval(id)
  }, [ms])

  return tick
}

// ── Pure math helpers ─────────────────────────────────────────────────
// Always resamples to exactly `width` points (no early return for small input).
// The old `if (values.length <= width) return values` caused the braille grid
// to only fill the first N pixel-columns when N < pxWidth.
function sample(values: number[], width: number): number[] {
  if (!values.length || width <= 0) {
    return []
  }

  if (width === 1) {
    return [values.at(-1) ?? 0]
  }

  return Array.from({ length: width }, (_, i) => {
    const pos = (i * (values.length - 1)) / (width - 1)

    return values[Math.round(pos)] ?? values.at(-1) ?? 0
  })
}

export function sparkline(values: number[], width: number): string {
  const pts = sample(values, width)

  if (!pts.length) {
    return ''
  }

  const lo = Math.min(...pts)
  const hi = Math.max(...pts)

  if (lo === hi) {
    return BARS[Math.floor(BARS.length / 2)]!.repeat(pts.length)
  }

  return pts
    .map(v => BARS[Math.max(0, Math.min(BARS.length - 1, Math.round(((v - lo) / (hi - lo)) * (BARS.length - 1))))]!)
    .join('')
}

export function wrapWindow<T>(values: T[], offset: number, width: number): T[] {
  if (!values.length || width <= 0) {
    return []
  }

  const start = ((offset % values.length) + values.length) % values.length

  return Array.from({ length: width }, (_, i) => values[(start + i) % values.length]!)
}

export function marquee(text: string, offset: number, width: number): string {
  if (!text || width <= 0) {
    return ''
  }

  const gap = '   '
  const source = text + gap + text
  const span = text.length + gap.length
  const start = ((offset % span) + span) % span

  return source.slice(start, start + width).padEnd(width, ' ')
}

// ── Braille line chart ────────────────────────────────────────────────
function brailleBit(dx: number, dy: number): number {
  return dx === 0 ? ([0x1, 0x2, 0x4, 0x40][dy] ?? 0) : ([0x8, 0x10, 0x20, 0x80][dy] ?? 0)
}

function drawLine(grid: boolean[][], x0: number, y0: number, x1: number, y1: number) {
  let x = x0
  let y = y0
  const dx = Math.abs(x1 - x0)
  const sx = x0 < x1 ? 1 : -1
  const dy = -Math.abs(y1 - y0)
  const sy = y0 < y1 ? 1 : -1
  let err = dx + dy

  while (true) {
    if (grid[y] && x >= 0 && x < grid[y]!.length) {
      grid[y]![x] = true
    }

    if (x === x1 && y === y1) {
      return
    }

    const e2 = err * 2

    if (e2 >= dy) {
      err += dy
      x += sx
    }

    if (e2 <= dx) {
      err += dx
      y += sy
    }
  }
}

export function plotLineRows(values: number[], width: number, height: number): string[] {
  if (!values.length || width <= 0 || height <= 0) {
    return []
  }

  const pxW = Math.max(2, width * 2)
  const pxH = Math.max(4, height * 4)
  const pts = sample(values, pxW)
  const lo = Math.min(...pts)
  const hi = Math.max(...pts)
  const grid = Array.from({ length: pxH }, () => new Array<boolean>(pxW).fill(false))

  const yFor = (v: number) => (hi === lo ? Math.floor((pxH - 1) / 2) : Math.round(((hi - v) / (hi - lo)) * (pxH - 1)))

  let prevY = yFor(pts[0] ?? 0)

  if (grid[prevY]) {
    grid[prevY]![0] = true
  }

  for (let x = 1; x < pts.length; x++) {
    const y = yFor(pts[x] ?? pts[x - 1] ?? 0)
    drawLine(grid, x - 1, prevY, x, y)
    prevY = y
  }

  return Array.from({ length: height }, (_, row) => {
    const top = row * 4

    return Array.from({ length: width }, (_, col) => {
      const left = col * 2
      let bits = 0

      for (let dy = 0; dy < 4; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          if (grid[top + dy]?.[left + dx]) {
            bits |= brailleBit(dx, dy)
          }
        }
      }

      return String.fromCodePoint(0x2800 + bits)
    }).join('')
  })
}

// ── Domain helpers ────────────────────────────────────────────────────
function pct(now: number, start: number): number {
  return !start ? 0 : ((now - start) / start) * 100
}

function deltaColor(delta: number, t: Theme): string {
  return delta > 0 ? t.color.statusGood : delta < 0 ? t.color.statusBad : t.color.dim
}

function money(v: number): string {
  return USD.format(v)
}

function changeStr(v: number): string {
  return `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`
}

export function cityTime(tz: string): string {
  try {
    return new Date().toLocaleTimeString('en-US', {
      hour: '2-digit',
      hour12: false,
      minute: '2-digit',
      second: '2-digit',
      timeZone: tz
    })
  } catch {
    return '--:--:--'
  }
}

function fToC(f: number): number {
  return Math.round(((f - 32) * 5) / 9)
}

// Smooth live points — always starts at series[0], adds an animated live
// endpoint oscillating ±0.8% so the chart never has discontinuous jumps.
export function livePoints(asset: Asset, tick: number): number[] {
  const last = asset.series.at(-1) ?? 0
  const phase = (tick * 0.3) % (Math.PI * 2)
  const live = Math.round(last * (1 + Math.sin(phase) * 0.008))

  return [...asset.series, live]
}

// ── Primitive components ──────────────────────────────────────────────
function LineChart({ color, height, values, width }: { color: any; height: number; values: number[]; width: number }) {
  const rows = plotLineRows(values, width, height)

  return (
    <Box flexDirection="column">
      {rows.map((row, i) => (
        <Text color={color} key={i}>
          {row}
        </Text>
      ))}
    </Box>
  )
}

// Simple widget frame system:
// - bordered widgets: default card chrome
// - bleed widgets: full-surface background with internal padding
function WidgetFrame({
  backgroundColor,
  bordered = true,
  borderColor,
  children,
  paddingX = 1,
  paddingY = 0,
  title,
  titleRight,
  titleTone,
  t
}: {
  backgroundColor?: any
  bordered?: boolean
  borderColor?: any
  children: ReactNode
  paddingX?: number
  paddingY?: number
  title: ReactNode
  titleRight?: ReactNode
  titleTone?: any
  t: Theme
}) {
  return (
    <Box
      backgroundColor={backgroundColor}
      borderColor={bordered ? (borderColor ?? t.color.bronze) : undefined}
      borderStyle={bordered ? 'round' : undefined}
      flexDirection="column"
      paddingX={paddingX}
      paddingY={paddingY}
    >
      <Box justifyContent="space-between">
        <Box flexDirection="row" flexShrink={1}>
          {typeof title === 'string' || typeof title === 'number' ? (
            <Text bold color={(titleTone ?? t.color.gold) as any} wrap="truncate-end">
              {title}
            </Text>
          ) : (
            title
          )}
        </Box>
        {titleRight ? (
          <Box flexDirection="row" flexShrink={0} marginLeft={1}>
            {typeof titleRight === 'string' || typeof titleRight === 'number' ? (
              <Text color={t.color.dim as any}>{titleRight}</Text>
            ) : (
              titleRight
            )}
          </Box>
        ) : null}
      </Box>
      {children}
    </Box>
  )
}

// ── Widgets ───────────────────────────────────────────────────────────

// Bloomberg-styled hero chart.
// Dark navy background for the chart cell so the green/red line pops.
// Custom layout (not Card) for full control over the title row structure.
// Compact single-line ticker strip for the dock region.
function TickerStrip({ cols, t, assetId }: WidgetCtx & { assetId?: string }) {
  const tick = useWidgetTicker(1200)

  const asset = ASSETS.find(a => a.label.toLowerCase() === assetId?.toLowerCase()) ?? ASSETS[tick % ASSETS.length]!

  const pts = livePoints(asset, tick)
  const last = pts.at(-1) ?? 0
  const first = pts[0] ?? 0
  const change = pct(last, first)
  const color = deltaColor(change, t)
  const sparkW = Math.max(8, Math.min(30, cols - 40))

  const others = ASSETS.filter(a => a.label !== asset.label)
    .map(a => {
      const c = pct(a.series.at(-1) ?? 0, a.series[0] ?? 0)

      return `${a.label} ${changeStr(c)}`
    })
    .join('  ')

  return (
    <Text color={t.color.dim as any} wrap="truncate-end">
      <Text bold color={t.color.gold as any}>
        {asset.label}
      </Text>
      <Text color={t.color.cornsilk as any}>{` ${money(last)} `}</Text>
      <Text bold color={color as any}>
        {changeStr(change)}
      </Text>
      <Text color={t.color.dim as any}>{`  ${sparkline(pts, sparkW)}  `}</Text>
      <Text color={t.color.dim as any}>{others}</Text>
    </Text>
  )
}

function WeatherCard({ t }: WidgetCtx) {
  const tick = useWidgetTicker(2000)
  const skyIdx = Math.floor(tick / 8) % SKY_DESC.length
  const desc = SKY_DESC[skyIdx]!
  const icon = SKY_ICON[skyIdx]!
  const temp = TEMP_DAY[tick % TEMP_DAY.length]!
  const wind = 9 + ((tick * 2) % 7)
  const hum = 64 + (tick % 8)

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.gold as any}>
        Weather
      </Text>
      <Text color={t.color.cornsilk as any} wrap="truncate-end">{`${icon} ${fToC(temp)}C · ${desc}`}</Text>
      <Text color={t.color.dim as any} wrap="truncate-end">{`Wind ${wind} km/h · Humidity ${hum}%`}</Text>
    </Box>
  )
}

// 2x2 clock grid in compact rows
function WorldClock({ cols, t }: WidgetCtx) {
  const tick = useWidgetTicker()
  const orb = ORBS[tick % ORBS.length]!
  const rows = [CLOCKS.slice(0, 2), CLOCKS.slice(2, 4)] as const
  const slotW = Math.max(12, Math.floor((Math.max(cols, 24) - 2) / 2))
  const cell = (label: string, tz: string) => `${label} ${cityTime(tz)}`

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.gold as any}>{`${orb} World Clock`}</Text>
      {rows.map((row, i) => (
        <Box flexDirection="row" key={i}>
          <Box marginRight={1} width={slotW}>
            <Text color={t.color.cornsilk as any} wrap="truncate-end">
              {cell(row[0]!.label, row[0]!.tz)}
            </Text>
          </Box>
          <Box width={slotW}>
            <Text color={t.color.cornsilk as any} wrap="truncate-end">
              {cell(row[1]!.label, row[1]!.tz)}
            </Text>
          </Box>
        </Box>
      ))}
    </Box>
  )
}

function HeartBeat({ t }: WidgetCtx) {
  const tick = useWidgetTicker(700)
  const bpm = 70 + (tick % 5)

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.gold as any}>
        Heartbeat
      </Text>
      <Text color={t.color.statusBad as any}>{`❤️ ${bpm} bpm`}</Text>
    </Box>
  )
}

// ── Widget catalog ────────────────────────────────────────────────────
export interface WidgetDef {
  id: string
  description: string
  region: WidgetRegion
  order: number
  defaultOn: boolean
  params?: string[]
}

export const WIDGET_CATALOG: WidgetDef[] = [
  {
    id: 'ticker',
    description: 'Live stock ticker strip',
    region: 'dock',
    order: 10,
    defaultOn: true,
    params: ['asset']
  },
  { id: 'world-clock', description: '2x2 world clock grid', region: 'sidebar', order: 10, defaultOn: true },
  { id: 'weather', description: 'Weather conditions', region: 'sidebar', order: 20, defaultOn: true },
  { id: 'heartbeat', description: 'Heartbeat monitor', region: 'sidebar', order: 30, defaultOn: true }
]

// ── Registry ──────────────────────────────────────────────────────────
export function buildWidgets(ctx: WidgetCtx, state?: WidgetRenderState): WidgetSpec[] {
  const bt = bloombergTheme(ctx.t)
  const enabled = state?.enabled
  const params = state?.params
  const on = (id: string) => (enabled ? (enabled[id] ?? false) : true)
  const param = (id: string, key: string) => params?.[id]?.[key]

  const all: WidgetSpec[] = [
    {
      id: 'ticker',
      node: <TickerStrip {...ctx} assetId={param('ticker', 'asset')} t={bt} />,
      order: 10,
      region: 'dock'
    },
    { id: 'world-clock', node: <WorldClock {...ctx} />, order: 10, region: 'sidebar' },
    { id: 'weather', node: <WeatherCard {...ctx} />, order: 20, region: 'sidebar' },
    { id: 'heartbeat', node: <HeartBeat {...ctx} />, order: 30, region: 'sidebar' }
  ]

  return all.filter(w => on(w.id))
}

export function widgetsInRegion(widgets: WidgetSpec[], region: WidgetRegion) {
  return [...widgets].filter(w => w.region === region).sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
}

export function WidgetHost({ region, widgets }: { region: WidgetRegion; widgets: WidgetSpec[] }) {
  const visible = widgetsInRegion(widgets, region)

  if (!visible.length) {
    return null
  }

  if (region === 'overlay') {
    return (
      <>
        {visible.map(w => (
          <Fragment key={w.id}>{w.node}</Fragment>
        ))}
      </>
    )
  }

  return (
    <Box flexDirection="column">
      {visible.map((w, i) => (
        <Box flexDirection="column" key={w.id} marginTop={i === 0 ? 0 : 1}>
          {w.node}
        </Box>
      ))}
    </Box>
  )
}
