import { Box, Text } from '@hermes/ink'
import { memo, type ReactNode, useEffect, useRef, useState } from 'react'

import type { GridAreaCell, GridTrackSize } from '../lib/widgetGrid.js'
import type { GridTestState } from '../sdk/apps/gridTestState.js'
import type { Theme } from '../theme.js'

import { GridAreas, type GridAreaWidget } from './widgetGrid.js'

// The streams demo tiles six independently-ticking panels through
// `layoutGridAreas`: one panel owns a promoted 2x2 slot, the rest auto-place
// densely around it. Promoting a different panel reorders/respans the widget
// list — but cells are keyed by id, so every stream keeps its history while
// the grid reshapes around it. That relayout-without-reset is the point of
// the demo.

const HEADER_ROWS = 3
const STREAM_ROWS = 3
const GRID_HEIGHT = HEADER_ROWS + STREAM_ROWS * 6

const SPARK_CHARS = '▁▂▃▄▅▆▇█'

interface StreamDef {
  id: string
  render: (inner: { height: number; t: Theme; width: number }) => ReactNode
  title: string
}

// ── tick + history hooks ────────────────────────────────────────────────────

const useTick = (ms: number) => {
  const [tick, setTick] = useState(0)

  useEffect(() => {
    const timer = setInterval(() => setTick(v => v + 1), ms)

    return () => clearInterval(timer)
  }, [ms])

  return tick
}

/** Ring-buffered sample history: one `sample()` per tick, capped at `cap`. */
const useHistory = (tick: number, sample: () => number, cap = 240) => {
  const historyRef = useRef<number[]>([])

  useEffect(() => {
    historyRef.current.push(sample())

    if (historyRef.current.length > cap) {
      historyRef.current.splice(0, historyRef.current.length - cap)
    }
    // The sampler is intentionally re-run per tick, not per identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick])

  return historyRef.current
}

/**
 * Render history as a column chart `rows` lines tall (top line first). Each
 * column gets `rows * 8` vertical levels — full blocks below the value, a
 * partial eighth-block at the value row, spaces above — so a promoted (taller)
 * cell genuinely gains chart resolution rather than just whitespace.
 */
const sparkRows = (history: number[], width: number, rows: number): string[] => {
  const window = history.slice(-Math.max(1, width))

  if (!window.length) {
    return Array.from({ length: rows }, () => '')
  }

  const min = Math.min(...window)
  const max = Math.max(...window)
  const range = max - min || 1
  const levels = window.map(v => Math.max(1, Math.round(((v - min) / range) * rows * 8)))

  return Array.from({ length: rows }, (_, lineIdx) => {
    const rowFromBottom = rows - 1 - lineIdx

    return levels
      .map(level => {
        const filled = Math.min(8, Math.max(0, level - rowFromBottom * 8))

        return filled === 0 ? ' ' : SPARK_CHARS[filled - 1]
      })
      .join('')
  })
}

// ── stream panels ───────────────────────────────────────────────────────────

const TOKEN_WORDS = (
  `Hermes streams tokens into the promoted cell while the grid reshapes around it. ` +
  `Cells are keyed by id, so promotion never resets a panel — history, cursors and ` +
  `tickers all survive the relayout. Row and column tracks re-solve to integer ` +
  `terminal cells on every change, spans bridge the gaps they cross, and dense ` +
  `auto-placement backfills the holes the promoted panel leaves behind. `
).split(' ')

function TokenStream({ height, t, width }: { height: number; t: Theme; width: number }) {
  const tick = useTick(90)
  const wordCount = tick % (TOKEN_WORDS.length * 4)
  // Keep roughly enough trailing text to fill the cell, on word boundaries.
  const budget = Math.max(16, width * height)
  const words: string[] = []
  let used = 0

  for (let i = wordCount; i >= 0 && used < budget; i--) {
    const word = TOKEN_WORDS[i % TOKEN_WORDS.length]!

    words.unshift(word)
    used += word.length + 1
  }

  return (
    <Text color={t.color.text} wrap="wrap">
      {words.join(' ')}
      <Text color={t.color.primary}>▌</Text>
    </Text>
  )
}

function SparkStream({
  color,
  format,
  height,
  interval,
  sample,
  t,
  width
}: {
  color: string
  format: (value: number) => string
  height: number
  interval: number
  sample: () => number
  t: Theme
  width: number
}) {
  const tick = useTick(interval)
  const history = useHistory(tick, sample)
  const current = history[history.length - 1] ?? 0
  // Chart height grows with the cell — a promoted cell gets a taller chart.
  const rows = Math.max(1, height - 1)

  return (
    <Box flexDirection="column">
      <Text color={t.color.muted} wrap="truncate">
        {format(current)}
      </Text>

      {sparkRows(history, width, rows).map((line, idx) => (
        <Text color={color} key={idx} wrap="truncate">
          {line}
        </Text>
      ))}
    </Box>
  )
}

const FAKE_TOOLS = [
  '┊ read_file src/app/chat.tsx',
  '┊ terminal npm run typecheck',
  '┊ search_files "GridAreas"',
  '┊ patch ui-tui/src/lib/widgetGrid.ts',
  '┊ web_search yoga absolute layout',
  '┊ delegate_task refactor sparkline',
  '┊ terminal scripts/run_tests.sh',
  '┊ vision_analyze screenshot.png'
]

function ToolTicker({ height, t }: { height: number; t: Theme; width: number }) {
  const tick = useTick(600)
  const visible = Math.max(1, height)

  const lines = Array.from({ length: Math.min(visible, tick + 1) }, (_, idx) => {
    const line = FAKE_TOOLS[(tick - idx) % FAKE_TOOLS.length]!

    return { line, recent: idx === 0 }
  }).reverse()

  return (
    <Box flexDirection="column">
      {lines.map(({ line, recent }, idx) => (
        <Text color={recent ? t.color.text : t.color.muted} key={idx} wrap="truncate">
          {line}
        </Text>
      ))}
    </Box>
  )
}

function MetaPanel({ t }: { height: number; t: Theme; width: number }) {
  const tick = useTick(1000)
  const startedRef = useRef(Date.now())
  const uptime = Math.floor((Date.now() - startedRef.current) / 1000)

  return (
    <Box flexDirection="column">
      <Text color={t.color.text}>{new Date().toLocaleTimeString()}</Text>
      <Text color={t.color.muted}>{`up ${Math.floor(uptime / 60)}m ${uptime % 60}s`}</Text>
      <Text color={t.color.muted}>{`${tick} ticks`}</Text>
    </Box>
  )
}

// ── stream registry ─────────────────────────────────────────────────────────

const sineSampler = () => (Math.sin(Date.now() / 700) + 1) / 2 + Math.random() * 0.15

let walkValue = 0.5

const walkSampler = () => {
  walkValue = Math.min(1, Math.max(0, walkValue + (Math.random() - 0.5) * 0.2))

  return walkValue
}

/** Exported so tests can assert the count matches GRID_STREAM_COUNT (focus wraps mod it). */
export const STREAM_DEFS: StreamDef[] = [
  {
    id: 'tokens',
    render: ({ height, t, width }) => <TokenStream height={height} t={t} width={width} />,
    title: 'token stream'
  },
  {
    id: 'throughput',
    render: ({ height, t, width }) => (
      <SparkStream
        color={t.color.accent}
        format={v => `${Math.round(20 + v * 40)} tok/s`}
        height={height}
        interval={150}
        sample={sineSampler}
        t={t}
        width={width}
      />
    ),
    title: 'throughput'
  },
  {
    id: 'heap',
    render: ({ height, t, width }) => (
      <SparkStream
        color={t.color.ok}
        format={v => `heap ${(v / 1024 / 1024).toFixed(1)} MB`}
        height={height}
        interval={500}
        sample={() => process.memoryUsage().heapUsed}
        t={t}
        width={width}
      />
    ),
    title: 'memory'
  },
  {
    id: 'latency',
    render: ({ height, t, width }) => (
      <SparkStream
        color={t.color.warn}
        format={v => `${Math.round(20 + v * 180)} ms`}
        height={height}
        interval={250}
        sample={walkSampler}
        t={t}
        width={width}
      />
    ),
    title: 'latency'
  },
  {
    id: 'tools',
    render: ({ height, t, width }) => <ToolTicker height={height} t={t} width={width} />,
    title: 'tool feed'
  },
  {
    id: 'meta',
    render: ({ height, t, width }) => <MetaPanel height={height} t={t} width={width} />,
    title: 'session'
  }
]

// ── the demo surface ────────────────────────────────────────────────────────

function StreamPanel({
  cell,
  focused,
  main,
  t,
  title,
  children
}: {
  cell: GridAreaCell
  children: (inner: { height: number; t: Theme; width: number }) => ReactNode
  focused: boolean
  main: boolean
  t: Theme
  title: string
}) {
  const innerWidth = Math.max(1, cell.width - 4)
  const innerHeight = Math.max(1, cell.height - 3)
  const borderColor = focused ? t.color.primary : main ? t.color.accent : t.color.border

  return (
    <Box
      borderColor={borderColor}
      borderStyle="round"
      flexDirection="column"
      height={cell.height}
      paddingX={1}
      width={cell.width}
    >
      <Text bold={focused} color={focused ? t.color.primary : t.color.label} wrap="truncate">
        {focused ? '▸ ' : '  '}
        {title}
        {main ? ' ·' : ''}
      </Text>

      <Box flexDirection="column" height={innerHeight} overflow="hidden" width={innerWidth}>
        {children({ height: innerHeight, t, width: innerWidth })}
      </Box>
    </Box>
  )
}

export const GridStreamsDemo = memo(function GridStreamsDemo({
  cols,
  state,
  t
}: {
  cols: number
  state: GridTestState
  t: Theme
}) {
  const columnTracks: GridTrackSize[] = [{ fr: 1 }, { fr: 1 }, { fr: 1 }]
  const main = STREAM_DEFS[state.streamMain % STREAM_DEFS.length]!

  // Promoted panel first so dense placement gives it the top-left 2x2; the
  // rest backfill in definition order. Ids never change, so React reconciles
  // each panel across promotions and its ticking state survives.
  const ordered = [main, ...STREAM_DEFS.filter(def => def.id !== main.id)]

  const widgets: GridAreaWidget[] = [
    {
      colSpan: 3,
      id: 'stream-header',
      render: cell => (
        <Box
          alignItems="center"
          borderColor={t.color.border}
          borderStyle="round"
          height={cell.height}
          justifyContent="space-between"
          paddingX={1}
          width={cell.width}
        >
          <Text bold color={t.color.primary}>
            hermes mission control
          </Text>
          <Text color={t.color.muted}>{`main: ${main.title}`}</Text>
        </Box>
      )
    },
    ...ordered.map((def, idx) => ({
      colSpan: idx === 0 ? 2 : 1,
      id: `stream-${def.id}`,
      render: (cell: GridAreaCell) => (
        <StreamPanel
          cell={cell}
          focused={STREAM_DEFS[state.streamFocus % STREAM_DEFS.length]!.id === def.id}
          main={def.id === main.id}
          t={t}
          title={def.title}
        >
          {def.render}
        </StreamPanel>
      ),
      rowSpan: idx === 0 ? 2 : 1
    }))
  ]

  return (
    <GridAreas
      columns={columnTracks}
      gap={1}
      height={GRID_HEIGHT}
      rowGap={0}
      rows={[HEADER_ROWS, { fr: 1 }, { fr: 1 }, { fr: 1 }]}
      widgets={widgets}
      width={cols}
    />
  )
})
