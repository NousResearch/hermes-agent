import { Box, NoSelect, ScrollBox, type ScrollBoxHandle, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useRef, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import { deriveStarmapPalette, fadeHex, fadeInk, type StarmapPalette } from '../lib/starmapPalette.js'
import type { Theme } from '../theme.js'

import { OverlayScrollbar } from './overlayScrollbar.js'

// A run is [text, styleKey, alpha?, hexOverride?] from learning_graph_render.py.
type Run = [string, string, number?, (string | null)?]

interface LegendItem {
  color?: string
  glyph: string
  label: string
  style?: string
}

interface BucketNode {
  body?: string
  fullLabel?: string
  glyph: string
  label: string
  meta: string
  style: string
}

interface BucketRow {
  category?: string | null
  color?: string | null
  date: string
  index: number
  label: string
  memories: number
  nodes: BucketNode[]
  skills: number
}

interface FramesResponse {
  axis: { end: string; start: string }
  buckets?: BucketRow[]
  categories?: LegendItem[]
  count: number
  frames: { grid: Run[][] }[]
  legend: LegendItem[]
  summary: string[]
}

interface JourneyProps {
  gw: GatewayClient
  onClose: () => void
  t: Theme
}

// Flattened timeline tree: a slice header followed by its chronological items.
type TreeRow =
  | { bucket: BucketRow; kind: 'slice' }
  | { bucket: BucketRow; kind: 'node'; last: boolean; node: BucketNode }

type Cell = { color?: string; text: string }

const MAX_CHART_ROWS = 8

const rowText = (row: Run[]) => row.map(run => run[0]).join('')

const treeLength = (buckets: BucketRow[]) => buckets.reduce((n, b) => n + 1 + b.nodes.length, 0)

const buildTree = (buckets: BucketRow[]): TreeRow[] => {
  const out: TreeRow[] = []

  for (const bucket of buckets) {
    out.push({ bucket, kind: 'slice' })
    bucket.nodes.forEach((node, j) => out.push({ bucket, kind: 'node', last: j === bucket.nodes.length - 1, node }))
  }

  return out
}

// Center a fixed-height window on the cursor, clamped to list bounds.
const windowStart = (cursor: number, len: number, h: number) =>
  Math.max(0, Math.min(Math.max(0, len - h), cursor - Math.floor(h / 2)))

function ChartRow({ palette, row }: { palette: StarmapPalette; row: Run[] }) {
  if (!row.length) {
    return <Text> </Text>
  }

  return (
    <Text>
      {row.map((run, i) => (
        <Text color={run[3] ? fadeHex(palette, run[3], run[2] ?? 1) : fadeInk(palette, run[1], run[2] ?? 1)} key={i}>
          {run[0]}
        </Text>
      ))}
    </Text>
  )
}

// Full-width selectable row, matching the /agents list treatment: the active
// row inverts and collapses every segment onto the accent foreground.
function ListRow({ active, cells, t }: { active: boolean; cells: Cell[]; t: Theme }) {
  const fg = active ? t.color.accent : t.color.text

  return (
    <Text bold={active} color={fg} inverse={active} wrap="truncate-end">
      {cells.map((c, i) => (
        <Text color={active ? fg : (c.color ?? t.color.text)} key={i}>
          {c.text}
        </Text>
      ))}
    </Text>
  )
}

export function Journey({ gw, onClose, t }: JourneyProps) {
  const { stdout } = useStdout()
  const cols = Math.max(40, (stdout?.columns ?? 90) - 3)
  const rows = Math.max(16, (stdout?.rows ?? 30) - 2)
  const chartRows = Math.max(5, Math.min(MAX_CHART_ROWS, Math.floor(rows * 0.32)))
  const page = Math.max(4, rows - 6)

  const palette = deriveStarmapPalette(t.color.primary, t.color.text)

  const [data, setData] = useState<FramesResponse | null>(null)
  const [err, setErr] = useState('')
  const [cursor, setCursor] = useState(0)
  const [mode, setMode] = useState<'item' | 'timeline'>('timeline')
  const [tick, setTick] = useState(0)
  const itemScroll = useRef<null | ScrollBoxHandle>(null)

  // The renderer is size-aware, so refetch when the terminal resizes.
  useEffect(() => {
    let alive = true
    setData(null)
    setErr('')

    gw.request<FramesResponse>('learning.frames', { cols, frames: 2, rows: chartRows })
      .then(r => {
        if (!alive) {
          return
        }

        setData(r)
        setCursor(Math.max(0, treeLength(r?.buckets ?? []) - 1)) // open on the newest entry
        setMode('timeline')
      })
      .catch((e: unknown) => alive && setErr(rpcErrorMessage(e)))

    return () => {
      alive = false
    }
  }, [gw, cols, chartRows])

  const tree = buildTree(data?.buckets ?? [])
  const activeRow = tree[Math.min(cursor, Math.max(0, tree.length - 1))]
  const activeNode = activeRow?.kind === 'node' ? activeRow.node : undefined
  const activeBucket = activeRow?.bucket

  useEffect(() => {
    if (mode === 'item') {
      itemScroll.current?.scrollTo(0)
      setTick(x => x + 1)
    }
  }, [mode, cursor])

  const scrollItem = (dy: number) => {
    itemScroll.current?.scrollBy(dy)
    setTick(x => x + 1)
  }

  useInput((ch, key) => {
    const back = key.escape || key.leftArrow || ch === 'h'

    if (ch === 'q') {
      return onClose()
    }

    if (mode === 'item') {
      if (back) {
        return setMode('timeline')
      }

      if (key.upArrow || ch === 'k') {
        return scrollItem(-2)
      }

      if (key.downArrow || ch === 'j') {
        return scrollItem(2)
      }

      if (key.pageUp || (key.ctrl && ch === 'u')) {
        return scrollItem(-page)
      }

      if (key.pageDown || (key.ctrl && ch === 'd') || ch === ' ') {
        return scrollItem(page)
      }

      if (ch === 'g') {
        itemScroll.current?.scrollTo(0)

        return setTick(x => x + 1)
      }

      if (ch === 'G') {
        itemScroll.current?.scrollToBottom()

        return setTick(x => x + 1)
      }

      return
    }

    if (back) {
      return onClose()
    }

    // Only memories carry body text; everything else is already fully shown
    // inline in the tree, so don't drill into an empty page.
    if ((key.return || key.rightArrow || ch === 'l') && activeNode?.body) {
      return setMode('item')
    }

    if (key.upArrow || ch === 'k') {
      return setCursor(v => Math.max(0, v - 1))
    }

    if (key.downArrow || ch === 'j') {
      return setCursor(v => Math.min(tree.length - 1, v + 1))
    }

    if (key.pageUp || (key.ctrl && ch === 'u')) {
      return setCursor(v => Math.max(0, v - page))
    }

    if (key.pageDown || (key.ctrl && ch === 'd')) {
      return setCursor(v => Math.min(tree.length - 1, v + page))
    }

    if (ch === 'g') {
      return setCursor(0)
    }

    if (ch === 'G') {
      return setCursor(Math.max(0, tree.length - 1))
    }
  })

  if (err) {
    return (
      <Shell t={t}>
        <Text color={t.color.error}>error: {err}</Text>
      </Shell>
    )
  }

  if (!data) {
    return (
      <Shell t={t}>
        <Text color={t.color.muted}>assembling your learning map…</Text>
      </Shell>
    )
  }

  if (!data.count) {
    return (
      <Shell t={t}>
        <Text color={t.color.muted}>
          No learning yet — your learned skills and memories will start mapping out here as you use Hermes.
        </Text>
      </Shell>
    )
  }

  // ── Item: a single memory, body scrolled via the shared ScrollBox ──
  if (mode === 'item' && activeBucket && activeNode) {
    const body = activeNode.body ? activeNode.body.split(/\r?\n/) : ['No additional detail recorded yet.']

    return (
      <Box alignItems="stretch" flexDirection="column" flexGrow={1} paddingX={1} paddingY={1}>
        <Box flexDirection="column" marginBottom={1}>
          <Text wrap="truncate-end">
            <Text bold color={fadeInk(palette, activeNode.style, 1)}>
              {activeNode.glyph} {activeNode.fullLabel || activeNode.label}
            </Text>
          </Text>
          <Text color={t.color.muted}>
            {activeBucket.label} · {activeNode.meta}
          </Text>
        </Box>

        <Box flexDirection="row" flexGrow={1} flexShrink={1} minHeight={0}>
          <ScrollBox flexDirection="column" flexGrow={1} flexShrink={1} ref={itemScroll}>
            <Box flexDirection="column" paddingBottom={2} paddingRight={1}>
              {body.map((line, i) => (
                <Text color={t.color.text} key={i} wrap="wrap">
                  {line || ' '}
                </Text>
              ))}
            </Box>
          </ScrollBox>
          <NoSelect flexShrink={0} marginLeft={1}>
            <OverlayScrollbar scrollRef={itemScroll} t={t} tick={tick} />
          </NoSelect>
        </Box>

        <Footer>
          <Hint t={t}>↑↓/jk scroll · PgUp/PgDn page · g/G top/bottom · Esc/← back · q close</Hint>
        </Footer>
      </Box>
    )
  }

  // ── Timeline: static chart overview + a chronological slice/item tree ──
  const axisGap = Math.max(1, cols - 2 - data.axis.start.length - data.axis.end.length)
  const dataGrid = data.frames.at(-1)?.grid.filter(r => !rowText(r).trimStart().startsWith('trajectory')) ?? []
  const chartGrid = dataGrid.slice(-MAX_CHART_ROWS)
  const listH = Math.max(3, rows - chartGrid.length - (data.categories?.length ? 11 : 10))
  const start = windowStart(cursor, tree.length, listH)

  return (
    <Box alignItems="stretch" flexDirection="column" flexGrow={1} paddingX={1} paddingY={1}>
      <Box flexDirection="column" marginBottom={1}>
        <Text wrap="truncate-end">
          <Text bold color={t.color.primary}>
            ✦ Journey
          </Text>
          <Text color={t.color.muted}>  learned skills &amp; memories over time</Text>
        </Text>
        <Text wrap="wrap">
          {data.legend.map((item, i) => (
            <Text key={item.label}>
              {i ? '   ' : ''}
              <Text color={fadeInk(palette, item.style ?? 'dim', 1)}>{item.glyph} </Text>
              <Text color={t.color.muted}>{item.label}</Text>
            </Text>
          ))}
        </Text>
        {data.categories?.length ? (
          <Text wrap="wrap">
            {data.categories.map((item, i) => (
              <Text key={item.label}>
                {i ? '  ' : ''}
                <Text color={item.color ? fadeHex(palette, item.color, 1) : t.color.muted}>{item.glyph} </Text>
                <Text color={t.color.muted}>{item.label}</Text>
              </Text>
            ))}
          </Text>
        ) : null}
      </Box>

      <Box flexDirection="column" marginBottom={1}>
        {chartGrid.map((row, i) => (
          <ChartRow key={i} palette={palette} row={row} />
        ))}
        <Text color={t.color.muted}>
          {data.axis.start}
          {' '.repeat(axisGap)}
          {data.axis.end}
        </Text>
      </Box>

      <Box flexDirection="column" flexGrow={1} flexShrink={1} minHeight={0} overflow="hidden">
        {tree.slice(start, start + listH).map((row, i) => (
          <TreeLine active={start + i === cursor} key={start + i} palette={palette} row={row} t={t} />
        ))}
      </Box>

      <Footer>
        {data.summary.length ? <Hint t={t}>{data.summary.join(' · ')}</Hint> : null}
        <Hint t={t}>
          ↑↓/jk move{activeNode?.body ? ' · Enter/→ open' : ''} · g/G top/bottom · q close
        </Hint>
      </Footer>
    </Box>
  )
}

function TreeLine({ active, palette, row, t }: { active: boolean; palette: StarmapPalette; row: TreeRow; t: Theme }) {
  if (row.kind === 'slice') {
    const { bucket } = row

    return (
      <ListRow
        active={active}
        cells={[
          { color: bucket.color ? fadeHex(palette, bucket.color, 0.85) : t.color.label, text: bucket.label },
          {
            color: t.color.muted,
            text: ` · ${bucket.skills} skills · ${bucket.memories} memories${bucket.category ? ` · ${bucket.category}` : ''}`
          }
        ]}
        t={t}
      />
    )
  }

  const { last, node } = row

  return (
    <ListRow
      active={active}
      cells={[
        { color: t.color.muted, text: ` ${last ? '└─' : '├─'} ` },
        { color: fadeInk(palette, node.style, 1), text: `${node.glyph} ${node.fullLabel || node.label}` },
        { color: t.color.muted, text: `  ${node.meta}${node.body ? '  ›' : ''}` }
      ]}
      t={t}
    />
  )
}

function Shell({ children, t }: { children: React.ReactNode; t: Theme }) {
  return (
    <Box flexDirection="column" paddingX={1} paddingY={1}>
      <Text bold color={t.color.primary}>
        ✦ Journey
      </Text>
      {children}
      <Text color={t.color.muted}>Esc/q close</Text>
    </Box>
  )
}

function Footer({ children }: { children: React.ReactNode }) {
  return (
    <Box flexDirection="column" marginTop={1}>
      {children}
    </Box>
  )
}

function Hint({ children, t }: { children: React.ReactNode; t: Theme }) {
  return (
    <Text color={t.color.muted} wrap="truncate-end">
      {children}
    </Text>
  )
}
