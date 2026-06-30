import { Box, NoSelect, ScrollBox, type ScrollBoxHandle, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useMemo, useRef, useState } from 'react'

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

type Mode = 'detail' | 'item' | 'timeline'
type Cell = { color?: string; text: string }

const MAX_CHART_ROWS = 8

const rowText = (row: Run[]) => row.map(run => run[0]).join('')

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

  const palette = useMemo(() => deriveStarmapPalette(t.color.primary, t.color.text), [t.color.primary, t.color.text])

  const [data, setData] = useState<FramesResponse | null>(null)
  const [err, setErr] = useState('')
  const [selectedRow, setSelectedRow] = useState(0)
  const [selectedNode, setSelectedNode] = useState(0)
  const [mode, setMode] = useState<Mode>('timeline')
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
        setSelectedRow(Math.max(0, (r?.buckets?.length ?? 1) - 1))
        setSelectedNode(0)
        setMode('timeline')
      })
      .catch((e: unknown) => alive && setErr(rpcErrorMessage(e)))

    return () => {
      alive = false
    }
  }, [gw, cols, chartRows])

  useEffect(() => setSelectedNode(0), [selectedRow])

  useEffect(() => {
    if (mode === 'item') {
      itemScroll.current?.scrollTo(0)
      setTick(x => x + 1)
    }
  }, [mode, selectedNode])

  const buckets = data?.buckets ?? []
  const selected = buckets.length ? buckets[Math.min(selectedRow, buckets.length - 1)] : null
  const nodes = selected?.nodes ?? []
  const activeNode = nodes[Math.min(selectedNode, Math.max(0, nodes.length - 1))]
  const page = Math.max(4, rows - 6)

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
        return setMode('detail')
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

    if (mode === 'detail') {
      if (back) {
        return setMode('timeline')
      }

      if ((key.return || key.rightArrow || ch === 'l') && nodes.length) {
        return setMode('item')
      }

      if (key.upArrow || ch === 'k') {
        return setSelectedNode(v => Math.max(0, v - 1))
      }

      if (key.downArrow || ch === 'j') {
        return setSelectedNode(v => Math.min(nodes.length - 1, v + 1))
      }

      if (key.pageUp || (key.ctrl && ch === 'u')) {
        return setSelectedNode(v => Math.max(0, v - page))
      }

      if (key.pageDown || (key.ctrl && ch === 'd')) {
        return setSelectedNode(v => Math.min(nodes.length - 1, v + page))
      }

      if (ch === 'g') {
        return setSelectedNode(0)
      }

      if (ch === 'G') {
        return setSelectedNode(Math.max(0, nodes.length - 1))
      }

      return
    }

    if (back) {
      return onClose()
    }

    if ((key.return || key.rightArrow || ch === 'l') && buckets.length) {
      return setMode('detail')
    }

    if (key.upArrow || ch === 'k') {
      return setSelectedRow(v => Math.max(0, v - 1))
    }

    if (key.downArrow || ch === 'j') {
      return setSelectedRow(v => Math.min(buckets.length - 1, v + 1))
    }

    if (ch === 'g') {
      return setSelectedRow(0)
    }

    if (ch === 'G') {
      return setSelectedRow(Math.max(0, buckets.length - 1))
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

  // ── Item: a single skill/memory, body scrolled via the shared ScrollBox ──
  if (mode === 'item' && selected && activeNode) {
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
            {selected.label} · {activeNode.meta} · item {selectedNode + 1}/{nodes.length}
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

  // ── Detail: the slice's skills + memories as a selectable list ──
  if (mode === 'detail' && selected) {
    const h = Math.max(4, rows - 6)
    const start = windowStart(selectedNode, nodes.length, h)

    return (
      <Box alignItems="stretch" flexDirection="column" flexGrow={1} paddingX={1} paddingY={1}>
        <Box flexDirection="column" marginBottom={1}>
          <Text wrap="truncate-end">
            <Text bold color={selected.color ? fadeHex(palette, selected.color, 1) : t.color.primary}>
              {selected.label}
            </Text>
            <Text color={t.color.muted}>
              {'  '}
              {selected.skills} skills · {selected.memories} memories
              {selected.category ? ` · ${selected.category}` : ''} · slice {selected.index + 1}/{buckets.length}
            </Text>
          </Text>
        </Box>

        <Box flexDirection="column" flexGrow={1} flexShrink={1} minHeight={0} overflow="hidden">
          {nodes.length ? (
            nodes.slice(start, start + h).map((node, i) => {
              const idx = start + i

              return (
                <ListRow
                  active={idx === selectedNode}
                  cells={[
                    { color: t.color.muted, text: ` ${String(idx + 1).padStart(2, ' ')} ` },
                    { color: fadeInk(palette, node.style, 1), text: `${node.glyph} ${node.fullLabel || node.label}` },
                    { color: t.color.muted, text: `  ${node.meta}` }
                  ]}
                  key={`${node.label}:${idx}`}
                  t={t}
                />
              )
            })
          ) : (
            <Text color={t.color.muted}>No objects in this slice.</Text>
          )}
        </Box>

        <Footer>
          <Hint t={t}>
            {nodes.length ? `${selectedNode + 1}/${nodes.length} · ` : ''}↑↓/jk move · Enter/→ open · g/G top/bottom · Esc/← back ·
            q close
          </Hint>
        </Footer>
      </Box>
    )
  }

  // ── Timeline: static chart overview + selectable slice list ──
  const axisGap = Math.max(1, cols - 2 - data.axis.start.length - data.axis.end.length)
  const dataGrid = data.frames.at(-1)?.grid.filter(r => !rowText(r).trimStart().startsWith('trajectory')) ?? []
  const chartGrid = dataGrid.slice(-MAX_CHART_ROWS)
  const listH = Math.max(3, rows - chartGrid.length - (data.categories?.length ? 11 : 10))
  const start = windowStart(selectedRow, buckets.length, listH)

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
        <Text color={t.color.muted}>Timeline slices</Text>
        {buckets.slice(start, start + listH).map((bucket, i) => {
          const idx = start + i
          const top = bucket.nodes[0]

          return (
            <ListRow
              active={idx === selectedRow}
              cells={[
                { color: t.color.muted, text: ` ${String(idx + 1).padStart(2, ' ')} ` },
                {
                  color: bucket.color ? fadeHex(palette, bucket.color, 0.85) : t.color.label,
                  text: bucket.label.padEnd(7, ' ')
                },
                {
                  color: t.color.muted,
                  text: ` ${bucket.skills} skills · ${bucket.memories} memories${
                    bucket.category ? ` · ${bucket.category}` : ''
                  }  ${top ? top.fullLabel || top.label : 'empty'}`
                }
              ]}
              key={`${bucket.label}:${idx}`}
              t={t}
            />
          )
        })}
      </Box>

      <Footer>
        {data.summary.length ? <Hint t={t}>{data.summary.join(' · ')}</Hint> : null}
        <Hint t={t}>↑↓/jk move · Enter/→ open · g/G top/bottom · q close</Hint>
      </Footer>
    </Box>
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
