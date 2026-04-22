import { Box, NoSelect, ScrollBox, type ScrollBoxHandle, Text, useInput, useStdout } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { type ReactNode, type RefObject, useEffect, useMemo, useRef, useState } from 'react'

import {
  $delegationState,
  $overlaySectionsOpen,
  applyDelegationStatus,
  toggleOverlaySection
} from '../app/delegationStore.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { $spawnDiff, $spawnHistory, clearDiffPair, type SpawnSnapshot } from '../app/spawnHistoryStore.js'
import { $turnState } from '../app/turnStore.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { DelegationPauseResponse, DelegationStatusResponse, SubagentInterruptResponse } from '../gatewayTypes.js'
import { asRpcResult } from '../lib/rpc.js'
import {
  buildSubagentTree,
  descendantIds,
  flattenTree,
  fmtCost,
  fmtTokens,
  formatSummary,
  hotnessBucket,
  peakHotness,
  sparkline,
  treeTotals,
  widthByDepth
} from '../lib/subagentTree.js'
import { compactPreview } from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { SubagentNode, SubagentProgress } from '../types.js'

// ── Types + lookup tables ────────────────────────────────────────────

type SortMode = 'depth-first' | 'duration-desc' | 'status' | 'tools-desc'
type FilterMode = 'all' | 'failed' | 'leaf' | 'running'
type Status = SubagentProgress['status']

const SORT_ORDER: readonly SortMode[] = ['depth-first', 'tools-desc', 'duration-desc', 'status']
const FILTER_ORDER: readonly FilterMode[] = ['all', 'running', 'failed', 'leaf']

const SORT_LABEL: Record<SortMode, string> = {
  'depth-first': 'spawn order',
  'duration-desc': 'slowest',
  status: 'status',
  'tools-desc': 'busiest'
}

const FILTER_LABEL: Record<FilterMode, string> = {
  all: 'all',
  failed: 'failed',
  leaf: 'leaves',
  running: 'running'
}

const STATUS_RANK: Record<Status, number> = {
  failed: 0,
  interrupted: 1,
  running: 2,
  queued: 3,
  completed: 4
}

const SORT_COMPARATORS: Record<SortMode, (a: SubagentNode, b: SubagentNode) => number> = {
  'depth-first': (a, b) => a.item.depth - b.item.depth || a.item.index - b.item.index,
  'tools-desc': (a, b) => b.aggregate.totalTools - a.aggregate.totalTools,
  'duration-desc': (a, b) => b.aggregate.totalDuration - a.aggregate.totalDuration,
  status: (a, b) => STATUS_RANK[a.item.status] - STATUS_RANK[b.item.status]
}

const FILTER_PREDICATES: Record<FilterMode, (n: SubagentNode) => boolean> = {
  all: () => true,
  leaf: n => n.children.length === 0,
  running: n => n.item.status === 'running' || n.item.status === 'queued',
  failed: n => n.item.status === 'failed' || n.item.status === 'interrupted'
}

const STATUS_GLYPH: Record<Status, { color: (t: Theme) => string; glyph: string }> = {
  running: { color: t => t.color.amber, glyph: '●' },
  queued: { color: t => t.color.dim, glyph: '○' },
  completed: { color: t => t.color.statusGood, glyph: '✓' },
  interrupted: { color: t => t.color.warn, glyph: '■' },
  failed: { color: t => t.color.error, glyph: '✗' }
}

// Heatmap palette — cold → hot, resolved against the active theme.
const heatPalette = (t: Theme) => [t.color.bronze, t.color.amber, t.color.gold, t.color.warn, t.color.error]

// ── Pure helpers ─────────────────────────────────────────────────────

const fmtDur = (seconds?: number): string => {
  if (!seconds || seconds <= 0) {
    return ''
  }

  if (seconds < 60) {
    return `${Math.round(seconds)}s`
  }

  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds - m * 60)

  return s === 0 ? `${m}m` : `${m}m ${s}s`
}

/** Server duration if present; else live edge from `startedAt` (running / queued). */
const displayElapsedSeconds = (item: SubagentProgress, nowMs: number): number | null => {
  if (item.durationSeconds != null) {
    return item.durationSeconds
  }

  if (item.startedAt != null && (item.status === 'running' || item.status === 'queued')) {
    return Math.max(0, (nowMs - item.startedAt) / 1000)
  }

  return null
}

/** Like fmtDur but allows 0s for just-started / still-running rows. */
const fmtElapsedLabel = (seconds: number): string => {
  if (seconds < 0) {
    return ''
  }

  if (seconds < 60) {
    return `${Math.max(0, Math.round(seconds))}s`
  }

  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds - m * 60)

  return s === 0 ? `${m}m` : `${m}m ${s}s`
}

const indentFor = (depth: number): string => '  '.repeat(Math.max(0, depth))
const formatRowId = (n: number): string => String(n + 1).padStart(2, ' ')
const cycle = <T,>(order: readonly T[], current: T): T => order[(order.indexOf(current) + 1) % order.length]!

const statusGlyph = (item: SubagentProgress, t: Theme) => {
  const g = STATUS_GLYPH[item.status]

  return { color: g.color(t), glyph: g.glyph }
}

const prepareRows = (tree: SubagentNode[], sort: SortMode, filter: FilterMode): SubagentNode[] =>
  tree.length === 0
    ? []
    : [...tree]
        .sort(SORT_COMPARATORS[sort])
        .flatMap(n => flattenTree([n]))
        .filter(FILTER_PREDICATES[filter])

// ── Sub-components ───────────────────────────────────────────────────

/**
 * Detail-pane scrollbar, polled on the parent tick.  `TranscriptScrollbar`
 * re-renders only on scroll events — fine for the main transcript, but the
 * overlay's content reflows on accordion toggle without any scroll, so the
 * thumb stays stale.  Ticking forces a re-read; always drawing the track
 * keeps the gutter visually stable for short content too.
 */
function OverlayScrollbar({
  scrollRef,
  t,
  tick
}: {
  scrollRef: RefObject<null | ScrollBoxHandle>
  t: Theme
  tick: number
}) {
  void tick // ensures re-render when the parent clock advances

  const [hover, setHover] = useState(false)
  const [grab, setGrab] = useState<null | number>(null)

  const s = scrollRef.current
  const vp = Math.max(0, s?.getViewportHeight() ?? 0)

  if (!vp) {
    return <Box width={1} />
  }

  const total = Math.max(vp, s?.getScrollHeight() ?? vp)
  const scrollable = total > vp
  const thumb = scrollable ? Math.max(1, Math.round((vp * vp) / total)) : vp
  const travel = Math.max(1, vp - thumb)
  const pos = Math.max(0, (s?.getScrollTop() ?? 0) + (s?.getPendingDelta() ?? 0))
  const thumbTop = scrollable ? Math.round((pos / Math.max(1, total - vp)) * travel) : 0
  const below = Math.max(0, vp - thumbTop - thumb)

  const trackLines = (n: number) => (n > 0 ? `${'│\n'.repeat(Math.max(0, n - 1))}│` : '')
  const thumbLines = `${'┃\n'.repeat(Math.max(0, thumb - 1))}┃`

  const thumbColor = grab !== null ? t.color.gold : hover ? t.color.amber : t.color.amber
  const trackColor = hover ? t.color.bronze : t.color.dim

  // Map a local row (0..vp-1) + grab offset to a scrollTop position.
  const jump = (row: number, offset: number) => {
    if (!s || !scrollable) {
      return
    }

    s.scrollTo(Math.round((Math.max(0, Math.min(travel, row - offset)) / travel) * Math.max(0, total - vp)))
  }

  return (
    <Box
      flexDirection="column"
      onMouseDown={(e: { localRow?: number }) => {
        const row = Math.max(0, Math.min(vp - 1, e.localRow ?? 0))
        const off = row >= thumbTop && row < thumbTop + thumb ? row - thumbTop : Math.floor(thumb / 2)
        setGrab(off)
        jump(row, off)
      }}
      onMouseDrag={(e: { localRow?: number }) =>
        jump(Math.max(0, Math.min(vp - 1, e.localRow ?? 0)), grab ?? Math.floor(thumb / 2))
      }
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onMouseUp={() => setGrab(null)}
      width={1}
    >
      {!scrollable ? (
        <Text color={trackColor} dim>
          {trackLines(vp)}
        </Text>
      ) : (
        <>
          {thumbTop > 0 ? (
            <Text color={trackColor} dim={!hover}>
              {trackLines(thumbTop)}
            </Text>
          ) : null}

          <Text color={thumbColor}>{thumbLines}</Text>

          {below > 0 ? (
            <Text color={trackColor} dim={!hover}>
              {trackLines(below)}
            </Text>
          ) : null}
        </>
      )}
    </Box>
  )
}

/**
 * Horizontal ASCII Gantt strip.  One bar per subagent, anchored by row id.
 * The ruler below maps screen positions to wall-clock seconds so a bar that
 * "ends in the middle" reads as "finished at ~Xs".
 */
function GanttStrip({
  cols,
  cursor,
  flatNodes,
  maxRows,
  now,
  t
}: {
  cols: number
  cursor: number
  flatNodes: SubagentNode[]
  maxRows: number
  now: number
  t: Theme
}) {
  const spans = flatNodes
    .map((node, idx) => {
      const started = node.item.startedAt ?? now

      const ended =
        node.item.durationSeconds != null && node.item.startedAt != null
          ? node.item.startedAt + node.item.durationSeconds * 1000
          : now

      return { endAt: ended, idx, node, startAt: started }
    })
    .filter(s => s.endAt >= s.startAt)

  if (!spans.length) {
    return null
  }

  const globalStart = Math.min(...spans.map(s => s.startAt))
  const globalEnd = Math.max(...spans.map(s => s.endAt))
  const totalSpan = Math.max(1, globalEnd - globalStart)
  const totalSeconds = (globalEnd - globalStart) / 1000

  // 5-col id gutter ("  12  ") so the bar doesn't press against the id.
  // 10-col right reserve: pad + up to `12m 30s`-style label without
  // truncate-end against a full-width bar.
  const idGutter = 5
  const labelReserve = 10
  const barWidth = Math.max(10, cols - idGutter - labelReserve)
  const startIdx = Math.max(0, Math.min(Math.max(0, spans.length - maxRows), cursor - Math.floor(maxRows / 2)))
  const shown = spans.slice(startIdx, startIdx + maxRows)

  const bar = (startAt: number, endAt: number) => {
    const s = Math.floor(((startAt - globalStart) / totalSpan) * barWidth)
    const e = Math.min(barWidth, Math.ceil(((endAt - globalStart) / totalSpan) * barWidth))
    const fill = Math.max(1, e - s)

    return ' '.repeat(s) + '█'.repeat(fill) + ' '.repeat(Math.max(0, barWidth - s - fill))
  }

  // Wall-clock axis: more ticks on short windows so the scale visibly
  // “counts up” with `now` instead of a single 0/10s pair.
  const charStep = totalSeconds < 20 && barWidth > 20 ? 5 : 10

  const ruler = Array.from({ length: barWidth }, (_, i) => {
    if (i > 0 && i % 10 === 0) {
      return '┼'
    }

    if (i > 0 && i % 5 === 0) {
      return '·'
    }

    return '─'
  }).join('')

  const rulerLabels = (() => {
    const chars = new Array(barWidth).fill(' ')

    for (let pos = 0; pos < barWidth; pos += charStep) {
      const secs = (pos / barWidth) * totalSeconds
      const label = pos === 0 ? '0' : secs >= 1 ? `${Math.round(secs)}s` : `${secs.toFixed(1)}s`

      for (let j = 0; j < label.length && pos + j < barWidth; j++) {
        chars[pos + j] = label[j]!
      }
    }

    return chars.join('')
  })()

  const windowLabel =
    spans.length > maxRows ? `  (${startIdx + 1}-${Math.min(spans.length, startIdx + maxRows)}/${spans.length})` : ''

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text color={t.color.dim}>
        Timeline · {fmtElapsedLabel(Math.max(0, totalSeconds))}
        {windowLabel}
      </Text>

      {shown.map(({ endAt, idx, node, startAt }) => {
        const active = idx === cursor
        const { color } = statusGlyph(node.item, t)
        const accent = active ? t.color.amber : t.color.dim

        const elSec = displayElapsedSeconds(node.item, now)
        const elLabel = elSec != null ? fmtElapsedLabel(elSec) : ''

        return (
          <Text key={node.item.id} wrap="truncate-end">
            <Text bold={active} color={accent}>
              {formatRowId(idx)}
              {'  '}
            </Text>

            <Text color={active ? t.color.amber : color}>{bar(startAt, endAt)}</Text>

            {elLabel ? (
              <Text color={accent}>
                {'   '}
                {elLabel}
              </Text>
            ) : null}
          </Text>
        )
      })}

      <Text color={t.color.dim} dim>
        {'    '}
        {ruler}
      </Text>

      {totalSeconds > 0 ? (
        <Text color={t.color.dim} dim>
          {'    '}
          {rulerLabels}
        </Text>
      ) : null}
    </Box>
  )
}

/**
 * A collapsible section.  Open-state lives on a shared atom so navigating
 * between agents / list ↔ detail / history doesn't reset accordions.
 */
function OverlaySection({
  children,
  count,
  defaultOpen = false,
  title,
  t
}: {
  children: ReactNode
  count?: number
  defaultOpen?: boolean
  title: string
  t: Theme
}) {
  const openMap = useStore($overlaySectionsOpen)
  const open = title in openMap ? openMap[title]! : defaultOpen

  return (
    <Box flexDirection="column" marginTop={1}>
      <Box onClick={() => toggleOverlaySection(title, defaultOpen)}>
        <Text color={t.color.label}>
          <Text color={t.color.amber}>{open ? '▾ ' : '▸ '}</Text>
          {title}
          {typeof count === 'number' ? ` (${count})` : ''}
        </Text>
      </Box>

      {open ? <Box flexDirection="column">{children}</Box> : null}
    </Box>
  )
}

/** `label · value` row with the detail-pane colour hierarchy. */
function Field({ name, t, value }: { name: string; t: Theme; value: ReactNode }) {
  return (
    <Text wrap="truncate-end">
      <Text color={t.color.label}>{name} · </Text>
      <Text color={t.color.cornsilk}>{value}</Text>
    </Text>
  )
}

function Detail({ id, node, t }: { id?: string; node: SubagentNode; t: Theme }) {
  const { aggregate: agg, item } = node
  const { color, glyph } = statusGlyph(item, t)

  const inputTokens = item.inputTokens ?? 0
  const outputTokens = item.outputTokens ?? 0
  const localTokens = inputTokens + outputTokens
  const subtreeTokens = agg.inputTokens + agg.outputTokens - localTokens
  const localCost = item.costUsd ?? 0
  const subtreeCost = agg.costUsd - localCost

  const filesRead = item.filesRead ?? []
  const filesWritten = item.filesWritten ?? []
  const outputTail = item.outputTail ?? []
  // Tool calls: prefer the live stream; for archived / post-turn views
  // that stream is often empty even when tool_count > 0, so fall back to
  // the tool names captured in outputTail at subagent.complete time.
  const toolLines = item.tools.length > 0 ? item.tools : outputTail.map(e => e.tool).filter(Boolean)

  const filesOverflow = Math.max(0, filesRead.length - 8) + Math.max(0, filesWritten.length - 8)

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.cornsilk} wrap="wrap">
        {id ? <Text color={t.color.amber}>#{id} </Text> : null}
        <Text color={color}>{glyph}</Text> {item.goal}
      </Text>

      <Box marginTop={1}>
        <Field name="depth" t={t} value={`${item.depth} · ${item.status}`} />
      </Box>

      {item.model ? <Field name="model" t={t} value={item.model} /> : null}

      {item.toolsets?.length ? <Field name="toolsets" t={t} value={item.toolsets.join(', ')} /> : null}

      <Field name="tools" t={t} value={`${item.toolCount ?? 0} (subtree ${agg.totalTools})`} />

      <Field
        name="subtree"
        t={t}
        value={`${agg.descendantCount} agent${agg.descendantCount === 1 ? '' : 's'} · d${agg.maxDepthFromHere} · ⚡${agg.activeCount}`}
      />

      {item.durationSeconds ? <Field name="elapsed" t={t} value={fmtDur(item.durationSeconds)} /> : null}

      {item.iteration != null ? <Field name="iteration" t={t} value={String(item.iteration)} /> : null}

      {item.apiCalls ? <Field name="api calls" t={t} value={String(item.apiCalls)} /> : null}

      {localTokens > 0 || localCost > 0 ? (
        <OverlaySection defaultOpen t={t} title="Budget">
          {localTokens > 0 ? (
            <Field
              name="tokens"
              t={t}
              value={
                <>
                  {fmtTokens(inputTokens)} in · {fmtTokens(outputTokens)} out
                  {item.reasoningTokens ? ` · ${fmtTokens(item.reasoningTokens)} reasoning` : ''}
                </>
              }
            />
          ) : null}

          {localCost > 0 ? (
            <Field
              name="cost"
              t={t}
              value={
                <>
                  {fmtCost(localCost)}
                  {subtreeCost >= 0.01 ? ` · subtree +${fmtCost(subtreeCost)}` : ''}
                </>
              }
            />
          ) : null}

          {subtreeTokens > 0 ? <Field name="subtree tokens" t={t} value={`+${fmtTokens(subtreeTokens)}`} /> : null}
        </OverlaySection>
      ) : null}

      {filesRead.length > 0 || filesWritten.length > 0 ? (
        <OverlaySection count={filesRead.length + filesWritten.length} t={t} title="Files">
          {filesWritten.slice(0, 8).map((p, i) => (
            <Text color={t.color.statusGood} key={`w-${i}`} wrap="truncate-end">
              +{p}
            </Text>
          ))}

          {filesRead.slice(0, 8).map((p, i) => (
            <Text color={t.color.cornsilk} key={`r-${i}`} wrap="truncate-end">
              <Text color={t.color.dim}>·</Text> {p}
            </Text>
          ))}

          {filesOverflow > 0 ? <Text color={t.color.dim}>…+{filesOverflow} more</Text> : null}
        </OverlaySection>
      ) : null}

      {toolLines.length > 0 ? (
        <OverlaySection count={toolLines.length} defaultOpen t={t} title="Tool calls">
          {toolLines.map((line, i) => (
            <Text color={t.color.cornsilk} key={i} wrap="wrap">
              <Text color={t.color.dim}>·</Text> {line}
            </Text>
          ))}
        </OverlaySection>
      ) : null}

      {outputTail.length > 0 ? (
        <OverlaySection count={outputTail.length} defaultOpen t={t} title="Output">
          {outputTail.map((entry, i) => (
            <Text color={entry.isError ? t.color.error : t.color.cornsilk} key={i} wrap="wrap">
              <Text bold color={entry.isError ? t.color.error : t.color.amber}>
                {entry.tool}
              </Text>{' '}
              {entry.preview}
            </Text>
          ))}
        </OverlaySection>
      ) : null}

      {item.notes.length ? (
        <OverlaySection count={item.notes.length} t={t} title="Progress">
          {item.notes.slice(-6).map((line, i) => (
            <Text color={t.color.cornsilk} key={i} wrap="wrap">
              <Text color={t.color.label}>·</Text> {line}
            </Text>
          ))}
        </OverlaySection>
      ) : null}

      {item.summary ? (
        <OverlaySection defaultOpen t={t} title="Summary">
          <Text color={t.color.cornsilk} wrap="wrap">
            {item.summary}
          </Text>
        </OverlaySection>
      ) : null}
    </Box>
  )
}

/** Pluck the label out of `formatToolCall` output: `Read_file("…")` → `Read_file`. */
const latestToolLabel = (tools: readonly string[]): string => {
  const last = tools[tools.length - 1]

  if (!last) {
    return ''
  }

  const paren = last.indexOf('(')

  return (paren > 0 ? last.slice(0, paren) : last).trim()
}

function ListRow({
  active,
  index,
  node,
  peak,
  t,
  width
}: {
  active: boolean
  index: number
  node: SubagentNode
  peak: number
  t: Theme
  width: number
}) {
  const { color, glyph } = statusGlyph(node.item, t)
  const palette = heatPalette(t)
  const heatIdx = hotnessBucket(node.aggregate.hotness, peak, palette.length)
  const heatMarker = heatIdx >= 2 ? palette[heatIdx]! : null

  const goal = compactPreview(node.item.goal || 'subagent', width - 28 - node.item.depth * 2)
  const toolsCount = node.aggregate.totalTools > 0 ? ` ·${node.aggregate.totalTools}t` : ''
  const kids = node.children.length ? ` ·${node.children.length}↓` : ''

  // Running rows replace the moving-number clock (timeline already has it)
  // with the most recent tool label — no per-tick re-render, but changes
  // as activity flows, so the list still conveys motion.
  const current = node.item.status === 'running' ? latestToolLabel(node.item.tools) : ''
  const trailing = current ? ` · ${compactPreview(current, 14)}` : ''

  // Selection pattern mirrors sessionPicker: inverse + amber for contrast
  // across any theme, body stays cornsilk, stats dim.
  const fg = active ? t.color.amber : t.color.cornsilk

  return (
    <Text bold={active} color={fg} inverse={active} wrap="truncate-end">
      {' '}
      <Text color={active ? fg : t.color.dim}>{formatRowId(index)} </Text>
      {indentFor(node.item.depth)}
      {heatMarker ? <Text color={heatMarker}>▍</Text> : null}
      <Text color={active ? fg : color}>{glyph}</Text> {goal}
      <Text color={active ? fg : t.color.dim}>
        {toolsCount}
        {kids}
        {trailing}
      </Text>
    </Text>
  )
}

function DiffPane({
  label,
  snapshot,
  t,
  totals,
  width
}: {
  label: string
  snapshot: SpawnSnapshot
  t: Theme
  totals: ReturnType<typeof treeTotals>
  width: number
}) {
  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.cornsilk}>
        {label}
      </Text>

      <Text color={t.color.dim} wrap="truncate-end">
        {snapshot.label}
      </Text>

      <Box marginTop={1}>
        <Text color={t.color.dim} wrap="truncate-end">
          {formatSummary(totals)}
        </Text>
      </Box>

      <Box flexDirection="column" marginTop={1}>
        {snapshot.subagents
          .filter(s => !s.parentId)
          .slice(0, 8)
          .map(s => {
            const { color, glyph } = statusGlyph(s, t)

            return (
              <Text color={t.color.dim} key={s.id} wrap="truncate-end">
                <Text color={color}>{glyph}</Text> {s.goal || 'subagent'}
              </Text>
            )
          })}
      </Box>
    </Box>
  )
}

function DiffView({
  cols,
  onClose,
  pair,
  t
}: {
  cols: number
  onClose: () => void
  pair: { baseline: SpawnSnapshot; candidate: SpawnSnapshot }
  t: Theme
}) {
  const aTotals = useMemo(() => treeTotals(buildSubagentTree(pair.baseline.subagents)), [pair.baseline])
  const bTotals = useMemo(() => treeTotals(buildSubagentTree(pair.candidate.subagents)), [pair.candidate])
  const paneWidth = Math.floor((cols - 4) / 2)

  useInput((ch, key) => {
    if (key.escape || ch === 'q') {
      onClose()
    }
  })

  const delta = (name: string, a: number, b: number, fmt: (n: number) => string): string => {
    const sign = b - a === 0 ? '' : b > a ? '+' : '-'

    return `${name}: ${fmt(a)} → ${fmt(b)}  (${sign}${fmt(Math.abs(b - a)) || '0'})`
  }

  const round = (n: number) => String(Math.round(n))
  const sumTokens = (x: typeof aTotals) => x.inputTokens + x.outputTokens
  const dollars = (n: number) => fmtCost(n) || '$0.00'

  return (
    <Box flexDirection="column" flexGrow={1} paddingX={1} paddingY={1}>
      <Box flexDirection="column" marginBottom={1}>
        <Text bold color={t.color.bronze}>
          Replay diff
        </Text>
        <Text color={t.color.dim}>baseline vs candidate · esc/q close</Text>
      </Box>

      <Box flexDirection="row" marginBottom={1}>
        <DiffPane label="A · baseline" snapshot={pair.baseline} t={t} totals={aTotals} width={paneWidth} />
        <Box width={2} />
        <DiffPane label="B · candidate" snapshot={pair.candidate} t={t} totals={bTotals} width={paneWidth} />
      </Box>

      <Box flexDirection="column" marginTop={1}>
        <Text bold color={t.color.amber}>
          Δ
        </Text>

        <Text color={t.color.cornsilk}>{delta('agents', aTotals.descendantCount, bTotals.descendantCount, round)}</Text>
        <Text color={t.color.cornsilk}>{delta('tools', aTotals.totalTools, bTotals.totalTools, round)}</Text>
        <Text color={t.color.cornsilk}>
          {delta('depth', aTotals.maxDepthFromHere, bTotals.maxDepthFromHere, round)}
        </Text>
        <Text color={t.color.cornsilk}>
          {delta('duration', aTotals.totalDuration, bTotals.totalDuration, n => `${n.toFixed(1)}s`)}
        </Text>
        <Text color={t.color.cornsilk}>{delta('tokens', sumTokens(aTotals), sumTokens(bTotals), fmtTokens)}</Text>
        <Text color={t.color.cornsilk}>{delta('cost', aTotals.costUsd, bTotals.costUsd, dollars)}</Text>
      </Box>
    </Box>
  )
}

// ── Main overlay ─────────────────────────────────────────────────────

export function AgentsOverlay({ gw, initialHistoryIndex = 0, onClose, t }: AgentsOverlayProps) {
  const turn = useStore($turnState)
  const delegation = useStore($delegationState)
  const history = useStore($spawnHistory)
  const diffPair = useStore($spawnDiff)
  const { stdout } = useStdout()

  // historyIndex === 0: live turn.  1..N pulls the Nth-most-recent archived
  // snapshot.  /replay passes N on open.
  const [historyIndex, setHistoryIndex] = useState(() =>
    Math.max(0, Math.min(history.length, Math.floor(initialHistoryIndex)))
  )

  const [sort, setSort] = useState<SortMode>('depth-first')
  const [filter, setFilter] = useState<FilterMode>('all')
  const [cursor, setCursor] = useState(0)
  const [flash, setFlash] = useState<string>('')
  const [now, setNow] = useState(() => Date.now())
  // cc-style view switching: list = full-width row picker, detail = full-width
  // scrollable pane.  Two panes side-by-side in Ink fought Yoga flex.
  const [mode, setMode] = useState<'detail' | 'list'>('list')

  const detailScrollRef = useRef<null | ScrollBoxHandle>(null)
  const prevLiveCountRef = useRef(turn.subagents.length)

  // ── Derived state ──────────────────────────────────────────────────

  const activeSnapshot = historyIndex > 0 ? history[historyIndex - 1] : null
  // Instant fallback to history[0] the moment the live list clears — avoids
  // a one-frame "no subagents" flash while the auto-follow effect fires.
  const justFinishedSnapshot = historyIndex === 0 && turn.subagents.length === 0 ? (history[0] ?? null) : null
  const effectiveSnapshot = activeSnapshot ?? justFinishedSnapshot
  const replayMode = effectiveSnapshot != null
  const subagents = replayMode ? effectiveSnapshot.subagents : turn.subagents

  const tree = useMemo(() => buildSubagentTree(subagents), [subagents])
  const totals = useMemo(() => treeTotals(tree), [tree])
  const widths = useMemo(() => widthByDepth(tree), [tree])
  const spark = useMemo(() => sparkline(widths), [widths])
  const peak = useMemo(() => peakHotness(tree), [tree])
  const rows = useMemo(() => prepareRows(tree, sort, filter), [tree, sort, filter])

  const selected = rows[cursor] ?? null

  const cols = stdout?.columns ?? 80
  const rowsH = Math.max(8, (stdout?.rows ?? 24) - 10)
  const listWindowStart = Math.max(0, cursor - Math.floor(rowsH / 2))

  // ── Effects ────────────────────────────────────────────────────────

  useEffect(() => {
    // Ticker drives both the live gantt and OverlayScrollbar content-reflow
    // detection.  Slower in replay (nothing's growing) but not stopped
    // because accordions still expand.
    const id = setInterval(() => setNow(Date.now()), replayMode ? 300 : 500)

    return () => clearInterval(id)
  }, [replayMode])

  useEffect(() => {
    // Clamp stale index when history grows/shrinks beneath us.
    if (historyIndex > history.length) {
      setHistoryIndex(history.length)
    }
  }, [history.length, historyIndex])

  useEffect(() => {
    // Auto-follow the just-finished turn onto history[1] so the user isn't
    // dropped into an empty live view.  Fires only when transitioning from
    // "had live subagents" → "live empty" while in live mode.
    const prev = prevLiveCountRef.current
    prevLiveCountRef.current = turn.subagents.length

    if (historyIndex === 0 && prev > 0 && turn.subagents.length === 0 && history.length > 0) {
      setHistoryIndex(1)
      setCursor(0)
      setFlash('turn finished · inspect freely · q to close')
    }
  }, [history.length, historyIndex, turn.subagents.length])

  useEffect(() => {
    // Reset detail scroll on navigation so the top of the new node shows.
    detailScrollRef.current?.scrollTo(0)
  }, [cursor, historyIndex, mode])

  useEffect(() => {
    // Warm caps + paused flag on open.
    gw.request<DelegationStatusResponse>('delegation.status', {})
      .then(r => applyDelegationStatus(asRpcResult<DelegationStatusResponse>(r)))
      .catch(() => {})
  }, [gw])

  useEffect(() => {
    if (cursor >= rows.length) {
      setCursor(Math.max(0, rows.length - 1))
    }
  }, [cursor, rows.length])

  // ── Actions ────────────────────────────────────────────────────────

  const guardLive = (action: () => void) => {
    if (replayMode) {
      setFlash('replay mode — controls disabled')
    } else {
      action()
    }
  }

  const interrupt = (id: string) => gw.request<SubagentInterruptResponse>('subagent.interrupt', { subagent_id: id })

  const killOne = (id: string) =>
    guardLive(() => {
      interrupt(id)
        .then(raw => {
          const r = asRpcResult<SubagentInterruptResponse>(raw)
          setFlash(r?.found ? `killing ${id}` : `not found: ${id}`)
        })
        .catch(() => setFlash(`kill failed: ${id}`))
    })

  const killSubtree = (node: SubagentNode) =>
    guardLive(() => {
      const ids = [node.item.id, ...descendantIds(node)]
      ids.forEach(id => interrupt(id).catch(() => {}))
      setFlash(`killing subtree · ${ids.length} node${ids.length === 1 ? '' : 's'}`)
    })

  const togglePause = () =>
    guardLive(() => {
      gw.request<DelegationPauseResponse>('delegation.pause', { paused: !delegation.paused })
        .then(raw => {
          const r = asRpcResult<DelegationPauseResponse>(raw)
          applyDelegationStatus({ paused: r?.paused })
          setFlash(r?.paused ? 'spawning paused' : 'spawning resumed')
        })
        .catch(() => setFlash('pause failed'))
    })

  const stepHistory = (delta: -1 | 1) =>
    setHistoryIndex(idx => {
      const next = Math.max(0, Math.min(history.length, idx + delta))

      if (next !== idx) {
        setCursor(0)
        setFlash(next === 0 ? 'live turn' : `replay · ${next}/${history.length}`)
      }

      return next
    })

  const closeWithCleanup = () => {
    clearDiffPair()
    onClose()
  }

  // ── Input ──────────────────────────────────────────────────────────

  const detailPageSize = Math.max(4, rowsH - 2)
  const scrollDetail = (dy: number) => detailScrollRef.current?.scrollBy(dy)

  useInput((ch, key) => {
    if (ch === 'q') {
      return closeWithCleanup()
    }

    if (key.escape) {
      return mode === 'detail' ? setMode('list') : closeWithCleanup()
    }

    // Shared actions (both modes).
    if (ch === '<' || ch === '[') {
      return stepHistory(1)
    }

    if (ch === '>' || ch === ']') {
      return stepHistory(-1)
    }

    if (ch === 'p') {
      return togglePause()
    }

    if (ch === 'x' && selected) {
      return killOne(selected.item.id)
    }

    if (ch === 'X' && selected) {
      return killSubtree(selected)
    }

    if (mode === 'detail') {
      if (key.leftArrow || ch === 'h') {
        return setMode('list')
      }

      if (key.pageUp || (key.ctrl && ch === 'u')) {
        return scrollDetail(-detailPageSize)
      }

      if (key.pageDown || (key.ctrl && ch === 'd')) {
        return scrollDetail(detailPageSize)
      }

      // Wheel = smooth pixel scroll; arrows = 2-row nudge.  Overlay's
      // useInput supersedes the global wheel handler so we re-bind here.
      if (key.wheelUp) {
        return scrollDetail(-3)
      }

      if (key.wheelDown) {
        return scrollDetail(3)
      }

      if (key.upArrow || ch === 'k') {
        return scrollDetail(-2)
      }

      if (key.downArrow || ch === 'j') {
        return scrollDetail(2)
      }

      if (ch === 'g') {
        return detailScrollRef.current?.scrollTo(0)
      }

      if (ch === 'G') {
        return detailScrollRef.current?.scrollToBottom?.()
      }

      return
    }

    // List mode.
    if ((key.return || key.rightArrow || ch === 'l') && selected) {
      return setMode('detail')
    }

    if (key.upArrow || ch === 'k' || key.wheelUp) {
      return setCursor(c => Math.max(0, c - 1))
    }

    if (key.downArrow || ch === 'j' || key.wheelDown) {
      return setCursor(c => Math.min(Math.max(0, rows.length - 1), c + 1))
    }

    if (ch === 'g') {
      return setCursor(0)
    }

    if (ch === 'G') {
      return setCursor(Math.max(0, rows.length - 1))
    }

    if (ch === 's') {
      return setSort(m => cycle(SORT_ORDER, m))
    }

    if (ch === 'f') {
      return setFilter(m => cycle(FILTER_ORDER, m))
    }
  })

  // ── Header assembly ────────────────────────────────────────────────

  const mix = Object.entries(
    subagents.reduce<Record<string, number>>((acc, it) => {
      const key = it.model ? it.model.split('/').pop()! : 'inherit'
      acc[key] = (acc[key] ?? 0) + 1

      return acc
    }, {})
  )
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([k, v]) => `${k}×${v}`)
    .join(' · ')

  const capsLabel = delegation.maxSpawnDepth
    ? `caps d${delegation.maxSpawnDepth}/${delegation.maxConcurrentChildren ?? '?'}`
    : ''

  // Single header line — title · metrics.  An earlier "title + subtitle"
  // variant wrapped on narrow terminals which looked like the header was
  // rendering twice, and a one-line header makes it obvious at a glance
  // whether the turn is live or finished.
  const title = (() => {
    if (!replayMode || !effectiveSnapshot) {
      return `Spawn tree${delegation.paused ? ' · ⏸ paused' : ''}`
    }

    const at = new Date(effectiveSnapshot.finishedAt).toLocaleTimeString()
    const position = historyIndex > 0 ? `Replay ${historyIndex}/${history.length}` : 'Last turn'

    return `${position} · finished ${at}`
  })()

  const metaLine = [formatSummary(totals), spark, capsLabel, mix ? `· ${mix}` : ''].filter(Boolean).join('  ')

  const controlsHint = replayMode
    ? ' · controls locked'
    : ` · x kill · X subtree · p ${delegation.paused ? 'resume' : 'pause'}`

  // ── Rendering ──────────────────────────────────────────────────────

  if (diffPair) {
    return <DiffView cols={cols} onClose={closeWithCleanup} pair={diffPair} t={t} />
  }

  return (
    <Box alignItems="stretch" flexDirection="column" flexGrow={1} paddingX={1} paddingY={1}>
      <Box flexDirection="column" marginBottom={1}>
        <Text wrap="truncate-end">
          <Text bold color={replayMode ? t.color.bronze : t.color.gold}>
            {title}
          </Text>
          {metaLine ? (
            <Text color={t.color.dim}>
              {'   '}
              {metaLine}
            </Text>
          ) : null}
        </Text>
      </Box>

      {rows.length === 0 ? (
        <Box flexDirection="column" flexGrow={1}>
          <Text color={t.color.dim}>No subagents this turn. Trigger delegate_task to populate the tree.</Text>
        </Box>
      ) : mode === 'list' ? (
        <Box flexDirection="column" flexGrow={1} flexShrink={1} minHeight={0}>
          <GanttStrip cols={cols} cursor={cursor} flatNodes={rows} maxRows={6} now={now} t={t} />

          <Box flexDirection="column" flexGrow={0} flexShrink={0} overflow="hidden">
            {rows.slice(listWindowStart, listWindowStart + rowsH).map((node, i) => (
              <ListRow
                active={listWindowStart + i === cursor}
                index={listWindowStart + i}
                key={node.item.id}
                node={node}
                peak={peak}
                t={t}
                width={cols}
              />
            ))}
          </Box>
        </Box>
      ) : (
        <Box flexDirection="row" flexGrow={1} flexShrink={1} minHeight={0}>
          <ScrollBox flexDirection="column" flexGrow={1} flexShrink={1} ref={detailScrollRef}>
            <Box flexDirection="column" paddingBottom={4} paddingRight={1}>
              {selected ? <Detail id={formatRowId(cursor).trim()} node={selected} t={t} /> : null}
            </Box>
          </ScrollBox>

          <NoSelect flexShrink={0} marginLeft={1}>
            <OverlayScrollbar scrollRef={detailScrollRef} t={t} tick={now} />
          </NoSelect>
        </Box>
      )}

      <Box flexDirection="column" marginTop={1}>
        {flash ? <Text color={t.color.amber}>{flash}</Text> : null}

        {mode === 'list' ? (
          <Text color={t.color.dim}>
            ↑↓/jk move · g/G top/bottom · Enter/→ open detail{controlsHint} · s sort:{SORT_LABEL[sort]} · f filter:
            {FILTER_LABEL[filter]}
            {history.length > 0 ? ` · [ / ] history ${historyIndex}/${history.length}` : ''}
            {' · q close'}
          </Text>
        ) : (
          <Text color={t.color.dim}>
            ↑↓/jk scroll · PgUp/PgDn page · g/G top/bottom · Esc/← back to list{controlsHint} · q close
          </Text>
        )}
      </Box>
    </Box>
  )
}

interface AgentsOverlayProps {
  gw: GatewayClient
  initialHistoryIndex?: number
  onClose: () => void
  t: Theme
}

export const closeAgentsOverlay = () => patchOverlayState({ agents: false })
export const openAgentsOverlay = () => patchOverlayState({ agents: true })
