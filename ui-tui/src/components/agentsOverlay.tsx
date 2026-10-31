import { Box, NoSelect, ScrollBox, type ScrollBoxHandle, Text, useInput, useStdout } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react'

import { useAgentRoster } from '../app/agentRoster.js'
import {
  $delegationState,
  $overlaySectionsOpen,
  applyDelegationStatus,
  toggleOverlaySection
} from '../app/delegationStore.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { type ProcessRow, useProcessRows } from '../app/processRoster.js'
import { $spawnDiff, $spawnHistory, clearDiffPair, type SpawnSnapshot } from '../app/spawnHistoryStore.js'
import { $uiSessionId } from '../app/uiStore.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { DelegationPauseResponse, DelegationStatusResponse, SubagentInterruptResponse } from '../gatewayTypes.js'
import { messages } from '../i18n/runtime.js'
import type { Translations } from '../i18n/types.js'
import { useT } from '../i18n/useT.js'
import { asRpcResult } from '../lib/rpc.js'
import { statusGlyph as agentStatusGlyph } from '../lib/subagentGlyph.js'
import {
  buildSubagentTree,
  descendantIds,
  flattenTree,
  fmtDuration,
  fmtTokens,
  formatSummary,
  hotnessBucket,
  peakHotness,
  sparkline,
  topLevelSubagents,
  treeTotals,
  widthByDepth
} from '../lib/subagentTree.js'
import { compactPreview } from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { SubagentNode, SubagentProgress } from '../types.js'

import { AgentLiveTail, AgentSteerForm, rosterViewport } from './agentControls.js'
import { buildProcessBlock, ProcessRowLine, processSummary } from './agentsPanel.js'
import { listRowStyle } from './overlayPrimitives.js'
import { OverlayScrollbar } from './overlayScrollbar.js'

// ── Types + lookup tables ────────────────────────────────────────────

type SortMode = 'depth-first' | 'duration-desc' | 'status' | 'tools-desc'
type FilterMode = 'all' | 'failed' | 'leaf' | 'running'
type Status = SubagentProgress['status']

const SORT_ORDER: readonly SortMode[] = ['depth-first', 'tools-desc', 'duration-desc', 'status']
const FILTER_ORDER: readonly FilterMode[] = ['all', 'running', 'failed', 'leaf']

type AgentsMessages = Translations['hubs']['agents']

// Mode → catalog leaf. Labels are resolved against the active language at
// render time (`sortLabel` / `filterLabel`), never at import time.
const SORT_KEY: Record<SortMode, keyof AgentsMessages['sort']> = {
  'depth-first': 'depthFirst',
  'duration-desc': 'durationDesc',
  status: 'status',
  'tools-desc': 'toolsDesc'
}

const FILTER_KEY: Record<FilterMode, keyof AgentsMessages['filter']> = {
  all: 'all',
  failed: 'failed',
  leaf: 'leaf',
  running: 'running'
}

export const sortLabel = (mode: SortMode, m: AgentsMessages = messages().hubs.agents): string => m.sort[SORT_KEY[mode]]

export const filterLabel = (mode: FilterMode, m: AgentsMessages = messages().hubs.agents): string =>
  m.filter[FILTER_KEY[mode]]

/** Agent state values are compared in code; only the display label is localised. */
const statusLabel = (status: string, m: AgentsMessages): string =>
  (m.status as Record<string, string>)[status] ?? status

const STATUS_RANK: Record<Status, number> = {
  error: 0,
  failed: 0,
  interrupted: 1,
  timeout: 1,
  running: 2,
  queued: 3,
  completed: 4
}

const statusRank = (status: string): number => STATUS_RANK[status as Status] ?? STATUS_RANK.error

const SORT_COMPARATORS: Record<SortMode, (a: SubagentNode, b: SubagentNode) => number> = {
  'depth-first': (a, b) => a.item.depth - b.item.depth || a.item.index - b.item.index,
  'tools-desc': (a, b) => b.aggregate.totalTools - a.aggregate.totalTools,
  'duration-desc': (a, b) => b.aggregate.totalDuration - a.aggregate.totalDuration,
  status: (a, b) => statusRank(a.item.status) - statusRank(b.item.status)
}

const FILTER_PREDICATES: Record<FilterMode, (n: SubagentNode) => boolean> = {
  all: () => true,
  leaf: n => n.children.length === 0,
  running: n => n.item.status === 'running' || n.item.status === 'queued',
  failed: n =>
    n.item.status === 'error' ||
    n.item.status === 'failed' ||
    n.item.status === 'interrupted' ||
    n.item.status === 'timeout'
}

// Heatmap palette — cold → hot, resolved against the active theme.
const heatPalette = (t: Theme) => [t.color.border, t.color.accent, t.color.primary, t.color.warn, t.color.error]

// ── Pure helpers ─────────────────────────────────────────────────────

const fmtDur = (seconds?: number) => (seconds == null || seconds <= 0 ? '' : fmtDuration(seconds))
const fmtElapsedLabel = (seconds: number) => (seconds < 0 ? '' : fmtDuration(seconds))

const displayElapsedSeconds = (item: SubagentProgress, nowMs: number): number | null => {
  if (item.durationSeconds != null) {
    return item.durationSeconds
  }

  if (item.startedAt != null && (item.status === 'running' || item.status === 'queued')) {
    return Math.max(0, (nowMs - item.startedAt) / 1000)
  }

  return null
}

const indentFor = (depth: number): string => '  '.repeat(Math.max(0, depth))
const formatRowId = (n: number): string => String(n + 1).padStart(2, ' ')
const cycle = <T,>(order: readonly T[], current: T): T => order[(order.indexOf(current) + 1) % order.length]!

const statusGlyph = (item: SubagentProgress, t: Theme) => agentStatusGlyph(item.status, t)

const prepareRows = (tree: SubagentNode[], sort: SortMode, filter: FilterMode): SubagentNode[] =>
  tree.length === 0 ? [] : flattenTree([...tree].sort(SORT_COMPARATORS[sort])).filter(FILTER_PREDICATES[filter])

const diffMetricLine = (name: string, a: number, b: number, fmt: (n: number) => string) => {
  const d = b - a
  const sign = d === 0 ? '' : d > 0 ? '+' : '-'

  return `${name}: ${fmt(a)} → ${fmt(b)}  (${sign}${fmt(Math.abs(d)) || '0'})`
}

// ── Sub-components ───────────────────────────────────────────────────

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
  const T = useT().hubs.agents

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
      <Text color={t.color.muted}>
        {T.timeline} · {fmtElapsedLabel(Math.max(0, totalSeconds))}
        {windowLabel}
      </Text>

      {shown.map(({ endAt, idx, node, startAt }) => {
        const active = idx === cursor
        const { color } = statusGlyph(node.item, t)
        const accent = active ? t.color.accent : t.color.muted

        const elSec = displayElapsedSeconds(node.item, now)
        const elLabel = elSec != null ? fmtElapsedLabel(elSec) : ''

        return (
          <Text key={node.item.id} wrap="truncate-end">
            <Text bold={active} color={accent}>
              {formatRowId(idx)}
              {'  '}
            </Text>

            <Text color={active ? t.color.accent : color}>{bar(startAt, endAt)}</Text>

            {elLabel ? (
              <Text color={accent}>
                {'   '}
                {elLabel}
              </Text>
            ) : null}
          </Text>
        )
      })}

      <Text color={t.color.muted} dim>
        {'    '}
        {ruler}
      </Text>

      {totalSeconds > 0 ? (
        <Text color={t.color.muted} dim>
          {'    '}
          {rulerLabels}
        </Text>
      ) : null}
    </Box>
  )
}

function OverlaySection({
  children,
  count,
  defaultOpen = false,
  id,
  title,
  t
}: {
  children: ReactNode
  count?: number
  defaultOpen?: boolean
  /** Locale-independent key for the open/closed store; `title` is the display label. */
  id: string
  title: string
  t: Theme
}) {
  const openMap = useStore($overlaySectionsOpen)
  const open = id in openMap ? openMap[id]! : defaultOpen

  return (
    <Box flexDirection="column" marginTop={1}>
      <Box onClick={() => toggleOverlaySection(id, defaultOpen)}>
        <Text color={t.color.label}>
          <Text color={t.color.accent}>{open ? '▾ ' : '▸ '}</Text>
          {title}
          {typeof count === 'number' ? ` (${count})` : ''}
        </Text>
      </Box>

      {open ? <Box flexDirection="column">{children}</Box> : null}
    </Box>
  )
}

/** Background processes owned by this session, listed under the spawn tree. They are
 * not part of the cursor roster (no per-row steer/tail); `/stop` ends them all. */
function ProcessesSection({ cols, rows, t }: { cols: number; rows: readonly ProcessRow[]; t: Theme }) {
  const T = useT().hubs.agents

  if (rows.length === 0) {
    return null
  }

  const block = buildProcessBlock(rows, 0)

  return (
    <Box flexDirection="column" flexShrink={0} marginTop={1}>
      <Text bold color={t.color.accent} wrap="truncate-end">
        {`${T.processes} · ${processSummary(block)}`}
      </Text>
      {block.rows.map(row => (
        <ProcessRowLine cols={cols} key={row.id} row={row} t={t} />
      ))}
    </Box>
  )
}

function Field({ name, t, value }: { name: string; t: Theme; value: ReactNode }) {
  return (
    <Text wrap="truncate-end">
      <Text color={t.color.label}>{name} · </Text>
      <Text color={t.color.text}>{value}</Text>
    </Text>
  )
}

function Detail({ id, node, t }: { id?: string; node: SubagentNode; t: Theme }) {
  const T = useT().hubs.agents
  const D = T.detail
  const { aggregate: agg, item } = node
  const { color, glyph } = statusGlyph(item, t)

  const inputTokens = item.inputTokens ?? 0
  const outputTokens = item.outputTokens ?? 0
  const localTokens = inputTokens + outputTokens
  const subtreeTokens = agg.inputTokens + agg.outputTokens - localTokens

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
      <Text bold color={t.color.text} wrap="wrap">
        {id ? <Text color={t.color.accent}>#{id} </Text> : null}
        <Text color={color}>{glyph}</Text> {item.goal}
      </Text>

      <Box flexDirection="column" marginTop={1}>
        <Field name={D.depth} t={t} value={`${item.depth} · ${statusLabel(item.status, T)}`} />
        {item.model ? <Field name={D.model} t={t} value={item.model} /> : null}
        {item.toolsets?.length ? <Field name={D.toolsets} t={t} value={item.toolsets.join(', ')} /> : null}
        <Field name={D.tools} t={t} value={D.toolsValue(item.toolCount ?? 0, agg.totalTools)} />
        <Field
          name={D.subtree}
          t={t}
          value={(agg.descendantCount === 1 ? D.subtreeValueOne : D.subtreeValueOther)(
            agg.descendantCount,
            agg.maxDepthFromHere,
            agg.activeCount
          )}
        />
        {item.durationSeconds ? <Field name={D.elapsed} t={t} value={fmtDur(item.durationSeconds)} /> : null}
        {item.iteration != null ? <Field name={D.iteration} t={t} value={String(item.iteration)} /> : null}
        {item.apiCalls ? <Field name={D.apiCalls} t={t} value={String(item.apiCalls)} /> : null}
      </Box>

      {localTokens > 0 ? (
        <OverlaySection defaultOpen id="budget" t={t} title={T.section.budget}>
          {localTokens > 0 ? (
            <Field
              name={D.tokens}
              t={t}
              value={
                <>
                  {D.tokensValue(fmtTokens(inputTokens), fmtTokens(outputTokens))}
                  {item.reasoningTokens ? D.reasoningSuffix(fmtTokens(item.reasoningTokens)) : ''}
                </>
              }
            />
          ) : null}

          {subtreeTokens > 0 ? <Field name={D.subtreeTokens} t={t} value={`+${fmtTokens(subtreeTokens)}`} /> : null}
        </OverlaySection>
      ) : null}

      {filesRead.length > 0 || filesWritten.length > 0 ? (
        <OverlaySection count={filesRead.length + filesWritten.length} id="files" t={t} title={T.section.files}>
          {filesWritten.slice(0, 8).map((p, i) => (
            <Text color={t.color.statusGood} key={`w-${i}`} wrap="truncate-end">
              +{p}
            </Text>
          ))}

          {filesRead.slice(0, 8).map((p, i) => (
            <Text color={t.color.text} key={`r-${i}`} wrap="truncate-end">
              <Text color={t.color.muted}>·</Text> {p}
            </Text>
          ))}

          {filesOverflow > 0 ? <Text color={t.color.muted}>{D.filesMore(filesOverflow)}</Text> : null}
        </OverlaySection>
      ) : null}

      {toolLines.length > 0 ? (
        <OverlaySection count={toolLines.length} defaultOpen id="toolCalls" t={t} title={T.section.toolCalls}>
          {toolLines.map((line, i) => (
            <Text color={t.color.text} key={i} wrap="wrap">
              <Text color={t.color.muted}>·</Text> {line}
            </Text>
          ))}
        </OverlaySection>
      ) : null}

      {outputTail.length > 0 ? (
        <OverlaySection count={outputTail.length} defaultOpen id="output" t={t} title={T.section.output}>
          {outputTail.map((entry, i) => (
            <Text color={entry.isError ? t.color.error : t.color.text} key={i} wrap="wrap">
              <Text bold color={entry.isError ? t.color.error : t.color.accent}>
                {entry.tool}
              </Text>{' '}
              {entry.preview}
            </Text>
          ))}
        </OverlaySection>
      ) : null}

      {item.notes.length ? (
        <OverlaySection count={item.notes.length} id="progress" t={t} title={T.section.progress}>
          {item.notes.slice(-6).map((line, i) => (
            <Text color={t.color.text} key={i} wrap="wrap">
              <Text color={t.color.label}>·</Text> {line}
            </Text>
          ))}
        </OverlaySection>
      ) : null}

      {item.summary ? (
        <OverlaySection defaultOpen id="summary" t={t} title={T.section.summary}>
          <Text color={t.color.text} wrap="wrap">
            {item.summary}
          </Text>
        </OverlaySection>
      ) : null}
    </Box>
  )
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
  const T = useT().hubs.agents
  const { color, glyph } = statusGlyph(node.item, t)
  const palette = heatPalette(t)
  const heatIdx = hotnessBucket(node.aggregate.hotness, peak, palette.length)
  const heatMarker = heatIdx >= 2 ? palette[heatIdx]! : null

  const goal = compactPreview(node.item.goal || T.subagentFallback, width - 28 - node.item.depth * 2)
  const toolsCount = node.aggregate.totalTools > 0 ? ` ·${node.aggregate.totalTools}t` : ''
  const kids = node.children.length ? ` ·${node.children.length}↓` : ''
  const line = node.item.status === 'running' ? node.item.tools.at(-1) : undefined
  const paren = line ? line.indexOf('(') : -1
  const toolShort = line ? (paren > 0 ? line.slice(0, paren) : line).trim() : ''
  const trailing = toolShort ? ` · ${compactPreview(toolShort, 14)}` : ''
  // Selection chip, not `inverse` — inverse swaps against the terminal's
  // unknowable defaults (black slab on transparent profiles).
  const row = listRowStyle(t, active)
  const fg = active ? (row.color ?? t.color.accent) : t.color.text

  return (
    <Text backgroundColor={row.backgroundColor} bold={active} color={fg} wrap="truncate-end">
      {' '}
      <Text color={active ? fg : t.color.muted}>{formatRowId(index)} </Text>
      {indentFor(node.item.depth)}
      {heatMarker ? <Text color={active ? fg : heatMarker}>▍</Text> : null}
      <Text color={active ? fg : color}>{glyph}</Text> {goal}
      <Text color={active ? fg : t.color.muted}>
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
  const T = useT().hubs.agents

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.text}>
        {label}
      </Text>

      <Text color={t.color.muted} wrap="truncate-end">
        {snapshot.label}
      </Text>

      <Box marginTop={1}>
        <Text color={t.color.muted} wrap="truncate-end">
          {formatSummary(totals)}
        </Text>
      </Box>

      <Box flexDirection="column" marginTop={1}>
        {topLevelSubagents(snapshot.subagents)
          .slice(0, 8)
          .map(s => {
            const { color, glyph } = statusGlyph(s, t)

            return (
              <Text color={t.color.muted} key={s.id} wrap="truncate-end">
                <Text color={color}>{glyph}</Text> {s.goal || T.subagentFallback}
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
  const T = useT().hubs.agents.diff
  const aTotals = useMemo(() => treeTotals(buildSubagentTree(pair.baseline.subagents)), [pair.baseline])
  const bTotals = useMemo(() => treeTotals(buildSubagentTree(pair.candidate.subagents)), [pair.candidate])
  const paneWidth = Math.floor((cols - 4) / 2)

  useInput((ch, key) => {
    if (key.escape || ch === 'q') {
      onClose()
    }
  })

  const round = (n: number) => String(Math.round(n))
  const sumTokens = (x: typeof aTotals) => x.inputTokens + x.outputTokens

  return (
    <Box flexDirection="column" flexGrow={1} paddingX={1} paddingY={1}>
      <Box flexDirection="column" marginBottom={1}>
        <Text bold color={t.color.border}>
          {T.title}
        </Text>
        <Text color={t.color.muted}>{T.subtitle}</Text>
      </Box>

      <Box flexDirection="row" marginBottom={1}>
        <DiffPane label={T.baseline} snapshot={pair.baseline} t={t} totals={aTotals} width={paneWidth} />
        <Box width={2} />
        <DiffPane label={T.candidate} snapshot={pair.candidate} t={t} totals={bTotals} width={paneWidth} />
      </Box>

      <Box flexDirection="column" marginTop={1}>
        <Text bold color={t.color.accent}>
          {T.delta}
        </Text>

        <Text color={t.color.text}>
          {diffMetricLine(T.agents, aTotals.descendantCount, bTotals.descendantCount, round)}
        </Text>
        <Text color={t.color.text}>{diffMetricLine(T.tools, aTotals.totalTools, bTotals.totalTools, round)}</Text>
        <Text color={t.color.text}>
          {diffMetricLine(T.depth, aTotals.maxDepthFromHere, bTotals.maxDepthFromHere, round)}
        </Text>
        <Text color={t.color.text}>
          {diffMetricLine(T.duration, aTotals.totalDuration, bTotals.totalDuration, n => `${n.toFixed(1)}s`)}
        </Text>
        <Text color={t.color.text}>{diffMetricLine(T.tokens, sumTokens(aTotals), sumTokens(bTotals), fmtTokens)}</Text>
      </Box>
    </Box>
  )
}

// ── Main overlay ─────────────────────────────────────────────────────

export function AgentsOverlay({ gw, initialHistoryIndex = 0, onClose, t }: AgentsOverlayProps) {
  const T = useT().hubs.agents
  const liveSubagents = useAgentRoster()
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
  const [mode, setMode] = useState<'detail' | 'list' | 'steer' | 'tail'>('list')
  const sid = useStore($uiSessionId)
  const processRows = useProcessRows(now)

  const detailScrollRef = useRef<null | ScrollBoxHandle>(null)
  const prevLiveCountRef = useRef(liveSubagents.length)

  // ── Derived state ──────────────────────────────────────────────────

  const activeSnapshot = historyIndex > 0 ? history[historyIndex - 1] : null
  // Instant fallback to history[0] the moment the live list clears — avoids
  // a one-frame "no subagents" flash while the auto-follow effect fires.
  const justFinishedSnapshot = historyIndex === 0 && liveSubagents.length === 0 ? (history[0] ?? null) : null
  const effectiveSnapshot = activeSnapshot ?? justFinishedSnapshot
  const replayMode = effectiveSnapshot != null
  const subagents = replayMode ? effectiveSnapshot.subagents : liveSubagents

  const tree = useMemo(() => buildSubagentTree(subagents), [subagents])
  const totals = useMemo(() => treeTotals(tree), [tree])
  const widths = useMemo(() => widthByDepth(tree), [tree])
  const spark = useMemo(() => sparkline(widths), [widths])
  const peak = useMemo(() => peakHotness(tree), [tree])
  const rows = useMemo(() => prepareRows(tree, sort, filter), [tree, sort, filter])

  const selected = rows[cursor] ?? null

  const cols = stdout?.columns ?? 80

  const {
    rows: rowsH,
    start: listWindowStart,
    timelineRows
  } = rosterViewport((stdout?.rows ?? 24) - (flash ? 1 : 0), rows.length, cursor)

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
    prevLiveCountRef.current = liveSubagents.length

    if (historyIndex === 0 && prev > 0 && liveSubagents.length === 0 && history.length > 0) {
      setHistoryIndex(1)
      setCursor(0)
      setFlash(messages().hubs.agents.flash.turnFinished)
    }
  }, [history.length, historyIndex, liveSubagents.length])

  useEffect(() => {
    // Reset detail scroll on navigation so the top of the new node shows.
    detailScrollRef.current?.scrollTo(0)
  }, [cursor, historyIndex, mode])

  useEffect(() => {
    // A control acknowledgement or newer hydration must win over this request.
    const initial = $delegationState.get()
    let active = true
    gw.request<DelegationStatusResponse>('delegation.status', {})
      .then(r => {
        if (active && $delegationState.get() === initial) {
          applyDelegationStatus(asRpcResult<DelegationStatusResponse>(r))
        }
      })
      .catch(() => {})

    return () => {
      active = false
    }
  }, [gw])

  useEffect(() => {
    if (cursor >= rows.length) {
      setCursor(Math.max(0, rows.length - 1))
    }
  }, [cursor, rows.length])

  // ── Actions ────────────────────────────────────────────────────────

  const guardLive = (action: () => void) => {
    if (replayMode) {
      setFlash(T.flash.replayLocked)
    } else {
      action()
    }
  }

  const interrupt = (id: string) =>
    gw.request<SubagentInterruptResponse>('subagent.interrupt', { session_id: sid, subagent_id: id })

  const killOne = (id: string) =>
    guardLive(() => {
      interrupt(id)
        .then(raw => {
          const r = asRpcResult<SubagentInterruptResponse>(raw)
          setFlash(r?.found ? T.flash.killing(id) : T.flash.notFound(id))
        })
        .catch(() => setFlash(T.flash.killFailed(id)))
    })

  const killSubtree = (node: SubagentNode) =>
    guardLive(() => {
      const ids = [node.item.id, ...descendantIds(node)]
      ids.forEach(id => interrupt(id).catch(() => {}))
      setFlash((ids.length === 1 ? T.flash.killingSubtreeOne : T.flash.killingSubtreeOther)(ids.length))
    })

  const togglePause = () =>
    guardLive(() => {
      gw.request<DelegationPauseResponse>('delegation.pause', { paused: !delegation.paused })
        .then(raw => {
          const r = asRpcResult<DelegationPauseResponse>(raw)
          applyDelegationStatus({ paused: r?.paused })
          setFlash(r?.paused ? T.flash.spawningPaused : T.flash.spawningResumed)
        })
        .catch(() => setFlash(T.flash.pauseFailed))
    })

  const stepHistory = (delta: -1 | 1) =>
    setHistoryIndex(idx => {
      const next = Math.max(0, Math.min(history.length, idx + delta))

      if (next !== idx) {
        setCursor(0)
        setFlash(next === 0 ? T.flash.liveTurn : T.flash.replay(next, history.length))
      }

      return next
    })

  const closeWithCleanup = () => {
    clearDiffPair()
    onClose()
  }

  // ── Input ──────────────────────────────────────────────────────────

  const detailPageSize = Math.max(4, rowsH - 2)
  const wheelDetailDy = 3
  const scrollDetail = (dy: number) => detailScrollRef.current?.scrollBy(dy)

  useInput((ch, key) => {
    if (mode === 'steer') {
      return
    }

    if (key.ctrl && ch === 't') {
      return closeWithCleanup()
    }

    if (ch === 'e' && selected && sid && !replayMode) {
      return setMode('steer')
    }

    if (ch === 't' && !key.ctrl && selected) {
      return setMode('tail')
    }

    if (ch === 'd' && !key.ctrl && selected) {
      return setMode('detail')
    }

    if (ch === 'q') {
      return closeWithCleanup()
    }

    if (key.escape) {
      return mode !== 'list' ? setMode('list') : closeWithCleanup()
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

    if (mode === 'detail' || mode === 'tail') {
      if (key.leftArrow || ch === 'h') {
        return setMode('list')
      }

      if (key.pageUp || (key.ctrl && ch === 'u')) {
        return scrollDetail(-detailPageSize)
      }

      if (key.pageDown || (key.ctrl && ch === 'd')) {
        return scrollDetail(detailPageSize)
      }

      if (key.wheelUp) {
        return scrollDetail(-wheelDetailDy)
      }

      if (key.wheelDown) {
        return scrollDetail(wheelDetailDy)
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
      return setMode(key.return && !replayMode ? 'tail' : 'detail')
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
      const key = it.model ? it.model.split('/').pop()! : T.inheritModel
      acc[key] = (acc[key] ?? 0) + 1

      return acc
    }, {})
  )
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([k, v]) => `${k}×${v}`)
    .join(' · ')

  const capsLabel = delegation.maxSpawnDepth
    ? T.caps(delegation.maxSpawnDepth, String(delegation.maxConcurrentChildren ?? '?'))
    : ''

  const title =
    replayMode && effectiveSnapshot
      ? `${historyIndex > 0 ? T.title.replay(historyIndex, history.length) : T.title.lastTurn}${T.title.finishedAt(
          new Date(effectiveSnapshot.finishedAt).toLocaleTimeString()
        )}`
      : `${T.title.spawnTree}${delegation.paused ? T.title.pausedSuffix : ''}`

  const metaLine = [formatSummary(totals), spark, capsLabel, mix ? `· ${mix}` : ''].filter(Boolean).join('  ')

  const controlsHint = replayMode
    ? T.hint.controlsLocked
    : T.hint.controls(delegation.paused ? T.hint.resume : T.hint.pause)

  // ── Rendering ──────────────────────────────────────────────────────

  if (diffPair) {
    return <DiffView cols={cols} onClose={closeWithCleanup} pair={diffPair} t={t} />
  }

  return (
    <Box alignItems="stretch" flexDirection="column" flexGrow={1} paddingX={1} paddingY={1}>
      <Box flexDirection="column" marginBottom={1}>
        <Text wrap="truncate-end">
          <Text bold color={replayMode ? t.color.border : t.color.primary}>
            {title}
          </Text>
          {metaLine ? (
            <Text color={t.color.muted}>
              {'   '}
              {metaLine}
            </Text>
          ) : null}
        </Text>
      </Box>

      {mode === 'steer' && selected && sid ? (
        <AgentSteerForm cols={cols} gw={gw} id={selected.item.id} onClose={() => setMode('detail')} sid={sid} t={t} />
      ) : rows.length === 0 ? (
        <Box flexDirection="column" flexGrow={1}>
          <Text color={t.color.muted}>{T.empty}</Text>
          <ProcessesSection cols={cols - 2} rows={processRows} t={t} />
        </Box>
      ) : mode === 'list' ? (
        <Box flexDirection="column" flexGrow={1} flexShrink={1} minHeight={0}>
          {timelineRows > 0 ? (
            <GanttStrip cols={cols - 2} cursor={cursor} flatNodes={rows} maxRows={timelineRows} now={now} t={t} />
          ) : null}

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
          <ProcessesSection cols={cols - 2} rows={processRows} t={t} />
        </Box>
      ) : (
        <Box flexDirection="row" flexGrow={1} flexShrink={1} minHeight={0}>
          <ScrollBox
            flexDirection="column"
            flexGrow={1}
            flexShrink={1}
            ref={detailScrollRef}
            stickyScroll={mode === 'tail'}
          >
            <Box flexDirection="column" paddingBottom={4} paddingRight={1}>
              {selected && mode === 'tail' && sid && !replayMode ? (
                <AgentLiveTail gw={gw} id={selected.item.id} key={selected.item.id} sid={sid} t={t} />
              ) : selected ? (
                <Detail id={selected.item.id} node={selected} t={t} />
              ) : null}
            </Box>
          </ScrollBox>

          <NoSelect flexShrink={0} marginLeft={1}>
            <OverlayScrollbar scrollRef={detailScrollRef} t={t} tick={now} />
          </NoSelect>
        </Box>
      )}

      <Box flexDirection="column" flexShrink={0} marginTop={1}>
        <Text color={t.color.accent} wrap="truncate-end">
          {replayMode ? T.hint.footerReplay : T.hint.footerLive}
        </Text>
        {flash ? (
          <Text color={t.color.accent} wrap="truncate-end">
            {flash}
          </Text>
        ) : null}

        {mode === 'list' ? (
          <Text color={t.color.muted} wrap="truncate-end">
            {T.hint.list(
              replayMode ? T.hint.listNavReplay : T.hint.listNavLive,
              controlsHint,
              sortLabel(sort, T),
              filterLabel(filter, T),
              history.length > 0 ? T.hint.history(historyIndex, history.length) : ''
            )}
          </Text>
        ) : (
          <Text color={t.color.muted} wrap="truncate-end">
            {T.hint.detail(controlsHint)}
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
