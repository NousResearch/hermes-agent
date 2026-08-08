import type { SubagentProgress } from '../types.js'

import { buildSubagentTree, flattenTree, fmtDuration, isRunning } from './subagentTree.js'

/**
 * Pure projection for the persistent agent dock — the compact live panel
 * between transcript and composer. Converts the turn's `SubagentProgress[]`
 * into a bounded, truthful view model so the Ink component stays a dumb
 * renderer and every behavior is testable without a terminal.
 *
 * Tones are theme *keys*, not colors: the component resolves them against the
 * active theme, and glyphs stay pairwise distinct so color is never the only
 * status signal. Ordering reuses `buildSubagentTree`/`flattenTree`, the same
 * spawn-order contract as the full `/agents` overlay.
 */

export const DOCK_MAX_ROWS = 3
/** Below this many columns the dock degrades to a one-line summary. */
export const DOCK_NARROW_WIDTH = 60

export type DockTone = 'accent' | 'error' | 'muted' | 'statusGood' | 'warn'

export interface DockRow {
  /** Plain-language current activity or terminal state; never raw arguments. */
  activity: string
  /** Bounded ASCII-safe name derived from the goal, with an indexed fallback. */
  callsign: string
  depth: number
  /** @deprecated Transitional alias for the pre-B3 renderer. */
  detail: string
  /** Formatted duration label, '' when nothing truthful is known. */
  elapsed: string
  glyph: string
  /** @deprecated Transitional alias for the pre-B3 renderer. */
  goal: string
  id: string
  /** True when `elapsed` derives from `nowMs` and needs periodic re-projection. */
  live: boolean
  status: SubagentProgress['status']
  tone: DockTone
}

export interface DockView {
  activeCount: number
  /** Truthful framed header counts, without the `agents` label. */
  header: string
  hidden: boolean
  overflow: number
  /** How many of the overflowed rows are still running/queued. */
  overflowActive: number
  /** Truthful aggregate for rows outside the three-row budget. */
  overflowSummary: string
  rows: DockRow[]
  summary: string
  summaryOnly: boolean
  totalCount: number
}

export interface DockOptions {
  forceSummary?: boolean
  maxRows?: number
  nowMs: number
  width: number
}

// Same glyph/tone semantics as the agents overlay, expressed as theme keys so
// this module needs no Theme import and stays render-free.
const STATUS_META: Record<SubagentProgress['status'], { glyph: string; tone: DockTone }> = {
  completed: { glyph: '✓', tone: 'statusGood' },
  error: { glyph: '!', tone: 'error' },
  failed: { glyph: '✗', tone: 'error' },
  interrupted: { glyph: '■', tone: 'warn' },
  queued: { glyph: '○', tone: 'muted' },
  running: { glyph: '●', tone: 'accent' },
  timeout: { glyph: 'T', tone: 'warn' }
}

const CALLSIGN_MAX = 12

const CALLSIGN_SAFE_WORDS = new Set([
  'analyze',
  'audit',
  'benchmark',
  'build',
  'check',
  'debug',
  'design',
  'document',
  'fix',
  'implement',
  'inspect',
  'investigate',
  'monitor',
  'patch',
  'plan',
  'profile',
  'read',
  'regression',
  'research',
  'review',
  'scan',
  'summarize',
  'test',
  'trace',
  'validate',
  'verify',
  'write'
])

interface DockCounts {
  blocked: number
  completed: number
  done: number
  queued: number
  ready: number
  running: number
}

const isFailed = (s: SubagentProgress): boolean =>
  s.status === 'error' || s.status === 'failed' || s.status === 'interrupted' || s.status === 'timeout'

const elapsedFor = (item: SubagentProgress, nowMs: number): { elapsed: string; live: boolean } => {
  if (item.durationSeconds != null) {
    return { elapsed: fmtDuration(item.durationSeconds), live: false }
  }

  if (item.startedAt != null && isRunning(item)) {
    return { elapsed: fmtDuration(Math.max(0, (nowMs - item.startedAt) / 1000)), live: true }
  }

  return { elapsed: '', live: false }
}

const callsignFor = (item: SubagentProgress): string => {
  const fallback = `agent ${Math.max(0, item.index) + 1}`.slice(0, CALLSIGN_MAX)
  const goal = item.goal.trim()
  const pathOrUrlShaped = goal.includes('/') || goal.includes('\\') || /^[a-z][a-z0-9+.-]*:(?!\s)/i.test(goal)

  if (pathOrUrlShaped) {
    return fallback
  }

  const asciiGoal = Array.from(goal.normalize('NFKD'), char =>
    (char.codePointAt(0) ?? Number.POSITIVE_INFINITY) <= 0x7f ? char : ' '
  ).join('')

  const tokens = asciiGoal
    .toLowerCase()
    .match(/[a-z0-9]+/g)

  const chosen = tokens?.find(token => CALLSIGN_SAFE_WORDS.has(token))

  if (chosen) {
    return chosen.slice(0, CALLSIGN_MAX)
  }

  return fallback
}

// Keys are formatToolCall()/toolTrailLabel() output lowercased, then
// stripped of any ("preview") suffix by activityFor. Keep this map keyed
// to the real Hermes tool vocabulary so dock activity never silently
// degrades to "working" for common tools.
const PLAIN_ACTIVITY: Record<string, string> = {
  // File + shell
  'close terminal': 'closing terminal',
  read: 'reading files',
  'read file': 'reading files',
  'read terminal': 'reading terminal',
  'search files': 'searching',
  write: 'writing files',
  'write file': 'writing files',
  patch: 'editing files',
  terminal: 'running commands',

  // Web + browser
  'browser back': 'browsing',
  'browser cdp': 'browsing',
  'browser click': 'browsing',
  'browser console': 'browsing',
  'browser dialog': 'browsing',
  'browser get images': 'browsing',
  'browser navigate': 'browsing',
  'browser press': 'browsing',
  'browser scroll': 'browsing',
  'browser snapshot': 'browsing',
  'browser type': 'browsing',
  'browser vision': 'browsing',
  'web extract': 'reading web',
  'web search': 'searching web',

  // Agent core
  clarify: 'asking',
  'computer use': 'using computer',
  cronjob: 'scheduling',
  'delegate task': 'delegating',
  'execute code': 'running code',
  memory: 'updating memory',
  process: 'managing process',
  'session search': 'searching sessions',
  'skill manage': 'managing skills',
  'skill view': 'reading skills',
  'skills list': 'listing skills',
  todo: 'updating todos',
  'text to speech': 'speaking',
  'vision analyze': 'analyzing image',
  'image generate': 'generating image',

  // Home Assistant
  'ha call service': 'controlling home',
  'ha get state': 'checking home',
  'ha list entities': 'listing home',
  'ha list services': 'listing home',

  // Kanban
  'kanban attach': 'attaching files',
  'kanban attach url': 'attaching files',
  'kanban attachments': 'listing attachments',
  'kanban block': 'blocking task',
  'kanban comment': 'commenting',
  'kanban complete': 'completing task',
  'kanban create': 'creating task',
  'kanban heartbeat': 'heartbeating',
  'kanban link': 'linking tasks',
  'kanban list': 'listing tasks',
  'kanban show': 'reading task',
  'kanban unblock': 'unblocking task',

  // Desktop / TUI panes
  'focus pane': 'focusing pane',
  'open preview': 'opening preview'
}

const TERMINAL_ACTIVITY: Partial<Record<SubagentProgress['status'], string>> = {
  error: 'failed',
  failed: 'failed',
  interrupted: 'interrupted',
  queued: 'queued',
  timeout: 'timed out'
}

const activityFor = (item: SubagentProgress): string => {
  if (item.status === 'completed') {
    return item.summary ? 'result ready' : 'done'
  }

  if (item.status !== 'running') {
    return TERMINAL_ACTIVITY[item.status] ?? 'blocked'
  }

  const raw = item.tools.at(-1)

  if (!raw) {
    return 'working'
  }

  const label = (raw.includes('("') ? raw.slice(0, raw.indexOf('("')) : raw).trim().toLowerCase()

  return PLAIN_ACTIVITY[label] ?? 'working'
}

const countsFor = (items: readonly SubagentProgress[]): DockCounts => {
  let blocked = 0
  let completed = 0
  let queued = 0
  let ready = 0
  let running = 0

  for (const item of items) {
    if (item.status === 'running') {
      running += 1
    } else if (item.status === 'queued') {
      queued += 1
    } else if (item.status === 'completed') {
      completed += 1

      if (item.summary) {
        ready += 1
      }
    } else {
      blocked += 1
    }
  }

  return { blocked, completed, done: completed - ready, queued, ready, running }
}

const plural = (count: number, singular: string): string => `${count} ${singular}`

const activeCountLabels = (counts: DockCounts): string[] => {
  const labels: string[] = []

  if (counts.running > 0) {
    labels.push(plural(counts.running, 'running'))
  }

  if (counts.queued > 0) {
    labels.push(plural(counts.queued, 'queued'))
  }

  if (counts.ready > 0) {
    labels.push(plural(counts.ready, 'ready'))
  }

  if (counts.done > 0) {
    labels.push(plural(counts.done, 'done'))
  }

  if (counts.blocked > 0) {
    labels.push(plural(counts.blocked, 'blocked'))
  }

  return labels
}

const headerFor = (items: readonly SubagentProgress[]): string => {
  const counts = countsFor(items)

  if (counts.running > 0 || counts.queued > 0) {
    return activeCountLabels(counts).join(' · ')
  }

  const labels: string[] = []

  if (counts.completed > 0) {
    labels.push(plural(counts.completed, 'done'))
  }

  if (counts.blocked > 0) {
    labels.push(plural(counts.blocked, 'blocked'))
  }

  return labels.join(' · ')
}

const overflowSummaryFor = (items: readonly SubagentProgress[]): string => {
  if (items.length === 0) {
    return ''
  }

  return [`${items.length} more`, ...activeCountLabels(countsFor(items))].join(' · ')
}

const buildSummary = (items: readonly SubagentProgress[], activeCount: number, nowMs: number): string => {
  const total = items.length
  const issues = items.filter(isFailed).length
  const pieces = [activeCount > 0 ? `${activeCount}/${total} active` : `${total} agent${total === 1 ? '' : 's'}`]

  if (issues > 0) {
    pieces.push(`${issues} issue${issues === 1 ? '' : 's'}`)
  }

  let longest = 0

  for (const item of items) {
    if (item.startedAt != null && isRunning(item)) {
      longest = Math.max(longest, (nowMs - item.startedAt) / 1000)
    }
  }

  if (longest > 0) {
    pieces.push(fmtDuration(longest))
  }

  return pieces.join(' · ')
}

export function projectAgentDock(subagents: readonly SubagentProgress[], opts: DockOptions): DockView {
  const { forceSummary = false, maxRows = DOCK_MAX_ROWS, nowMs, width } = opts
  const totalCount = subagents.length

  if (totalCount === 0) {
    return {
      activeCount: 0,
      header: '',
      hidden: true,
      overflow: 0,
      overflowActive: 0,
      overflowSummary: '',
      rows: [],
      summary: '',
      summaryOnly: false,
      totalCount: 0
    }
  }

  const activeCount = subagents.filter(isRunning).length
  const header = headerFor(subagents)
  const summary = buildSummary(subagents, activeCount, nowMs)

  if (forceSummary || width < DOCK_NARROW_WIDTH) {
    return {
      activeCount,
      header,
      hidden: false,
      overflow: totalCount,
      overflowActive: activeCount,
      overflowSummary: `${totalCount} more`,
      rows: [],
      summary,
      summaryOnly: true,
      totalCount
    }
  }

  const ordered = flattenTree(buildSubagentTree(subagents))

  // Preserve spawn/tree order within each class, but never bury live work
  // behind old completed rows when the three-row budget is full.
  const prioritized = [
    ...ordered.filter(node => isRunning(node.item)),
    ...ordered.filter(node => !isRunning(node.item) && isFailed(node.item)),
    ...ordered.filter(node => !isRunning(node.item) && !isFailed(node.item))
  ]

  const selectedIds = new Set(prioritized.slice(0, maxRows).map(node => node.item.id))
  // Priority decides which rows earn the bounded slots. Tree order decides
  // how those selected rows render, so a live child never jumps above a
  // selected completed parent.
  const visible = ordered.filter(node => selectedIds.has(node.item.id))
  const overflowNodes = prioritized.slice(maxRows)
  const overflowItems = overflowNodes.map(node => node.item)
  const visibleItems = new Map(visible.map(node => [node.item.id, node.item]))

  const visibleDepth = (item: SubagentProgress): number => {
    let depth = 0
    let parentId = item.parentId
    const seen = new Set<string>()

    while (parentId && visibleItems.has(parentId) && !seen.has(parentId)) {
      seen.add(parentId)
      depth += 1
      parentId = visibleItems.get(parentId)?.parentId ?? null
    }

    return depth
  }

  const rows: DockRow[] = visible.map(node => {
    const item = node.item
    // Defensive fallback for cross-version snapshots with unknown statuses.
    const meta = STATUS_META[item.status] ?? STATUS_META.error
    const { elapsed, live } = elapsedFor(item, nowMs)
    const activity = activityFor(item)
    const callsign = callsignFor(item)

    return {
      activity,
      callsign,
      depth: visibleDepth(item),
      detail: activity,
      elapsed,
      glyph: meta.glyph,
      goal: callsign,
      id: item.id,
      live,
      status: item.status,
      tone: meta.tone
    }
  })

  return {
    activeCount,
    header,
    hidden: false,
    overflow: overflowNodes.length,
    overflowActive: overflowNodes.filter(n => isRunning(n.item)).length,
    overflowSummary: overflowSummaryFor(overflowItems),
    rows,
    summary,
    summaryOnly: false,
    totalCount
  }
}
