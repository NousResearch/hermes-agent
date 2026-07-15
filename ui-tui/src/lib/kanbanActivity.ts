import { stringWidth } from '@hermes/ink'

import type {
  KanbanActivityBoard,
  KanbanActivityResponse,
  KanbanActivityRun,
  KanbanActivityTask,
  KanbanRunOutcome,
  KanbanTaskStatus
} from '../gatewayTypes.js'

export type ActivityTone = 'accent' | 'error' | 'muted' | 'neutral' | 'success' | 'warning'
export type HeartbeatFreshness = 'fresh' | 'missing' | 'not-applicable' | 'stale'

export interface ActivityRun {
  endedAt: null | number
  lastHeartbeatAt: null | number
  maxRuntimeSeconds: null | number
  outcome: KanbanRunOutcome | null
  profile: null | string
  runId: null | number
  startedAt: null | number
}

export interface ActivityTask {
  assignee: null | string
  blockReason: null | string
  children: ActivityTask[]
  id: string
  parents: string[]
  run: ActivityRun | null
  status: KanbanTaskStatus
  title: string
}

export interface ActivityBoard {
  checkedAt: number
  error: null | string
  name: string
  roots: ActivityTask[]
  truncated: boolean
}

export interface KanbanActivityModel {
  boards: ActivityBoard[]
  checkedAt: number
  diagnostics: string[]
}

export interface ActivityCounts {
  active: number
  attention: number
  completed: number
  queued: number
  total: number
}

// Truncation-aware label segments. `title` renders in the default foreground,
// `owner` dim, and `state` in the row tone — the segments exist so the
// component can paint each signal on its own channel.
export interface ActivityLabelParts {
  owner: null | string
  state: null | string
  title: string
}

export interface ActivityTaskRow {
  board: string
  depth: number
  glyph: '!' | '×' | '○' | '●' | '◉'
  kind: 'task'
  label: string
  parts: ActivityLabelParts
  prefix: string
  stateLabel: string
  task: ActivityTask
  tone: ActivityTone
}

// `label` is the width-budgeted board name (possibly empty at extreme widths)
// and `suffix` the warn-toned error marker; together they always fit `width`.
export interface ActivityBoardRow {
  board: string
  error: boolean
  kind: 'board'
  label: string
  suffix: null | string
}

export interface ActivitySummaryRow {
  board: string
  kind: 'summary'
  label: string
  prefix: string
  tone: 'muted'
}

export type KanbanActivityRowModel = ActivityBoardRow | ActivitySummaryRow | ActivityTaskRow

const STATUSES = new Set<KanbanTaskStatus>([
  'archived',
  'blocked',
  'done',
  'ready',
  'review',
  'running',
  'scheduled',
  'todo',
  'triage'
])

const FAILED_OUTCOMES = new Set(['crashed', 'failed', 'gave_up', 'spawn_failed', 'stopped', 'timed_out'])
const DEFAULT_STALE_SECONDS = 300
const DEFAULT_COMPLETION_LINGER_SECONDS = 300
const MAX_DEPTH = 32
const MAX_TASKS = 1000


function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' ? (value as Record<string, unknown>) : null
}

function finiteNumber(value: unknown, fallback: null | number = null): null | number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function cleanText(value: unknown, fallback = '', max = 160): string {
  const cleaned = typeof value === 'string' ? value.replace(/\s+/g, ' ').trim() : ''

  return (cleaned || fallback).slice(0, max)
}

function normalizeRun(value: unknown): ActivityRun | null {
  const raw = record(value)

  if (!raw) {
    return null
  }

  const runId = finiteNumber(raw.run_id)

  return {
    endedAt: finiteNumber(raw.ended_at),
    lastHeartbeatAt: finiteNumber(raw.last_heartbeat_at),
    maxRuntimeSeconds: finiteNumber(raw.max_runtime_seconds),
    outcome: cleanText(raw.outcome, '', 40) || null,
    profile: cleanText(raw.profile, '', 80) || null,
    runId,
    startedAt: finiteNumber(raw.started_at)
  }
}


export function normalizeKanbanActivity(payload: unknown, now = Date.now() / 1000): KanbanActivityModel {
  const raw = record(payload)
  const diagnostics: string[] = []
  let visited = 0

  const normalizeTask = (
    value: unknown,
    pathIds: ReadonlySet<string>,
    pathObjects: ReadonlySet<object>,
    depth: number
  ): ActivityTask | null => {
    const rawTask = record(value)

    if (!rawTask) {
      diagnostics.push('malformed-task')

      return null
    }

    const id = cleanText(rawTask.task_id, '', 128)

    if (!id) {
      diagnostics.push('missing-task-id')

      return null
    }

    if (pathIds.has(id) || pathObjects.has(rawTask)) {
      diagnostics.push(`cycle:${id}`)

      return null
    }

    if (depth > MAX_DEPTH || visited >= MAX_TASKS) {
      diagnostics.push('task-limit')

      return null
    }

    visited += 1
    const status = STATUSES.has(rawTask.status as KanbanTaskStatus) ? (rawTask.status as KanbanTaskStatus) : 'todo'

    if (status !== rawTask.status) {
      diagnostics.push(`invalid-status:${id}`)
    }

    const nextIds = new Set(pathIds).add(id)
    const nextObjects = new Set(pathObjects).add(rawTask)
    const rawChildren = Array.isArray(rawTask.children) ? rawTask.children : []

    if (rawTask.children !== undefined && !Array.isArray(rawTask.children)) {
      diagnostics.push(`malformed-children:${id}`)
    }

    const children = rawChildren
      .map(child => normalizeTask(child, nextIds, nextObjects, depth + 1))
      .filter((child): child is ActivityTask => child !== null)

    return {
      assignee: cleanText(rawTask.assignee, '', 80) || null,
      blockReason: cleanText(rawTask.block_reason, '', 120) || null,
      children,
      id,
      parents: Array.isArray(rawTask.parents)
        ? rawTask.parents
            .map(parent => cleanText(parent, '', 128))
            .filter(Boolean)
            .sort()
        : [],
      run: normalizeRun(rawTask.run),
      status,
      title: cleanText(rawTask.title, 'Untitled task', 160)
    }
  }

  const boardsSource = Array.isArray(raw?.boards) ? raw.boards : []

  const boards = boardsSource
    .map((value, index): ActivityBoard | null => {
      const rawBoard = record(value)

      if (!rawBoard) {
        diagnostics.push(`malformed-board:${index}`)

        return null
      }

      const name = cleanText(rawBoard.board, `board-${index}`, 80)

      const roots = (Array.isArray(rawBoard.roots) ? rawBoard.roots : [])
        .map(root => normalizeTask(root, new Set(), new Set(), 0))
        .filter((root): root is ActivityTask => root !== null)

      return {
        checkedAt: finiteNumber(rawBoard.checked_at, now) ?? now,
        error: cleanText(rawBoard.error, '', 160) || null,
        name,
        roots,
        truncated: rawBoard.truncated === true
      }
    })
    .filter((board): board is ActivityBoard => board !== null)
    .sort((a, b) => a.name.localeCompare(b.name))

  return {
    boards,
    checkedAt: finiteNumber(raw?.checked_at, now) ?? now,
    diagnostics: [
      ...new Set([
        ...(Array.isArray(raw?.diagnostics) ? raw.diagnostics.map(item => cleanText(item)).filter(Boolean) : []),
        ...diagnostics
      ])
    ]
  }
}

export function heartbeatFreshness(
  task: ActivityTask,
  now: number,
  staleAfterSeconds = DEFAULT_STALE_SECONDS
): HeartbeatFreshness {
  if (task.status !== 'running' || (task.run?.endedAt !== null && task.run?.endedAt !== undefined)) {
    return 'not-applicable'
  }

  if (!task.run || task.run.lastHeartbeatAt === null) {
    return 'missing'
  }

  return now - task.run.lastHeartbeatAt > staleAfterSeconds ? 'stale' : 'fresh'
}

function isFailed(task: ActivityTask): boolean {
  return Boolean(task.run?.outcome && FAILED_OUTCOMES.has(task.run.outcome))
}

// Lifecycle channel only: glyph + state word + tone. Topology (the rail) is
// built separately and stays muted, and accent is reserved for live work —
// queued states (ready/review) render neutral so triage reads at a glance.
export function activityPresentation(
  task: ActivityTask,
  now: number,
  staleAfterSeconds = DEFAULT_STALE_SECONDS
): Pick<ActivityTaskRow, 'glyph' | 'stateLabel' | 'tone'> {
  if (isFailed(task)) {
    return { glyph: '×', stateLabel: task.run!.outcome!, tone: 'error' }
  }

  if (task.status === 'blocked') {
    const reason = task.blockReason ? `blocked: ${task.blockReason}` : 'blocked'

    return { glyph: '!', stateLabel: reason, tone: 'warning' }
  }

  if (task.status === 'triage') {
    return { glyph: '○', stateLabel: 'triage', tone: 'muted' }
  }

  if (task.status === 'running') {
    const freshness = heartbeatFreshness(task, now, staleAfterSeconds)

    if (freshness === 'stale') {
      return { glyph: '!', stateLabel: 'heartbeat stale', tone: 'warning' }
    }

    if (freshness === 'missing') {
      return { glyph: '!', stateLabel: 'heartbeat unavailable', tone: 'warning' }
    }

    return { glyph: '◉', stateLabel: 'running', tone: 'accent' }
  }

  if (task.status === 'review') {
    return { glyph: '○', stateLabel: 'review queued', tone: 'neutral' }
  }

  if (task.status === 'scheduled') {
    return { glyph: '○', stateLabel: 'scheduled', tone: 'muted' }
  }

  if (task.status === 'ready') {
    return { glyph: '○', stateLabel: 'ready', tone: 'neutral' }
  }

  if (task.status === 'done') {
    return { glyph: '●', stateLabel: 'completed', tone: 'success' }
  }

  if (task.status === 'archived') {
    return { glyph: '●', stateLabel: 'archived', tone: 'muted' }
  }

  return { glyph: '○', stateLabel: task.status, tone: 'muted' }
}

function isRecentTerminal(task: ActivityTask, now: number, lingerSeconds: number): boolean {
  const endedAt = task.run?.endedAt

  return endedAt !== null && endedAt !== undefined && now - endedAt >= 0 && now - endedAt <= lingerSeconds
}

function needsAttention(
  task: ActivityTask,
  now: number,
  staleAfterSeconds: number,
  completionLingerSeconds: number
): boolean {
  if (task.status === 'blocked') {
    return true
  }

  const freshness = heartbeatFreshness(task, now, staleAfterSeconds)

  return freshness === 'missing' || freshness === 'stale' || (isFailed(task) && isRecentTerminal(task, now, completionLingerSeconds))
}

export function aggregateKanbanActivity(
  model: KanbanActivityModel,
  now = model.checkedAt,
  staleAfterSeconds = DEFAULT_STALE_SECONDS,
  completionLingerSeconds = DEFAULT_COMPLETION_LINGER_SECONDS
): ActivityCounts {
  const counts: ActivityCounts = { active: 0, attention: 0, completed: 0, queued: 0, total: 0 }

  const walk = (tasks: readonly ActivityTask[]) => {
    for (const task of tasks) {
      counts.total += 1

      if (task.status === 'running' && !isFailed(task)) {
        counts.active += 1
      }

      if (needsAttention(task, now, staleAfterSeconds, completionLingerSeconds)) {
        counts.attention += 1
      }

      if (
        (task.status === 'done' || task.status === 'archived') &&
        !isFailed(task) &&
        isRecentTerminal(task, now, completionLingerSeconds)
      ) {
        counts.completed += 1
      }

      if (['ready', 'review', 'scheduled', 'todo', 'triage'].includes(task.status)) {
        counts.queued += 1
      }

      walk(task.children)
    }
  }

  for (const board of model.boards) {
    if (board.error) {
      counts.attention += 1
    }

    walk(board.roots)
  }

  return counts
}

export function truncateActivityLabel(value: string, width: number): string {
  const clean = value.replace(/\s+/g, ' ').trim()

  if (width <= 0) {
    return ''
  }

  if (stringWidth(clean) <= width) {
    return clean
  }

  if (width === 1) {
    return '…'
  }

  const chars = Array.from(clean)

  while (chars.length && stringWidth(chars.join('')) > width - 1) {
    chars.pop()
  }

  return `${chars.join('')}…`
}

const MIN_TITLE_WIDTH = 8

function shortActivityState(task: ActivityTask, now: number, staleAfterSeconds = DEFAULT_STALE_SECONDS): string {
  return task.status === 'blocked' ? 'blocked' : activityPresentation(task, now, staleAfterSeconds).stateLabel
}

export function joinActivityLabelParts(parts: ActivityLabelParts): string {
  const owner = parts.owner === null ? '' : ` — ${parts.owner}`
  const state = parts.state === null ? '' : ` · ${parts.state}`

  return `${parts.title}${owner}${state}`
}

// Truncation drops the least important signal first: blocked-reason detail,
// then the owner, then title characters. The state word goes last — the glyph
// and tone still encode it, but the word is the primary carrier.
export function composeActivityLabel(
  task: ActivityTask,
  width: number,
  now: number,
  staleAfterSeconds = DEFAULT_STALE_SECONDS
): ActivityLabelParts {
  const { stateLabel } = activityPresentation(task, now, staleAfterSeconds)
  const owner = task.assignee ?? task.run?.profile ?? null
  const shortState = shortActivityState(task, now, staleAfterSeconds)

  const candidates: ActivityLabelParts[] = [
    { owner, state: stateLabel, title: task.title },
    { owner, state: shortState, title: task.title },
    { owner: null, state: shortState, title: task.title }
  ]

  for (const candidate of candidates) {
    if (stringWidth(joinActivityLabelParts(candidate)) <= width) {
      return candidate
    }
  }

  const titleWidth = width - stringWidth(` · ${shortState}`)

  if (titleWidth >= MIN_TITLE_WIDTH) {
    return { owner: null, state: shortState, title: truncateActivityLabel(task.title, titleWidth) }
  }

  return { owner: null, state: null, title: truncateActivityLabel(task.title, width) }
}

export function focusedActivityLabel(
  task: ActivityTask,
  width: number,
  now: number,
  staleAfterSeconds = DEFAULT_STALE_SECONDS
): string {
  return joinActivityLabelParts(composeActivityLabel(task, width, now, staleAfterSeconds))
}

const MIN_HEADLINE_WIDTH = 12

export interface ActivityDockSegment {
  text: string
  tone: ActivityTone
}

function segmentDockCounts(label: string, counts: ActivityCounts): ActivityDockSegment[] {
  const markers: ActivityDockSegment[] = counts.attention
    ? [
        { text: String(counts.attention), tone: 'warning' },
        { text: String(counts.active), tone: 'accent' }
      ]
    : counts.active
      ? [{ text: String(counts.active), tone: 'accent' }]
      : [{ text: String(counts.completed), tone: 'success' }]

  const segments: ActivityDockSegment[] = []
  let cursor = 0

  const push = (text: string, tone: ActivityTone) => {
    if (!text) {
      return
    }

    const previous = segments.at(-1)

    if (previous?.tone === tone) {
      previous.text += text
    } else {
      segments.push({ text, tone })
    }
  }

  for (const marker of markers) {
    const index = label.indexOf(marker.text, cursor)

    if (index < 0) {
      continue
    }

    push(label.slice(cursor, index), 'neutral')
    push(marker.text, marker.tone)
    cursor = index + marker.text.length
  }

  push(label.slice(cursor), 'neutral')

  return segments
}

// Narrow-dock shorthand never shows a zero-count badge: attention and active
// appear only when present, and completed-only recent activity reports itself
// as '●N' instead of a false '◉0'.
function shorthandBadges(counts: ActivityCounts): ActivityDockSegment[] {
  const badges: ActivityDockSegment[] = []

  if (counts.attention) {
    badges.push({ text: `!${counts.attention}`, tone: 'warning' })
  }

  if (counts.active) {
    badges.push({ text: `◉${counts.active}`, tone: 'accent' })
  }

  if (!counts.attention && !counts.active && counts.completed) {
    badges.push({ text: `●${counts.completed}`, tone: 'success' })
  }

  return badges
}

// Tone-tagged view of the collapsed dock so the component can paint 'K'
// neutral, '!N' warning, and '◉N' accent as separate segments below the
// medium breakpoint. Above it the single label renders in the default
// foreground behind the dock's lamp glyph.
export function collapsedActivitySegments(
  model: KanbanActivityModel,
  width: number,
  now = model.checkedAt,
  layoutWidth = width
): ActivityDockSegment[] {
  const counts = aggregateKanbanActivity(model, now)

  if (!counts.total && !counts.attention) {
    return [{ text: truncateActivityLabel('Kanban · idle', width), tone: 'muted' }]
  }

  if (layoutWidth >= 28) {
    return segmentDockCounts(collapsedActivityLabel(model, width, now, layoutWidth), counts)
  }

  const badges = shorthandBadges(counts)

  const segments: ActivityDockSegment[] =
    width >= 14
      ? [{ text: 'K', tone: 'neutral' }, ...badges.map(badge => ({ ...badge, text: ` ${badge.text}` }))]
      : badges

  const total = segments.reduce((sum, segment) => sum + stringWidth(segment.text), 0)

  if (total > width) {
    return [{ text: collapsedActivityLabel(model, width, now), tone: badges[0]?.tone ?? 'muted' }]
  }

  return segments
}

// The wide-dock headline is the single most attention-worthy task: a fresh
// failure outranks a block, which outranks a degraded heartbeat, which
// outranks ordinary running work.
function headlineTask(tasks: readonly ActivityTask[], now: number): ActivityTask | null {
  const ranks: ((task: ActivityTask) => boolean)[] = [
    task => isFailed(task) && isRecentTerminal(task, now, DEFAULT_COMPLETION_LINGER_SECONDS),
    task => task.status === 'blocked',
    task => {
      const freshness = heartbeatFreshness(task, now)

      return freshness === 'missing' || freshness === 'stale'
    },
    task => task.status === 'running' && !isFailed(task),
    task => isRecentTerminal(task, now, DEFAULT_COMPLETION_LINGER_SECONDS)
  ]

  for (const matches of ranks) {
    const found = tasks.find(matches)

    if (found) {
      return found
    }
  }

  return tasks[0] ?? null
}

export function collapsedActivityLabel(
  model: KanbanActivityModel,
  width: number,
  now = model.checkedAt,
  layoutWidth = width
): string {
  const counts = aggregateKanbanActivity(model, now)

  if (!counts.total && !counts.attention) {
    return truncateActivityLabel('Kanban · idle', width)
  }

  if (layoutWidth < 14) {
    return truncateActivityLabel(
      shorthandBadges(counts)
        .map(badge => badge.text)
        .join(''),
      width
    )
  }

  if (layoutWidth < 28) {
    return truncateActivityLabel(
      ['K', ...shorthandBadges(counts).map(badge => badge.text)].join(' '),
      width
    )
  }

  const lead = counts.attention
    ? `${counts.attention} need attention · ${counts.active} active`
    : counts.active
      ? `${counts.active} active`
      : `${counts.completed} completed`

  const medium = `Kanban · ${lead}`

  if (layoutWidth < 48) {
    return truncateActivityLabel(medium, width)
  }

  const tasks: ActivityTask[] = []

  const collect = (nodes: readonly ActivityTask[]) => {
    for (const task of nodes) {
      tasks.push(task)
      collect(task.children)
    }
  }

  model.boards.forEach(board => collect(board.roots))

  const headline = headlineTask(tasks, now)

  if (!headline) {
    return truncateActivityLabel(medium, width)
  }

  const headlineText = needsAttention(headline, now, DEFAULT_STALE_SECONDS, DEFAULT_COMPLETION_LINGER_SECONDS)
    ? `${headline.title} ${shortActivityState(headline, now)}`
    : headline.title

  const wide = `${medium} · ${headlineText}`

  if (stringWidth(wide) <= width) {
    return wide
  }

  // A headline fragment under ~12 cells reads as noise — fall back to counts.
  const headlineWidth = width - stringWidth(`${medium} · `)

  if (headlineWidth >= MIN_HEADLINE_WIDTH) {
    return `${medium} · ${truncateActivityLabel(headlineText, headlineWidth)}`
  }

  return truncateActivityLabel(medium, width)
}

export function branchChildren<T>(children: readonly T[], visibleLimit = 5): { omitted: number; visible: T[] } {
  const limit = Math.max(0, Math.floor(visibleLimit))

  return { omitted: Math.max(0, children.length - limit), visible: children.slice(0, limit) }
}

const RAIL_PREFIX = '│ '
const BRANCH_CONNECTOR = '├── '
const LAST_CONNECTOR = '└── '
const CONTINUE_SEGMENT = '│   '
const GAP_SEGMENT = '    '

// One '│   '/'    ' segment per ancestor level below the root keeps branch
// lines attached to their parent at depth 2+ ('│   ├── …'). The topology
// prefix is a single light rail — lifecycle never changes its weight.
function continuationPrefix(ancestors: readonly boolean[]): string {
  return ancestors.map(hasMore => (hasMore ? CONTINUE_SEGMENT : GAP_SEGMENT)).join('')
}

const BOARD_ERROR_SUFFIX = ' · unavailable'
const MIN_BOARD_NAME_WIDTH = 2

// Name and error suffix share one width budget so the row can never wrap.
// When the budget cannot hold both, the error signal outranks identity.
function boardRowPresentation(name: string, error: boolean, width: number): Pick<ActivityBoardRow, 'label' | 'suffix'> {
  if (!error) {
    return { label: truncateActivityLabel(name, width), suffix: null }
  }

  const nameWidth = width - stringWidth(BOARD_ERROR_SUFFIX)

  if (nameWidth >= MIN_BOARD_NAME_WIDTH) {
    return { label: truncateActivityLabel(name, nameWidth), suffix: BOARD_ERROR_SUFFIX }
  }

  return { label: '', suffix: truncateActivityLabel('unavailable', width) }
}

export function buildKanbanActivityRows(
  model: KanbanActivityModel,
  options: { maxChildren?: number; now?: number; staleAfterSeconds?: number; width?: number } = {}
): KanbanActivityRowModel[] {
  const rows: KanbanActivityRowModel[] = []
  const now = options.now ?? model.checkedAt
  const staleAfterSeconds = options.staleAfterSeconds ?? DEFAULT_STALE_SECONDS
  const maxChildren = options.maxChildren ?? 5
  const width = options.width ?? 120
  const showBoards = model.boards.length > 1
  const taskTitles = new Map<string, string>()

  const indexTasks = (tasks: readonly ActivityTask[]) => {
    for (const task of tasks) {
      taskTitles.set(task.id, task.title)
      indexTasks(task.children)
    }
  }

  model.boards.forEach(board => indexTasks(board.roots))

  const walk = (
    board: string,
    tasks: readonly ActivityTask[],
    omittedAfter: number,
    depth: number,
    parentId: null | string,
    ancestors: readonly boolean[]
  ) => {
    tasks.forEach((task, index) => {
      const moreAfter = index < tasks.length - 1 || omittedAfter > 0
      const presentation = activityPresentation(task, now, staleAfterSeconds)

      const prefix =
        depth === 0 ? RAIL_PREFIX : `${continuationPrefix(ancestors)}${moreAfter ? BRANCH_CONNECTOR : LAST_CONNECTOR}`

      const labelWidth = Math.max(1, width - stringWidth(`${prefix}${presentation.glyph} `))
      const parts = composeActivityLabel(task, labelWidth, now, staleAfterSeconds)
      rows.push({
        board,
        depth,
        ...presentation,
        kind: 'task',
        label: joinActivityLabelParts(parts),
        parts,
        prefix,
        task
      })

      const childAncestors = depth === 0 ? [] : [...ancestors, moreAfter]
      const { omitted, visible } = branchChildren(task.children, maxChildren)
      const linkedParents = task.parents.filter(id => id !== parentId)
      linkedParents.forEach((linkedParent, linkedIndex) => {
        const followed = linkedIndex < linkedParents.length - 1 || visible.length > 0 || omitted > 0
        rows.push({
          board,
          kind: 'summary',
          label: `↳ also linked from ${taskTitles.get(linkedParent) ?? linkedParent}`,
          prefix: `${continuationPrefix(childAncestors)}${followed ? BRANCH_CONNECTOR : LAST_CONNECTOR}`,
          tone: 'muted'
        })
      })

      walk(board, visible, omitted, depth + 1, task.id, childAncestors)

      if (omitted) {
        rows.push({
          board,
          kind: 'summary',
          label: `+${omitted} more tasks`,
          prefix: `${continuationPrefix(childAncestors)}${LAST_CONNECTOR}`,
          tone: 'muted'
        })
      }
    })
  }

  for (const board of model.boards) {
    if (showBoards || board.error) {
      rows.push({
        board: board.name,
        error: Boolean(board.error),
        kind: 'board',
        ...boardRowPresentation(board.name, Boolean(board.error), width)
      })
    }

    walk(board.name, board.roots, 0, 0, null, [])

    if (board.truncated) {
      rows.push({
        board: board.name,
        kind: 'summary',
        label: '+ more activity not shown',
        prefix: LAST_CONNECTOR,
        tone: 'muted'
      })
    }
  }

  return rows
}

// Compile-time alignment checks for the wire contract. These are intentionally
// dormant: no gateway request or live TUI import is introduced here.
export type KanbanActivityWireContract = KanbanActivityResponse
export type KanbanActivityBoardWireContract = KanbanActivityBoard
export type KanbanActivityTaskWireContract = KanbanActivityTask
export type KanbanActivityRunWireContract = KanbanActivityRun
