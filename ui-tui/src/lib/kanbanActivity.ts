import { stringWidth } from '@hermes/ink'

import type {
  KanbanActivityBoard,
  KanbanActivityResponse,
  KanbanActivityRun,
  KanbanActivityTask,
  KanbanRunOutcome,
  KanbanTaskStatus
} from '../gatewayTypes.js'

export type ActivityTone = 'accent' | 'error' | 'muted' | 'success' | 'warning'
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

export interface ActivityTaskRow {
  board: string
  connector: '' | '├──' | '└──'
  depth: number
  glyph: '!' | '×' | '○' | '●' | '◉'
  kind: 'task'
  label: string
  rail: '┃' | '│'
  stateLabel: string
  task: ActivityTask
  tone: ActivityTone
}

export interface ActivityBoardRow {
  board: string
  kind: 'board'
  label: string
}

export interface ActivitySummaryRow {
  board: string
  connector: '├──' | '└──'
  depth: number
  kind: 'summary'
  label: string
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

export function activityPresentation(
  task: ActivityTask,
  now: number,
  staleAfterSeconds = DEFAULT_STALE_SECONDS
): Pick<ActivityTaskRow, 'glyph' | 'rail' | 'stateLabel' | 'tone'> {
  if (isFailed(task)) {
    return { glyph: '×', rail: '┃', stateLabel: task.run!.outcome!, tone: 'error' }
  }

  if (task.status === 'blocked') {
    const reason = task.blockReason ? `blocked: ${task.blockReason}` : 'blocked'

    return { glyph: '!', rail: '│', stateLabel: reason, tone: 'warning' }
  }

  if (task.status === 'triage') {
    return { glyph: '○', rail: '│', stateLabel: 'triage', tone: 'muted' }
  }

  if (task.status === 'running') {
    const freshness = heartbeatFreshness(task, now, staleAfterSeconds)

    if (freshness === 'stale') {
      return { glyph: '!', rail: '│', stateLabel: 'heartbeat stale', tone: 'warning' }
    }

    if (freshness === 'missing') {
      return { glyph: '!', rail: '│', stateLabel: 'heartbeat unavailable', tone: 'warning' }
    }

    return { glyph: '◉', rail: '│', stateLabel: 'running', tone: 'accent' }
  }

  if (task.status === 'review') {
    return { glyph: '○', rail: '│', stateLabel: 'review queued', tone: 'accent' }
  }

  if (task.status === 'scheduled') {
    return { glyph: '○', rail: '│', stateLabel: 'scheduled', tone: 'muted' }
  }

  if (task.status === 'ready') {
    return { glyph: '○', rail: '│', stateLabel: 'ready', tone: 'accent' }
  }

  if (task.status === 'done') {
    return { glyph: '●', rail: '┃', stateLabel: 'completed', tone: 'success' }
  }

  if (task.status === 'archived') {
    return { glyph: '●', rail: '┃', stateLabel: 'archived', tone: 'muted' }
  }

  return { glyph: '○', rail: '│', stateLabel: task.status, tone: 'muted' }
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

export function focusedActivityLabel(
  task: ActivityTask,
  width: number,
  now: number,
  staleAfterSeconds = DEFAULT_STALE_SECONDS
): string {
  const { stateLabel } = activityPresentation(task, now, staleAfterSeconds)
  const owner = task.assignee ?? task.run?.profile ?? 'unassigned'

  return truncateActivityLabel(`${task.title} — ${owner} · ${stateLabel}`, width)
}

export function collapsedActivityLabel(model: KanbanActivityModel, width: number, now = model.checkedAt): string {
  const counts = aggregateKanbanActivity(model, now)

  if (!counts.total && !counts.attention) {
    return truncateActivityLabel('Kanban · idle', width)
  }

  const lead = counts.attention
    ? `${counts.attention} needs attention · ${counts.active} active`
    : `${counts.active} active`

  const tasks: ActivityTask[] = []

  const collect = (nodes: readonly ActivityTask[]) => {
    for (const task of nodes) {
      tasks.push(task)
      collect(task.children)
    }
  }

  model.boards.forEach(board => collect(board.roots))

  const headline =
    tasks.find(task => needsAttention(task, now, DEFAULT_STALE_SECONDS, DEFAULT_COMPLETION_LINGER_SECONDS)) ??
    tasks.find(task => task.status === 'running') ??
    tasks.find(task => isRecentTerminal(task, now, DEFAULT_COMPLETION_LINGER_SECONDS)) ??
    tasks[0]

  const wide = headline ? `Kanban · ${lead} · ${headline.title}` : `Kanban · ${lead}`
  const medium = `Kanban · ${lead}`
  const narrow = counts.attention ? `K · !${counts.attention} · A${counts.active}` : `K · ${counts.active} active`
  const tiny = counts.attention ? `K !${counts.attention} A${counts.active}` : `K ${counts.active}`
  const source = width >= 48 ? wide : width >= 28 ? medium : width >= 14 ? narrow : tiny

  return truncateActivityLabel(source, width)
}

export function branchChildren<T>(children: readonly T[], visibleLimit = 5): { omitted: number; visible: T[] } {
  const limit = Math.max(0, Math.floor(visibleLimit))

  return { omitted: Math.max(0, children.length - limit), visible: children.slice(0, limit) }
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

  const walk = (board: string, tasks: readonly ActivityTask[], depth: number, parentId: null | string = null) => {
    tasks.forEach((task, index) => {
      const presentation = activityPresentation(task, now, staleAfterSeconds)
      const connector = depth === 0 ? '' : index === tasks.length - 1 ? '└──' : '├──'
      const prefixWidth = 2 + (depth > 0 ? depth * 2 + 4 : 0)
      rows.push({
        board,
        connector,
        depth,
        ...presentation,
        kind: 'task',
        label: focusedActivityLabel(task, Math.max(1, width - prefixWidth), now, staleAfterSeconds),
        task
      })

      for (const linkedParent of task.parents.filter(id => id !== parentId)) {
        rows.push({
          board,
          connector: '├──',
          depth: depth + 1,
          kind: 'summary',
          label: `↳ also linked from ${taskTitles.get(linkedParent) ?? linkedParent}`,
          tone: 'muted'
        })
      }

      const { omitted, visible } = branchChildren(task.children, maxChildren)
      walk(board, visible, depth + 1, task.id)

      if (omitted) {
        rows.push({
          board,
          connector: '└──',
          depth: depth + 1,
          kind: 'summary',
          label: `+${omitted} more tasks`,
          tone: 'muted'
        })
      }
    })
  }

  for (const board of model.boards) {
    if (showBoards || board.error) {
      rows.push({ board: board.name, kind: 'board', label: board.error ? `${board.name} · unavailable` : board.name })
    }

    walk(board.name, board.roots, 0)

    if (board.truncated) {
      rows.push({
        board: board.name,
        connector: '└──',
        depth: 0,
        kind: 'summary',
        label: '+ more activity not shown',
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
