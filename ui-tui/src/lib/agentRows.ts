import type { AsyncDelegationRecord } from '../gatewayTypes.js'
import type { SubagentProgress } from '../types.js'

// Pure merge logic for the docked agents panel. Kept ink-free so it is unit
// testable on its own; agentsPanel.tsx owns only the presentation.

// One merged panel row — either a live in-turn subagent or a background
// async delegation, normalised to the same shape so the view stays dumb.
export interface AgentRow {
  detail: string
  elapsedSeconds: null | number
  goal: string
  key: string
  name: string
  resultReady: boolean
  status: string
}

const RESULT_READY = new Set(['completed', 'done'])

// Live subagent elapsed: prefer a settled duration, else clock from startedAt
// while still running. Mirrors the overlay's displayElapsedSeconds.
const liveElapsed = (item: SubagentProgress, nowMs: number): null | number => {
  if (item.durationSeconds != null) {
    return item.durationSeconds
  }

  if (item.startedAt != null && (item.status === 'running' || item.status === 'queued')) {
    return Math.max(0, (nowMs - item.startedAt) / 1000)
  }

  return null
}

/** Merge live in-turn subagents with background async delegations into one
 * ordered, view-ready row list. Live rows first (they carry the freshest
 * tool/elapsed signal), then background/done rows. */
export const buildAgentRows = (
  subagents: SubagentProgress[],
  asyncDelegations: readonly AsyncDelegationRecord[],
  nowMs: number
): { done: number; rows: AgentRow[]; running: number } => {
  const rows: AgentRow[] = []
  let running = 0
  let done = 0

  for (const s of subagents) {
    const isRunning = s.status === 'running' || s.status === 'queued'

    if (isRunning) {
      running += 1
    } else if (s.status === 'completed') {
      done += 1
    }

    rows.push({
      detail: s.tools.at(-1) ?? '',
      elapsedSeconds: liveElapsed(s, nowMs),
      goal: s.goal || 'agent',
      key: `live:${s.id}`,
      name: '',
      resultReady: false,
      status: s.status
    })
  }

  for (const d of asyncDelegations) {
    const status = d.status ?? 'running'
    const resultReady = RESULT_READY.has(status)

    if (status === 'running') {
      running += 1
    } else if (resultReady) {
      done += 1
    }

    const endMs = resultReady && d.completed_at != null ? d.completed_at * 1000 : nowMs
    const elapsedSeconds = d.dispatched_at != null ? Math.max(0, (endMs - d.dispatched_at * 1000) / 1000) : null

    rows.push({
      detail: resultReady ? 'result ready' : status,
      elapsedSeconds,
      goal: d.goal ?? '',
      key: `async:${d.delegation_id}`,
      name: d.role ?? 'agent',
      resultReady,
      status
    })
  }

  return { done, rows, running }
}
