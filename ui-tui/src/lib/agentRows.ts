import type { AsyncDelegationRecord } from '../gatewayTypes.js'
import type { SubagentProgress } from '../types.js'

// Pure merge + layout logic for the docked agents panel. Kept ink-free so it is
// unit testable on its own; agentsPanel.tsx owns only the presentation.

// One merged panel row — either a live in-turn subagent or a background
// async delegation, normalised to the same shape so the view stays dumb.
export interface AgentRow {
  detail: string
  elapsedSeconds: null | number
  goal: string
  /** Shortest unambiguous prefix of the agent id — the token the user types
   * back as `@<id> steer text`. Empty when the row carries no id. */
  id: string
  key: string
  name: string
  resultReady: boolean
  status: string
}

export interface AgentRows {
  done: number
  /** Rows that exist but were cut by the height bound, so the panel can say so
   * instead of silently lying about how many agents are in flight. */
  hidden: number
  rows: AgentRow[]
  running: number
}

/** Hard height bound for the docked panel. It sits between the transcript and
 * the composer as fixed chrome, so every row it paints is a row the transcript
 * loses forever — it is never allowed to grow with history. The full list
 * lives in the `/agents` overlay. */
export const PANEL_MAX_ROWS = 5

export const dockRowLimit = (height: number): number =>
  Math.max(1, Math.min(PANEL_MAX_ROWS, Math.floor((height - 10) / 6)))

/** Statuses that mean "this agent is still going" — the union of the live
 * subagent vocabulary and the async delegation one. */
const IN_FLIGHT = new Set(['dispatched', 'finalizing', 'queued', 'running'])

const RESULT_READY = new Set(['completed', 'done'])

/** Never abbreviate an id below this many characters — shorter prefixes are
 * too easy to typo into a different agent. */
const MIN_ID = 4

/** Shortest prefix of `id` that no other live id shares, floored at MIN_ID.
 * The panel prints this and `resolveSteerTargetId` accepts it, so what the user
 * reads is exactly what they can type back. */
export const shortAgentId = (id: string, all: readonly string[]): string => {
  for (let n = MIN_ID; n < id.length; n += 1) {
    const p = id.slice(0, n)

    if (!all.some(other => other !== id && other.startsWith(p))) {
      return p
    }
  }

  return id
}

// Live subagent elapsed: prefer a settled duration, else clock from startedAt
// while still running. Mirrors the overlay's displayElapsedSeconds.
const liveElapsed = (item: SubagentProgress, nowMs: number): null | number => {
  if (item.durationSeconds != null) {
    return item.durationSeconds
  }

  if (item.startedAt != null && IN_FLIGHT.has(item.status)) {
    return Math.max(0, (nowMs - item.startedAt) / 1000)
  }

  return null
}

/** Drop the batch rows whose own children are already on screen as live rows.
 *
 * A background fan-out is ONE registry record covering N children, but those
 * children also stream live subagent events, so the naive merge paints N+1
 * rows and counts N+1 agents for N agents. While the children are reporting
 * they are the better row (per-child goal, tool and elapsed, and each is
 * individually steerable by its own `@id`), so the batch row is suppressed.
 * Once they clear at the turn boundary — or when the batch finishes and the
 * `result ready ⏎` cue is the whole point — the batch row comes back. */
const dropCoveredBatches = (
  subagents: readonly SubagentProgress[],
  asyncDelegations: readonly AsyncDelegationRecord[]
): readonly AsyncDelegationRecord[] => {
  const live = new Set(subagents.filter(s => IN_FLIGHT.has(s.status)).map(s => s.id))

  if (live.size === 0) {
    return asyncDelegations
  }

  return asyncDelegations.filter(d => {
    const covered = IN_FLIGHT.has(d.status ?? 'running') && (d.subagent_ids ?? []).some(id => live.has(id))

    return !covered
  })
}

/** Merge live in-turn subagents with background async delegations into one
 * ordered, height-bounded row list. In-flight rows first (they carry the
 * freshest tool/elapsed signal), then recently finished rows newest-first. */
export const buildAgentRows = (
  subagents: SubagentProgress[],
  asyncDelegations: readonly AsyncDelegationRecord[],
  nowMs: number,
  maxRows: number = PANEL_MAX_ROWS
): AgentRows => {
  const delegations = dropCoveredBatches(subagents, asyncDelegations)
  const ids = [...subagents.map(s => s.id), ...delegations.map(d => d.delegation_id)]
  const active: AgentRow[] = []
  let running = 0
  let done = 0

  for (const s of subagents) {
    const row: AgentRow = {
      detail: s.notes.at(-1) || s.outputTail?.at(-1)?.preview || s.tools.at(-1) || 'Starting…',
      elapsedSeconds: liveElapsed(s, nowMs),
      goal: s.goal || 'agent',
      id: shortAgentId(s.id, ids),
      key: `live:${s.id}`,
      name: '',
      resultReady: false,
      status: s.status
    }

    if (IN_FLIGHT.has(s.status)) {
      running += 1
      active.push(row)
    } else {
      if (RESULT_READY.has(s.status)) {
        done += 1
      }
    }
  }

  for (const d of delegations) {
    const status = d.status ?? 'running'
    const inFlight = IN_FLIGHT.has(status)
    const resultReady = RESULT_READY.has(status)
    const endMs = !inFlight && d.completed_at != null ? d.completed_at * 1000 : nowMs
    const elapsedSeconds = d.dispatched_at != null ? Math.max(0, (endMs - d.dispatched_at * 1000) / 1000) : null

    const row: AgentRow = {
      detail: resultReady ? 'result ready' : status,
      elapsedSeconds,
      goal: d.goal ?? '',
      id: shortAgentId(d.delegation_id, ids),
      key: `async:${d.delegation_id}`,
      name: d.role ?? 'agent',
      resultReady,
      status
    }

    if (inFlight) {
      running += 1
      active.push(row)

      continue
    }

    if (resultReady) {
      done += 1
    }
  }

  const candidates = active
  const rows = maxRows > 0 ? candidates.slice(0, maxRows) : candidates

  return { done, hidden: candidates.length - rows.length, rows, running }
}
