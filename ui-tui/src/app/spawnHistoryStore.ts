import { atom } from 'nanostores'

import type { SpawnTreeLoadResponse } from '../gatewayTypes.js'
import type { SubagentProgress, SubagentStatus } from '../types.js'

export interface SpawnSnapshot {
  finishedAt: number
  fromDisk?: boolean
  id: string
  label: string
  path?: string
  sessionId: null | string
  startedAt: number
  subagents: SubagentProgress[]
}

export interface SpawnDiffPair {
  baseline: SpawnSnapshot
  candidate: SpawnSnapshot
}

const HISTORY_LIMIT = 10

const KNOWN_SUBAGENT_STATUSES = new Set<SubagentStatus>([
  'completed',
  'error',
  'failed',
  'interrupted',
  'queued',
  'running',
  'timeout'
])

const normalizeSubagentStatus = (status: unknown, fallback: SubagentStatus): SubagentStatus => {
  if (typeof status !== 'string') {
    return fallback
  }

  const normalized = status.toLowerCase() as SubagentStatus

  return KNOWN_SUBAGENT_STATUSES.has(normalized) ? normalized : fallback
}

export const $spawnHistory = atom<SpawnSnapshot[]>([])
export const $spawnDiff = atom<null | SpawnDiffPair>(null)

export const getSpawnHistory = () => $spawnHistory.get()
export const getSpawnDiff = () => $spawnDiff.get()

export const clearSpawnHistory = () => $spawnHistory.set([])
export const clearDiffPair = () => $spawnDiff.set(null)
export const setDiffPair = (pair: SpawnDiffPair) => $spawnDiff.set(pair)

/**
 * Commit a finished turn's spawn tree to history.  Keeps the last 10
 * non-empty snapshots — empty turns (no subagents) are dropped.
 *
 * Why in-memory?  The primary investigation loop is "I just ran a fan-out,
 * it misbehaved, let me look at what happened" — same-session debugging.
 * Disk persistence across process restarts is a natural extension but
 * adds RPC surface for a less-common path.
 */
export const pushSnapshot = (
  subagents: readonly SubagentProgress[],
  meta: { sessionId?: null | string; startedAt?: null | number }
) => {
  if (!subagents.length) {
    return
  }

  const now = Date.now()
  const started = meta.startedAt ?? Math.min(...subagents.map(s => s.startedAt ?? now))

  const snap: SpawnSnapshot = {
    finishedAt: now,
    id: `snap-${now.toString(36)}`,
    label: summarizeLabel(subagents),
    sessionId: meta.sessionId ?? null,
    startedAt: Number.isFinite(started) ? started : now,
    subagents: subagents.map(item => ({ ...item }))
  }

  const next = [snap, ...$spawnHistory.get()].slice(0, HISTORY_LIMIT)
  $spawnHistory.set(next)
}

function summarizeLabel(subagents: readonly SubagentProgress[]): string {
  const top = subagents
    .filter(s => s.parentId == null || subagents.every(o => o.id !== s.parentId))
    .slice(0, 2)
    .map(s => s.goal || 'subagent')
    .join(' · ')

  return top || `${subagents.length} agent${subagents.length === 1 ? '' : 's'}`
}

/**
 * Push a disk-loaded snapshot onto the front of the history stack so the
 * overlay can pick it up at index 1 via /replay load.  Normalises the
 * server payload (arbitrary list) into the same SubagentProgress shape
 * used for live data — defensive against cross-version reads.
 */
export const pushDiskSnapshot = (r: SpawnTreeLoadResponse, path: string) => {
  const raw = Array.isArray(r.subagents) ? r.subagents : []
  const normalised = raw.map(normaliseSubagent)

  if (!normalised.length) {
    return
  }

  const snap: SpawnSnapshot = {
    finishedAt: (r.finished_at ?? Date.now() / 1000) * 1000,
    fromDisk: true,
    id: `disk-${path}`,
    label: r.label || `${normalised.length} subagents`,
    path,
    sessionId: r.session_id ?? null,
    startedAt: (r.started_at ?? r.finished_at ?? Date.now() / 1000) * 1000,
    subagents: normalised
  }

  const next = [snap, ...$spawnHistory.get()].slice(0, HISTORY_LIMIT)
  $spawnHistory.set(next)
}

function normaliseSubagent(raw: unknown): SubagentProgress {
  const o = raw as Record<string, unknown>
  const s = (v: unknown) => (typeof v === 'string' ? v : undefined)
  const n = (v: unknown) => (typeof v === 'number' ? v : undefined)
  const arr = <T>(v: unknown): T[] | undefined => (Array.isArray(v) ? (v as T[]) : undefined)

  return {
    apiCalls: n(o.apiCalls),
    costUsd: n(o.costUsd),
    depth: typeof o.depth === 'number' ? o.depth : 0,
    durationSeconds: n(o.durationSeconds),
    filesRead: arr<string>(o.filesRead),
    filesWritten: arr<string>(o.filesWritten),
    goal: s(o.goal) ?? 'subagent',
    id: s(o.id) ?? `sa-${Math.random().toString(36).slice(2, 8)}`,
    index: typeof o.index === 'number' ? o.index : 0,
    inputTokens: n(o.inputTokens),
    iteration: n(o.iteration),
    model: s(o.model),
    notes: (arr<string>(o.notes) ?? []).filter(x => typeof x === 'string'),
    outputTail: arr(o.outputTail) as SubagentProgress['outputTail'],
    outputTokens: n(o.outputTokens),
    parentId: s(o.parentId) ?? null,
    reasoningTokens: n(o.reasoningTokens),
    startedAt: n(o.startedAt),
    status: normalizeSubagentStatus(o.status, 'completed'),
    summary: s(o.summary),
    taskCount: typeof o.taskCount === 'number' ? o.taskCount : 1,
    thinking: (arr<string>(o.thinking) ?? []).filter(x => typeof x === 'string'),
    toolCount: typeof o.toolCount === 'number' ? o.toolCount : 0,
    tools: (arr<string>(o.tools) ?? []).filter(x => typeof x === 'string'),
    toolsets: arr<string>(o.toolsets)
  }
}
