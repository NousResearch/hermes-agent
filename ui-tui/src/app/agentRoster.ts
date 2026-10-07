import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useMemo } from 'react'

import type { SubagentListResponse } from '../gatewayTypes.js'
import type { SubagentProgress } from '../types.js'

import { useTurnSelector } from './turnStore.js'
import { $uiState } from './uiStore.js'

// Session-local presentation only; never persisted to config.
export const $agentDockCollapsed = atom(false)

const EMPTY: SubagentListResponse = { subagents: [], delegations: [] }
export const $agentSnapshot = atom<{ sid: string | null; data: SubagentListResponse }>({ sid: null, data: EMPTY })

export function applyAgentSnapshot(sid: string | null, data: SubagentListResponse = EMPTY) {
  const previous = $agentSnapshot.get()

  if (previous.sid !== sid || JSON.stringify(previous.data) !== JSON.stringify(data)) {
    $agentSnapshot.set({ sid, data })
  }
}

export function mergeAgentRoster(events: SubagentProgress[], data: SubagentListResponse): SubagentProgress[] {
  const merged = new Map(events.map(s => [s.id, s]))

  for (const [index, s] of data.subagents.entries()) {
    const previous = merged.get(s.subagent_id)
    merged.set(s.subagent_id, {
      depth: s.depth ?? 0,
      index,
      parentId: s.parent_id ?? null,
      notes: [],
      thinking: [],
      taskCount: 1,
      ...previous,
      id: s.subagent_id,
      goal: s.goal || previous?.goal || 'Starting agent',
      delegationId: s.delegation_id ?? previous?.delegationId,
      model: s.model ?? previous?.model,
      startedAt: s.started_at != null ? s.started_at * 1000 : previous?.startedAt,
      // Snapshot replies may predate progress/completion events already rendered.
      status: previous && previous.status !== 'queued' ? previous.status : s.status === 'queued' ? 'queued' : 'running',
      toolCount: Math.max(s.tool_count ?? 0, previous?.toolCount ?? 0),
      tools: previous?.tools.length ? previous.tools : s.last_tool ? [s.last_tool] : []
    })
  }

  // Async records are completion units, not agents. Independent completions can
  // give them different IDs from the actual children; never invent extra rows.

  return [...merged.values()]
}

export function useAgentRoster() {
  const events = useTurnSelector(s => s.subagents)
  const snapshot = useStore($agentSnapshot)
  const { sid } = useStore($uiState)

  return useMemo(() => mergeAgentRoster(events, snapshot.sid === sid ? snapshot.data : EMPTY), [events, snapshot, sid])
}
