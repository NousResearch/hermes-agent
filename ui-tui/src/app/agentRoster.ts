import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useMemo } from 'react'

import type { SubagentListResponse } from '../gatewayTypes.js'
import type { SubagentProgress } from '../types.js'

import { useTurnSelector } from './turnStore.js'
import { $uiState } from './uiStore.js'

const EMPTY: SubagentListResponse = { subagents: [], delegations: [] }
export const $agentSnapshot = atom<{ sid: string | null; data: SubagentListResponse }>({ sid: null, data: EMPTY })

export function applyAgentSnapshot(sid: string | null, data: SubagentListResponse = EMPTY) {
  const previous = $agentSnapshot.get()

  if (previous.sid !== sid || JSON.stringify(previous.data) !== JSON.stringify(data)) {$agentSnapshot.set({ sid, data })}
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
      status: s.status === 'queued' ? 'queued' : 'running',
      toolCount: s.tool_count ?? previous?.toolCount ?? 0,
      tools: previous?.tools.length ? previous.tools : s.current_tool ? [s.current_tool] : []
    })
  }

  for (const d of data.delegations) {
    if (!['running', 'finalizing', 'dispatched', 'queued'].includes(d.status ?? '')) {continue}

    if ([...merged.values()].some(s => s.delegationId === d.delegation_id || d.subagent_ids?.includes(s.id))) {continue}
    merged.set(d.delegation_id, {
      id: d.delegation_id,
      goal: d.goal || 'Starting delegation',
      depth: 0,
      index: merged.size,
      parentId: null,
      notes: [],
      thinking: [],
      tools: [],
      taskCount: 1,
      toolCount: 0,
      status: 'queued',
      startedAt: d.dispatched_at == null ? undefined : d.dispatched_at * 1000
    })
  }

  return [...merged.values()]
}

export function useAgentRoster() {
  const events = useTurnSelector(s => s.subagents)
  const snapshot = useStore($agentSnapshot)
  const { sid } = useStore($uiState)

  return useMemo(() => mergeAgentRoster(events, snapshot.sid === sid ? snapshot.data : EMPTY), [events, snapshot, sid])
}
