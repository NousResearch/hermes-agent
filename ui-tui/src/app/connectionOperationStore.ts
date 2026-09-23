import type {
  ConnectionOperationTarget,
  ConnectionRequestPayload,
  ConnectionUpdatePayload
} from '@hermes/shared/gateway-events'
import { atom } from 'nanostores'

import { patchOverlayState } from './overlayStore.js'

export interface ConnectionOperationSnapshot {
  deadlineAt: number
  opId: string
  seq: number
  targets: ConnectionOperationTarget[]
  toolCallId: null | string
}

export interface ConnectionOverlayState {
  opId: string
}

export const $connectionOperation = atom<ConnectionOperationSnapshot | null>(null)

const settledOperationIds = new Set<string>()

export function applyConnectionRequest(payload: ConnectionRequestPayload): void {
  const current = $connectionOperation.get()

  if (settledOperationIds.has(payload.op_id)) {
    return
  }

  if (current?.opId === payload.op_id && payload.seq <= current.seq) {
    return
  }

  $connectionOperation.set({
    deadlineAt: payload.deadline_at,
    opId: payload.op_id,
    seq: payload.seq,
    targets: payload.targets,
    toolCallId: payload.tool_call_id ?? null
  })
  patchOverlayState({ connection: { opId: payload.op_id } })
}

export function applyConnectionUpdate(payload: ConnectionUpdatePayload): void {
  const current = $connectionOperation.get()

  if (!current || current.opId !== payload.op_id || payload.seq <= current.seq) {
    return
  }

  if (payload.settled) {
    settledOperationIds.add(payload.op_id)
    clearConnectionOperation()

    return
  }

  $connectionOperation.set({
    ...current,
    deadlineAt: payload.deadline_at,
    seq: payload.seq,
    targets: payload.targets
  })
}

export function clearConnectionOperation(): void {
  $connectionOperation.set(null)
  patchOverlayState({ connection: null })
}

export function resetConnectionOperationsForTests(): void {
  settledOperationIds.clear()
  clearConnectionOperation()
}
