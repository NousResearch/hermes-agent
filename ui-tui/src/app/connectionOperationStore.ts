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

const OPERATION_MEMORY = 64
const settledOperationIds: string[] = []
const dismissedOperationIds: string[] = []

function remember(opIds: string[], opId: string): void {
  if (opIds.includes(opId)) {
    return
  }

  opIds.push(opId)

  if (opIds.length > OPERATION_MEMORY) {
    opIds.shift()
  }
}

export const isSettledOperation = (opId: string): boolean => settledOperationIds.includes(opId)

export const isDismissedOperation = (opId: string): boolean => dismissedOperationIds.includes(opId)

const outcomeWord = (target: ConnectionOperationTarget): string => {
  if (target.state === 'connected') {
    return 'connected'
  }

  return target.state === 'skipped' ? 'skipped' : 'not connected'
}

export function applyConnectionRequest(payload: ConnectionRequestPayload): void {
  if (isSettledOperation(payload.op_id) || isDismissedOperation(payload.op_id)) {
    return
  }

  const current = $connectionOperation.get()
  const older = current?.opId === payload.op_id && payload.seq <= current.seq

  if (!older) {
    $connectionOperation.set({
      deadlineAt: payload.deadline_at,
      opId: payload.op_id,
      seq: payload.seq,
      targets: payload.targets,
      toolCallId: payload.tool_call_id ?? null
    })
  }

  patchOverlayState({ connection: { opId: payload.op_id } })
}

export function applyConnectionUpdate(payload: ConnectionUpdatePayload): string[] {
  const current = $connectionOperation.get()
  const shown = current !== null && current.opId === payload.op_id

  if (payload.settled) {
    remember(settledOperationIds, payload.op_id)

    if (!shown && !isDismissedOperation(payload.op_id)) {
      return []
    }

    if (shown) {
      clearConnectionOperation()
    }

    return payload.targets.map(target => `${target.name}: ${outcomeWord(target)}`)
  }

  if (!shown || !current || payload.seq <= current.seq) {
    return []
  }

  $connectionOperation.set({
    ...current,
    deadlineAt: payload.deadline_at,
    seq: payload.seq,
    targets: payload.targets
  })
  patchOverlayState({ connection: { opId: payload.op_id } })

  return []
}

export function clearConnectionOperation(): void {
  $connectionOperation.set(null)
  patchOverlayState({ connection: null })
}

/** Esc on the card: the outcome is still wanted, but this card must never come back. */
export function dismissConnectionOperation(opId: string): void {
  remember(dismissedOperationIds, opId)
  clearConnectionOperation()
}

export function resetConnectionOperationsForTests(): void {
  settledOperationIds.length = 0
  dismissedOperationIds.length = 0
  clearConnectionOperation()
}
