import type {
  DetachedTurnTask,
  SessionDetachTurnAckResponse,
  SessionDetachTurnConsumedResponse
} from '../gatewayTypes.js'

import type { GatewayRpc } from './interfaces.js'
import { getUiState, patchUiState } from './uiStore.js'

const displayedTaskKeys = new Set<string>()
const completedPromptTaskKeys = new Set<string>()
const consumptionAcks = new Map<string, Promise<void>>()
const registrationAcks = new Map<string, Promise<void>>()
const MAX_DISPLAYED_TASKS = 512
const MAX_CONSUMPTION_ATTEMPTS = 3
const REGISTRATION_RETRY_DELAYS_MS = [0, 50, 150] as const
let trackingGeneration = 0

export const registerDetachedTasks = (tasks: DetachedTurnTask[]) =>
  patchUiState(state => ({
    ...state,
    bgTasks: new Set([...state.bgTasks, ...tasks.map(task => task.task_id)])
  }))

export const registerBackgroundTask = (taskId: string, ownerSid: null | string) => {
  const key = `${ownerSid ?? ''}:${taskId}`

  // A very fast prompt.background worker can emit completion before the start
  // RPC resolves. Its terminal line has already been displayed, so registering
  // it now would create an immortal footer entry.
  if (completedPromptTaskKeys.has(key)) {
    completedPromptTaskKeys.delete(key)

    return false
  }

  patchUiState(state => ({ ...state, bgTasks: new Set(state.bgTasks).add(taskId) }))

  return true
}

const dropTask = (taskId: string) => {
  const next = new Set(getUiState().bgTasks)
  next.delete(taskId)
  patchUiState({ bgTasks: next })
}

const rememberDisplayed = (key: string) => {
  displayedTaskKeys.add(key)

  if (displayedTaskKeys.size > MAX_DISPLAYED_TASKS) {
    displayedTaskKeys.delete(displayedTaskKeys.values().next().value as string)
  }
}

const rememberPromptCompletion = (key: string) => {
  completedPromptTaskKeys.add(key)

  if (completedPromptTaskKeys.size > MAX_DISPLAYED_TASKS) {
    completedPromptTaskKeys.delete(completedPromptTaskKeys.values().next().value as string)
  }
}

const acknowledgeConsumption = (key: string, taskId: string, ownerSid: string, rpc: GatewayRpc) => {
  if (consumptionAcks.has(key)) {
    return
  }

  let attempts = 0

  const attempt = (): Promise<void> => {
    attempts += 1

    return rpc<SessionDetachTurnConsumedResponse>('session.detach_turn_consumed', {
      session_id: ownerSid,
      task_id: taskId
    })
      .then(result => {
        if (!result?.consumed) {
          throw new Error('detached task consumption was not confirmed')
        }

        dropTask(taskId)
      })
      .catch((error: unknown) => {
        if (attempts < MAX_CONSUMPTION_ATTEMPTS) {
          return attempt()
        }

        throw error
      })
  }

  const pending = attempt()
    .catch(() => undefined)
    .finally(() => consumptionAcks.delete(key))

  consumptionAcks.set(key, pending)
}

export const consumeDetachedTask = (
  task: DetachedTurnTask,
  ownerSid: string,
  rpc: GatewayRpc,
  sys: (text: string) => void
) => {
  const state = getUiState()
  const detached = Boolean(task.source_session_id)

  // Detached turns use the active session plus registered presentation id as
  // their ownership boundary. Legacy prompt.background completions carry no
  // source_session_id and remain unconditional, including the fast-completion
  // case where their start RPC has not returned yet.
  if (detached && (state.sid !== ownerSid || !state.bgTasks.has(task.task_id))) {
    return false
  }

  const displayKey = `${ownerSid}:${task.task_id}`

  if (!displayedTaskKeys.has(displayKey)) {
    rememberDisplayed(displayKey)
    const status = task.status && task.status !== 'complete' ? `${task.status}: ` : ''
    sys(`[bg ${task.task_id}] ${status}${task.text || task.status || 'complete'}`)
  }

  if (detached) {
    acknowledgeConsumption(displayKey, task.task_id, ownerSid, rpc)
  } else {
    // Existing prompt.background tasks are not retained by the detach
    // registry, but they still share the completion event shape.
    rememberPromptCompletion(displayKey)
    dropTask(task.task_id)
  }

  return true
}

export const acknowledgeDetachedTask = (
  task: DetachedTurnTask,
  ownerSid: string,
  rpc: GatewayRpc,
  sys: (text: string) => void
) => {
  const key = `${ownerSid}:${task.task_id}`

  if (registrationAcks.has(key)) {
    return registrationAcks.get(key)!
  }

  const generation = trackingGeneration

  const attempt = async (index: number): Promise<void> => {
    if (generation !== trackingGeneration || getUiState().sid !== ownerSid) {
      return
    }

    if (REGISTRATION_RETRY_DELAYS_MS[index]! > 0) {
      await new Promise(resolve => setTimeout(resolve, REGISTRATION_RETRY_DELAYS_MS[index]))
    }

    if (generation !== trackingGeneration || getUiState().sid !== ownerSid) {
      return
    }

    try {
      const result = await rpc<SessionDetachTurnAckResponse>('session.detach_turn_ack', {
        session_id: ownerSid,
        task_id: task.task_id
      })

      if (!result?.task) {
        throw new Error('detached task registration was not confirmed')
      }

      if (result.task.status !== 'running') {
        consumeDetachedTask(result.task, ownerSid, rpc, sys)
      }
    } catch (error: unknown) {
      const next = index + 1

      if (
        next < REGISTRATION_RETRY_DELAYS_MS.length &&
        generation === trackingGeneration &&
        getUiState().sid === ownerSid
      ) {
        return attempt(next)
      }

      if (generation === trackingGeneration && getUiState().sid === ownerSid) {
        sys(`warning: could not restore background task ${task.task_id}: ${String(error)}`)
      }
    }
  }

  const pending = attempt(0).finally(() => registrationAcks.delete(key))
  registrationAcks.set(key, pending)

  return pending
}

export const resetDetachedTaskTrackingForTests = () => {
  trackingGeneration += 1
  displayedTaskKeys.clear()
  completedPromptTaskKeys.clear()
  consumptionAcks.clear()
  registrationAcks.clear()
}
