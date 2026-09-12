import { randomUUID } from 'node:crypto'

import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { isServerQueued } from '../app/pendingBubbles.js'
import { captureDestination, type SubmissionDestination } from '../app/submissionDestination.js'
import { $uiState, getUiState, patchUiState } from '../app/uiStore.js'
import {
  loadPendingInputs,
  pendingDestinationKey,
  pendingInputOwner,
  pendingInputRevision,
  removePendingInput,
  savePendingInput
} from '../lib/pendingInputs.js'

export interface QueueItem {
  display: string
  text: string
  queued?: boolean
  createdAt?: number
  submissionId?: string
  destination?: SubmissionDestination
  ownerDestination?: SubmissionDestination
  inFlight?: boolean
  failed?: boolean
  attachments?: Array<{ path: string; mime: string }>
  controlMethod?: 'session.steer' | 'session.redirect'
  executionGeneration?: number
  preparedText?: string
  legacyAttempted?: boolean
  settle?: (accepted: boolean) => void
}

export const queueItem = (text: string, display = text): QueueItem => ({ display, text })

export function prependQueueItem(queue: QueueItem[], item: QueueItem): void {
  queue.unshift(item)
}

export function takeQueueItem(queue: QueueItem[], index: number, editedDisplay?: string): QueueItem | undefined {
  if (index < 0 || index >= queue.length) {
    return undefined
  }

  const [item] = queue.splice(index, 1)

  if (!item || editedDisplay === undefined) {
    return item
  }

  const text = editedDisplay.includes(item.display) ? editedDisplay.replace(item.display, item.text) : editedDisplay

  return text === item.text ? { ...item, display: editedDisplay } : { display: editedDisplay, text }
}

// Mutates `arr` in place; returned reference is the same input array, kept
// so callers can chain. Use `Array.prototype.toSpliced` if you need a copy.
export function removeAtInPlace<T>(arr: T[], i: number): T[] {
  if (i < 0 || i >= arr.length) {
    return arr
  }

  arr.splice(i, 1)

  return arr
}

interface PendingQueue {
  revision?: number
  destination?: SubmissionDestination
  edit: number | null
  items: QueueItem[]
}

// `gw` owns server-queued rows: the authority's pending fanout is rendered
// after this client's unconfirmed local items, and deleting/editing one goes
// through `prompt.cancel` — the next fanout, not a local splice, retires it.
export function useQueue(gw?: { request: (method: string, params: Record<string, unknown>) => Promise<unknown> }) {
  const ui = useStore($uiState)
  const queues = useRef(new Map<string, PendingQueue>())
  const unbound = useRef<PendingQueue>({ edit: null, items: [] })
  const [, refresh] = useState(0)

  const getQueue = useCallback((destination = captureDestination()) => {
    // Apply compression before resolving a queue, including through stale callbacks.
    for (const [oldKey, source] of queues.current) {
      const owner = pendingInputOwner(source.destination!)
      const newKey = pendingDestinationKey(owner)

      if (newKey === oldKey) {
        continue
      }

      const target = queues.current.get(newKey) ?? { destination: owner, edit: null, items: [] }
      const edited = source.edit === null ? undefined : source.items[source.edit]

      for (const item of source.items) {
        item.ownerDestination = owner

        if (!item.inFlight && !item.failed) {
          item.destination = owner
        }
      }

      const retained = new Map([...target.items, ...source.items].map(item => [item.submissionId, item]))
      target.items = loadPendingInputs(owner).map(item => retained.get(item.submissionId) ?? item)

      if (edited) {
        target.edit = target.items.indexOf(edited)
      }

      queues.current.delete(oldKey)
      queues.current.set(newKey, target)
    }

    destination = pendingInputOwner(destination)
    const { sid } = destination

    if (!sid) {
      return unbound.current
    }

    const key = pendingDestinationKey(destination)
    let queue = queues.current.get(key)

    if (!queue) {
      queue = { destination, edit: null, items: loadPendingInputs(destination) }
      queues.current.set(key, queue)
    }

    if (queue.revision !== pendingInputRevision) {
      const retained = new Map(queue.items.map(item => [item.submissionId, item]))
      const edited = queue.edit === null ? undefined : queue.items[queue.edit]
      queue.items = loadPendingInputs(destination).map(item => retained.get(item.submissionId) ?? item)
      queue.edit = edited ? queue.items.indexOf(edited) : null
      queue.revision = pendingInputRevision
    }

    // Input typed before any session exists belongs to the next attachment,
    // unlike a bound session's queue, which must never migrate on navigation.
    if (unbound.current.items.length) {
      const offset = queue.items.length

      for (const item of unbound.current.items) {
        item.destination = destination
        savePendingInput(item)
        queue.items.push(item)
      }

      if (unbound.current.edit !== null) {
        queue.edit = offset + unbound.current.edit
      }

      unbound.current = { edit: null, items: [] }
    }

    return queue
  }, [])

  // Resolve on access, not in an effect: navigation and a drain can happen
  // before React renders again, including through an older input callback.
  const queueRef = useMemo(
    () => ({
      get current() {
        return getQueue().items
      }
    }),
    [getQueue]
  )

  const queueEditRef = useMemo(
    () => ({
      get current() {
        return getQueue().edit
      },
      set current(value: number | null) {
        getQueue().edit = value
      }
    }),
    [getQueue]
  )

  const serverRows = (ui.info?.pending_submissions ?? []).filter(isServerQueued)

  const queuedDisplay = [
    ...queueRef.current.map(item => `${item.failed ? '[unconfirmed · Alt+K retry] ' : ''}${item.display}`),
    ...serverRows.map(row => `[${row.status}] ${row.user}`)
  ]

  // Indexes past the local items address server rows (read live, not from
  // render state: key handlers run between renders).
  const serverRowAt = useCallback(
    (index: number) => {
      const local = queueRef.current.length

      return index >= local ? (getUiState().info?.pending_submissions ?? []).filter(isServerQueued)[index - local] : undefined
    },
    [queueRef]
  )

  const queueDraft = useCallback(
    (index: number) => serverRowAt(index)?.user ?? queueRef.current[index]?.display ?? '',
    [queueRef, serverRowAt]
  )

  // Resolves once the authority has retired the row; the rejection carries the
  // store's refusal (stale_generation, …) so callers decide what the edited
  // or deleted text becomes.
  const cancelServerRow = useCallback(
    (row: { admission_id: string; status?: string; execution_generation?: number | null }) => {
      const session_id = getUiState().sid
      const unknown = row.status === 'unknown'

      return (gw?.request(unknown ? 'prompt.resolve_unknown' : 'prompt.cancel', {
        session_id, admission_id: row.admission_id,
        ...(unknown ? { execution_generation: row.execution_generation } : {})
      }) ?? Promise.resolve()).catch((error: Error) => {
        if (getUiState().sid === session_id) { patchUiState({ status: `discard failed: ${error.message}` }) }

        throw error
      })
    },
    [gw]
  )

  const queueEditIdx = queueEditRef.current
  const syncQueue = useCallback(() => refresh(version => version + 1), [])

  const setQueueEdit = useCallback(
    (idx: number | null) => {
      queueEditRef.current = idx
      syncQueue()
    },
    [queueEditRef, syncQueue]
  )

  const enqueue = useCallback(
    (text: string, display = text, destination?: SubmissionDestination) => {
      const owner = pendingInputOwner(destination ?? captureDestination())
      const queue = getQueue(owner)

      const item = {
        ...queueItem(text, display),
        submissionId: randomUUID(),
        destination: owner,
        createdAt: Math.max(Date.now(), (queue.items.at(-1)?.createdAt ?? 0) + 1)
      }

      savePendingInput(item)
      queue.items.push(item)
      syncQueue()

      return item
    },
    [getQueue, syncQueue]
  )

  const prependQ = useCallback(
    (item: QueueItem, destination?: SubmissionDestination) => {
      const queue = getQueue(destination)
      item.inFlight = false
      item.submissionId ??= randomUUID()
      item.destination ??= destination ?? captureDestination()
      savePendingInput(item)

      if (!queue.items.includes(item)) {
        prependQueueItem(queue.items, item)
      }

      syncQueue()
    },
    [getQueue, syncQueue]
  )

  const claim = useCallback(
    (queue: PendingQueue, item: QueueItem) => {
      item.submissionId ??= randomUUID()
      item.destination ??= captureDestination()
      item.inFlight = true
      item.failed = false
      savePendingInput(item)
      let confirmed = false

      item.settle = accepted => {
        if (confirmed) {
          return
        }

        // Resolve migrations while the attempted state still protects the receipt target.
        getQueue()
        confirmed = accepted
        item.inFlight = false
        item.failed = !accepted

        if (accepted) {
          removePendingInput(item)

          for (const pending of queues.current.values()) {
            removeAtInPlace(pending.items, pending.items.indexOf(item))
          }

          removeAtInPlace(queue.items, queue.items.indexOf(item))
        } else {
          savePendingInput(item)
        }

        syncQueue()
      }

      syncQueue()

      return item
    },
    [getQueue, syncQueue]
  )

  useEffect(() => {
    const queue = getQueue()

    for (const receipt of getUiState().info?.pending_submissions ?? []) {
      const item = queue.items.find(
        item =>
          item.submissionId === (receipt.input_id ?? receipt.admission_id) &&
          Boolean(item.destination?.storedSid) &&
          item.destination?.storedSid === receipt.target_session_id &&
          item.destination?.profileHome === receipt.target_profile_home
      )

      if (!item) {
        continue
      }

      if (item.settle) {
        item.settle(true)
      } else {
        removePendingInput(item)
        removeAtInPlace(queue.items, queue.items.indexOf(item))
        syncQueue()
      }
    }
  }, [ui.info, getQueue, syncQueue])

  const stage = useCallback(
    (text: string, display = text, destination = captureDestination()) => {
      const item = enqueue(text, display, destination)
      item.queued = false

      return claim(getQueue(destination), item)
    },
    [enqueue, claim, getQueue]
  )

  const dequeue = useCallback(
    (retry = false) => {
      if (getUiState().gatewayConnected === false) { return undefined }
      const queue = getQueue()
      const item = queue.items[0]

      if (!item || item.inFlight || (item.failed && !retry)) {
        return undefined
      }

      return claim(queue, item)
    },
    [getQueue, claim]
  )

  const takeQ = useCallback(
    (i: number, editedDisplay?: string) => {
      const queue = getQueue()
      const server = serverRowAt(i)

      // Editing a durable row re-admits the edited text as a new input once
      // the authority has retired the original. Until then the edit is a
      // durable local row; a refused retirement leaves it as an unconfirmed
      // draft (Alt+K) instead of admitting a second copy behind the original.
      if (server) {
        if (server.status === 'unknown') {
          patchUiState({ status: 'unknown execution — Ctrl+X to discard before retrying' })

          return undefined
        }

        const text = editedDisplay ?? server.user
        const item = enqueue(text, text, queue.destination)
        item.inFlight = true
        savePendingInput(item)

        return cancelServerRow(server).then(
          () => claim(queue, item),
          () => {
            item.inFlight = false
            item.failed = true
            savePendingInput(item)
            syncQueue()

            return undefined
          })
      }

      if (queue.items[i]?.inFlight) {
        return undefined
      }

      const previous = queue.items[i]
      const item = takeQueueItem(queue.items, i, editedDisplay)

      if (!item) {
        return undefined
      }

      if (previous && previous.submissionId !== item.submissionId) {
        removePendingInput(previous)
      }

      queue.items.splice(i, 0, item)

      return claim(queue, item)
    },
    [getQueue, claim, cancelServerRow, enqueue, serverRowAt, syncQueue]
  )

  const removeQ = useCallback(
    (i: number) => {
      const server = serverRowAt(i)

      if (server) {
        return void cancelServerRow(server).catch(() => undefined)
      }

      if (queueRef.current[i]?.inFlight) {
        return
      }

      const item = queueRef.current[i]

      if (item) {
        removePendingInput(item)
      }

      removeAtInPlace(queueRef.current, i)
      syncQueue()
    },
    [queueRef, syncQueue, cancelServerRow, serverRowAt]
  )

  return {
    dequeue,
    stage,
    enqueue,
    prependQ,
    queueDraft,
    queueEditIdx,
    queueEditRef,
    queueRef,
    queuedDisplay,
    removeQ,
    setQueueEdit,
    takeQ
  }
}
