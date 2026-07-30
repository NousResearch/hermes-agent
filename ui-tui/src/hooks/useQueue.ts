import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef, useState } from 'react'

import { $uiSessionId } from '../app/uiStore.js'

// Bucket for prompts submitted before any session is live. `dispatchSubmission`
// in useSubmission deliberately queues while `sid` is null, and the drain effect
// in useMainApp flushes the queue the moment `ui.sid` appears — so this bucket is
// promoted into the arriving session, never abandoned.
const NO_SESSION_KEY = '__no_session__'

export interface QueueItem {
  display: string
  text: string
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

  return {
    display: editedDisplay,
    text: editedDisplay.includes(item.display) ? editedDisplay.replace(item.display, item.text) : editedDisplay
  }
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

/**
 * Per-session prompt queues, kept outside React so the isolation and promotion
 * rules are unit-testable without rendering the hook — the same reason
 * `removeAtInPlace` lives here as a pure export.
 *
 * `currentRef` is what `useQueue` hands out as `queueRef`, so consumers that
 * mutate the queue in place (useInputHandlers, useSubmission,
 * slash/commands/core) keep working untouched: a session switch swaps which
 * array `currentRef.current` points at, it does not change the ref's shape.
 */
export function createSessionQueueManager() {
  const buckets = new Map<string, QueueItem[]>()

  const bucketFor = (key: string): QueueItem[] => {
    let bucket = buckets.get(key)

    if (!bucket) {
      bucket = []
      buckets.set(key, bucket)
    }

    return bucket
  }

  const currentRef: MutableRefObject<QueueItem[]> = { current: bucketFor(NO_SESSION_KEY) }
  let sessionKey = NO_SESSION_KEY

  /**
   * Point the active queue at `sid`'s bucket and return it.
   *
   * Switching between two live sessions never moves prompts — carrying them
   * over is the leak this manager exists to close. The one exception is the
   * no-session -> live transition: prompts queued before a session existed have
   * no owner yet, so they are appended to the arriving session's own backlog
   * and the drain effect picks them up exactly as it did before bucketing.
   */
  const setSession = (sid: null | string): QueueItem[] => {
    const nextKey = sid || NO_SESSION_KEY

    if (sessionKey === NO_SESSION_KEY && nextKey !== NO_SESSION_KEY) {
      const pending = bucketFor(NO_SESSION_KEY)

      if (pending.length > 0) {
        bucketFor(nextKey).push(...pending)
        pending.length = 0
      }
    }

    sessionKey = nextKey
    currentRef.current = bucketFor(nextKey)

    return currentRef.current
  }

  return {
    currentRef,
    dequeue: () => currentRef.current.shift()?.text,
    display: () => currentRef.current.map(item => item.display),
    enqueue: (item: QueueItem) => {
      currentRef.current.push(item)
    },
    prepend: (item: QueueItem) => prependQueueItem(currentRef.current, item),
    setSession,
    take: (i: number, editedDisplay?: string) => takeQueueItem(currentRef.current, i, editedDisplay)
  }
}

export function useQueue() {
  const sid = useStore($uiSessionId)
  const managerRef = useRef<null | ReturnType<typeof createSessionQueueManager>>(null)

  if (!managerRef.current) {
    managerRef.current = createSessionQueueManager()
  }

  const manager = managerRef.current
  const queueRef = manager.currentRef
  const [queuedDisplay, setQueuedDisplay] = useState<string[]>(() => manager.display())
  const queueEditRef = useRef<number | null>(null)
  const [queueEditIdx, setQueueEditIdx] = useState<number | null>(null)

  const syncQueue = useCallback(() => setQueuedDisplay(manager.display()), [manager])

  const setQueueEdit = useCallback((idx: number | null) => {
    queueEditRef.current = idx
    setQueueEditIdx(idx)
  }, [])

  // Runs before useMainApp's drain effect (hooks flush in call order, and
  // useComposerState is called first), so the drain always reads the bucket that
  // belongs to the session it is about to send into. The queue-edit index points
  // into the outgoing session's array, so it cannot survive the swap.
  useEffect(() => {
    manager.setSession(sid)
    setQueueEdit(null)
    syncQueue()
  }, [manager, setQueueEdit, sid, syncQueue])

  const enqueue = useCallback(
    (text: string, display = text) => {
      manager.enqueue(queueItem(text, display))
      syncQueue()
    },
    [manager, syncQueue]
  )

  const prependQ = useCallback(
    (item: QueueItem) => {
      manager.prepend(item)
      syncQueue()
    },
    [manager, syncQueue]
  )

  const dequeue = useCallback(() => {
    const head = manager.dequeue()
    syncQueue()

    return head
  }, [manager, syncQueue])

  const takeQ = useCallback(
    (i: number, editedDisplay?: string) => {
      const item = manager.take(i, editedDisplay)

      if (item) {
        syncQueue()
      }

      return item
    },
    [manager, syncQueue]
  )

  const removeQ = useCallback(
    (i: number) => {
      takeQ(i)
    },
    [takeQ]
  )

  return {
    dequeue,
    enqueue,
    prependQ,
    queueEditIdx,
    queueEditRef,
    queueRef,
    queuedDisplay,
    removeQ,
    setQueueEdit,
    takeQ
  }
}
