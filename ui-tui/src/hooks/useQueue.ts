import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { $uiSessionId } from '../app/uiStore.js'

const FALLBACK_SESSION_KEY = '__no_session__'

// Mutates `arr` in place; returned reference is the same input array, kept
// so callers can chain. Use `Array.prototype.toSpliced` if you need a copy.
export function removeAtInPlace<T>(arr: T[], i: number): T[] {
  if (i < 0 || i >= arr.length) {
    return arr
  }

  arr.splice(i, 1)

  return arr
}

export function createSessionQueueManager() {
  const queues = new Map<string, string[]>()
  const currentRef = { current: [] as string[] }
  let sessionKey = FALLBACK_SESSION_KEY

  const bucketFor = (sid: string | null | undefined): string[] => {
    const key = sid || FALLBACK_SESSION_KEY
    let bucket = queues.get(key)

    if (!bucket) {
      bucket = []
      queues.set(key, bucket)
    }

    return bucket
  }

  const setSession = (sid: string | null | undefined) => {
    sessionKey = sid || FALLBACK_SESSION_KEY
    currentRef.current = bucketFor(sessionKey)

    return currentRef.current
  }

  setSession(sessionKey)

  return {
    current: () => currentRef.current,
    currentRef,
    dequeue: () => currentRef.current.shift(),
    display: () => [...currentRef.current],
    enqueue: (text: string) => currentRef.current.push(text),
    remove: (i: number) => removeAtInPlace(currentRef.current, i),
    replace: (i: number, text: string) => {
      currentRef.current[i] = text
    },
    setSession,
    sync: () => [...currentRef.current]
  }
}

export function useQueue() {
  const sid = useStore($uiSessionId)
  const managerRef = useRef<ReturnType<typeof createSessionQueueManager> | null>(null)

  if (!managerRef.current) {
    managerRef.current = createSessionQueueManager()
  }

  const manager = managerRef.current
  const queueRef = manager.currentRef
  const [queuedDisplay, setQueuedDisplay] = useState<string[]>(() => manager.display())
  const queueEditRef = useRef<number | null>(null)
  const [queueEditIdx, setQueueEditIdx] = useState<number | null>(null)

  const syncQueue = useCallback(() => setQueuedDisplay(manager.sync()), [manager])

  const setQueueEdit = useCallback((idx: number | null) => {
    queueEditRef.current = idx
    setQueueEditIdx(idx)
  }, [])

  useEffect(() => {
    manager.setSession(sid)
    setQueueEdit(null)
    syncQueue()
  }, [manager, setQueueEdit, sid, syncQueue])

  const enqueue = useCallback(
    (text: string) => {
      manager.enqueue(text)
      syncQueue()
    },
    [manager, syncQueue]
  )

  const dequeue = useCallback(() => {
    const head = manager.dequeue()
    syncQueue()

    return head
  }, [manager, syncQueue])

  const replaceQ = useCallback(
    (i: number, text: string) => {
      manager.replace(i, text)
      syncQueue()
    },
    [manager, syncQueue]
  )

  const removeQ = useCallback(
    (i: number) => {
      const before = queueRef.current.length

      manager.remove(i)

      if (queueRef.current.length !== before) {
        syncQueue()
      }
    },
    [manager, queueRef, syncQueue]
  )

  return {
    dequeue,
    enqueue,
    queueEditIdx,
    queueEditRef,
    queueRef,
    queuedDisplay,
    removeQ,
    replaceQ,
    setQueueEdit,
    syncQueue
  }
}
