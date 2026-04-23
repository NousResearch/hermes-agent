import { useCallback, useRef, useState } from 'react'

export function useQueue() {
  const queueRef = useRef<string[]>([])
  const [queuedDisplay, setQueuedDisplay] = useState<string[]>([])
  const queueEditRef = useRef<number | null>(null)
  const [queueEditIdx, setQueueEditIdx] = useState<number | null>(null)

  const syncQueue = useCallback(() => setQueuedDisplay([...queueRef.current]), [])

  const setQueueEdit = useCallback((idx: number | null) => {
    queueEditRef.current = idx
    setQueueEditIdx(idx)
  }, [])

  const enqueue = useCallback(
    (text: string) => {
      queueRef.current.push(text)
      syncQueue()
    },
    [syncQueue]
  )

  const dequeue = useCallback(() => {
    const head = queueRef.current.shift()
    syncQueue()

    return head
  }, [syncQueue])

  const replaceQ = useCallback(
    (i: number, text: string) => {
      queueRef.current[i] = text
      syncQueue()
    },
    [syncQueue]
  )

  return {
    dequeue,
    enqueue,
    queueEditIdx,
    queueEditRef,
    queueRef,
    queuedDisplay,
    replaceQ,
    setQueueEdit,
    syncQueue
  }
}
