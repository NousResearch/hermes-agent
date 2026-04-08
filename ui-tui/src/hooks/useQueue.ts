import { useRef, useState } from 'react'

export function useQueue() {
  const queueRef = useRef<string[]>([])
  const [queuedDisplay, setQueuedDisplay] = useState<string[]>([])
  const queueEditRef = useRef<number | null>(null)
  const [queueEditIdx, setQueueEditIdx] = useState<number | null>(null)

  const syncQueue = () => setQueuedDisplay([...queueRef.current])

  const setQueueEdit = (idx: number | null) => {
    queueEditRef.current = idx
    setQueueEditIdx(idx)
  }

  const enqueue = (text: string) => {
    queueRef.current.push(text)
    syncQueue()
  }

  const dequeue = () => {
    const [head, ...rest] = queueRef.current
    queueRef.current = rest
    syncQueue()

    return head
  }

  const replaceQ = (i: number, text: string) => {
    queueRef.current[i] = text
    syncQueue()
  }

  return { queueRef, queueEditRef, queuedDisplay, queueEditIdx, enqueue, dequeue, replaceQ, setQueueEdit, syncQueue }
}
