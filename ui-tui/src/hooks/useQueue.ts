import { useCallback, useRef, useState } from 'react'

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

export function useQueue() {
  const queueRef = useRef<QueueItem[]>([])
  const [queuedDisplay, setQueuedDisplay] = useState<string[]>([])
  const queueEditRef = useRef<number | null>(null)
  const [queueEditIdx, setQueueEditIdx] = useState<number | null>(null)

  const syncQueue = useCallback(() => setQueuedDisplay(queueRef.current.map(item => item.display)), [])

  const setQueueEdit = useCallback((idx: number | null) => {
    queueEditRef.current = idx
    setQueueEditIdx(idx)
  }, [])

  const enqueue = useCallback(
    (text: string, display = text) => {
      queueRef.current.push(queueItem(text, display))
      syncQueue()
    },
    [syncQueue]
  )

  const prependQ = useCallback(
    (item: QueueItem) => {
      prependQueueItem(queueRef.current, item)
      syncQueue()
    },
    [syncQueue]
  )

  const dequeue = useCallback(() => {
    const head = queueRef.current.shift()?.text
    syncQueue()

    return head
  }, [syncQueue])

  const takeQ = useCallback(
    (i: number, editedDisplay?: string) => {
      const item = takeQueueItem(queueRef.current, i, editedDisplay)

      if (item) {
        syncQueue()
      }

      return item
    },
    [syncQueue]
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
