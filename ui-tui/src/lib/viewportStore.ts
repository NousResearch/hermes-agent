import type { ScrollBoxHandle } from '@hermes/ink'
import type { RefObject } from 'react'
import { useCallback, useSyncExternalStore } from 'react'

export interface ViewportSnapshot {
  atBottom: boolean
  bottom: number
  pending: number
  scrollHeight: number
  top: number
  viewportHeight: number
}

const EMPTY: ViewportSnapshot = {
  atBottom: true,
  bottom: 0,
  pending: 0,
  scrollHeight: 0,
  top: 0,
  viewportHeight: 0
}

export function getViewportSnapshot(s?: ScrollBoxHandle | null): ViewportSnapshot {
  if (!s) {
    return EMPTY
  }

  const pending = s.getPendingDelta()
  const top = Math.max(0, s.getScrollTop() + pending)
  const viewportHeight = Math.max(0, s.getViewportHeight())
  const scrollHeight = Math.max(viewportHeight, s.getScrollHeight())
  const bottom = top + viewportHeight

  return {
    atBottom: s.isSticky() || bottom >= scrollHeight - 2,
    bottom,
    pending,
    scrollHeight,
    top,
    viewportHeight
  }
}

export function viewportSnapshotKey(v: ViewportSnapshot) {
  return `${v.atBottom ? 1 : 0}:${v.top}:${v.viewportHeight}:${v.scrollHeight}:${v.pending}`
}

export function useViewportSnapshot(scrollRef: RefObject<ScrollBoxHandle | null>): ViewportSnapshot {
  const key = useSyncExternalStore(
    useCallback((cb: () => void) => scrollRef.current?.subscribe(cb) ?? (() => {}), [scrollRef]),
    () => viewportSnapshotKey(getViewportSnapshot(scrollRef.current)),
    () => viewportSnapshotKey(EMPTY)
  )

  void key

  return getViewportSnapshot(scrollRef.current)
}
