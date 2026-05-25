import type { ScrollBoxHandle } from '@hermes/ink'
import type { RefObject } from 'react'
import { useCallback, useMemo, useSyncExternalStore } from 'react'

export interface ViewportSnapshot {
  atBottom: boolean
  bottom: number
  pending: number
  scrollHeight: number
  top: number
  viewportHeight: number
}

export interface ScrollbarSnapshot {
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

const EMPTY_SCROLLBAR: ScrollbarSnapshot = {
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
  const cachedScrollHeight = Math.max(viewportHeight, s.getScrollHeight())
  let scrollHeight = cachedScrollHeight
  const bottom = top + viewportHeight
  let atBottom = s.isSticky() || bottom >= scrollHeight - 2

  if (!atBottom) {
    scrollHeight = Math.max(viewportHeight, s.getFreshScrollHeight?.() ?? cachedScrollHeight)
    atBottom = s.isSticky() || bottom >= scrollHeight - 2
  }

  return {
    atBottom,
    bottom,
    pending,
    scrollHeight,
    top,
    viewportHeight
  }
}

export function viewportSnapshotKey(v: ViewportSnapshot) {
  return `${v.atBottom ? 1 : 0}:${Math.ceil(v.top / 8) * 8}:${v.viewportHeight}:${Math.ceil(v.scrollHeight / 8) * 8}:${v.pending}`
}

export function getScrollbarSnapshot(s?: ScrollBoxHandle | null): ScrollbarSnapshot {
  if (!s) {
    return EMPTY_SCROLLBAR
  }

  const viewportHeight = Math.max(0, s.getViewportHeight())
  const scrollHeight = Math.max(viewportHeight, s.getScrollHeight())
  const maxTop = Math.max(0, scrollHeight - viewportHeight)

  return {
    scrollHeight,
    top: Math.max(0, Math.min(maxTop, s.getScrollTop())),
    viewportHeight
  }
}

export function scrollbarSnapshotKey(v: ScrollbarSnapshot) {
  return `${v.top}:${v.viewportHeight}:${v.scrollHeight}`
}

export function useViewportSnapshot(scrollRef: RefObject<ScrollBoxHandle | null>): ViewportSnapshot {
  const key = useSyncExternalStore(
    useCallback((cb: () => void) => scrollRef.current?.subscribe(cb) ?? (() => {}), [scrollRef]),
    () => viewportSnapshotKey(getViewportSnapshot(scrollRef.current)),
    () => viewportSnapshotKey(EMPTY)
  )

  return useMemo(() => {
    const [atBottom = '1', top = '0', viewportHeight = '0', scrollHeight = '0', pending = '0'] = key.split(':')

    return {
      atBottom: atBottom === '1',
      bottom: Number(top) + Number(viewportHeight),
      pending: Number(pending),
      scrollHeight: Number(scrollHeight),
      top: Number(top),
      viewportHeight: Number(viewportHeight)
    }
  }, [key])
}

export function useScrollbarSnapshot(scrollRef: RefObject<ScrollBoxHandle | null>): ScrollbarSnapshot {
  const key = useSyncExternalStore(
    useCallback((cb: () => void) => scrollRef.current?.subscribe(cb) ?? (() => {}), [scrollRef]),
    () => scrollbarSnapshotKey(getScrollbarSnapshot(scrollRef.current)),
    () => scrollbarSnapshotKey(EMPTY_SCROLLBAR)
  )

  return useMemo(() => {
    const [top = '0', viewportHeight = '0', scrollHeight = '0'] = key.split(':')

    return {
      scrollHeight: Number(scrollHeight),
      top: Number(top),
      viewportHeight: Number(viewportHeight)
    }
  }, [key])
}
