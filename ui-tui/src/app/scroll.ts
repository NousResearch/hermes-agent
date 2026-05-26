import type { ScrollBoxHandle } from '@hermes/ink'

import type { SelectionApi } from './interfaces.js'

export interface SelectionSnap {
  anchor?: { row: number } | null
  focus?: { row: number } | null
  isDragging?: boolean
}

export interface ScrollWithSelectionOptions {
  readonly scrollRef: { readonly current: ScrollBoxHandle | null }
  readonly selection: SelectionApi
}

function scrollBoundsForDelta(s: ScrollBoxHandle, cur: number, delta: number) {
  const viewport = Math.max(0, s.getViewportHeight())
  const cachedHeight = Math.max(viewport, s.getScrollHeight())
  let max = Math.max(0, cachedHeight - viewport)

  // getScrollHeight() is render-time cached. After the streaming tail is
  // committed into virtual history, the Yoga height can be fresher than the
  // cached value; if we clamp only against the cached fake bottom, wheel-down
  // becomes a no-op and no render is scheduled to reveal the real tail.
  if (delta > 0 && cur + delta >= max - 1) {
    const freshHeight = Math.max(viewport, s.getFreshScrollHeight())
    max = Math.max(0, freshHeight - viewport)
  }

  return { max, viewport }
}

export function scrollWithSelectionBy(delta: number, { scrollRef, selection }: ScrollWithSelectionOptions): void {
  const s = scrollRef.current

  if (!s) {
    return
  }

  const cur = s.getScrollTop() + s.getPendingDelta()
  const { max, viewport } = scrollBoundsForDelta(s, cur, delta)
  const actual = Math.max(0, Math.min(max, cur + delta)) - cur

  if (actual === 0) {
    return
  }

  const sel = selection.getState() as null | SelectionSnap
  const top = s.getViewportTop()
  const bottom = top + viewport - 1

  if (
    sel?.anchor &&
    sel.focus &&
    sel.anchor.row >= top &&
    sel.anchor.row <= bottom &&
    (sel.isDragging || (sel.focus.row >= top && sel.focus.row <= bottom))
  ) {
    const shift = sel.isDragging ? selection.shiftAnchor : selection.shiftSelection

    if (actual > 0) {
      selection.captureScrolledRows(top, top + actual - 1, 'above')
    } else {
      selection.captureScrolledRows(bottom + actual + 1, bottom, 'below')
    }

    shift(-actual, top, bottom)
  }

  s.scrollBy(actual)
}
