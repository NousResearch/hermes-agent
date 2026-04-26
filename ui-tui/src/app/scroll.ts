import type { ScrollBoxHandle } from '@hermes/ink'

import type { SelectionApi } from './interfaces.js'
import { markScrolling } from './interactionMode.js'

export interface SelectionSnap {
  anchor?: { row: number } | null
  focus?: { row: number } | null
  isDragging?: boolean
}

export interface ScrollWithSelectionOptions {
  readonly scrollRef: { readonly current: ScrollBoxHandle | null }
  readonly selection: SelectionApi
}

export function scrollWithSelectionBy(delta: number, { scrollRef, selection }: ScrollWithSelectionOptions): void {
  const s = scrollRef.current

  if (!s) {
    return
  }

  const cur = s.getScrollTop() + s.getPendingDelta()
  const viewport = Math.max(0, s.getViewportHeight())
  const max = Math.max(0, s.getScrollHeight() - viewport)
  const actual = Math.max(0, Math.min(max, cur + delta)) - cur

  if (actual === 0) {
    return
  }

  markScrolling()

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
