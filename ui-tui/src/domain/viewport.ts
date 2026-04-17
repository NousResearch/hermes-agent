import type { Msg } from '../types.js'

import { userDisplay } from './messages.js'

const upperBound = (offsets: ArrayLike<number>, target: number) => {
  let lo = 0
  let hi = offsets.length

  while (lo < hi) {
    const mid = (lo + hi) >> 1

    offsets[mid]! <= target ? (lo = mid + 1) : (hi = mid)
  }

  return lo
}

export const stickyPromptFromViewport = (
  messages: readonly Msg[],
  offsets: ArrayLike<number>,
  top: number,
  sticky: boolean
) => {
  if (sticky || !messages.length) {
    return ''
  }

  const first = Math.max(0, Math.min(messages.length - 1, upperBound(offsets, top) - 1))
  const aboveViewport = (i: number) => (offsets[i] ?? 0) + 1 < top

  // Walk backward from the first visible row. The nearest user message wins:
  // if it's still on screen, no sticky is needed; if it's already scrolled
  // above the top, its text becomes the floating breadcrumb.
  for (let i = first; i >= 0; i--) {
    if (messages[i]?.role !== 'user') {
      continue
    }

    return aboveViewport(i) ? userDisplay(messages[i]!.text.trim()).replace(/\s+/g, ' ').trim() : ''
  }

  return ''
}
