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
  bottom: number,
  top: number,
  sticky: boolean
) => {
  if (sticky || !messages.length) {
    return ''
  }

  const first = Math.max(0, Math.min(messages.length - 1, upperBound(offsets, top) - 1))
  const last = Math.max(first, Math.min(messages.length - 1, upperBound(offsets, bottom) - 1))

  for (let i = first; i <= last; i++) {
    if (messages[i]?.role === 'user') {
      return ''
    }
  }

  for (let i = first - 1; i >= 0; i--) {
    if (messages[i]?.role !== 'user') {
      continue
    }

    return (offsets[i] ?? 0) + 1 < top ? userDisplay(messages[i]!.text.trim()).replace(/\s+/g, ' ').trim() : ''
  }

  return ''
}
