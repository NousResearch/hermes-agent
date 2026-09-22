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

/** Return the scroll offset of the adjacent user turn in `direction`. */
export const userMessageScrollTarget = (
  messages: readonly Msg[],
  offsets: ArrayLike<number>,
  top: number,
  direction: -1 | 1
): null | number => {
  if (direction < 0) {
    for (let i = messages.length - 1; i >= 0; i--) {
      const offset = offsets[i]

      if (messages[i]?.role === 'user' && offset !== undefined && offset < top - 0.5) {
        return offset
      }
    }
  } else {
    for (let i = 0; i < messages.length; i++) {
      const offset = offsets[i]

      if (messages[i]?.role === 'user' && offset !== undefined && offset > top + 0.5) {
        return offset
      }
    }
  }

  return null
}

export const stickyPromptFromViewport = (
  messages: readonly Msg[],
  offsets: ArrayLike<number>,
  top: number,
  bottom: number,
  sticky: boolean
) => {
  if (sticky || !messages.length) {
    return ''
  }

  const first = Math.max(0, upperBound(offsets, top) - 1)
  const last = Math.max(first, upperBound(offsets, bottom) - 1)
  const visibleStart = Math.min(messages.length, first)
  const visibleEnd = Math.min(messages.length - 1, last)

  for (let i = visibleStart; i <= visibleEnd; i++) {
    if (messages[i]?.role === 'user') {
      return ''
    }
  }

  for (let i = Math.min(messages.length - 1, visibleStart - 1); i >= 0; i--) {
    if (messages[i]?.role !== 'user') {
      continue
    }

    return (offsets[i + 1] ?? (offsets[i] ?? 0) + 1) <= top
      ? userDisplay(messages[i]!.text.trim()).replace(/\s+/g, ' ').trim()
      : ''
  }

  return ''
}
