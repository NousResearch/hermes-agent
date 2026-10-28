import type { Msg } from '../types.js'

import { userDisplay } from './messages.js'

export interface TurnAnchor {
  msg: Msg
  mode: 'live' | 'history'
  turnIndex: number
}

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

// Pinned turn anchor (approach A): the user message that opens the turn the
// viewport is looking at — 'live' while that turn streams, 'history' when the
// user scrolled into older content. Reuses the offset math of
// stickyPromptFromViewport (kept for `display.turn_anchor: echo`).
export const turnAnchorFromViewport = (
  messages: readonly Msg[],
  offsets: ArrayLike<number>,
  top: number,
  bottom: number,
  sticky: boolean,
  streaming: boolean
): TurnAnchor | null => {
  if (!messages.length) {
    return null
  }

  if (sticky && !streaming) {
    return null
  }

  const firstVisible = Math.max(0, upperBound(offsets, top) - 1)
  const lastVisible = Math.max(firstVisible, upperBound(offsets, bottom) - 1)

  // A user message already on screen anchors itself — no bar needed.
  for (let i = Math.min(messages.length - 1, firstVisible); i <= Math.min(messages.length - 1, lastVisible); i++) {
    if (messages[i]?.role === 'user') {
      return null
    }
  }

  // Nearest user message strictly above the viewport opens the visible turn.
  let lastUserIdx = -1

  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === 'user') {
      lastUserIdx = i

      break
    }
  }

  for (let i = Math.min(messages.length - 1, firstVisible - 1); i >= 0; i--) {
    if (messages[i]?.role !== 'user') {
      continue
    }

    const blockEnd = offsets[i + 1] ?? (offsets[i] ?? 0) + 1

    if (blockEnd > top) {
      return null
    }

    const mode: TurnAnchor['mode'] = streaming && i === lastUserIdx ? 'live' : 'history'

    return { msg: messages[i]!, mode, turnIndex: i }
  }

  return null
}
