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

export type PromptNavigationDirection = 'next' | 'previous'

export interface PromptAnchor {
  index: number
  top: number
}

export interface PromptJumpState {
  bottomMode: 'center' | 'hidden' | 'left'
  hasNextPrompt: boolean
  hasPreviousPrompt: boolean
}

export const promptAnchorFromViewport = (
  messages: readonly Msg[],
  offsets: ArrayLike<number>,
  top: number,
  direction: PromptNavigationDirection
): null | PromptAnchor => {
  if (!messages.length || offsets.length < 2) {
    return null
  }

  if (direction === 'previous') {
    for (let i = Math.min(messages.length - 1, upperBound(offsets, top - 1) - 1); i >= 0; i--) {
      if (messages[i]?.role === 'user') {
        return { index: i, top: offsets[i] ?? 0 }
      }
    }

    return null
  }

  for (let i = Math.max(0, upperBound(offsets, top)); i < messages.length; i++) {
    if (messages[i]?.role === 'user') {
      return { index: i, top: offsets[i] ?? 0 }
    }
  }

  return null
}

export const promptJumpStateFromViewport = (
  messages: readonly Msg[],
  offsets: ArrayLike<number>,
  top: number,
  atBottom: boolean,
  bottom = top
): PromptJumpState => {
  if (atBottom) {
    return { bottomMode: 'hidden', hasNextPrompt: false, hasPreviousPrompt: false }
  }

  const previous = promptAnchorFromViewport(messages, offsets, top, 'previous')

  const latestPromptIndex = (() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i]?.role === 'user') {
        return i
      }
    }

    return -1
  })()

  const latestPromptTop = latestPromptIndex >= 0 ? (offsets[latestPromptIndex] ?? 0) : null
  const scrolledAboveLatestPrompt = latestPromptTop !== null && bottom <= latestPromptTop
  const next = scrolledAboveLatestPrompt ? promptAnchorFromViewport(messages, offsets, top, 'next') : null

  return {
    bottomMode: next ? 'left' : previous ? 'center' : 'left',
    hasNextPrompt: Boolean(next),
    hasPreviousPrompt: Boolean(previous)
  }
}

export const stickyPromptAnchorFromViewport = (
  messages: readonly Msg[],
  offsets: ArrayLike<number>,
  top: number,
  bottom: number,
  sticky: boolean
): null | PromptAnchor => {
  if (sticky || !messages.length) {
    return null
  }

  const first = Math.max(0, upperBound(offsets, top) - 1)
  const last = Math.max(first, upperBound(offsets, bottom) - 1)
  const visibleStart = Math.min(messages.length, first)
  const visibleEnd = Math.min(messages.length - 1, last)

  for (let i = visibleStart; i <= visibleEnd; i++) {
    if (messages[i]?.role === 'user') {
      return null
    }
  }

  for (let i = Math.min(messages.length - 1, visibleStart - 1); i >= 0; i--) {
    if (messages[i]?.role !== 'user') {
      continue
    }

    return (offsets[i + 1] ?? (offsets[i] ?? 0) + 1) <= top ? { index: i, top: offsets[i] ?? 0 } : null
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
  const anchor = stickyPromptAnchorFromViewport(messages, offsets, top, bottom, sticky)

  return anchor ? userDisplay(messages[anchor.index]!.text.trim()).replace(/\s+/g, ' ').trim() : ''
}
