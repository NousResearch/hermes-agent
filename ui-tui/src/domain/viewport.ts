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

/**
 * Prompt navigation for the transcript — the terminal answer to the desktop's
 * right-edge prompt rail.
 *
 * Row indexes here are MESSAGE indexes: the virtualizer keeps one entry per
 * message, so `offsets[rows[i]]` is that prompt's top row in document space and
 * `scrollTo` on it lands the prompt at the top of the viewport. Kept pure and
 * DOM-free so the rail's ticks and the hotkeys' steps can never disagree about
 * which prompt you are on.
 */

/**
 * Turns the harness injects AS user messages: background-process notices,
 * truncation notices, the preserved-task note. The backend names the same set
 * in `agent/conversation_compression.py` (`_SYNTHETIC_USER_PREFIXES`) and the
 * desktop rail filters them out too — without this the rail ticks a finished
 * background job as if you had typed it.
 */
const SYNTHETIC_USER_PREFIXES = [
  '[IMPORTANT: Background process ',
  '[System: Your previous response was truncated',
  '[System: The previous response was cut off',
  '[System: Your previous tool call',
  '[Your active task list was preserved across context compression]'
]

/** A turn you actually wrote, rather than one the harness injected. */
const isOwnPrompt = (msg: Msg): boolean =>
  msg.role === 'user' &&
  msg.text.trim() !== '' &&
  !SYNTHETIC_USER_PREFIXES.some(prefix => msg.text.startsWith(prefix))

/** Message-row indexes of the user prompts, oldest first. */
export const promptRowIndexes = (messages: readonly Msg[]): number[] => {
  const rows: number[] = []

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]

    if (msg && isOwnPrompt(msg)) {
      rows.push(i)
    }
  }

  return rows
}

/**
 * Index INTO `rows` of the prompt the viewport sits in: the last prompt at or
 * above the viewport top. Pinned to the bottom the viewport is past every
 * prompt by construction, so the newest one answers without a walk — the same
 * fast path the desktop rail takes while it is following the tail. -1 when the
 * session has no prompts yet.
 */
export const activePromptIndex = (
  offsets: ArrayLike<number>,
  rows: readonly number[],
  top: number,
  sticky: boolean,
  slack = 1
): number => {
  if (!rows.length) {
    return -1
  }

  if (sticky) {
    return rows.length - 1
  }

  let active = 0

  for (let i = 0; i < rows.length; i++) {
    // Offsets are a monotone prefix sum, so the first prompt BELOW the
    // viewport top ends the walk.
    if ((offsets[rows[i]!] ?? 0) <= top + slack) {
      active = i
    } else {
      break
    }
  }

  return active
}

/**
 * Absolute scroll offset for stepping one prompt from where the viewport is
 * now — the row to land ON. null at either end (and with no prompts at all) so
 * the caller leaves the viewport alone rather than bouncing it.
 */
export const steppedPromptOffset = (
  offsets: ArrayLike<number>,
  rows: readonly number[],
  top: number,
  sticky: boolean,
  dir: -1 | 1
): number | null => {
  const active = activePromptIndex(offsets, rows, top, sticky)

  if (active < 0) {
    return null
  }

  const next = active + dir

  if (next < 0 || next >= rows.length) {
    return null
  }

  return offsets[rows[next]!] ?? null
}

export interface PromptTick {
  /** Prompt index INTO `rows` — 0 is the oldest. */
  index: number
  /** Rail row the tick draws on, 0-based from the top of the viewport. */
  row: number
}

/**
 * Map every prompt's document offset onto a row of the right-edge rail.
 * Proportional, like the scrollbar thumb beside it, oldest at the top. Prompts
 * that round onto the same row collapse to ONE tick — and the ACTIVE prompt
 * wins the collision, so the marker telling you where you are can never be
 * absorbed by a neighbour in a dense session.
 */
export const promptTicks = (
  offsets: ArrayLike<number>,
  rows: readonly number[],
  total: number,
  viewportHeight: number,
  activeIndex: number
): PromptTick[] => {
  if (!rows.length || viewportHeight <= 0) {
    return []
  }

  const last = viewportHeight - 1
  const span = Math.max(1, total)
  const byRow = new Map<number, number>()

  for (let i = 0; i < rows.length; i++) {
    const scaled = Math.round(((offsets[rows[i]!] ?? 0) / span) * last)
    const row = Math.max(0, Math.min(last, scaled))
    const held = byRow.get(row)

    if (held === undefined || i === activeIndex) {
      byRow.set(row, i)
    }
  }

  return [...byRow].map(([row, index]) => ({ index, row })).sort((a, b) => a.row - b.row)
}

/**
 * Document offset a rail row stands for, so clicking a tick lands on the prompt
 * EXACTLY instead of on the thumb's proportional guess. null when the row holds
 * no tick, which leaves the plain scrub behaviour untouched.
 */
export const promptOffsetAtRailRow = (
  offsets: ArrayLike<number>,
  rows: readonly number[],
  total: number,
  viewportHeight: number,
  activeIndex: number,
  row: number
): number | null => {
  const hit = promptTicks(offsets, rows, total, viewportHeight, activeIndex).find(t => t.row === row)

  return hit ? (offsets[rows[hit.index]!] ?? null) : null
}
