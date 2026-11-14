import { stringWidth, wrapAnsi } from '@hermes/ink'

import type { Msg } from '../types.js'

/** One wrapped row of the backtrack transcript pager; `msg` is the index of the message it belongs to. */
export interface PagerLine {
  msg: number
  text: string
  user: boolean
}

const USER_PREFIX = '› '
const REPLY_PREFIX = '• '
const BODY_PREFIX = '  '

/** The prompts and replies a backtrack pager shows (no intro, panels, slash echoes or tool trails). */
export const pagerMessages = (items: readonly Msg[]): Msg[] =>
  items.filter(msg => !msg.kind && (msg.role === 'user' || msg.role === 'assistant') && msg.text.trim())

/** Wrap every message to `width` columns, a blank row between messages. */
export function pagerLines(messages: readonly Msg[], width: number): PagerLine[] {
  const body = Math.max(8, width - USER_PREFIX.length)
  const lines: PagerLine[] = []

  messages.forEach((msg, index) => {
    const user = msg.role === 'user'
    let lead = user ? USER_PREFIX : REPLY_PREFIX

    if (index > 0) {
      lines.push({ msg: -1, text: '', user: false })
    }

    for (const raw of msg.text.trimEnd().split('\n')) {
      for (const row of wrapAnsi(raw, body, { hard: true, trim: false }).split('\n')) {
        lines.push({ msg: index, text: lead + row, user })
        lead = BODY_PREFIX
      }
    }
  })

  return lines
}

/** The first and last pager row of message `msg`, or null when it has none. */
export function messageRange(lines: readonly PagerLine[], msg: number): [number, number] | null {
  const first = lines.findIndex(line => line.msg === msg)

  return first < 0 ? null : [first, lines.findLastIndex(line => line.msg === msg)]
}

export const clampOffset = (offset: number, total: number, rows: number) =>
  Math.max(0, Math.min(offset, Math.max(0, total - rows)))

/** Scroll the least from `offset` so rows `range` are on screen (its top wins when it is taller than the window). */
export function ensureVisible(offset: number, range: [number, number] | null, rows: number, total: number) {
  if (!range) {
    return clampOffset(offset, total, rows)
  }

  const [start, end] = range

  if (start < offset || end - start + 1 > rows) {
    return clampOffset(start, total, rows)
  }

  return clampOffset(end >= offset + rows ? end - rows + 1 : offset, total, rows)
}

/**
 * The browse footer for `width` cells, shedding hints the way Codex's prompt-navigation footer does:
 * full, then scroll/prompt keys without their words, then without fork, then only edit + back, then
 * back, then the label. `hint` is the localized full line, `label · keys word · …` (5 hints).
 */
export function fitBrowseHint(hint: string, width: number): string {
  const [label = '', ...hints] = hint.split(' · ')

  if (hints.length !== 5 || stringWidth(hint) <= width) {
    return hint
  }

  const [scroll, prompts, edit, fork, back] = hints as [string, string, string, string, string]
  const keys = (segment: string) => segment.split(' ')[0]!
  const nav = `${keys(scroll)} ${keys(prompts)}`

  const ladder = [[nav, edit, fork, back], [nav, edit, back], [edit, back], [back], []]

  for (const rung of ladder) {
    const line = [label, ...rung].join(' · ')

    if (stringWidth(line) <= width) {
      return line
    }
  }

  return label
}
