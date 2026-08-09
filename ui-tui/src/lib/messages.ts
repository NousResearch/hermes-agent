import { MAX_HISTORY } from '../config/limits.js'
import type { Msg, Role } from '../types.js'

import { appendToolShelfMessage } from './liveProgress.js'

export const appendTranscriptMessage = (prev: Msg[], msg: Msg): Msg[] => appendToolShelfMessage(prev, msg)

export const capTranscriptHistory = (items: Msg[]): Msg[] => {
  if (items.length <= MAX_HISTORY) {
    return items
  }

  return items[0]?.kind === 'intro' ? [items[0], ...items.slice(-(MAX_HISTORY - 1))] : items.slice(-MAX_HISTORY)
}

export const upsert = (prev: Msg[], role: Role, text: string): Msg[] =>
  prev.at(-1)?.role === role ? [...prev.slice(0, -1), { role, text }] : [...prev, { role, text }]
