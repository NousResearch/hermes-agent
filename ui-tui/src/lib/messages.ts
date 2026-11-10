import { MAX_HISTORY } from '../config/limits.js'
import type { Msg, Role } from '../types.js'

import { appendToolShelfMessage } from './liveProgress.js'

// Stamp live rows AT APPEND (wall clock, Unix seconds) rather than later:
// a message's authoring time is when it entered the transcript, not when it
// happened to be persisted or re-rendered (#82840-class rule). Rehydrated
// rows arrive with their persisted `createdAt` and keep it.
export const appendTranscriptMessage = (prev: Msg[], msg: Msg): Msg[] =>
  appendToolShelfMessage(prev, msg.createdAt === undefined ? { ...msg, createdAt: Date.now() / 1000 } : msg)

export const capTranscriptHistory = (items: Msg[]): Msg[] => {
  if (items.length <= MAX_HISTORY) {
    return items
  }

  return items[0]?.kind === 'intro' ? [items[0], ...items.slice(-(MAX_HISTORY - 1))] : items.slice(-MAX_HISTORY)
}

export const upsert = (prev: Msg[], role: Role, text: string): Msg[] =>
  prev.at(-1)?.role === role ? [...prev.slice(0, -1), { role, text }] : [...prev, { role, text }]
