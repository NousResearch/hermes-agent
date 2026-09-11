import { MAX_HISTORY } from '../config/limits.js'
import type { Msg, Role } from '../types.js'

import { appendToolShelfMessage } from './liveProgress.js'

// Stamp live rows AT APPEND (wall clock, Unix seconds) rather than later:
// a message's authoring time is when it entered the transcript, not when it
// happened to be persisted or re-rendered (#82840-class rule). Rehydrated
// rows arrive with their persisted `createdAt` and keep it.
export const appendTranscriptMessage = (prev: Msg[], msg: Msg): Msg[] => {
  // Transcript-snapshot + live-tail race guard (#88362): a session
  // re-activate can hand us the persisted transcript snapshot and then
  // replay the same tail events, appending an identical user+assistant
  // pair twice (the DB stays clean — pure render duplication; the
  // assistant-side variants of this race were fixed in #59634/#59673,
  // the pairwise variant reproduced here). Skip an exact adjacent
  // replay: same role+text as the previous row, both plain messages.
  // Rows with a special `kind` are exempt — tool-shelf/trail rows merge
  // into holders and legitimately repeat. A user cannot legitimately
  // submit the same text twice in a row anyway: the composer is busy
  // until the assistant reply lands between the two rows.
  const last = prev.at(-1)
  if (
    last !== undefined &&
    msg.kind === undefined &&
    last.kind === undefined &&
    last.role === msg.role &&
    last.text === msg.text
  ) {
    return prev
  }
  return appendToolShelfMessage(prev, msg.createdAt === undefined ? { ...msg, createdAt: Date.now() / 1000 } : msg)
}

export const capTranscriptHistory = (items: Msg[]): Msg[] => {
  if (items.length <= MAX_HISTORY) {
    return items
  }

  return items[0]?.kind === 'intro' ? [items[0], ...items.slice(-(MAX_HISTORY - 1))] : items.slice(-MAX_HISTORY)
}

export const upsert = (prev: Msg[], role: Role, text: string): Msg[] =>
  prev.at(-1)?.role === role ? [...prev.slice(0, -1), { role, text }] : [...prev, { role, text }]
