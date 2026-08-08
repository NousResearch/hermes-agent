import { MAX_HISTORY } from '../config/limits.js'
import type { Msg, Role } from '../types.js'

import { appendToolShelfMessage } from './liveProgress.js'

/** Add a display-only local timestamp to non-empty human transcript rows. */
export const stampHumanMessage = (msg: Msg, now = Date.now()): Msg =>
  (msg.role === 'user' || msg.role === 'assistant') && msg.text.trim() && msg.timestamp === undefined
    ? { ...msg, timestamp: now }
    : msg

export const streamingAssistantMessage = (text: string, timestamp: number, tools: string[] = []): Msg =>
  stampHumanMessage({ role: 'assistant', text, ...(tools.length && { tools }) }, timestamp)

export const appendTranscriptMessage = (prev: Msg[], msg: Msg): Msg[] => appendToolShelfMessage(prev, msg)

export const capTranscriptHistory = (items: Msg[]): Msg[] => {
  if (items.length <= MAX_HISTORY) {
    return items
  }

  return items[0]?.kind === 'intro' ? [items[0], ...items.slice(-(MAX_HISTORY - 1))] : items.slice(-MAX_HISTORY)
}

export const upsert = (prev: Msg[], role: Role, text: string): Msg[] =>
  prev.at(-1)?.role === role ? [...prev.slice(0, -1), { role, text }] : [...prev, { role, text }]
