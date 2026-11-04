import { atom } from 'nanostores'

import { PASTE_SNIPPET_RE } from '../protocol/paste.js'
import type { Msg, PasteSection } from '../types.js'

import type { ComposerToken } from './interfaces.js'

const MAX_STAGED = 64

// Payloads of paste chips just submitted, waiting for their user row. A row
// can be painted later than the submit (a busy-time admission paints when the
// authority starts it), so they wait here by label, oldest first, and each row
// takes the ones its own text names. Two edits of one paste can share a label;
// FIFO keeps each row on the payload it actually sent.
const staged = new Map<string, string[]>()

/** Ctrl+P with no chip in the composer: open (or close) every sent paste section. */
export const $pasteSectionsOpen = atom(false)

export const togglePasteSections = () => $pasteSectionsOpen.set(!$pasteSectionsOpen.get())

export function stagePastes(tokens: readonly ComposerToken[]) {
  for (const token of tokens) {
    if (token.kind === 'paste') {
      // Re-insert so the Map's order stays oldest-first for eviction.
      const queue = [...(staged.get(token.label) ?? []), token.text]
      staged.delete(token.label)
      staged.set(token.label, queue)
    }
  }

  // A row that never shows its chip (a gateway that echoes the expanded text)
  // must not pin payloads forever.
  for (const label of staged.keys()) {
    if (staged.size <= MAX_STAGED) {
      break
    }

    staged.delete(label)
  }
}

/** The user row with the payload of every staged chip its text shows. */
export function attachPastes(msg: Msg): Msg {
  if (msg.role !== 'user' || !staged.size) {
    return msg
  }

  const pastes: PasteSection[] = []

  for (const label of msg.text.match(new RegExp(PASTE_SNIPPET_RE.source, 'g')) ?? []) {
    const queue = staged.get(label)
    const text = queue?.shift()

    if (text === undefined) {
      continue
    }

    pastes.push({ label, text })

    if (!queue?.length) {
      staged.delete(label)
    }
  }

  return pastes.length ? { ...msg, pastes } : msg
}
