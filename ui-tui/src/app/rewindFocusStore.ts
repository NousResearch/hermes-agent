import { atom } from 'nanostores'

import { splitImageNotes } from '../lib/imageNotes.js'
import { PASTE_SNIPPET_RE } from '../protocol/paste.js'
import type { PasteSection } from '../types.js'

/** The prompt backtrack browsing has selected; the transcript scrolls to it and draws it in reverse video.
 *  `rowId` is its durable message row, `turn` its ordinal among user turns, `text` its stored text. */
export interface RewindFocus {
  rowId?: number
  text: string
  turn: number
}

export const $rewindFocus = atom<null | RewindFocus>(null)

/** Set by Esc ("back") just before browsing closes: the transcript returns to where it was before browsing. */
export const $rewindScrollRestore = atom(false)

interface FocusRow {
  pastes?: PasteSection[]
  role: string
  rowId?: number
  text: string
}

const squash = (text: string) => text.replace(/\s+/g, ' ').trim()

/** A transcript row's prompt as stored: its paste chips expanded, image chips dropped (the image travels
 *  beside the text). Rows painted at send time carry no row id, so this is how they are recognised. */
const rowPrompt = (row: FocusRow) => {
  const pastes = new Map<string, string[]>()

  for (const paste of row.pastes ?? []) {
    pastes.set(paste.label, [...(pastes.get(paste.label) ?? []), paste.text])
  }

  return squash(row.text.replace(PASTE_SNIPPET_RE, label => pastes.get(label)?.shift() ?? ''))
}

/** The stored prompt as typed: the image notes the model got removed. */
const storedPrompt = (text: string) => squash(splitImageNotes(text).text.replace(PASTE_SNIPPET_RE, ''))

/** The transcript row of the focused prompt: the row carrying its row id; else the `turn`-th user row when
 *  its prompt matches, else the last user row with that prompt (a display projection can fold or drop
 *  rows), else -1. */
export function rewindFocusIndex(rows: readonly FocusRow[], focus: null | RewindFocus) {
  if (!focus) {
    return -1
  }

  if (focus.rowId !== undefined) {
    const byId = rows.findIndex(row => row.role === 'user' && row.rowId === focus.rowId)

    if (byId >= 0) {
      return byId
    }
  }

  const want = storedPrompt(focus.text)
  const users = rows.flatMap((row, index) => (row.role === 'user' ? [index] : []))
  const byTurn = users[focus.turn - 1]

  if (byTurn !== undefined && rowPrompt(rows[byTurn]!) === want) {
    return byTurn
  }

  return users.findLast(index => rowPrompt(rows[index]!) === want) ?? -1
}
