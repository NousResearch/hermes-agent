import { atom } from 'nanostores'

/** The prompt backtrack browsing has selected; the transcript scrolls to it and draws it in reverse video.
 *  `turn` is the prompt's ordinal among user turns, `text` disambiguates when rows differ. */
export interface RewindFocus {
  text: string
  turn: number
}

export const $rewindFocus = atom<null | RewindFocus>(null)

/** Set by Esc ("back") just before browsing closes: the transcript returns to where it was before browsing. */
export const $rewindScrollRestore = atom(false)

/** The transcript row of the focused prompt: the `turn`-th user row when its text matches, else the
 *  last user row with that text (a display projection can fold or drop rows), else -1. */
export function rewindFocusIndex(rows: readonly { role: string; text: string }[], focus: null | RewindFocus) {
  if (!focus) {
    return -1
  }

  const want = focus.text.trim()
  const users = rows.flatMap((row, index) => (row.role === 'user' ? [index] : []))
  const byTurn = users[focus.turn - 1]

  if (byTurn !== undefined && rows[byTurn]!.text.trim() === want) {
    return byTurn
  }

  return users.findLast(index => rows[index]!.text.trim() === want) ?? -1
}
