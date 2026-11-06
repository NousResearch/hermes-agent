import { atom } from 'nanostores'

export interface InputSelection {
  clear: () => void
  collapseToEnd: () => void
  copy: () => void
  /** Where the caret is: the moving end of a selection, unlike `start`/`end`. */
  cursor: number
  cut: () => void
  end: number
  start: number
  value: string
}

export const $inputSelection = atom<InputSelection | null>(null)

export const setInputSelection = (next: InputSelection | null) => $inputSelection.set(next)

export const getInputSelection = () => $inputSelection.get()

// A cursor the composer's owner wants with the next value it sets (Ctrl+P
// opening or folding a paste); the input otherwise puts an outside value's
// cursor at the end of the draft.
let placed: null | { cursor: number; value: string } = null

export const placeInputCursor = (value: string, cursor: number) => {
  placed = { cursor, value }
}

/** The placed cursor if `value` is the value it was placed for (consumed either way). */
export const takeInputCursor = (value: string) => {
  const hit = placed?.value === value ? placed.cursor : undefined
  placed = null

  return hit
}
