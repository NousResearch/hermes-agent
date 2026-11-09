import { atom } from 'nanostores'

export interface InputSelection {
  canMoveUp?: boolean
  canMoveDown?: boolean
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

/** History only owns arrows when no rendered row remains in that direction. */
export function atHistoryBoundary(text: string, direction: -1 | 1, mode: 'auto' | 'cursor') {
  if (mode === 'cursor') {
    return false
  }

  const selection = getInputSelection()
  const canMove = direction < 0 ? selection?.canMoveUp : selection?.canMoveDown

  if (canMove !== undefined) {
    return !canMove
  }

  if (!text) {
    return true
  }

  if (!selection || selection.start !== selection.end) {
    return false
  }

  return direction < 0
    ? text.lastIndexOf('\n', Math.max(0, selection.start - 1)) < 0
    : text.indexOf('\n', selection.start) < 0
}

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
