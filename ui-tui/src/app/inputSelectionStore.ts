import { atom } from 'nanostores'

export interface InputSelection {
  canMoveUp?: boolean
  canMoveDown?: boolean
  clear: () => void
  collapseToEnd: () => void
  copy: () => void
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
