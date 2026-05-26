import { atom } from 'nanostores'

export interface InputSelection {
  clear: () => void
  collapseToEnd: () => void
  end: number
  start: number
  value: string
}

export const $inputSelection = atom<InputSelection | null>(null)

export const setInputSelection = (next: InputSelection | null) => $inputSelection.set(next)

export const getInputSelection = () => $inputSelection.get()
