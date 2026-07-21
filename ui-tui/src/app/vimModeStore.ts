import { atom } from 'nanostores'
import type { VimMode } from './vimMode.js'

/** Current vim mode (normal/insert) */
export const $vimMode = atom<VimMode>('normal')

/** Whether vim mode is enabled in the TUI */
export const $vimEnabled = atom(false)

/**
 * External cursor position request.
 * When set to a non-null number, textInput should move cursor to this position.
 * The textInput will reset it to null after processing.
 */
export const $vimPendingCursor = atom<number | null>(null)

export function setVimMode(mode: VimMode): void {
  $vimMode.set(mode)
}

export function setVimEnabled(enabled: boolean): void {
  $vimEnabled.set(enabled)
}

export function toggleVimEnabled(): void {
  $vimEnabled.set(!$vimEnabled.get())
}

export function getVimMode(): VimMode {
  return $vimMode.get()
}

export function getVimEnabled(): boolean {
  return $vimEnabled.get()
}

/** Request the textInput to move its cursor to the given position */
export function requestCursorMove(pos: number): void {
  $vimPendingCursor.set(pos)
}
