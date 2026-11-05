// Codex-style Esc backtrack, the two states that live outside the browse overlay:
//  - PRIME: the first Esc on an idle, empty composer. No time window: it lasts until the next key
//    that is not Esc, a click, a paste, a turn starting or a session switch. The second Esc opens
//    browse mode (`overlay.rewind`). `count` is the number of earlier prompts once the prefetch
//    answers; the hint shows only when there is at least one.
//  - PENDING EDIT: Enter in browse mode put the picked prompt into the main composer. Sending a
//    non-empty, non-slash message while it is armed rewinds the session to just before that prompt
//    and sends the edit (`rewindAndSend`); nothing is cut before that send.

import { atom } from 'nanostores'

import { t } from '../i18n/runtime.js'

import type { RewindTarget } from './interfaces.js'

export interface BacktrackPrime {
  count: null | number
  sid: string
}

export interface PendingRewind {
  send: (text: string) => void
  sid: string
  target: RewindTarget
}

export const $backtrackPrime = atom<BacktrackPrime | null>(null)
export const $pendingRewind = atom<PendingRewind | null>(null)

export const primeBacktrack = (sid: string) => $backtrackPrime.set({ count: null, sid })

export const cancelBacktrackPrime = () => {
  if ($backtrackPrime.get()) {
    $backtrackPrime.set(null)
  }
}

/** The prefetch answered: record how many earlier prompts there are, if this prime is still live. */
export const setPrimeCount = (sid: string, count: number) => {
  const prime = $backtrackPrime.get()

  if (prime?.sid === sid) {
    $backtrackPrime.set({ count, sid })
  }
}

export const armPendingRewind = (pending: PendingRewind) => $pendingRewind.set(pending)

/** Drop the armed edit; nothing was cut. `sys` prints the cancel notice (explicit cancels only). */
export function cancelPendingRewind(sys?: (text: string) => void): boolean {
  if (!$pendingRewind.get()) {
    return false
  }

  $pendingRewind.set(null)
  sys?.(t('slashCmd.core.rewind.cancelled'))

  return true
}

/** Submit while an edit is armed: a non-empty, non-slash, non-shell message for the same session
 *  goes to `rewindAndSend`; anything else disarms it (and is handled normally by the caller).
 *  True when the text was consumed. */
export function sendPendingRewind(text: string, sid: null | string): boolean {
  const pending = $pendingRewind.get()

  if (!pending) {
    return false
  }

  $pendingRewind.set(null)

  if (pending.sid !== sid || !text.trim() || /^[/!]/.test(text.trimStart())) {
    return false
  }

  pending.send(text)

  return true
}
