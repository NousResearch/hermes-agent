import { COMPOSER_SUBMIT_LOCK_MS } from '../config/timing.js'

let lockedUntil = 0

/**
 * Incoming transcript rows (assistant reply, interim bubble, bg/btw notice)
 * remount/reflow the composer. That redraw can flush a leftover CR into
 * TextInput as Enter and send whatever the user was still typing.
 *
 * Lock user-Enter submit for a short window after those events. Voice,
 * slash, and billing go through submitRef, not TextInput, so they stay live.
 */
export function lockComposerSubmit(now = Date.now(), ms = COMPOSER_SUBMIT_LOCK_MS): void {
  lockedUntil = Math.max(lockedUntil, now + ms)
}

export function isComposerSubmitLocked(now = Date.now()): boolean {
  return now < lockedUntil
}

export function resetComposerSubmitGuard(): void {
  lockedUntil = 0
}

export function hasComposerDraft(input: string, inputBuf: readonly string[]): boolean {
  return Boolean(input.trim() || inputBuf.length)
}

/**
 * Queue auto-drain fires on busy → false (a chat message just settled).
 * Skip it while the composer still has unsent text — that's the user
 * mid-draft, not a ready follow-up.
 */
export function shouldDrainQueuedFollowUp(opts: {
  busy: boolean
  composerDraft: boolean
  queueEdit: number | null
  queueLength: number
  sid: string | null | undefined
}): boolean {
  return Boolean(
    opts.sid && !opts.busy && opts.queueEdit === null && opts.queueLength > 0 && !opts.composerDraft
  )
}
