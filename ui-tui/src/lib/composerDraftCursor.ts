/**
 * Draft-cursor memory for classic-style Up/Down history in the TUI composer.
 *
 * TextInput parks the cursor at end-of-value on external `value` changes unless
 * we request a restore. Climbing Up through rows, and browsing history entries,
 * must not overwrite the last real edit position on the working draft.
 */

type DraftCursorSlot = {
  pos: number | null
  /** True while Up-climbing visual rows toward history. */
  suppress: boolean
  /** True while browsing history (not on the working draft). */
  frozen: boolean
  pendingRestore: number | null
}

const slot: DraftCursorSlot = {
  pos: null,
  suppress: false,
  frozen: false,
  pendingRestore: null
}

export function rememberComposerDraftCursor(pos: number): void {
  if (slot.suppress || slot.frozen) {
    return
  }

  slot.pos = Math.max(0, pos)
}

export function getComposerDraftCursor(): number | null {
  return slot.pos
}

export function setComposerDraftCursorSuppress(on: boolean): void {
  slot.suppress = on
}

/** Lock memory while history entries are shown so their cursors cannot clobber it. */
export function setComposerDraftCursorFrozen(on: boolean): void {
  slot.frozen = on
}

/** Ask TextInput to use this cursor on the next external `value` change. */
export function requestComposerCursorRestore(pos: number | null): void {
  slot.pendingRestore = pos === null ? null : Math.max(0, pos)
}

export function takeComposerCursorRestore(fallback: number): number {
  if (slot.pendingRestore === null) {
    return fallback
  }

  const next = slot.pendingRestore
  slot.pendingRestore = null

  return next
}

/** Test helper — reset module state between cases. */
export function resetComposerDraftCursorStateForTests(): void {
  slot.pos = null
  slot.suppress = false
  slot.frozen = false
  slot.pendingRestore = null
}
