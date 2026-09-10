/**
 * Copy-on-select gating for the TUI.
 *
 * iTerm/Claude-Code-style copy-on-select: when mouse tracking is enabled the
 * terminal forwards drags to the TUI, so the native (shift+drag) selection
 * path is unavailable. Once a drag creates a *stable* TUI selection we want
 * to write it to the system clipboard on release. This is pure decision
 * logic kept out of the React effect so it can be unit-tested without
 * rendering the TUI.
 *
 * The gate is intentionally platform-agnostic: the macOS-only restriction was
 * removed (Terminal.app swallows Cmd+C for fullscreen TUIs — the same
 * copy-on-select is also the right UX on Linux/Windows, which cannot take the
 * native selection path while mouse tracking is on).
 */

export interface StableSelectionInput {
  /** Whether a TUI selection exists at all. */
  hasSelection: boolean
  /** True while the user is still dragging (mid-selection). */
  isDragging: boolean | undefined
  /** Monotonic selection version; changes whenever the selection changes. */
  version: number
  /** Version of the last selection we already copied to the clipboard. */
  lastCopiedVersion: number | null
}

/**
 * Decide whether a selection-release event should copy the selection to the
 * system clipboard.
 *
 * Returns true only when all of:
 *   - there is a selection,
 *   - the drag has ended (a `false`/undefined isDragging — we do not copy
 *     mid-drag, only once the selection is stable),
 *   - the selection version differs from the last one already copied
 *     (de-dupe: prevents spamming the clipboard backend on every drag-move or
 *     repeated release of an unchanged selection).
 */
export function shouldCopyStableSelection(input: StableSelectionInput): boolean {
  if (!input.hasSelection) {
    return false
  }

  if (input.isDragging) {
    return false
  }

  if (input.version === input.lastCopiedVersion) {
    return false
  }

  return true
}
