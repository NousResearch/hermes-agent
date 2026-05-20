import { createContext } from 'react'

/**
 * Notify Ink that the physical terminal cursor was advanced by an
 * out-of-band stdout.write (e.g. the TextInput fast-echo path).
 *
 * This is a two-part notification — calling it updates both:
 *
 *   1. Ink's cached `displayCursor` (the basis log-update uses to
 *      compute relative cursor moves for the next frame's preamble).
 *      Without this, the next frame's preamble starts from a stale
 *      parked position and the diff is rendered N cells offset.
 *      This half is SKIPPED on alt-screen — every alt-screen frame
 *      begins with CSI H which absolutely repositions the cursor, so
 *      the relative-move basis is reset for free.
 *
 *   2. Ink's active `cursorDeclaration` (the target the cursor parks
 *      at after every frame, set by `useDeclaredCursor`). Without
 *      this, an unrelated component re-rendering before the deferred
 *      React state catches up would publish a stale declaration and
 *      visually undo the fast-echo's advance. This half applies to
 *      BOTH main-screen and alt-screen — on alt-screen the cursor-
 *      park branch in onRender emits an absolute CUP to
 *      `rect.x + decl.relativeX`, so a stale declaration there is
 *      still wrong even though displayCursor is skipped.
 *
 * `dx`/`dy` are deltas in terminal cells (positive = right/down,
 * negative = left/up). The caller is responsible for ensuring the
 * physical cursor really did move by that amount.
 */
export type CursorAdvanceNotifier = (dx: number, dy?: number) => void

const CursorAdvanceContext = createContext<CursorAdvanceNotifier>(() => {})

export default CursorAdvanceContext
