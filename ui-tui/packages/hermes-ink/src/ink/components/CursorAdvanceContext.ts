import { createContext } from 'react'

/**
 * Notify Ink that the physical terminal cursor was advanced by an
 * out-of-band stdout.write (e.g. the TextInput fast-echo path) so that
 * Ink's `displayCursor` tracking stays in sync.
 *
 * Without this, log-update's relative cursor moves on the next frame
 * compute from a stale `parked` position, and the hardware cursor parks
 * `dx` columns to the right of where the caret should be — visible as
 * "extra space after my word as I type" on long sessions where unrelated
 * components (status bar timer, streaming reasoning, etc.) re-render
 * between fast-echo writes and the deferred composer re-render.
 *
 * `dx`/`dy` are deltas in terminal cells (positive = right/down,
 * negative = left/up). The caller is responsible for ensuring the
 * physical cursor really did move by that amount.
 */
export type CursorAdvanceNotifier = (dx: number, dy?: number) => void

const CursorAdvanceContext = createContext<CursorAdvanceNotifier>(() => {})

export default CursorAdvanceContext
