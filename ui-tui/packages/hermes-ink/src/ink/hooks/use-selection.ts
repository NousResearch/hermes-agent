import { useContext, useMemo, useSyncExternalStore } from 'react'

import StdinContext from '../components/StdinContext.js'
import instances from '../instances.js'
import { type FocusMove, type SelectionState, shiftAnchor } from '../selection.js'

/**
 * Access to text selection operations on the Ink instance (fullscreen only).
 * Returns no-op functions when fullscreen mode is disabled.
 */
export function useSelection(): {
  copySelection: () => Promise<string>
  /** Copy without clearing the highlight (for copy-on-select). */
  copySelectionNoClear: () => Promise<string>
  clearSelection: () => void
  hasSelection: () => boolean
  /** Read the raw mutable selection state (for drag-to-scroll). */
  getState: () => SelectionState | null
  /** Subscribe to selection mutations (start/update/finish/clear). */
  subscribe: (cb: () => void) => () => void
  /** Shift the anchor row by dRow, clamped to [minRow, maxRow]. */
  shiftAnchor: (dRow: number, minRow: number, maxRow: number) => void
  /** Shift anchor AND focus by dRow (keyboard scroll: whole selection
   *  tracks content). Clamped points get col reset to the full-width edge
   *  since their content was captured by captureScrolledRows. Reads
   *  screen.width from the ink instance for the col-reset boundary. */
  shiftSelection: (dRow: number, minRow: number, maxRow: number) => void
  /** Keyboard selection extension (shift+arrow): move focus, anchor fixed.
   *  Left/right wrap across rows; up/down clamp at viewport edges. */
  moveFocus: (move: FocusMove) => void
  /** Capture text from rows about to scroll out of the viewport (call
   *  BEFORE scrollBy so the screen buffer still has the outgoing rows). */
  captureScrolledRows: (firstRow: number, lastRow: number, side: 'above' | 'below') => void
  /** Set the selection highlight bg color (theme-piping; solid bg
   *  replaces the old SGR-7 inverse so syntax highlighting stays readable
   *  under selection). Call once on mount + whenever theme changes. */
  setSelectionBgColor: (color: string) => void
  /** Monotonic counter incremented on every selection mutation. */
  version: () => number
} {
  // Look up the Ink instance via stdout — same pattern as instances map.
  // StdinContext is available (it's always provided), and the Ink instance
  // is keyed by stdout which we can get from process.stdout since there's
  // only one Ink instance per process in practice.
  useContext(StdinContext) // anchor to App subtree for hook rules
  const ink = instances.get(process.stdout)

  // Memoize so callers can safely use the return value in dependency arrays.
  // ink is a singleton per stdout — stable across renders.
  return useMemo(() => {
    if (!ink) {
      return {
        copySelection: async () => '',
        copySelectionNoClear: async () => '',
        clearSelection: () => {},
        hasSelection: () => false,
        getState: () => null,
        subscribe: () => () => {},
        shiftAnchor: () => {},
        shiftSelection: () => {},
        moveFocus: () => {},
        captureScrolledRows: () => {},
        setSelectionBgColor: () => {},
        version: () => 0
      }
    }

    return {
      copySelection: () => ink.copySelection(),
      copySelectionNoClear: () => ink.copySelectionNoClear(),
      clearSelection: () => ink.clearTextSelection(),
      hasSelection: () => ink.hasTextSelection(),
      getState: () => ink.selection,
      subscribe: (cb: () => void) => ink.subscribeToSelectionChange(cb),
      shiftAnchor: (dRow: number, minRow: number, maxRow: number) => shiftAnchor(ink.selection, dRow, minRow, maxRow),
      shiftSelection: (dRow, minRow, maxRow) => ink.shiftSelectionForScroll(dRow, minRow, maxRow),
      moveFocus: (move: FocusMove) => ink.moveSelectionFocus(move),
      captureScrolledRows: (firstRow, lastRow, side) => ink.captureScrolledRows(firstRow, lastRow, side),
      setSelectionBgColor: (color: string) => ink.setSelectionBgColor(color),
      version: () => ink.getSelectionVersion()
    }
  }, [ink])
}

const NO_SUBSCRIBE = () => () => {}
const ALWAYS_FALSE = () => false

/**
 * Reactive selection-exists state. Re-renders the caller when a text
 * selection is created or cleared. Always returns false outside
 * fullscreen mode (selection is only available in alt-screen).
 */
export function useHasSelection(): boolean {
  useContext(StdinContext)
  const ink = instances.get(process.stdout)

  return useSyncExternalStore(
    ink ? ink.subscribeToSelectionChange : NO_SUBSCRIBE,
    ink ? ink.hasTextSelection : ALWAYS_FALSE
  )
}
