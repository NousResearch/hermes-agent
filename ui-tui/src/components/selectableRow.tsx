import { Box } from '@hermes/ink'
import { type ReactNode, useRef } from 'react'

// Matches the composer's MULTI_CLICK_MS (textInput.tsx) so double-click feel
// is consistent across the TUI.
export const ROW_DOUBLE_CLICK_MS = 500

export interface RowClickState {
  at: number
  index: number
}

/**
 * Decide whether a click on a menu row should move the highlight (`'select'`)
 * or activate the row (`'activate'`, the mouse equivalent of Enter).
 *
 * Activate when the row is already highlighted, or when this is a second
 * click on the same row within the double-click window — so a single click on
 * an unhighlighted row selects it, and a follow-up click (or a click on the
 * current highlight) confirms. Predictable across every picker and safe for
 * ones whose Enter has side effects.
 *
 * Pure so it can be unit-tested without a terminal; `now`/`prev` are passed in
 * rather than read from the clock.
 */
export function decideRowClick(
  now: number,
  prev: RowClickState,
  index: number,
  isActive: boolean
): 'activate' | 'select' {
  const isRepeat = now - prev.at < ROW_DOUBLE_CLICK_MS && prev.index === index

  return isActive || isRepeat ? 'activate' : 'select'
}

interface SelectableRowProps {
  /** Row's index in the full list (not the visible window). */
  index: number
  /** True when this row is the current keyboard highlight. */
  isActive: boolean
  /** Move the highlight to this row (mouse equivalent of ↑/↓). */
  onSelect: (index: number) => void
  /** Activate this row (mouse equivalent of Enter). */
  onActivate?: (index: number) => void
  /** Row width so the whole line is a click target, not just the glyphs. */
  width?: number
  children: ReactNode
}

/**
 * Wrap a picker/menu row so it responds to the mouse without disturbing
 * keyboard navigation. Clicking a row highlights it (`onSelect`); clicking the
 * highlighted row — or double-clicking — activates it (`onActivate`). The
 * picker keeps its existing `useInput`; this only adds a pointer path, so
 * every menu becomes click-to-use with no command or config.
 *
 * `stopImmediatePropagation` keeps the click from bubbling to the transcript's
 * background click handler (which would clear the selection).
 */
export function SelectableRow({ children, index, isActive, onActivate, onSelect, width }: SelectableRowProps) {
  const lastClick = useRef<RowClickState>({ at: 0, index: -1 })

  return (
    <Box
      onClick={(e: { stopImmediatePropagation?: () => void }) => {
        e.stopImmediatePropagation?.()
        const now = Date.now()
        const action = decideRowClick(now, lastClick.current, index, isActive)
        lastClick.current = { at: now, index }

        if (action === 'activate') {
          onActivate?.(index)

          return
        }

        onSelect(index)
      }}
      width={width}
    >
      {children}
    </Box>
  )
}
