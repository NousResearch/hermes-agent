import { useContext, useMemo } from 'react'

import StdinContext from '../components/StdinContext.js'
import type { DOMElement } from '../dom.js'
import instances from '../instances.js'
import type { MatchPosition } from '../render-to-screen.js'

/**
 * Set the search highlight query on the Ink instance. Non-empty → all
 * visible occurrences are inverted on the next frame (SGR 7, screen-buffer
 * overlay, same damage machinery as selection). Empty → clears.
 *
 * This is a screen-space highlight — it matches the RENDERED text, not the
 * source message text. Works for anything visible (bash output, file paths,
 * error messages) regardless of where it came from in the message tree. A
 * query that matched in source but got truncated/ellipsized in rendering
 * won't highlight; that's acceptable — we highlight what you see.
 */
export function useSearchHighlight(): {
  setQuery: (query: string) => void
  /** Paint an existing DOM subtree (from the MAIN tree) to a fresh
   *  Screen at its natural height, scan. Element-relative positions
   *  (row 0 = element top). Zero context duplication — the element
   *  IS the one built with all real providers. */
  scanElement: (el: DOMElement) => MatchPosition[]
  /** Position-based CURRENT highlight. Every frame writes yellow at
   *  positions[currentIdx] + rowOffset. The scan-highlight (inverse on
   *  all matches) still runs — this overlays on top. rowOffset tracks
   *  scroll; positions stay stable (message-relative). null clears. */
  setPositions: (
    state: {
      positions: MatchPosition[]
      rowOffset: number
      currentIdx: number
    } | null
  ) => void
} {
  useContext(StdinContext) // anchor to App subtree for hook rules
  const ink = instances.get(process.stdout)

  return useMemo(() => {
    if (!ink) {
      return {
        setQuery: () => {},
        scanElement: () => [],
        setPositions: () => {}
      }
    }

    return {
      setQuery: (query: string) => ink.setSearchHighlight(query),
      scanElement: (el: DOMElement) => ink.scanElementSubtree(el),
      setPositions: state => ink.setSearchPositions(state)
    }
  }, [ink])
}
