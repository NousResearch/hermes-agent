import { buildRowText } from './text-builder.js'
import { cellAtIndex, CellWidth, type Screen, setCellStyleId, type StylePool } from './screen.js'

/**
 * Highlight all visible occurrences of `query` in the screen buffer by
 * inverting cell styles (SGR 7). Post-render, same damage-tracking machinery
 * as applySelectionOverlay — the diff picks up highlighted cells as ordinary
 * changes, LogUpdate stays a pure diff engine.
 *
 * Case-insensitive. Handles wide characters (CJK, emoji) by building a
 * col-of-char map per row — the Nth character isn't at col N when wide chars
 * are present (each occupies 2 cells: head + SpacerTail).
 *
 * This ONLY inverts — there is no "current match" logic here. The yellow
 * current-match overlay is handled separately by applyPositionedHighlight
 * (render-to-screen.ts), which writes on top using positions scanned from
 * the target message's DOM subtree.
 *
 * Returns true if any match was highlighted (damage gate — caller forces
 * full-frame damage when true).
 */
export function applySearchHighlight(screen: Screen, query: string, stylePool: StylePool): boolean {
  if (!query) {
    return false
  }

  const lq = query.toLowerCase()
  const qlen = lq.length
  const w = screen.width
  const noSelect = screen.noSelect
  const height = screen.height

  let applied = false

  for (let row = 0; row < height; row++) {
    const rowOff = row * w
    const { text, colOf, codeUnitToCell } = buildRowText(screen, row)

    let pos = text.indexOf(lq)

    while (pos >= 0) {
      applied = true
      const startCi = codeUnitToCell[pos]!
      const endCi = codeUnitToCell[pos + qlen - 1]!

      for (let ci = startCi; ci <= endCi; ci++) {
        const col = colOf[ci]!
        const cell = cellAtIndex(screen, rowOff + col)
        setCellStyleId(screen, col, row, stylePool.withInverse(cell.styleId))
      }

      // Non-overlapping advance (less/vim/grep/Ctrl+F). pos+1 would find
      // 'aa' at 0 AND 1 in 'aaa' → double-invert cell 1.
      pos = text.indexOf(lq, pos + qlen)
    }
  }

  return applied
}
