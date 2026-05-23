import { cellAtIndex, CellWidth, type Screen } from './screen.js'

/**
 * Builds a string from a screen row, mapping code units to their original
 * screen cell positions. Used for search scanning and highlighting.
 *
 * It skips:
 *  - SpacerTail (part of wide chars)
 *  - SpacerHead (end-of-line padding)
 *  - noSelect (gutters, line numbers)
 *
 * @returns { text: string, colOf: number[], codeUnitToCell: number[] }
 *   - text: the lowercased text content of the row
 *   - colOf: maps an index (0 to non-skipped cells) to the screen column
 *   - codeUnitToCell: maps a string index in `text` to the `colOf` index
 */
export function buildRowText(
  screen: Screen,
  row: number
): { text: string; colOf: number[]; codeUnitToCell: number[] } {
  const w = screen.width
  const noSelect = screen.noSelect
  const rowOff = row * w

  let text = ''
  const colOf: number[] = []
  const codeUnitToCell: number[] = []

  for (let col = 0; col < w; col++) {
    const idx = rowOff + col
    const cell = cellAtIndex(screen, idx)

    if (cell.width === CellWidth.SpacerTail || cell.width === CellWidth.SpacerHead || noSelect[idx] === 1) {
      continue
    }

    const lc = cell.char.toLowerCase()
    const cellIdx = colOf.length

    for (let i = 0; i < lc.length; i++) {
      codeUnitToCell.push(cellIdx)
    }

    text += lc
    colOf.push(col)
  }

  return { text, colOf, codeUnitToCell }
}
