import { stringWidth } from '@hermes/ink'

let _seg: Intl.Segmenter | null = null
const seg = () => (_seg ??= new Intl.Segmenter(undefined, { granularity: 'grapheme' }))

/**
 * Mirrors the char-wrap behavior used by the composer TextInput.
 * Returns the zero-based visual line and column of the cursor cell.
 */
export function cursorLayout(value: string, cursor: number, cols: number) {
  const pos = Math.max(0, Math.min(cursor, value.length))
  const w = Math.max(1, cols)

  let col = 0,
    line = 0

  for (const { segment, index } of seg().segment(value)) {
    if (index >= pos) {
      break
    }

    if (segment === '\n') {
      line++
      col = 0

      continue
    }

    const sw = stringWidth(segment)

    if (!sw) {
      continue
    }

    if (col + sw > w) {
      line++
      col = 0
    }

    col += sw
  }

  // trailing cursor-cell overflows to the next row at the wrap column
  if (col >= w) {
    line++
    col = 0
  }

  return { column: col, line }
}

export function inputVisualHeight(value: string, columns: number) {
  return cursorLayout(value, value.length, columns).line + 1
}

export function stableComposerColumns(totalCols: number, promptWidth: number) {
  // Physical render/wrap width. Always reserve outer composer padding and
  // prompt prefix. Only reserve the transcript scrollbar gutter when the
  // terminal is wide enough; on narrow panes, preserving input columns beats
  // keeping gutters visually aligned.
  return Math.max(1, totalCols - promptWidth - 2 - (totalCols - promptWidth >= 24 ? 2 : 0))
}
