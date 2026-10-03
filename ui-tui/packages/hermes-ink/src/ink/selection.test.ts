import { describe, expect, it } from 'vitest'

import Output from './output.js'
import { cellAt, CellWidth, CharPool, createScreen, HyperlinkPool, setCellAt, type Screen, StylePool } from './screen.js'
import {
  applySelectionOverlay,
  createSelectionState,
  getSelectedText,
  startSelection,
  updateSelection
} from './selection.js'
import { wrapTextWithTrim } from './wrap-text.js'

const screenWithText = () => {
  const styles = new StylePool()
  const screen = createScreen(10, 3, styles, new CharPool(), new HyperlinkPool())

  setCellAt(screen, 2, 1, { char: 'h', hyperlink: undefined, styleId: screen.emptyStyleId, width: CellWidth.Narrow })
  setCellAt(screen, 3, 1, { char: 'i', hyperlink: undefined, styleId: screen.emptyStyleId, width: CellWidth.Narrow })

  return { screen, styles }
}

describe('selection whitespace handling', () => {
  it('does not copy whitespace-only selections', () => {
    const { screen } = screenWithText()
    const selection = createSelectionState()

    startSelection(selection, 0, 0)
    updateSelection(selection, 9, 0)

    expect(getSelectedText(selection, screen)).toBe('')
  })

  it('trims outer drag padding while preserving selected content', () => {
    const { screen } = screenWithText()
    const selection = createSelectionState()

    startSelection(selection, 0, 1)
    updateSelection(selection, 9, 1)

    expect(getSelectedText(selection, screen)).toBe('hi')
  })

  it('preserves selected indentation when spaces are rendered content', () => {
    const styles = new StylePool()
    const screen = createScreen(10, 1, styles, new CharPool(), new HyperlinkPool())
    const selection = createSelectionState()

    setCellAt(screen, 0, 0, { char: ' ', hyperlink: undefined, styleId: screen.emptyStyleId, width: CellWidth.Narrow })
    setCellAt(screen, 1, 0, { char: ' ', hyperlink: undefined, styleId: screen.emptyStyleId, width: CellWidth.Narrow })
    setCellAt(screen, 2, 0, { char: 'x', hyperlink: undefined, styleId: screen.emptyStyleId, width: CellWidth.Narrow })

    startSelection(selection, 0, 0)
    updateSelection(selection, 9, 0)

    expect(getSelectedText(selection, screen)).toBe('  x')
  })

  it('clamps copied selection bounds to screen width', () => {
    const { screen } = screenWithText()
    const selection = createSelectionState()

    startSelection(selection, 0, 1)
    updateSelection(selection, 99, 1)

    expect(getSelectedText(selection, screen)).toBe('hi')
  })

  it('does not paint selection background on leading/trailing empty cells or empty rows', () => {
    const { screen, styles } = screenWithText()
    const selection = createSelectionState()

    startSelection(selection, 0, 0)
    updateSelection(selection, 9, 2)
    applySelectionOverlay(screen, selection, styles)

    expect(cellAt(screen, 0, 0)?.styleId).toBe(screen.emptyStyleId)
    expect(cellAt(screen, 0, 1)?.styleId).toBe(screen.emptyStyleId)
    expect(cellAt(screen, 2, 1)?.styleId).not.toBe(screen.emptyStyleId)
    expect(cellAt(screen, 4, 1)?.styleId).toBe(screen.emptyStyleId)
    expect(cellAt(screen, 0, 2)?.styleId).toBe(screen.emptyStyleId)
  })
})

/** Render a wrap-trim paragraph exactly like the markdown path does
 *  (wrapTextWithTrim → per-line 0/1/2 marks → Output.write), then drag
 *  across the soft-wrap boundary and read back the copy. */
function renderWrapTrim(src: string, wrapWidth: number) {
  const width = 30
  const height = 5
  const stylePool = new StylePool()
  const screen = createScreen(width, height, stylePool, new CharPool(), new HyperlinkPool())
  const entry = wrapTextWithTrim(src, wrapWidth, 'wrap-trim')
  const output = new Output({ height, screen, stylePool, width })

  output.write(
    0,
    0,
    entry.text,
    [0, ...entry.trimmed.map(t => (t ? 2 : 1))]
  )

  return output.get()
}

function dragCopy(screen: Screen, from: [number, number], to: [number, number]): string {
  const selection = createSelectionState()

  startSelection(selection, from[0], from[1])
  updateSelection(selection, to[0], to[1])

  return getSelectedText(selection, screen)
}

describe('wrap-trim drag-copy round-trip (#118395)', () => {
  it('restores the boundary space glued by wrap-trim', () => {
    const screen = renderWrapTrim('alpha beta', 7)

    expect(dragCopy(screen, [0, 0], [29, 1])).toBe('alpha beta')
  })

  it('restores a longer paragraph across several boundaries', () => {
    const src = 'lorem ipsum dolor sit amet'
    const screen = renderWrapTrim(src, 10)

    expect(dragCopy(screen, [0, 0], [29, 4])).toBe(src)
  })

  it('keeps hard mid-word splits glued (long tokens, URLs)', () => {
    const src = 'abcdefghij'
    const screen = renderWrapTrim(src, 7)

    expect(dragCopy(screen, [0, 0], [29, 1])).toBe(src)
  })
})
