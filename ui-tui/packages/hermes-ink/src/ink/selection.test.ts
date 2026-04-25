import { describe, expect, it } from 'vitest'

import { cellAt, CellWidth, CharPool, createScreen, HyperlinkPool, setCellAt, StylePool } from './screen.js'
import {
  applySelectionOverlay,
  createSelectionState,
  getSelectedText,
  startSelection,
  updateSelection
} from './selection.js'

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
