import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { resetBidiDetectionForTest } from './bidi.js'
import Output from './output.js'
import { blitRegion, cellAt, CharPool, createScreen, HyperlinkPool, type Screen, shiftRows, StylePool } from './screen.js'
import { createSelectionState, getSelectedText, startSelection, updateSelection } from './selection.js'

const HEIGHT = 4
const WIDTH = 24

function renderLine(line: string, y = 0) {
  const stylePool = new StylePool()
  const screen = createScreen(WIDTH, HEIGHT, stylePool, new CharPool(), new HyperlinkPool())
  const output = new Output({ width: WIDTH, height: HEIGHT, screen, stylePool })

  output.write(0, y, line)

  return output.get()
}

function rowText(screen: Screen, row = 0): string {
  let text = ''

  for (let col = 0; col < WIDTH; col++) {
    text += cellAt(screen, col, row)?.char ?? ''
  }

  return text
}

/** What a drag across row `row`, columns [colStart, colEnd], puts on the clipboard. */
function copyRow(screen: Screen, row = 0, colStart = 0, colEnd = WIDTH - 1): string {
  const selection = createSelectionState()

  startSelection(selection, colStart, row)
  updateSelection(selection, colEnd, row)

  return getSelectedText(selection, screen)
}

beforeEach(() => {
  // Force software reordering regardless of the CI terminal's env.
  process.env['HERMES_TUI_BIDI'] = '1'
  resetBidiDetectionForTest()
})

afterEach(() => {
  delete process.env['HERMES_TUI_BIDI']
  resetBidiDetectionForTest()
})

describe('bidi selection copy', () => {
  it('leaves the drawn cells in visual order', () => {
    // The fix must not change what the terminal paints — only what copy returns.
    expect(rowText(renderLine('שלום')).trimEnd()).toBe('םולש')
  })

  it('copies a pure-RTL line back in logical order', () => {
    expect(copyRow(renderLine('שלום'))).toBe('שלום')
  })

  it('copies a mixed line back in logical order', () => {
    expect(copyRow(renderLine('abc שלום def'))).toBe('abc שלום def')
  })

  it('copies a partial selection in logical order', () => {
    // Visual layout of 'שלום עולם' is 'םלוע םולש'; a drag over the run's
    // right-hand half starts at visual column 5 and must yield 'שלום'.
    const screen = renderLine('שלום עולם')

    expect(copyRow(screen, 0, 5, 8)).toBe('שלום')
    expect(copyRow(screen, 0, 0, 3)).toBe('עולם')
    expect(copyRow(screen, 0, 5, 8) + ' ' + copyRow(screen, 0, 0, 3)).toBe('שלום עולם')
  })

  it('mirrors brackets back to their logical glyphs on copy', () => {
    const screen = renderLine('שלום (עולם)')

    expect(rowText(screen).trimEnd()).toBe('(םלוע) םולש')
    expect(copyRow(screen)).toBe('שלום (עולם)')
  })

  it('does not remap a row a second writer shares', () => {
    const stylePool = new StylePool()
    const screen = createScreen(WIDTH, HEIGHT, stylePool, new CharPool(), new HyperlinkPool())
    const output = new Output({ width: WIDTH, height: HEIGHT, screen, stylePool })

    output.write(0, 0, 'שלום')
    output.write(10, 0, 'X')

    const rendered = output.get()

    // No unambiguous mapping for a shared row — fall back to the visual bytes
    // (today's behaviour) rather than a mapping that may not hold.
    expect(copyRow(rendered)).toContain('םולש')
    expect(copyRow(rendered)).toContain('X')
  })

  it('survives a blit into the next frame', () => {
    const source = renderLine('שלום')
    const stylePool = new StylePool()
    const target = createScreen(WIDTH, HEIGHT, stylePool, source.charPool, source.hyperlinkPool)

    blitRegion(target, source, 0, 0, WIDTH, HEIGHT)

    expect(copyRow(target)).toBe('שלום')
  })

  it('carries the mapping through a scroll shift', () => {
    const screen = renderLine('שלום', 1)

    // SU by 1: row 1's line (and its record) moves to row 0.
    shiftRows(screen, 0, HEIGHT - 1, 1)

    expect(copyRow(screen, 0)).toBe('שלום')
    expect(copyRow(screen, 1)).toBe('')
  })
})
