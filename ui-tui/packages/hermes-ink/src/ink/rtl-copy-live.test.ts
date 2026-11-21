/**
 * Live-shape RTL copy scenarios — the cases the happy-path tests in
 * bidi-copy.test.ts do NOT cover, but the fullscreen TUI hits every frame:
 * incremental (streaming) rewrites of a row, a second writer padding the
 * row's tail, and multi-row drags. These must copy LOGICAL order; a mirrored
 * result here is the "pastes backwards" bug.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { resetBidiDetectionForTest } from './bidi.js'
import Output from './output.js'
import { cellAt, CharPool, createScreen, HyperlinkPool, type Screen, StylePool } from './screen.js'
import { createSelectionState, getSelectedText, startSelection, updateSelection } from './selection.js'

const HEIGHT = 6
const WIDTH = 32

function makeOutput() {
  const stylePool = new StylePool()
  const screen = createScreen(WIDTH, HEIGHT, stylePool, new CharPool(), new HyperlinkPool())

  return { output: new Output({ width: WIDTH, height: HEIGHT, screen, stylePool }), stylePool }
}

function rowText(screen: Screen, row = 0): string {
  let text = ''

  for (let col = 0; col < WIDTH; col++) {
    text += cellAt(screen, col, row)?.char ?? ''
  }

  return text
}

function copyRows(screen: Screen, r0: number, r1: number, c0 = 0, c1 = WIDTH - 1): string {
  const selection = createSelectionState()

  startSelection(selection, c0, r0)
  updateSelection(selection, c1, r1)

  return getSelectedText(selection, screen)
}

beforeEach(() => {
  process.env['HERMES_TUI_BIDI'] = '1'
  resetBidiDetectionForTest()
})

afterEach(() => {
  delete process.env['HERMES_TUI_BIDI']
  resetBidiDetectionForTest()
})

describe('live-shape RTL copy', () => {
  it('copies logically after a streaming rewrite of the same row', () => {
    const { output } = makeOutput()

    // Assistant text arrives in chunks and the row is rewritten each frame.
    output.write(0, 0, 'שלום')
    output.write(0, 0, 'שלום עו')
    output.write(0, 0, 'שלום עולם')

    expect(copyRows(output.get(), 0, 0)).toBe('שלום עולם')
  })

  it('copies logically when the row tail is padded by a second writer', () => {
    const { output } = makeOutput()

    output.write(0, 0, 'שלום')
    // Clearing the rest of the row is a separate write, as the renderer does.
    output.write(4, 0, ' '.repeat(WIDTH - 4))

    expect(copyRows(output.get(), 0, 0).trimEnd()).toBe('שלום')
  })

  it('copies logically when LTR chrome shares the row', () => {
    const { output } = makeOutput()

    output.write(0, 0, 'שלום')
    output.write(10, 0, '3.9M')

    expect(copyRows(output.get(), 0, 0)).toContain('שלום')
  })

  it('copies a multi-row drag logically', () => {
    const { output } = makeOutput()

    output.write(0, 0, 'שלום עולם')
    output.write(0, 1, 'מה שלומך')

    const text = copyRows(output.get(), 0, 1)

    expect(text.split('\n')[0]!.trimEnd()).toBe('שלום עולם')
    expect(text.split('\n')[1]!.trimEnd()).toBe('מה שלומך')
  })

  it('copies logically when the record is verified after a partial rewrite', () => {
    const { output } = makeOutput()

    // Streaming rewrite: the renderer clears the row, then writes the shorter
    // line. The replacement re-records its own span, so copy stays logical.
    output.write(0, 0, 'שלום עולם')
    output.write(0, 0, ' '.repeat(WIDTH))
    output.write(0, 0, 'שלום')

    expect(copyRows(output.get(), 0, 0).trimEnd()).toBe('שלום')
  })
})
