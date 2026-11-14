import { EventEmitter } from 'events'

import React from 'react'
import { describe, expect, it } from 'vitest'

import Box from './components/Box.js'
import Text from './components/Text.js'
import Ink from './ink.js'
import { startSelection, updateSelection } from './selection.js'

class FakeTty extends EventEmitter {
  chunks: string[] = []
  columns = 40
  rows = 10
  isTTY = true

  write(chunk: string | Uint8Array, cb?: (err?: Error | null) => void): boolean {
    this.chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))
    cb?.()

    return true
  }
}

// Drive the real render pipeline (Text → render-node-to-output → Output →
// screen.softWrap) and copy from the painted front frame, the same buffer a
// fullscreen mouse drag + right-click copies from (#118395).
const dragCopy = (node: React.ReactElement, from: [number, number] = [0, 0], to?: [number, number]): string => {
  const stdout = new FakeTty()
  const stdin = new FakeTty()
  const stderr = new FakeTty()

  const ink = new Ink({
    exitOnCtrlC: false,
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  ink.setAltScreenActive(true)
  ink.render(node)
  ink.onRender()
  startSelection(ink.selection, from[0], from[1])
  updateSelection(ink.selection, ...(to ?? [stdout.columns - 1, stdout.rows - 1]))
  const copied = ink.getTextSelectionText()
  ink.unmount()

  return copied
}

const paragraph = (wrap: 'wrap' | 'wrap-trim', children: React.ReactNode) => (
  <Box width={12}>
    <Text wrap={wrap}>{children}</Text>
  </Box>
)

describe('drag-copy across wrap-trim soft-wrap boundaries', () => {
  it('keeps the word separator a wrap-trim boundary removed from the screen', () => {
    expect(dragCopy(paragraph('wrap-trim', 'lorem ipsum dolor sit amet consectetur'))).toBe(
      'lorem ipsum dolor sit amet consectetur'
    )
  })

  it('keeps separators in multi-segment (styled) markdown prose', () => {
    expect(
      dragCopy(
        paragraph(
          'wrap-trim',
          <>
            lorem <Text bold>ipsum dolor</Text> sit amet
          </>
        )
      )
    ).toBe('lorem ipsum dolor sit amet')
  })

  it('preserves the extra space when only one of several boundary spaces was trimmed', () => {
    expect(dragCopy(paragraph('wrap-trim', 'lorem ipsum  dolor'))).toBe('lorem ipsum  dolor')
  })

  it('adds no leading space when the drag starts in the padding after a wrapped row', () => {
    // Width 12 paints 'lorem ipsum' / 'dolor'; start in the cell past 'ipsum'.
    expect(dragCopy(paragraph('wrap-trim', 'lorem ipsum dolor'), [11, 0], [4, 1])).toBe('dolor')
  })

  it('keeps hard mid-word splits glued', () => {
    expect(dragCopy(paragraph('wrap-trim', 'abcdefghijklmnopqrstuvwx'))).toBe('abcdefghijklmnopqrstuvwx')
  })

  it('leaves plain wrap mode copies unchanged', () => {
    expect(dragCopy(paragraph('wrap', 'lorem ipsum dolor sit amet'))).toBe('lorem ipsum dolor sit amet')
  })

  it('keeps hard newlines as newlines and leading indentation as content', () => {
    expect(dragCopy(paragraph('wrap-trim', 'lorem ipsum dolor\nsit amet'))).toBe('lorem ipsum dolor\nsit amet')
    expect(dragCopy(paragraph('wrap-trim', '  lorem\nipsum'))).toBe('  lorem\nipsum')
  })
})
