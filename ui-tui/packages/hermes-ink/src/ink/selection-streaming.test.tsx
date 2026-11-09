import { PassThrough } from 'stream'

import React, { createRef } from 'react'
import { describe, expect, it } from 'vitest'

import Box from './components/Box.js'
import ScrollBox, { type ScrollBoxHandle } from './components/ScrollBox.js'
import Text from './components/Text.js'
import Ink from './ink.js'
import { finishSelection, startSelection, updateSelection } from './selection.js'

const makeStream = (columns = 40, rows = 8) => {
  const stream = new PassThrough()

  Object.assign(stream, { columns, isTTY: false, rows })
  stream.on('data', () => {})

  return stream
}

const transcript = (count: number, scrollRef: React.RefObject<ScrollBoxHandle | null>) => (
  <Box flexDirection="column" height={8}>
    <ScrollBox flexDirection="column" height={4} ref={scrollRef} stickyScroll>
      {Array.from({ length: count }, (_, index) => (
        <Text key={index}>line-{index}</Text>
      ))}
    </ScrollBox>
  </Box>
)

describe('streaming while text is selected', () => {
  it('pauses transcript following until the text selection is cleared', () => {
    const stdout = makeStream()
    const stdin = makeStream()
    const stderr = makeStream()
    const scrollRef = createRef<ScrollBoxHandle>()

    const ink = new Ink({
      exitOnCtrlC: false,
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    })

    try {
      ink.setAltScreenActive(true, 'all')
      ink.render(transcript(8, scrollRef))
      ink.onRender()

      const scrollTop = scrollRef.current!.getScrollTop()

      startSelection(ink.selection, 0, 2)
      updateSelection(ink.selection, 5, 2)
      finishSelection(ink.selection)
      ink.onRender()

      const selectedText = ink.getTextSelectionText()

      expect(selectedText).toBe('line-6')

      ink.render(transcript(14, scrollRef))
      ink.onRender()

      expect(scrollRef.current!.getScrollTop()).toBe(scrollTop)
      expect(ink.hasTextSelection()).toBe(true)
      expect(ink.getTextSelectionText()).toBe(selectedText)

      ink.clearTextSelection()
      ink.onRender()

      expect(scrollRef.current!.getScrollTop()).toBe(
        scrollRef.current!.getScrollHeight() - scrollRef.current!.getViewportHeight()
      )
    } finally {
      ink.unmount()
    }
  })
})
