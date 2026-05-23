import { PassThrough } from 'stream'

import React, { useLayoutEffect, useRef } from 'react'
import { describe, expect, it } from 'vitest'

import Box from './components/Box.js'
import ScrollBox, { type ScrollBoxHandle } from './components/ScrollBox.js'
import Text from './components/Text.js'
import { renderSync } from './root.js'

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

const makeStreams = () => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 12 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', () => {})

  return { stderr, stdin, stdout }
}

function TallOverflowHarness({ expose }: { expose: React.MutableRefObject<ScrollBoxHandle | null> }) {
  const scrollRef = useRef<ScrollBoxHandle | null>(null)

  useLayoutEffect(() => {
    expose.current = scrollRef.current
  })

  return (
    <ScrollBox height={10} ref={scrollRef} stickyScroll>
      <Box flexDirection="column">
        <Box flexShrink={0} height={40}>
          <Text>tall overflow child</Text>
        </Box>
      </Box>
    </ScrollBox>
  )
}

describe('ScrollBox overflow height', () => {
  it('counts overflowing child extents in scrollHeight', async () => {
    const expose = { current: null as ScrollBoxHandle | null }
    const streams = makeStreams()

    const instance = renderSync(<TallOverflowHarness expose={expose} />, {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    try {
      await delay(30)

      expect(expose.current!.getViewportHeight()).toBe(10)
      expect(expose.current!.getScrollHeight()).toBe(40)
      expect(expose.current!.getFreshScrollHeight()).toBe(40)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })
})
