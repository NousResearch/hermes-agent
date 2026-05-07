import { PassThrough } from 'stream'

import { Box, renderSync, ScrollBox, type ScrollBoxHandle, Text } from '@hermes/ink'
import React, { useLayoutEffect, useRef } from 'react'
import { describe, expect, it } from 'vitest'

import { useVirtualHistory } from '../hooks/useVirtualHistory.js'

interface Item {
  height: number
  key: string
}

interface Exposed {
  scroll: ScrollBoxHandle | null
  virtualHistory: ReturnType<typeof useVirtualHistory>
}

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

const makeStreams = () => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 20 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', () => {})

  return { stderr, stdin, stdout }
}

const mountedSpan = (items: readonly Item[], virtualHistory: ReturnType<typeof useVirtualHistory>) => {
  let height = 0

  for (let index = virtualHistory.start; index < virtualHistory.end; index++) {
    height += items[index]?.height ?? 0
  }

  return { bottom: virtualHistory.topSpacer + height, top: virtualHistory.topSpacer }
}

const viewportIsMounted = (items: readonly Item[], virtualHistory: ReturnType<typeof useVirtualHistory>, scroll: ScrollBoxHandle) => {
  const span = mountedSpan(items, virtualHistory)
  const top = scroll.getScrollTop()
  const bottom = top + scroll.getViewportHeight()

  return top >= span.top && bottom <= span.bottom
}

function Harness({ expose, items }: { expose: React.MutableRefObject<Exposed | null>; items: readonly Item[] }) {
  const scrollRef = useRef<ScrollBoxHandle | null>(null)

  const virtualHistory = useVirtualHistory(scrollRef, items, 80, {
    coldStartCount: 16,
    estimateHeight: index => items[index]?.height ?? 1,
    maxMounted: 16,
    overscan: 2
  })

  useLayoutEffect(() => {
    expose.current = { scroll: scrollRef.current, virtualHistory }
  })

  return React.createElement(
    ScrollBox,
    { flexDirection: 'column', height: 10, ref: scrollRef, stickyScroll: true },
    React.createElement(
      Box,
      { flexDirection: 'column', width: '100%' },
      virtualHistory.topSpacer > 0 ? React.createElement(Box, { height: virtualHistory.topSpacer }) : null,
      ...items
        .slice(virtualHistory.start, virtualHistory.end)
        .map(item =>
          React.createElement(
            Box,
            { height: item.height, key: item.key, ref: virtualHistory.measureRef(item.key) },
            React.createElement(Text, null, item.key)
          )
        ),
      virtualHistory.bottomSpacer > 0 ? React.createElement(Box, { height: virtualHistory.bottomSpacer }) : null
    )
  )
}

describe('useVirtualHistory offset cache reuse', () => {
  it('recomputes offsets after a mounted row height changes', async () => {
    const tall = [
      { height: 6, key: 'a' },
      { height: 6, key: 'b' },
      { height: 6, key: 'c' }
    ]

    const short = tall.map(item => ({ ...item, height: 2 }))
    const expose = { current: null as Exposed | null }
    const streams = makeStreams()

    const instance = renderSync(React.createElement(Harness, { expose, items: tall }), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    try {
      await delay(20)
      expect(expose.current!.virtualHistory.offsets[tall.length]).toBe(18)

      instance.rerender(React.createElement(Harness, { expose, items: short }))
      await delay(40)

      expect(expose.current!.virtualHistory.offsets[short.length]).toBe(6)
      expect(expose.current!.virtualHistory.bottomSpacer).toBe(0)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('ignores stale reused offset-array entries after the item count shrinks', async () => {
    const beforeShrink = Array.from({ length: 1400 }, (_, index) => ({ height: 1, key: `old${index}` }))
    const afterShrink = Array.from({ length: 800 }, (_, index) => ({ height: 7, key: `new${index}` }))
    const expose = { current: null as Exposed | null }
    const streams = makeStreams()

    const instance = renderSync(React.createElement(Harness, { expose, items: beforeShrink }), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    try {
      await delay(20)
      instance.rerender(React.createElement(Harness, { expose, items: afterShrink }))
      await delay(20)

      const scroll = expose.current!.scroll!
      const transcriptHeight = expose.current!.virtualHistory.offsets[afterShrink.length] ?? 0

      expect(transcriptHeight).toBe(5600)
      expect(scroll.getScrollTop()).toBe(transcriptHeight - scroll.getViewportHeight())

      scroll.scrollBy(-1)
      await delay(80)

      expect(scroll.getPendingDelta()).toBe(0)
      expect(viewportIsMounted(afterShrink, expose.current!.virtualHistory, scroll)).toBe(true)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })
})
