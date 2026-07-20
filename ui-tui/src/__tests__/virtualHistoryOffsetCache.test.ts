import { PassThrough } from 'stream'

import { Box, renderSync, ScrollBox, type ScrollBoxHandle, Text } from '@hermes/ink'
import React, { useLayoutEffect, useRef, useState } from 'react'
import { describe, expect, it } from 'vitest'

import { useVirtualHistory, virtualHistorySnapshotKey } from '../hooks/useVirtualHistory.js'

interface Item {
  height: number
  heightAfterResize?: number
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

const viewportIsMounted = (
  items: readonly Item[],
  virtualHistory: ReturnType<typeof useVirtualHistory>,
  scroll: ScrollBoxHandle
) => {
  const span = mountedSpan(items, virtualHistory)
  const top = scroll.getScrollTop()
  const bottom = top + scroll.getViewportHeight()

  return top >= span.top && bottom <= span.bottom
}

const itemHeightForColumns = (item: Item | undefined, columns: number) =>
  columns >= 80 ? (item?.heightAfterResize ?? item?.height ?? 1) : (item?.height ?? 1)

const makeScrollHandle = () => {
  const listeners = new Set<() => void>()
  let clampMin: number | undefined
  let clampMax: number | undefined
  let scrollTop = 0
  let pendingDelta = 0
  const viewportHeight = 10
  let sticky = false
  let lastManualScrollAt = 0

  return {
    getClampBounds: () => ({ max: clampMax, min: clampMin }),
    getLastManualScrollAt: () => lastManualScrollAt,
    getPendingDelta: () => pendingDelta,
    getScrollTop: () => scrollTop,
    getViewportHeight: () => viewportHeight,
    isSticky: () => sticky,
    scrollBy: (delta: number) => {
      pendingDelta += delta
      listeners.forEach(listener => listener())
    },
    scrollTo: (nextTop: number) => {
      scrollTop = nextTop
      listeners.forEach(listener => listener())
    },
    setClampBounds: (min: number | undefined, max: number | undefined) => {
      clampMin = min
      clampMax = max
    },
    subscribe: (listener: () => void) => {
      listeners.add(listener)

      return () => listeners.delete(listener)
    }
  }
}

function ClampHarness({
  expose,
  items,
  measured = false
}: {
  expose: React.MutableRefObject<{ scroll: ReturnType<typeof makeScrollHandle>; virtualHistory: ReturnType<typeof useVirtualHistory> } | null>
  items: readonly Item[]
  measured?: boolean
}) {
  const [scrollHandle] = useState(() => makeScrollHandle())
  const scrollRef = useRef(scrollHandle)

  const virtualHistory = useVirtualHistory(scrollRef as unknown as React.RefObject<ScrollBoxHandle>, items, 80, {
    estimateHeight: index => itemHeightForColumns(items[index], 80),
    maxMounted: 16,
    overscan: 2
  })

  useLayoutEffect(() => {
    expose.current = { scroll: scrollRef.current, virtualHistory }
  })

  return React.createElement(
    ScrollBox,
    { flexDirection: 'column', height: 10 },
    React.createElement(
      Box,
      { flexDirection: 'column' },
      ...items.slice(virtualHistory.start, virtualHistory.end).map(item =>
        React.createElement(
          Box,
          {
            key: item.key,
            ref: (el: unknown) => {
              const measure = virtualHistory.measureRef(item.key)

              if (measured) {
                measure({ yogaNode: { getComputedHeight: () => itemHeightForColumns(item, 80) } })
              } else {
                measure({ yogaNode: { getComputedHeight: () => 0 } })
              }
            }
          },
          React.createElement(Text, null, item.key)
        )
      )
    )
  )
}

function Harness({
  columns = 80,
  expose,
  height = 10,
  items,
  maxMounted = 16
}: {
  columns?: number
  expose: React.MutableRefObject<Exposed | null>
  height?: number
  items: readonly Item[]
  maxMounted?: number
}) {
  const scrollRef = useRef<ScrollBoxHandle | null>(null)

  const virtualHistory = useVirtualHistory(scrollRef, items, columns, {
    coldStartCount: 16,
    estimateHeight: index => itemHeightForColumns(items[index], columns),
    maxMounted,
    overscan: 2
  })

  useLayoutEffect(() => {
    expose.current = { scroll: scrollRef.current, virtualHistory }
  })

  return React.createElement(
    ScrollBox,
    { flexDirection: 'column', height, ref: scrollRef, stickyScroll: true },
    React.createElement(
      Box,
      { flexDirection: 'column', width: '100%' },
      virtualHistory.topSpacer > 0 ? React.createElement(Box, { height: virtualHistory.topSpacer }) : null,
      ...items.slice(virtualHistory.start, virtualHistory.end).map(item =>
        React.createElement(
          Box,
          {
            height: itemHeightForColumns(item, columns),
            key: item.key,
            ref: virtualHistory.measureRef(item.key)
          },
          React.createElement(Text, null, item.key)
        )
      ),
      virtualHistory.bottomSpacer > 0 ? React.createElement(Box, { height: virtualHistory.bottomSpacer }) : null
    )
  )
}

describe('useVirtualHistory offset cache reuse', () => {
  it('includes viewport height in the external-store snapshot key', () => {
    const base = {
      getPendingDelta: () => 0,
      getScrollTop: () => 20,
      isSticky: () => false
    }

    const short = virtualHistorySnapshotKey({
      ...base,
      getViewportHeight: () => 5
    } as ScrollBoxHandle)

    const tall = virtualHistorySnapshotKey({
      ...base,
      getViewportHeight: () => 25
    } as ScrollBoxHandle)

    expect(short).not.toBe(tall)
  })

  it('remounts enough tail rows after the scroll viewport grows', async () => {
    const items = Array.from({ length: 100 }, (_, index) => ({ height: 1, key: `item-${index}` }))
    const expose = { current: null as Exposed | null }
    const streams = makeStreams()

    const instance = renderSync(React.createElement(Harness, { expose, height: 4, items, maxMounted: 80 }), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    try {
      await delay(20)
      instance.rerender(React.createElement(Harness, { expose, height: 9, items, maxMounted: 80 }))
      await delay(80)

      expect(viewportIsMounted(items, expose.current!.virtualHistory, expose.current!.scroll!)).toBe(true)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('recomputes tail coverage when wrapped rows shrink after a width resize', async () => {
    const items = Array.from({ length: 100 }, (_, index) => ({
      height: 4,
      heightAfterResize: 1,
      key: `item-${index}`
    }))

    const expose = { current: null as Exposed | null }
    const streams = makeStreams()

    const instance = renderSync(
      React.createElement(Harness, { columns: 40, expose, height: 10, items, maxMounted: 80 }),
      {
        patchConsole: false,
        stderr: streams.stderr as NodeJS.WriteStream,
        stdin: streams.stdin as NodeJS.ReadStream,
        stdout: streams.stdout as NodeJS.WriteStream
      }
    )

    try {
      await delay(20)
      instance.rerender(React.createElement(Harness, { columns: 80, expose, height: 10, items, maxMounted: 80 }))
      await delay(80)

      const resizedItems = items.map(item => ({ height: item.heightAfterResize!, key: item.key }))

      expect(viewportIsMounted(resizedItems, expose.current!.virtualHistory, expose.current!.scroll!)).toBe(true)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('keeps sticky scroll at the bottom when one tall tail row resizes', async () => {
    const items = [{ height: 90, heightAfterResize: 50, key: 'tail' }]
    const expose = { current: null as Exposed | null }
    const streams = makeStreams()

    const instance = renderSync(
      React.createElement(Harness, { columns: 70, expose, height: 18, items, maxMounted: 80 }),
      {
        patchConsole: false,
        stderr: streams.stderr as NodeJS.WriteStream,
        stdin: streams.stdin as NodeJS.ReadStream,
        stdout: streams.stdout as NodeJS.WriteStream
      }
    )

    try {
      await delay(20)
      instance.rerender(React.createElement(Harness, { columns: 120, expose, height: 36, items, maxMounted: 80 }))
      await delay(80)

      const scroll = expose.current!.scroll!

      expect(scroll.getScrollTop()).toBe(scroll.getScrollHeight() - scroll.getViewportHeight())
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

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

  it('defers clamp bounds until mounted rows have measured heights, not just live nodes', async () => {
    const items = Array.from({ length: 20 }, (_, index) => ({ height: 10, key: `item-${index}` }))
    const expose = { current: null as { scroll: ReturnType<typeof makeScrollHandle>; virtualHistory: ReturnType<typeof useVirtualHistory> } | null }
    const streams = makeStreams()

    const instance = renderSync(
      React.createElement(ClampHarness, { expose, items, measured: false }),
      {
        patchConsole: false,
        stderr: streams.stderr as NodeJS.WriteStream,
        stdin: streams.stdin as NodeJS.ReadStream,
        stdout: streams.stdout as NodeJS.WriteStream
      }
    )

    try {
      await delay(500)
      // Live nodes are present but their heights have not been measured (0).
      // A node-presence-only gate would install clamps here; the measured-height
      // gate must keep bounds cleared until real heights are committed.
      expect(expose.current!.scroll.getClampBounds()).toEqual({ min: undefined, max: undefined })

      instance.rerender(React.createElement(ClampHarness, { expose, items, measured: true }))
      await delay(20)

      // The hook's clamp effect runs only when the mounted range changes. Scrolling
      // a few items at a time commits the measured heights and then gives the next
      // commit a matching offset version so clamps can be installed.
      expose.current!.scroll.scrollTo(15)
      await delay(20)
      expose.current!.scroll.scrollTo(25)
      await delay(20)

      const bounds = expose.current!.scroll.getClampBounds()
      expect(bounds.min).toBeDefined()
      expect(bounds.max).toBeDefined()
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })
})
