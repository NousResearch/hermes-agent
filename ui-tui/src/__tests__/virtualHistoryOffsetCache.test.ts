import { PassThrough } from 'stream'

import { Box, renderSync, ScrollBox, type ScrollBoxHandle, Text } from '@hermes/ink'
import React, { useLayoutEffect, useRef } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { MAX_HISTORY } from '../config/limits.js'
import { pruneVirtualHeightCache, useVirtualHistory, virtualHistorySnapshotKey } from '../hooks/useVirtualHistory.js'

interface Item {
  height: number
  heightAfterResize?: number
  key: string
  text?: string
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

function Harness({
  columns = 80,
  expose,
  height = 10,
  generation = 0,
  initialHeights,
  items,
  maxMounted = 16
}: {
  columns?: number
  expose: React.MutableRefObject<Exposed | null>
  height?: number
  generation?: number
  initialHeights?: ReadonlyMap<string, number>
  items: readonly Item[]
  maxMounted?: number
}) {
  const scrollRef = useRef<ScrollBoxHandle | null>(null)

  const virtualHistory = useVirtualHistory(scrollRef, items, columns, {
    coldStartCount: 16,
    estimateHeight: index => itemHeightForColumns(items[index], columns),
    generation,
    initialHeights,
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
          React.createElement(Text, null, item.text ?? item.key)
        )
      ),
      virtualHistory.bottomSpacer > 0 ? React.createElement(Box, { height: virtualHistory.bottomSpacer }) : null
    )
  )
}

describe('useVirtualHistory offset cache reuse', () => {
  it('prunes stable-session external height caches to active history keys', () => {
    const cache = new Map([
      ['outgoing', 9],
      ['active', 3]
    ])

    pruneVirtualHeightCache(cache, [{ key: 'active' }])

    expect([...cache]).toEqual([['active', 3]])
  })

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

  it('adjusts the committed viewport without consuming pending scroll intent', async () => {
    const items = Array.from({ length: 20 }, (_, index) => ({ height: 2, key: `item-${index}` }))
    const expose = { current: null as Exposed | null }
    const streams = makeStreams()

    const instance = renderSync(React.createElement(Harness, { expose, items }), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    try {
      await delay(20)
      const scroll = expose.current!.scroll!

      scroll.scrollTo(3)
      scroll.scrollBy(2)
      scroll.adjustScrollTop(4)

      expect(scroll.getScrollTop()).toBe(7)
      expect(scroll.getPendingDelta()).toBe(2)
      expect(scroll.isSticky()).toBe(false)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('keeps the tail clamp open while a manual non-sticky tail grows', async () => {
    const before = Array.from({ length: 20 }, (_, index) => ({ height: 2, key: `item-${index}` }))
    const after = before.map((item, index) => (index === before.length - 1 ? { ...item, height: 8 } : item))
    const expose = { current: null as Exposed | null }
    const streams = makeStreams()
    const initialHeights = new Map(before.map(item => [item.key, item.height]))

    const instance = renderSync(React.createElement(Harness, { expose, initialHeights, items: before }), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    try {
      await delay(20)
      const scroll = expose.current!.scroll!
      const setClampBounds = vi.spyOn(scroll, 'setClampBounds')

      scroll.scrollTo(28)
      await delay(20)
      instance.rerender(React.createElement(Harness, { expose, initialHeights, items: after }))
      await delay(60)

      expect(scroll.isSticky()).toBe(false)
      expect(setClampBounds.mock.calls.some(([, max]) => max === Number.POSITIVE_INFINITY)).toBe(true)

      scroll.scrollTo(36)
      await delay(20)
      expect(scroll.getScrollTop()).toBe(36)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('quarantines invalid measured heights before cache and compensation', async () => {
    const items = Array.from({ length: 20 }, (_, index) => ({ height: 2, key: `item-${index}` }))
    const expose = { current: null as Exposed | null }
    const streams = makeStreams()
    const initialHeights = new Map(items.map(item => [item.key, item.height]))

    const instance = renderSync(React.createElement(Harness, { expose, initialHeights, items }), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    try {
      await delay(20)
      const scroll = expose.current!.scroll!

      scroll.scrollTo(5)
      await delay(20)
      const adjustScrollTop = vi.spyOn(scroll, 'adjustScrollTop')
      const ref = expose.current!.virtualHistory.measureRef('item-1')

      for (const height of [Number.NaN, Number.POSITIVE_INFINITY, -1, 1_000_000_000]) {
        ref({ yogaNode: { getComputedHeight: () => height } })
        ref(null)
      }

      expect(adjustScrollTop).not.toHaveBeenCalled()
      expect(expose.current!.virtualHistory.offsets[items.length]).toBe(40)
      expect(Number.isFinite(scroll.getScrollTop())).toBe(true)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('preserves the visual anchor when a measured row above the viewport changes height', async () => {
    const before = Array.from({ length: 20 }, (_, index) => ({ height: 2, key: `item-${index}` }))
    const after = before.map((item, index) => (index === 0 ? { ...item, height: 5 } : item))
    const expose = { current: null as Exposed | null }
    const streams = makeStreams()
    const initialHeights = new Map(before.map(item => [item.key, item.height]))

    const instance = renderSync(React.createElement(Harness, { expose, initialHeights, items: before }), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    try {
      await delay(20)
      expose.current!.scroll!.scrollTo(3)
      await delay(20)

      instance.rerender(React.createElement(Harness, { expose, initialHeights, items: after }))
      await delay(40)

      expect(expose.current!.scroll!.getScrollTop()).toBe(6)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('keeps a compensated near-tail viewport manual', async () => {
    const before = Array.from({ length: 20 }, (_, index) => ({ height: 2, key: `item-${index}` }))
    const after = before.map((item, index) => (index === 13 ? { ...item, height: 5 } : item))
    const expose = { current: null as Exposed | null }
    const streams = makeStreams()
    const initialHeights = new Map(before.map(item => [item.key, item.height]))

    const instance = renderSync(React.createElement(Harness, { expose, initialHeights, items: before }), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    try {
      await delay(20)
      const scroll = expose.current!.scroll!

      scroll.scrollTo(29)
      await delay(20)
      instance.rerender(React.createElement(Harness, { expose, initialHeights, items: after }))

      expect(scroll.getScrollTop()).toBe(32)
      expect(scroll.isSticky()).toBe(false)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('ignores stale unmount measurement from the previous width layout', async () => {
    const items = Array.from({ length: 20 }, (_, index) => ({
      height: 4,
      heightAfterResize: index === 0 ? 5 : 2,
      key: `item-${index}`
    }))

    const expose = { current: null as Exposed | null }
    const streams = makeStreams()
    const initialHeights = new Map(items.map(item => [item.key, item.height]))

    const instance = renderSync(React.createElement(Harness, { columns: 40, expose, initialHeights, items }), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    try {
      await delay(20)
      const scroll = expose.current!.scroll!

      scroll.scrollTo(0)
      await delay(20)
      scroll.scrollTo(5)
      const adjustScrollTop = vi.spyOn(scroll, 'adjustScrollTop')

      instance.rerender(React.createElement(Harness, { columns: 80, expose, initialHeights, items }))
      await delay(40)

      expect(adjustScrollTop).not.toHaveBeenCalled()
      expect(scroll.getScrollTop()).toBe(5)
      expect(scroll.isSticky()).toBe(false)
      expect(expose.current!.virtualHistory.start).toBeGreaterThan(0)
      expect(expose.current!.virtualHistory.offsets[1]).toBe(2)
      expect(expose.current!.virtualHistory.offsets[items.length]).toBe(40)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('does not let outgoing transcript refs compensate a new layout generation', async () => {
    const outgoing = Array.from({ length: 20 }, (_, index) => ({ height: 2, key: `old-${index}` }))
    const incoming = Array.from({ length: 20 }, (_, index) => ({ height: 2, key: `new-${index}` }))
    const expose = { current: null as Exposed | null }
    const streams = makeStreams()
    const initialHeights = new Map(outgoing.map(item => [item.key, item.height]))

    const instance = renderSync(React.createElement(Harness, { expose, initialHeights, items: outgoing }), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    try {
      await delay(20)
      const scroll = expose.current!.scroll!

      scroll.scrollTo(5)
      await delay(20)
      const adjustScrollTop = vi.spyOn(scroll, 'adjustScrollTop')

      const replacementCache = new Map<string, number>([
        ...outgoing.map(item => [item.key, item.key === 'old-1' ? 1 : item.height] as const),
        ...incoming.map(item => [item.key, item.height] as const)
      ])

      instance.rerender(
        React.createElement(Harness, {
          expose,
          generation: 1,
          initialHeights: replacementCache,
          items: incoming
        })
      )
      await delay(40)

      expect(adjustScrollTop).not.toHaveBeenCalled()
      expect(scroll.getScrollTop()).toBe(5)
      expect(expose.current!.virtualHistory.offsets[incoming.length]).toBe(40)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('corrects and compensates a same-layout row measured at unmount', async () => {
    const items = Array.from({ length: 20 }, (_, index) => ({ height: 2, key: `item-${index}` }))
    const expose = { current: null as Exposed | null }
    const streams = makeStreams()
    const initialHeights = new Map(items.map(item => [item.key, item.height]))

    const instance = renderSync(React.createElement(Harness, { expose, initialHeights, items }), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    try {
      await delay(20)
      const scroll = expose.current!.scroll!

      scroll.scrollTo(0)
      await delay(20)
      scroll.scrollTo(5)
      const adjustScrollTop = vi.spyOn(scroll, 'adjustScrollTop')
      const staleHeights = new Map(initialHeights)

      staleHeights.set(items[0]!.key, 1)
      instance.rerender(React.createElement(Harness, { expose, initialHeights: staleHeights, items }))
      await delay(40)

      expect(adjustScrollTop).toHaveBeenCalledOnce()
      expect(adjustScrollTop).toHaveBeenCalledWith(1)
      expect(scroll.getScrollTop()).toBe(6)
      expect(scroll.isSticky()).toBe(false)
      expect(expose.current!.virtualHistory.start).toBeGreaterThan(0)
      expect(expose.current!.virtualHistory.offsets[1]).toBe(2)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('does not compensate for measured height changes in or below the viewport', async () => {
    const before = Array.from({ length: 20 }, (_, index) => ({ height: 2, key: `item-${index}` }))
    const visibleChanged = before.map((item, index) => (index === 1 ? { ...item, height: 5 } : item))
    const belowChanged = visibleChanged.map((item, index) => (index === 8 ? { ...item, height: 5 } : item))
    const expose = { current: null as Exposed | null }
    const streams = makeStreams()
    const initialHeights = new Map(before.map(item => [item.key, item.height]))

    const instance = renderSync(React.createElement(Harness, { expose, initialHeights, items: before }), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    try {
      await delay(20)
      expose.current!.scroll!.scrollTo(3)
      await delay(20)

      instance.rerender(React.createElement(Harness, { expose, initialHeights, items: visibleChanged }))
      await delay(40)
      expect(expose.current!.scroll!.getScrollTop()).toBe(3)

      instance.rerender(React.createElement(Harness, { expose, initialHeights, items: belowChanged }))
      await delay(40)
      expect(expose.current!.scroll!.getScrollTop()).toBe(3)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('does not compensate measured heights while sticky at the live tail', async () => {
    const before = Array.from({ length: 20 }, (_, index) => ({ height: 2, key: `item-${index}` }))
    const after = before.map((item, index) => (index === 14 ? { ...item, height: 5 } : item))
    const expose = { current: null as Exposed | null }
    const streams = makeStreams()
    const initialHeights = new Map(before.map(item => [item.key, item.height]))

    const instance = renderSync(React.createElement(Harness, { expose, initialHeights, items: before }), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    try {
      await delay(20)
      const adjustScrollTop = vi.spyOn(expose.current!.scroll!, 'adjustScrollTop')

      instance.rerender(React.createElement(Harness, { expose, initialHeights, items: after }))
      await delay(40)

      expect(adjustScrollTop).not.toHaveBeenCalled()
      expect(expose.current!.scroll!.isSticky()).toBe(true)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('ignores stale reused offset-array entries after the item count shrinks', async () => {
    const beforeShrink = Array.from({ length: 1400 }, (_, index) => ({ height: 1, key: `old${index}` }))
    const afterShrink = Array.from({ length: MAX_HISTORY }, (_, index) => ({ height: 7, key: `new${index}` }))
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
