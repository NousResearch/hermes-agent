import { PassThrough } from 'stream'

import { Box, renderSync, ScrollBox, type ScrollBoxHandle, Text } from '@hermes/ink'
import React, { useLayoutEffect, useRef } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { ensureVirtualItemHeight, useVirtualHistory } from '../hooks/useVirtualHistory.js'

describe('ensureVirtualItemHeight', () => {
  it('reuses cached heights without invoking the estimator', () => {
    const heights = new Map([['a', 7]])
    const estimateHeight = vi.fn(() => 99)

    expect(ensureVirtualItemHeight(heights, 'a', 0, 4, estimateHeight)).toBe(7)
    expect(estimateHeight).not.toHaveBeenCalled()
    expect(heights.get('a')).toBe(7)
  })

  it('lazily seeds missing heights from the estimator', () => {
    const heights = new Map<string, number>()
    const estimateHeight = vi.fn((index: number) => 10 + index)

    expect(ensureVirtualItemHeight(heights, 'b', 2, 4, estimateHeight)).toBe(12)
    expect(estimateHeight).toHaveBeenCalledTimes(1)
    expect(estimateHeight).toHaveBeenCalledWith(2, 'b')
    expect(heights.get('b')).toBe(12)
  })

  it('falls back to the default estimate when no estimator is provided', () => {
    const heights = new Map<string, number>()

    expect(ensureVirtualItemHeight(heights, 'c', 0, 4)).toBe(4)
    expect(heights.get('c')).toBe(4)
  })

  it('normalizes non-positive estimates to a minimum of one row', () => {
    const heights = new Map<string, number>()
    const estimateHeight = vi.fn(() => 0)

    expect(ensureVirtualItemHeight(heights, 'd', 0, 0, estimateHeight)).toBe(1)
    expect(heights.get('d')).toBe(1)
  })

  it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, -4, 1_000_000_000])(
    'quarantines invalid cached height %s and reseeds it',
    cached => {
      const heights = new Map([['bad', cached]])

      expect(ensureVirtualItemHeight(heights, 'bad', 0, 4, () => 7)).toBe(7)
      expect(heights.get('bad')).toBe(7)
    }
  )

  it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, -4, 1_000_000_000])(
    'falls back when the estimator returns invalid height %s',
    estimate => {
      const heights = new Map<string, number>()

      expect(ensureVirtualItemHeight(heights, 'bad', 0, 4, () => estimate)).toBe(4)
      expect(heights.get('bad')).toBe(4)
    }
  )
})

// Issue #55594: long assistant responses scroll out of the mounted range
// and the clamp holds the viewport at the edge of mounted content while
// the user catches up.  The default `maxMounted` cap (raised 120 → 300)
// exists so longer responses stay reachable.  Rather than freezing the
// constant, assert the behavioural contract under the hook's DEFAULT
// options across a range of transcript sizes: the tail of a long response
// must remain mounted (reachable) and the viewport must stay covered so
// nothing "disappears" while the user scrolls up.
interface Item {
  height: number
  key: string
}

interface Exposed {
  scroll: ScrollBoxHandle | null
  virtualHistory: ReturnType<typeof useVirtualHistory>
}

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

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

function DefaultOptionsHarness({
  expose,
  items
}: {
  expose: React.MutableRefObject<Exposed | null>
  items: readonly Item[]
}) {
  const scrollRef = useRef<ScrollBoxHandle | null>(null)

  // No options: exercise the hook's DEFAULT estimate, overscan and
  // maxMounted behaviour (the maxMounted default is what #55594 raised).
  const virtualHistory = useVirtualHistory(scrollRef, items, 80)

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
      ...items.slice(virtualHistory.start, virtualHistory.end).map(item =>
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

// Height of the rows actually mounted between topSpacer and bottomSpacer.
const mountedContentBottom = (
  items: readonly Item[],
  virtualHistory: ReturnType<typeof useVirtualHistory>
) => {
  let height = 0

  for (let index = virtualHistory.start; index < virtualHistory.end; index++) {
    height += items[index]?.height ?? 0
  }

  return virtualHistory.topSpacer + height
}

describe('useVirtualHistory default options', () => {
  it('keeps long responses reachable when scrolling up (default maxMounted)', async () => {
    // Issue #55594: a long assistant response scrolls out of the mounted
    // range and the clamp pins the viewport at the edge of mounted content,
    // so the response appears to "disappear" with no way back. The default
    // maxMounted cap exists so the mounted range still spans the response a
    // user is catching up through. Sweep a range of response sizes so the
    // test asserts behaviour under varying input, not a frozen snapshot.
    for (const rows of [200, 320, 500, 1000]) {
      const longResponse = Array.from({ length: rows }, (_, index) => ({ height: 2, key: `resp-${index}` }))
      const tail = { height: 4, key: 'tail' }
      const items = [...longResponse, tail]
      const expose = { current: null as Exposed | null }
      const streams = makeStreams()

      const instance = renderSync(React.createElement(DefaultOptionsHarness, { expose, items }), {
        patchConsole: false,
        stderr: streams.stderr as NodeJS.WriteStream,
        stdin: streams.stdin as NodeJS.ReadStream,
        stdout: streams.stdout as NodeJS.WriteStream
      })

      try {
        await delay(20)
        const scroll = expose.current!.scroll!

        // Start pinned to the bottom (sticky) and scroll up into the long
        // response — the scenario from #55594.
        scroll.scrollBy(-(rows * 2) + 4)
        await delay(80)

        const scrollTop = scroll.getScrollTop()
        const viewportBottom = scrollTop + scroll.getViewportHeight()
        const mountedBottom = mountedContentBottom(items, expose.current!.virtualHistory)

        // The viewport must land on real mounted response rows, not blank
        // spacer — otherwise the response has "disappeared" behind an
        // unmountable gap. Allow a little slack for the clamp edge.
        expect(viewportBottom).toBeLessThanOrEqual(mountedBottom + 2)
        expect(scrollTop).toBeLessThan(mountedBottom)
      } finally {
        instance.unmount()
        instance.cleanup()
      }
    }
  })
})
