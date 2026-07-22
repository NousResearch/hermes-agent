import { PassThrough } from 'stream'

import { Box, renderSync, ScrollBox, type ScrollBoxHandle, Text } from '@hermes/ink'
import React, { useLayoutEffect, useRef } from 'react'
import { describe, expect, it, vi } from 'vitest'

import SourceBox from '../../packages/hermes-ink/src/ink/components/Box.js'
import SourceScrollBox from '../../packages/hermes-ink/src/ink/components/ScrollBox.js'
import SourceText from '../../packages/hermes-ink/src/ink/components/Text.js'
import type { DOMElement } from '../../packages/hermes-ink/src/ink/dom.js'
import Output from '../../packages/hermes-ink/src/ink/output.js'
import { scrollFastPathStats as sourceScrollFastPathStats } from '../../packages/hermes-ink/src/ink/render-node-to-output.js'
import { renderSync as renderSourceSync } from '../../packages/hermes-ink/src/ink/root.js'
import { useVirtualHistory } from '../hooks/useVirtualHistory.js'

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

function CorruptGeometryHarness({ expose, tick }: { expose: React.MutableRefObject<DOMElement[]>; tick: number }) {
  const nodes = useRef<DOMElement[]>([])

  useLayoutEffect(() => {
    expose.current = nodes.current
  })

  return React.createElement(
    ScrollBox,
    { flexDirection: 'column', height: 8 },
    ...['nan-top', 'positive-infinity', 'negative-infinity', 'billion-rows', 'clipped-huge-fill'].map((label, index) =>
      React.createElement(
        Box,
        {
          backgroundColor: 'blue',
          borderStyle: label === 'clipped-huge-fill' ? 'single' : undefined,
          height: 1,
          key: label,
          opaque: true,
          ref: node => {
            if (node) {
              nodes.current[index] = node
            }
          },
          width: '100%'
        },
        React.createElement(Text, null, `${label}-${tick}`)
      )
    ),
    React.createElement(
      Box,
      {
        height: 1,
        ref: node => {
          if (node) {
            nodes.current[5] = node
          }
        }
      },
      React.createElement(Text, null, React.createElement(Text, null, `nested-corrupt-${tick}`))
    ),
    React.createElement(Text, null, `adjacent-valid-${tick}`),
    React.createElement(Box, { height: 20 }, React.createElement(Text, null, `tail-${tick}`))
  )
}

interface FastPathRepairExpose {
  adjacent: DOMElement | null
  dirtyChild: DOMElement | null
  overlay: DOMElement | null
  scroll: ScrollBoxHandle | null
  scrollBox: DOMElement | null
}

function FastPathRepairHarness({
  expose,
  tick,
  dirtyTick = tick,
  includeOverlay = true
}: {
  dirtyTick?: number
  expose: React.MutableRefObject<FastPathRepairExpose | null>
  includeOverlay?: boolean
  tick: number
}) {
  return React.createElement(
    SourceBox,
    { flexDirection: 'column', height: 12, width: 40 },
    React.createElement(
      SourceScrollBox,
      {
        flexDirection: 'column',
        height: 8,
        ref: scroll => {
          if (expose.current) {
            expose.current.scroll = scroll
          }
        },
        width: 40
      },
      React.createElement(SourceBox, { height: 2 }, React.createElement(SourceText, null, 'head-row')),
      React.createElement(
        SourceBox,
        {
          height: 2,
          ref: dirtyChild => {
            if (expose.current) {
              expose.current.dirtyChild = dirtyChild
              expose.current.scrollBox = dirtyChild?.parentNode?.parentNode ?? null
            }
          }
        },
        React.createElement(SourceText, null, `dirty-row-${dirtyTick}`)
      ),
      React.createElement(SourceBox, { height: 20 }, React.createElement(SourceText, null, 'tail-row'))
    ),
    includeOverlay
      ? React.createElement(
          SourceBox,
          {
            height: 2,
            left: 0,
            position: 'absolute',
            ref: overlay => {
              if (expose.current) {
                expose.current.overlay = overlay
              }
            },
            top: 2,
            width: 40
          },
          React.createElement(SourceText, null, 'overlay-row')
        )
      : null,
    React.createElement(
      SourceBox,
      {
        height: 1,
        ref: adjacent => {
          if (expose.current) {
            expose.current.adjacent = adjacent
          }
        }
      },
      React.createElement(SourceText, null, `adjacent-fast-path-${tick}`)
    )
  )
}

function guardFastPathRepairAllocations(maxWidth: number, maxHeight: number) {
  const originalArrayFill = Array.prototype.fill
  const originalBlit = Output.prototype.blit
  const originalClear = Output.prototype.clear
  const originalRepeat = String.prototype.repeat
  const originalWrite = Output.prototype.write

  const observed = {
    largestArrayRows: 0,
    largestBlitHeight: 0,
    largestBlitWidth: 0,
    largestClearHeight: 0,
    largestClearWidth: 0,
    largestRepeat: 0,
    largestWrite: 0,
    repairWhitespaceWrites: 0
  }

  vi.spyOn(String.prototype, 'repeat').mockImplementation(function (count: number) {
    observed.largestRepeat = Math.max(observed.largestRepeat, count)

    if (!Number.isSafeInteger(count) || count < 0 || count > maxWidth) {
      throw new Error(`unbounded fast-path repeat: ${count}`)
    }

    return originalRepeat.call(this, count)
  })
  vi.spyOn(Array.prototype, 'fill').mockImplementation(function (
    this: unknown[],
    value: unknown,
    start?: number,
    end?: number
  ) {
    observed.largestArrayRows = Math.max(observed.largestArrayRows, this.length)

    if (this.length > maxHeight) {
      throw new Error(`unbounded fast-path row array: ${this.length}`)
    }

    return Reflect.apply(originalArrayFill, this, [value, start, end])
  } as typeof Array.prototype.fill)
  vi.spyOn(Output.prototype, 'blit').mockImplementation(function (...args: Parameters<Output['blit']>) {
    observed.largestBlitWidth = Math.max(observed.largestBlitWidth, args[3])
    observed.largestBlitHeight = Math.max(observed.largestBlitHeight, args[4])

    return originalBlit.apply(this, args)
  })
  vi.spyOn(Output.prototype, 'clear').mockImplementation(function (...args: Parameters<Output['clear']>) {
    observed.largestClearWidth = Math.max(observed.largestClearWidth, args[0].width)
    observed.largestClearHeight = Math.max(observed.largestClearHeight, args[0].height)

    return originalClear.apply(this, args)
  })
  vi.spyOn(Output.prototype, 'write').mockImplementation(function (...args: Parameters<Output['write']>) {
    observed.largestWrite = Math.max(observed.largestWrite, args[2].length)

    if (args[2].length > 0 && /^[ \n]+$/.test(args[2])) {
      observed.repairWhitespaceWrites++
    }

    if (args[2].length > maxWidth * maxHeight + maxHeight) {
      throw new Error(`unbounded fast-path Output.write input: ${args[2].length}`)
    }

    return originalWrite.apply(this, args)
  })

  return observed
}

describe('ScrollBox renderer bounds', () => {
  it('rejects invalid imperative geometry without poisoning scroll state', async () => {
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

      scroll.scrollTo(4)
      scroll.scrollTo(Number.NaN)
      scroll.scrollBy(Number.POSITIVE_INFINITY)
      scroll.adjustScrollTop(Number.NEGATIVE_INFINITY)
      scroll.setClampBounds(Number.NaN, Number.POSITIVE_INFINITY)
      await delay(20)

      expect(scroll.getScrollTop()).toBe(4)
      expect(scroll.getPendingDelta()).toBe(0)
      expect(Number.isFinite(scroll.getScrollHeight())).toBe(true)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('fails closed on corrupt ScrollBox child geometry and keeps adjacent rows renderable', async () => {
    const expose = { current: [] as DOMElement[] }
    const streams = makeStreams()
    const originalRepeat = String.prototype.repeat
    const originalWrite = Output.prototype.write
    let largestWriteInput = 0
    let largestWrite = 0
    let output = ''

    vi.spyOn(String.prototype, 'repeat').mockImplementation(function (count: number) {
      if (!Number.isSafeInteger(count) || count < 0 || count > 10_000) {
        throw new Error(`unbounded string repeat: ${count}`)
      }

      return originalRepeat.call(this, count)
    })
    vi.spyOn(Output.prototype, 'write').mockImplementation(function (...args: Parameters<Output['write']>) {
      largestWriteInput = Math.max(largestWriteInput, args[2].length)

      if (args[2].length > 10_000) {
        throw new Error(`unbounded Output.write input: ${args[2].length}`)
      }

      return originalWrite.apply(this, args)
    })

    streams.stdout.removeAllListeners('data')
    streams.stdout.on('data', chunk => {
      largestWrite = Math.max(largestWrite, chunk.length)
      output += chunk.toString()
    })

    const instance = renderSync(React.createElement(CorruptGeometryHarness, { expose, tick: 0 }), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    try {
      await delay(20)

      const [nanTop, positiveInfinity, negativeInfinity, billionRows, clippedHugeFill, nestedTextWrapper] =
        expose.current

      expect(nanTop?.yogaNode).toBeDefined()
      expect(positiveInfinity?.yogaNode).toBeDefined()
      expect(negativeInfinity?.yogaNode).toBeDefined()
      expect(billionRows?.yogaNode).toBeDefined()
      expect(clippedHugeFill?.yogaNode).toBeDefined()
      expect(nestedTextWrapper?.yogaNode).toBeDefined()

      const nestedText = nestedTextWrapper!.childNodes.find(child => child.nodeName === 'ink-text') as
        | DOMElement
        | undefined

      const nestedTextChild = nestedText?.childNodes[0]

      expect(nestedTextChild).toBeDefined()

      vi.spyOn(nanTop!.yogaNode!, 'getComputedTop').mockReturnValue(Number.NaN)
      vi.spyOn(positiveInfinity!.yogaNode!, 'getComputedHeight').mockReturnValue(Number.POSITIVE_INFINITY)
      vi.spyOn(negativeInfinity!.yogaNode!, 'getComputedHeight').mockReturnValue(Number.NEGATIVE_INFINITY)
      vi.spyOn(billionRows!.yogaNode!, 'getComputedHeight').mockReturnValue(1_000_000_000)
      vi.spyOn(clippedHugeFill!.yogaNode!, 'getComputedHeight').mockReturnValue(100_000_000)
      vi.spyOn(clippedHugeFill!.yogaNode!, 'getComputedWidth').mockReturnValue(100_000_000)

      let nestedOffsetX = 0
      let nestedOffsetY = 0

      nestedTextChild!.yogaNode = {
        getComputedLeft: () => nestedOffsetX,
        getComputedTop: () => nestedOffsetY
      } as DOMElement['yogaNode']

      output = ''
      largestWrite = 0
      largestWriteInput = 0

      const corruptNestedOffsets = [
        [Number.NaN, 0],
        [Number.POSITIVE_INFINITY, 0],
        [Number.NEGATIVE_INFINITY, 0],
        [-1, 0],
        [0.5, 0],
        [100_000_000, 0],
        [0, Number.NaN],
        [0, Number.POSITIVE_INFINITY],
        [0, Number.NEGATIVE_INFINITY],
        [0, -1],
        [0, 0.5],
        [0, 100_000_000]
      ] as const

      for (const [index, [offsetX, offsetY]] of corruptNestedOffsets.entries()) {
        nestedOffsetX = offsetX
        nestedOffsetY = offsetY

        expect(() => {
          instance.rerender(React.createElement(CorruptGeometryHarness, { expose, tick: index + 1 }))
        }).not.toThrow()
        await delay(5)
      }

      await delay(40)

      expect(output).toContain(`adjacent-valid-${corruptNestedOffsets.length}`)
      expect(largestWriteInput).toBeLessThan(10_000)
      expect(largestWrite).toBeLessThan(10_000)
      expect(output.length).toBeLessThan(50_000)
    } finally {
      vi.restoreAllMocks()
      instance.unmount()
      instance.cleanup()
    }
  })

  it('clips corrupt dirty-child DECSTBM repairs before allocating or recursing', async () => {
    const expose = {
      current: {
        adjacent: null,
        dirtyChild: null,
        overlay: null,
        scroll: null,
        scrollBox: null
      } as FastPathRepairExpose
    }

    const streams = makeStreams()
    let output = ''

    const instance = renderSourceSync(
      React.createElement(FastPathRepairHarness, { expose, includeOverlay: false, tick: 0 }),
      {
        patchConsole: false,
        stderr: streams.stderr as NodeJS.WriteStream,
        stdin: streams.stdin as NodeJS.ReadStream,
        stdout: streams.stdout as NodeJS.WriteStream
      }
    )

    try {
      await delay(20)
      const dirtyChild = expose.current!.dirtyChild!

      expect(dirtyChild, streams.stderr.read()?.toString()).not.toBeNull()
      expect(dirtyChild.yogaNode).toBeDefined()
      vi.spyOn(dirtyChild.yogaNode!, 'getComputedTop').mockReturnValue(-99_999_995)
      vi.spyOn(dirtyChild.yogaNode!, 'getComputedWidth').mockReturnValue(100_000_000)
      vi.spyOn(dirtyChild.yogaNode!, 'getComputedHeight').mockReturnValue(100_000_000)

      instance.rerender(React.createElement(FastPathRepairHarness, { expose, includeOverlay: false, tick: 1 }))
      await delay(20)

      streams.stdout.removeAllListeners('data')
      streams.stdout.on('data', chunk => {
        output += chunk.toString()
      })

      const observed = guardFastPathRepairAllocations(80, 20)
      const fastPathsBefore = sourceScrollFastPathStats.taken
      const capturedBefore = sourceScrollFastPathStats.captured

      expect(() => expose.current!.scroll!.scrollTo(1)).not.toThrow()
      expect(() =>
        instance.rerender(React.createElement(FastPathRepairHarness, { expose, includeOverlay: false, tick: 2 }))
      ).not.toThrow()
      await delay(40)

      expect(sourceScrollFastPathStats.captured, JSON.stringify(sourceScrollFastPathStats)).toBeGreaterThan(
        capturedBefore
      )
      expect(sourceScrollFastPathStats.taken, JSON.stringify(sourceScrollFastPathStats)).toBeGreaterThan(
        fastPathsBefore
      )
      expect(observed.repairWhitespaceWrites).toBeGreaterThan(0)
      expect(observed.largestRepeat).toBeGreaterThan(0)
      expect(observed.largestArrayRows).toBeGreaterThan(0)

      output = ''
      expect(() =>
        instance.rerender(
          React.createElement(FastPathRepairHarness, {
            dirtyTick: 2,
            expose,
            includeOverlay: false,
            tick: 3
          })
        )
      ).not.toThrow()
      await delay(40)

      expect(observed.largestRepeat).toBeLessThanOrEqual(80)
      expect(observed.largestArrayRows).toBeLessThanOrEqual(20)
      expect(observed.largestBlitWidth).toBeLessThanOrEqual(80)
      expect(observed.largestBlitHeight).toBeLessThanOrEqual(20)
      expect(observed.largestClearWidth).toBeLessThanOrEqual(80)
      expect(observed.largestClearHeight).toBeLessThanOrEqual(20)
      expect(observed.largestWrite).toBeLessThanOrEqual(1_620)
      expect(expose.current!.scroll!.getScrollTop()).toBe(1)
      expect(expose.current!.scroll!.getViewportHeight()).toBeGreaterThan(0)
      expect(output).toContain('adjacent-fast-path-3')
      expect(streams.stderr.read()?.toString() ?? '').toBe('')
    } finally {
      vi.restoreAllMocks()
      instance.unmount()
      instance.cleanup()
    }
  })

  it('clips corrupt absolute-overlay DECSTBM repairs before allocating or recursing', async () => {
    const expose = {
      current: {
        adjacent: null,
        dirtyChild: null,
        overlay: null,
        scroll: null,
        scrollBox: null
      } as FastPathRepairExpose
    }

    const streams = makeStreams()
    let output = ''

    const instance = renderSourceSync(React.createElement(FastPathRepairHarness, { expose, tick: 0 }), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    try {
      await delay(20)
      const overlay = expose.current!.overlay!
      const scrollBox = expose.current!.scrollBox!

      expect(overlay, streams.stderr.read()?.toString()).not.toBeNull()
      expect(overlay.yogaNode).toBeDefined()
      expect(scrollBox.yogaNode).toBeDefined()
      vi.spyOn(scrollBox.yogaNode!, 'getComputedWidth').mockReturnValue(100_000_000)
      vi.spyOn(overlay.yogaNode!, 'getComputedWidth').mockReturnValue(100_000_000)
      vi.spyOn(overlay.yogaNode!, 'getComputedHeight').mockReturnValue(100_000_000)

      instance.rerender(React.createElement(FastPathRepairHarness, { expose, tick: 1 }))
      await delay(20)
      instance.rerender(React.createElement(FastPathRepairHarness, { expose, tick: 1 }))
      await delay(20)

      streams.stdout.removeAllListeners('data')
      streams.stdout.on('data', chunk => {
        output += chunk.toString()
      })

      const observed = guardFastPathRepairAllocations(80, 20)
      const fastPathsBefore = sourceScrollFastPathStats.taken
      const capturedBefore = sourceScrollFastPathStats.captured

      expect(() => expose.current!.scroll!.scrollTo(1)).not.toThrow()
      expect(expose.current!.scroll!.getScrollTop()).toBe(1)
      await delay(40)

      expect(sourceScrollFastPathStats.captured, JSON.stringify(sourceScrollFastPathStats)).toBeGreaterThan(
        capturedBefore
      )
      expect(sourceScrollFastPathStats.taken, JSON.stringify(sourceScrollFastPathStats)).toBeGreaterThan(
        fastPathsBefore
      )
      expect(observed.repairWhitespaceWrites).toBeGreaterThan(0)
      expect(observed.largestRepeat).toBeGreaterThan(0)
      expect(observed.largestArrayRows).toBeGreaterThan(0)

      output = ''
      expect(() =>
        instance.rerender(React.createElement(FastPathRepairHarness, { dirtyTick: 1, expose, tick: 2 }))
      ).not.toThrow()
      await delay(40)

      expect(observed.largestRepeat).toBeLessThanOrEqual(80)
      expect(observed.largestArrayRows).toBeLessThanOrEqual(20)
      expect(observed.largestBlitWidth).toBeLessThanOrEqual(80)
      expect(observed.largestBlitHeight).toBeLessThanOrEqual(20)
      expect(observed.largestClearWidth).toBeLessThanOrEqual(80)
      expect(observed.largestClearHeight).toBeLessThanOrEqual(20)
      expect(observed.largestWrite).toBeLessThanOrEqual(1_620)
      expect(output.length).toBeLessThan(2_000)
      expect(expose.current!.scroll!.getViewportHeight()).toBeGreaterThan(0)
      expect(output).toContain('adjacent-fast-path-2')
      expect(streams.stderr.read()?.toString() ?? '').toBe('')
    } finally {
      vi.restoreAllMocks()
      instance.unmount()
      instance.cleanup()
    }
  })
})
