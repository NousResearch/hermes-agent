// Deterministic virtual-history benchmark. The file intentionally uses only
// APIs present before the performance candidate so the exact same script can
// be copied/run on base and candidate checkouts.
//
// Run from ui-tui:
//   npx tsx scripts/bench-history-scroll.tsx
//   npx tsx scripts/bench-history-scroll.tsx --warmups=2 --samples=5 --items=100,1000,10000
//
// In addition to the virtual-history workloads, every run mounts one
// oversized bordered/fill box at each RENDERER_EXTENT inside the fixed
// viewport. Keeping that tree to a few Yoga nodes isolates renderer clipping
// from node-construction cost and makes the workload revision-comparable.

import { PassThrough } from 'stream'

import { Box, renderSync, ScrollBox, type ScrollBoxHandle, Text } from '@hermes/ink'
import React, { useLayoutEffect, useRef } from 'react'

import { useVirtualHistory } from '../src/hooks/useVirtualHistory.js'

const DEFAULT_WORKLOADS = [100, 1_000, 10_000]
const RENDERER_EXTENTS = [100, 1_000, 10_000]
const DEFAULT_WARMUPS = 1
const DEFAULT_SAMPLES = 5
const COLUMNS = 100
const ROWS = 30
const MAX_MOUNTED = 120

interface BenchItem {
  height: number
  key: string
  text: string
}

interface Exposed {
  scroll: ScrollBoxHandle | null
  virtual: ReturnType<typeof useVirtualHistory>
}

interface Sample {
  anchorError: number
  heapDeltaBytes: number | null
  invalidOffsets: number
  measuredHeightReconciliationMs: number
  mountMs: number
  mountedRowsMax: number
  nonMonotoneOffsets: number
  rerenderMs: number
  scrollMs: number
  terminalBytes: number
  terminalWrites: number
}

interface WorkloadResult {
  distributions: {
    anchorError: ReturnType<typeof distribution>
    heapDeltaBytes: ReturnType<typeof distribution>
    measuredHeightReconciliationMs: ReturnType<typeof distribution>
    mountMs: ReturnType<typeof distribution>
    mountedRowsMax: ReturnType<typeof distribution>
    rerenderMs: ReturnType<typeof distribution>
    scrollMs: ReturnType<typeof distribution>
    terminalBytes: ReturnType<typeof distribution>
    terminalWrites: ReturnType<typeof distribution>
  }
  invalidOffsets: number
  itemCount: number
  nonMonotoneOffsets: number
  samples: Sample[]
}

interface OversizedRendererSample {
  freshMountRenderMs: number
  terminalBytes: number
  terminalWrites: number
}

interface OversizedRendererResult {
  distributions: {
    freshMountRenderMs: ReturnType<typeof distribution>
    terminalBytes: ReturnType<typeof distribution>
    terminalWrites: ReturnType<typeof distribution>
  }
  extent: number
  samples: OversizedRendererSample[]
}

class CountingStream extends PassThrough {
  columns = COLUMNS
  rows = ROWS
  isTTY = false
  bytes = 0
  writes = 0

  override _write(chunk: Buffer | string, encoding: BufferEncoding, callback: (error?: Error | null) => void) {
    this.bytes += Buffer.byteLength(chunk)
    this.writes++
    callback()
  }
}

const immediate = () => new Promise<void>(resolve => setImmediate(resolve))

async function settle(frames = 4) {
  for (let frame = 0; frame < frames; frame++) {
    await immediate()
  }
}

async function waitUntil(predicate: () => boolean, attempts = 40) {
  for (let attempt = 0; attempt < attempts; attempt++) {
    if (predicate()) {
      return true
    }

    await immediate()
  }

  return predicate()
}

function makeItems(count: number): BenchItem[] {
  return Array.from({ length: count }, (_, index) => ({
    height: 1 + ((index * 17) % 4),
    key: `row-${index}`,
    text: `row ${index} ${'history '.repeat(2 + (index % 5))}`
  }))
}

function Harness({ expose, items }: { expose: React.MutableRefObject<Exposed | null>; items: readonly BenchItem[] }) {
  const scrollRef = useRef<ScrollBoxHandle | null>(null)

  const virtual = useVirtualHistory(scrollRef, items, COLUMNS, {
    coldStartCount: 30,
    estimateHeight: index => items[index]?.height ?? 1,
    maxMounted: MAX_MOUNTED,
    overscan: 20
  })

  useLayoutEffect(() => {
    expose.current = { scroll: scrollRef.current, virtual }
  })

  return (
    <ScrollBox flexDirection="column" height={ROWS} ref={scrollRef} stickyScroll>
      <Box flexDirection="column" width="100%">
        {virtual.topSpacer > 0 ? <Box height={virtual.topSpacer} /> : null}
        {items.slice(virtual.start, virtual.end).map(item => (
          <Box height={item.height} key={item.key} ref={virtual.measureRef(item.key)}>
            <Text>{item.text}</Text>
          </Box>
        ))}
        {virtual.bottomSpacer > 0 ? <Box height={virtual.bottomSpacer} /> : null}
      </Box>
    </ScrollBox>
  )
}

function OversizedRendererHarness({ extent }: { extent: number }) {
  return (
    <ScrollBox flexDirection="column" height={ROWS} width={COLUMNS}>
      <Box backgroundColor="ansi:blue" borderStyle="single" flexShrink={0} height={extent} opaque width={COLUMNS}>
        <Text>deterministic oversized height workload</Text>
      </Box>
      <Box backgroundColor="ansi:magenta" borderStyle="single" height={ROWS} opaque position="absolute" width={extent}>
        <Text>deterministic oversized width workload</Text>
      </Box>
    </ScrollBox>
  )
}

function inspectOffsets(offsets: ArrayLike<number>, count: number) {
  let invalidOffsets = 0
  let nonMonotoneOffsets = 0

  for (let index = 0; index <= count; index++) {
    const value = offsets[index]

    if (!Number.isFinite(value)) {
      invalidOffsets++
    }

    if (index > 0 && value! < offsets[index - 1]!) {
      nonMonotoneOffsets++
    }
  }

  return { invalidOffsets, nonMonotoneOffsets }
}

async function runSample(itemCount: number): Promise<Sample> {
  const stdout = new CountingStream()
  const stderr = new CountingStream()
  const stdin = new PassThrough()
  const expose = { current: null as Exposed | null }
  let items = makeItems(itemCount)
  const heapBefore = process.memoryUsage?.().heapUsed ?? null
  const mountStart = performance.now()

  const instance = renderSync(<Harness expose={expose} items={items} />, {
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  await waitUntil(() => expose.current?.scroll !== null)
  await settle()
  const mountMs = performance.now() - mountStart
  let mountedRowsMax = expose.current!.virtual.end - expose.current!.virtual.start

  const rerenderItems = items.map((item, index) =>
    index === items.length - 1 ? { ...item, text: `${item.text} rerender` } : item
  )

  const rerenderStart = performance.now()

  instance.rerender(<Harness expose={expose} items={rerenderItems} />)
  await settle()
  const rerenderMs = performance.now() - rerenderStart
  items = rerenderItems
  mountedRowsMax = Math.max(mountedRowsMax, expose.current!.virtual.end - expose.current!.virtual.start)

  const scroll = expose.current!.scroll!
  const total = expose.current!.virtual.offsets[itemCount] ?? 0
  const scrollStart = performance.now()

  scroll.scrollTo(Math.max(0, Math.floor(total * 0.55)))
  await settle(8)
  const scrollMs = performance.now() - scrollStart
  mountedRowsMax = Math.max(mountedRowsMax, expose.current!.virtual.end - expose.current!.virtual.start)

  const beforeOffsets = expose.current!.virtual.offsets
  const beforeTop = scroll.getScrollTop()
  let measuredIndex = expose.current!.virtual.start

  while (measuredIndex + 1 < expose.current!.virtual.end && (beforeOffsets[measuredIndex + 1] ?? 0) > beforeTop) {
    measuredIndex++
  }

  if ((beforeOffsets[measuredIndex + 1] ?? Number.POSITIVE_INFINITY) > beforeTop) {
    measuredIndex = Math.max(expose.current!.virtual.start, measuredIndex - 1)
  }

  const heightDelta = 3
  const oldTotal = beforeOffsets[itemCount] ?? 0

  const measuredItems = items.map((item, index) =>
    index === measuredIndex ? { ...item, height: item.height + heightDelta } : item
  )

  const reconcileStart = performance.now()

  instance.rerender(<Harness expose={expose} items={measuredItems} />)
  await waitUntil(() => (expose.current!.virtual.offsets[itemCount] ?? 0) === oldTotal + heightDelta)
  await settle(2)
  const measuredHeightReconciliationMs = performance.now() - reconcileStart
  const measuredWasAbove = (beforeOffsets[measuredIndex + 1] ?? 0) <= beforeTop
  const expectedTop = beforeTop + (measuredWasAbove ? heightDelta : 0)
  const anchorError = Math.abs(scroll.getScrollTop() - expectedTop)
  mountedRowsMax = Math.max(mountedRowsMax, expose.current!.virtual.end - expose.current!.virtual.start)

  const offsetHealth = inspectOffsets(expose.current!.virtual.offsets, itemCount)
  const heapAfter = process.memoryUsage?.().heapUsed ?? null
  const heapDeltaBytes = heapBefore === null || heapAfter === null ? null : heapAfter - heapBefore
  const terminalBytes = stdout.bytes
  const terminalWrites = stdout.writes

  instance.unmount()
  instance.cleanup()
  stdin.destroy()
  stdout.destroy()
  stderr.destroy()

  return {
    anchorError,
    heapDeltaBytes,
    ...offsetHealth,
    measuredHeightReconciliationMs,
    mountMs,
    mountedRowsMax,
    rerenderMs,
    scrollMs,
    terminalBytes,
    terminalWrites
  }
}

async function runOversizedRendererSample(extent: number): Promise<OversizedRendererSample> {
  const stdout = new CountingStream()
  const stderr = new CountingStream()
  const stdin = new PassThrough()
  const mountStart = performance.now()
  let instance: ReturnType<typeof renderSync> | undefined

  try {
    instance = renderSync(<OversizedRendererHarness extent={extent} />, {
      patchConsole: false,
      stderr: stderr as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    })

    const rendered = await waitUntil(() => stdout.writes > 0)

    if (!rendered) {
      throw new Error(`oversized renderer extent ${extent} did not produce a terminal frame`)
    }

    return {
      freshMountRenderMs: performance.now() - mountStart,
      terminalBytes: stdout.bytes,
      terminalWrites: stdout.writes
    }
  } finally {
    instance?.unmount()
    instance?.cleanup()

    stdin.destroy()
    stdout.destroy()
    stderr.destroy()
  }
}

function distribution(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b)

  const percentile = (p: number) =>
    sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * p) - 1))] ?? 0

  return {
    max: sorted.at(-1) ?? 0,
    mean: sorted.reduce((sum, value) => sum + value, 0) / Math.max(1, sorted.length),
    min: sorted[0] ?? 0,
    p50: percentile(0.5),
    p95: percentile(0.95),
    p99: percentile(0.99)
  }
}

function numericArg(name: string, fallback: number) {
  const raw = process.argv
    .slice(2)
    .find(arg => arg.startsWith(`--${name}=`))
    ?.split('=', 2)[1]

  const parsed = Number(raw)

  return Number.isSafeInteger(parsed) && parsed >= 0 ? parsed : fallback
}

function workloadsArg() {
  const raw = process.argv
    .slice(2)
    .find(arg => arg.startsWith('--items='))
    ?.split('=', 2)[1]

  if (!raw) {
    return DEFAULT_WORKLOADS
  }

  const parsed = raw.split(',').map(Number)

  if (parsed.some(value => !Number.isSafeInteger(value) || value <= 0)) {
    throw new Error(`invalid --items workload list: ${raw}`)
  }

  return parsed
}

async function main() {
  const workloads = workloadsArg()
  const warmups = numericArg('warmups', DEFAULT_WARMUPS)
  const samplesPerWorkload = numericArg('samples', DEFAULT_SAMPLES)
  const results: WorkloadResult[] = []
  const oversizedRendererResults: OversizedRendererResult[] = []

  for (const itemCount of workloads) {
    for (let warmup = 0; warmup < warmups; warmup++) {
      await runSample(itemCount)
    }

    const samples: Sample[] = []

    for (let sample = 0; sample < samplesPerWorkload; sample++) {
      samples.push(await runSample(itemCount))
    }

    results.push({
      itemCount,
      distributions: {
        anchorError: distribution(samples.map(sample => sample.anchorError)),
        heapDeltaBytes: distribution(samples.flatMap(sample => sample.heapDeltaBytes ?? [])),
        measuredHeightReconciliationMs: distribution(samples.map(sample => sample.measuredHeightReconciliationMs)),
        mountMs: distribution(samples.map(sample => sample.mountMs)),
        mountedRowsMax: distribution(samples.map(sample => sample.mountedRowsMax)),
        rerenderMs: distribution(samples.map(sample => sample.rerenderMs)),
        scrollMs: distribution(samples.map(sample => sample.scrollMs)),
        terminalBytes: distribution(samples.map(sample => sample.terminalBytes)),
        terminalWrites: distribution(samples.map(sample => sample.terminalWrites))
      },
      invalidOffsets: samples.reduce((sum, sample) => sum + sample.invalidOffsets, 0),
      nonMonotoneOffsets: samples.reduce((sum, sample) => sum + sample.nonMonotoneOffsets, 0),
      samples
    })
  }

  for (const extent of RENDERER_EXTENTS) {
    for (let warmup = 0; warmup < warmups; warmup++) {
      await runOversizedRendererSample(extent)
    }

    const samples: OversizedRendererSample[] = []

    for (let sample = 0; sample < samplesPerWorkload; sample++) {
      samples.push(await runOversizedRendererSample(extent))
    }

    oversizedRendererResults.push({
      distributions: {
        freshMountRenderMs: distribution(samples.map(sample => sample.freshMountRenderMs)),
        terminalBytes: distribution(samples.map(sample => sample.terminalBytes)),
        terminalWrites: distribution(samples.map(sample => sample.terminalWrites))
      },
      extent,
      samples
    })
  }

  const scaling = results.slice(1).map((result, index) => {
    const previous = results[index]!

    const ratio = (metric: keyof (typeof result)['distributions']) =>
      result.distributions[metric].p50 / Math.max(Number.EPSILON, previous.distributions[metric].p50)

    return {
      fromItems: previous.itemCount,
      itemFactor: result.itemCount / previous.itemCount,
      measuredHeightReconciliationP50Factor: ratio('measuredHeightReconciliationMs'),
      mountP50Factor: ratio('mountMs'),
      rerenderP50Factor: ratio('rerenderMs'),
      scrollP50Factor: ratio('scrollMs'),
      terminalBytesP50Factor: ratio('terminalBytes'),
      toItems: result.itemCount
    }
  })

  const oversizedRendererScaling = oversizedRendererResults.slice(1).map((result, index) => {
    const previous = oversizedRendererResults[index]!

    const ratio = (metric: keyof (typeof result)['distributions']) =>
      result.distributions[metric].p50 / Math.max(Number.EPSILON, previous.distributions[metric].p50)

    return {
      extentFactor: result.extent / previous.extent,
      freshMountRenderP50Factor: ratio('freshMountRenderMs'),
      fromExtent: previous.extent,
      terminalBytesP50Factor: ratio('terminalBytes'),
      terminalWritesP50Factor: ratio('terminalWrites'),
      toExtent: result.extent
    }
  })

  process.stdout.write(
    `${JSON.stringify(
      {
        config: { columns: COLUMNS, maxMounted: MAX_MOUNTED, rows: ROWS, samples: samplesPerWorkload, warmups },
        oversizedRenderer: {
          extents: RENDERER_EXTENTS,
          results: oversizedRendererResults,
          scaling: oversizedRendererScaling
        },
        results,
        scaling,
        workloads
      },
      null,
      2
    )}\n`
  )
}

await main()
