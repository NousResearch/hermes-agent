// Compare the native primary-buffer transcript with the virtualized ScrollBox path.
//
// Run from ui-tui:
//   npx tsx scripts/bench-native-history.tsx --warmups=1 --samples=3 --items=100,1000,10000
//
// Native scrolling is handled by the terminal after rows are written, so this
// compares the work before scrolling begins: mount, rerender, memory, and output.

import { PassThrough } from 'stream'

import { Box, renderSync, ScrollBox, type ScrollBoxHandle, Text } from '@hermes/ink'
import React from 'react'

import { useVirtualHistory } from '../src/hooks/useVirtualHistory.js'

const DEFAULT_WORKLOADS = [100, 1_000, 10_000]
const DEFAULT_WARMUPS = 1
const DEFAULT_SAMPLES = 5
const COLUMNS = 100
const ROWS = 30

interface Item {
  height: number
  key: string
  text: string
}

interface Sample {
  heapDeltaBytes: number | null
  mountMs: number
  rerenderMs: number
  terminalBytes: number
  terminalWrites: number
}

type ItemsProps = { items: readonly Item[] }
type VirtualProps = ItemsProps & { scrollRef: React.MutableRefObject<ScrollBoxHandle | null> }

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

function makeItems(count: number): Item[] {
  return Array.from({ length: count }, (_, index) => ({
    height: 1 + ((index * 17) % 4),
    key: `row-${index}`,
    text: `row ${index} ${'history '.repeat(2 + (index % 5))}`
  }))
}

function NativeHarness({ items }: ItemsProps) {
  return (
    <Box flexDirection="column" width="100%">
      {items.map(item => (
        <Box flexDirection="column" key={item.key} minHeight={item.height}>
          <Text>{item.text}</Text>
        </Box>
      ))}
    </Box>
  )
}

function VirtualHarness({ items, scrollRef }: VirtualProps) {
  const virtual = useVirtualHistory(scrollRef!, items, COLUMNS, {
    coldStartCount: 30,
    estimateHeight: index => items[index]?.height ?? 1,
    maxMounted: 120,
    overscan: 20
  })

  return (
    <ScrollBox flexDirection="column" height={ROWS} ref={scrollRef} stickyScroll>
      <Box flexDirection="column" width="100%">
        {virtual.topSpacer > 0 ? <Box height={virtual.topSpacer} /> : null}
        {items.slice(virtual.start, virtual.end).map(item => (
          <Box flexDirection="column" key={item.key} minHeight={item.height} ref={virtual.measureRef(item.key)}>
            <Text>{item.text}</Text>
          </Box>
        ))}
        {virtual.bottomSpacer > 0 ? <Box height={virtual.bottomSpacer} /> : null}
      </Box>
    </ScrollBox>
  )
}

async function runSample(mode: 'native' | 'virtual', itemCount: number): Promise<Sample> {
  const stdout = new CountingStream()
  const stderr = new CountingStream()
  const stdin = new PassThrough()
  const scrollRef = { current: null as ScrollBoxHandle | null }

  const items = makeItems(itemCount)

  const renderHarness = (nextItems: readonly Item[]) =>
    mode === 'native' ? (
      <NativeHarness items={nextItems} />
    ) : (
      <VirtualHarness items={nextItems} scrollRef={scrollRef} />
    )

  const heapBefore = process.memoryUsage?.().heapUsed ?? null
  const mountStart = performance.now()

  const instance = renderSync(renderHarness(items), {
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  await settle()
  const mountMs = performance.now() - mountStart

  const rerenderItems = items.map((item, index) =>
    index === items.length - 1 ? { ...item, text: `${item.text} rerender` } : item
  )

  const rerenderStart = performance.now()
  instance.rerender(renderHarness(rerenderItems))
  await settle()
  const rerenderMs = performance.now() - rerenderStart
  const heapAfter = process.memoryUsage?.().heapUsed ?? null

  const sample = {
    heapDeltaBytes: heapBefore === null || heapAfter === null ? null : heapAfter - heapBefore,
    mountMs,
    rerenderMs,
    terminalBytes: stdout.bytes,
    terminalWrites: stdout.writes
  }

  instance.unmount()
  instance.cleanup()
  stdin.destroy()
  stdout.destroy()
  stderr.destroy()

  return sample
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
    p95: percentile(0.95)
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

function summarize(samples: Sample[]) {
  return {
    heapDeltaBytes: distribution(samples.flatMap(sample => sample.heapDeltaBytes ?? [])),
    mountMs: distribution(samples.map(sample => sample.mountMs)),
    rerenderMs: distribution(samples.map(sample => sample.rerenderMs)),
    terminalBytes: distribution(samples.map(sample => sample.terminalBytes)),
    terminalWrites: distribution(samples.map(sample => sample.terminalWrites))
  }
}

async function main() {
  const workloads = workloadsArg()
  const warmups = numericArg('warmups', DEFAULT_WARMUPS)
  const sampleCount = numericArg('samples', DEFAULT_SAMPLES)
  const results = []

  for (const itemCount of workloads) {
    for (let warmup = 0; warmup < warmups; warmup++) {
      await runSample('virtual', itemCount)
      await runSample('native', itemCount)
    }

    const virtual: Sample[] = []
    const native: Sample[] = []

    for (let sample = 0; sample < sampleCount; sample++) {
      virtual.push(await runSample('virtual', itemCount))
      native.push(await runSample('native', itemCount))
    }

    results.push({
      itemCount,
      native: { samples: native, summary: summarize(native) },
      virtual: { samples: virtual, summary: summarize(virtual) }
    })
  }

  process.stdout.write(
    `${JSON.stringify({ config: { columns: COLUMNS, rows: ROWS, samples: sampleCount, warmups }, results, workloads }, null, 2)}\n`
  )
}

await main()
