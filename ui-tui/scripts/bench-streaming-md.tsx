// Benchmark: streamed-markdown render strategies for the TUI.
//
// Replays a newline-terminated, block-heavy synthetic stream at width 80
// through a real Ink render (renderSync + rerender per update) and compares:
//
//   naive       — <Md text={full}/> per update (re-tokenizes everything)
//   monolithic  — the previous StreamingMd: one memoized stable-prefix <Md>
//                 plus a tail <Md>. O(total) fence scan per update and a
//                 full prefix re-parse every time the boundary advances.
//   per-block   — current StreamingMd: append-only settled block array +
//                 incremental scanner. Each block parses exactly once.
//
// Each strategy/size pair runs in its OWN child process (orchestrator mode)
// so the parse-tree LRU in markdown.tsx and GC pressure from one strategy
// can't distort another's numbers. Each run also gets a unique text salt and
// a fresh theme object (its own WeakMap cache bucket).
//
// Run:          npx tsx scripts/bench-streaming-md.tsx
// Single case:  npx tsx scripts/bench-streaming-md.tsx <naive|monolithic|per-block> <blocks>

import { execFileSync } from 'child_process'
import { PassThrough } from 'stream'

import { Box, renderSync } from '@hermes/ink'
import React, { memo, useRef } from 'react'

import { Md } from '../src/components/markdown.js'
import { StreamingMd } from '../src/components/streamingMarkdown.js'
import { DEFAULT_THEME } from '../src/theme.js'

// ---- previous implementation (monolithic stable prefix), for comparison ----

const fenceOpenAt = (s: string, end: number) => {
  let codeOpen = false
  let mathOpen = false
  let mathOpener: '$$' | '\\[' | null = null
  let i = 0

  while (i < end) {
    const nl = s.indexOf('\n', i)
    const lineEnd = nl < 0 || nl > end ? end : nl
    const line = s.slice(i, lineEnd).trim()

    if (/^(?:`{3,}|~{3,})/.test(line)) {
      codeOpen = !codeOpen
    } else if (!codeOpen) {
      if (!mathOpen && /^\$\$/.test(line)) {
        if (!(line.length >= 4 && /\$\$$/.test(line))) {
          mathOpen = true
          mathOpener = '$$'
        }
      } else if (!mathOpen && /^\\\[/.test(line)) {
        if (!/\\\]$/.test(line)) {
          mathOpen = true
          mathOpener = '\\['
        }
      } else if (mathOpen && mathOpener === '$$' && /\$\$$/.test(line)) {
        mathOpen = false
        mathOpener = null
      } else if (mathOpen && mathOpener === '\\[' && /\\\]$/.test(line)) {
        mathOpen = false
        mathOpener = null
      }
    }

    if (nl < 0 || nl >= end) {
      break
    }

    i = nl + 1
  }

  return codeOpen || mathOpen
}

const findStableBoundaryOld = (text: string) => {
  let idx = text.length

  while (idx > 0) {
    const boundary = text.lastIndexOf('\n\n', idx - 1)

    if (boundary < 0) {
      return -1
    }

    const splitAt = boundary + 2

    if (!fenceOpenAt(text, splitAt)) {
      return splitAt
    }

    idx = boundary
  }

  return -1
}

const MonolithicStreamingMd = memo(function MonolithicStreamingMd({
  t,
  text
}: {
  t: typeof DEFAULT_THEME
  text: string
}) {
  const stablePrefixRef = useRef('')

  if (!text.startsWith(stablePrefixRef.current)) {
    stablePrefixRef.current = ''
  }

  const boundary = findStableBoundaryOld(text)

  if (boundary > stablePrefixRef.current.length) {
    stablePrefixRef.current = text.slice(0, boundary)
  }

  const stablePrefix = stablePrefixRef.current
  const unstableSuffix = text.slice(stablePrefix.length)

  if (!stablePrefix) {
    return <Md t={t} text={unstableSuffix} />
  }

  if (!unstableSuffix) {
    return <Md t={t} text={stablePrefix} />
  }

  return (
    <Box flexDirection="column">
      <Md t={t} text={stablePrefix} />
      <Md t={t} text={unstableSuffix} />
    </Box>
  )
})

// ---- synthetic stream ----

const makeBlocks = (count: number, salt: string) => {
  const blocks: string[] = []

  for (let i = 0; i < count; i++) {
    switch (i % 4) {
      case 0:
        blocks.push(`Paragraph ${salt}-${i} explaining step ${i} with **bold** and \`code\` inline.\n`)

        break

      case 1:
        blocks.push(`- item one ${salt}-${i}\n- item two with _emphasis_\n- item three\n`)

        break

      case 2:
        blocks.push(`\`\`\`ts\nconst v${i} = compute${salt}(${i})\nif (v${i} > 0) {\n  emit(v${i})\n}\n\`\`\`\n`)

        break

      default:
        blocks.push(`### Heading ${salt}-${i}\n\nSome follow-up prose for section ${i}.\n`)
    }
  }

  return blocks
}

// Newline-terminated updates: the stream grows one line per rerender.
const makeUpdates = (blocks: string[]) => {
  const full = blocks.join('\n')
  const updates: string[] = []

  let pos = 0

  while (pos < full.length) {
    const nl = full.indexOf('\n', pos)

    pos = nl < 0 ? full.length : nl + 1
    updates.push(full.slice(0, pos))
  }

  return updates
}

const nullStream = () => {
  const s = new PassThrough()

  Object.assign(s, { columns: 80, isTTY: false, rows: 24 })
  s.on('data', () => {})

  return s
}

const bench = (
  label: string,
  updates: string[],
  node: (t: typeof DEFAULT_THEME, text: string) => React.ReactNode,
  captureSeries = false
) => {
  // Fresh theme per run → fresh (collectable) mdCache bucket.
  const runTheme = { ...DEFAULT_THEME }

  const instance = renderSync(node(runTheme, ''), {
    patchConsole: false,
    stderr: nullStream() as unknown as NodeJS.WriteStream,
    stdin: nullStream() as unknown as NodeJS.ReadStream,
    stdout: nullStream() as unknown as NodeJS.WriteStream
  })

  const times: number[] = []
  const start = performance.now()

  let n = 0

  for (const text of updates) {
    const t0 = captureSeries ? performance.now() : 0

    instance.rerender(node(runTheme, text))

    if (captureSeries) {
      times.push(performance.now() - t0)
    }

    // Something in the render path emits performance measures; unbounded,
    // the entry buffer itself becomes a memory leak over thousands of
    // rerenders and skews long runs.
    if (++n % 64 === 0) {
      performance.clearMeasures()
      performance.clearMarks()
    }
  }

  const elapsed = performance.now() - start

  instance.unmount()
  instance.cleanup()

  return { elapsed, label, times }
}

const STRATEGIES = {
  monolithic: (t: typeof DEFAULT_THEME, text: string) => <MonolithicStreamingMd t={t} text={text} />,
  naive: (t: typeof DEFAULT_THEME, text: string) => <Md t={t} text={text} />,
  'per-block': (t: typeof DEFAULT_THEME, text: string) => <StreamingMd t={t} text={text} />
} as const

const [strategyArg, sizeArg] = process.argv.slice(2)

if (strategyArg === 'series') {
  // Series mode: per-append render times for one strategy/size, as JSON.
  // Usage: npx tsx scripts/bench-streaming-md.tsx series <strategy> <blocks>
  const [, seriesStrategy, seriesSize] = process.argv.slice(2)
  const size = Number(seriesSize)
  const updates = makeUpdates(makeBlocks(size, `${seriesStrategy}${size}`))
  const { elapsed, times } = bench(
    seriesStrategy!,
    updates,
    STRATEGIES[seriesStrategy as keyof typeof STRATEGIES],
    true
  )

  console.log(JSON.stringify({ appends: updates.length, elapsed, times }))
} else if (strategyArg) {
  // Child mode: run one strategy/size and print elapsed ms as JSON.
  const size = Number(sizeArg)
  const updates = makeUpdates(makeBlocks(size, `${strategyArg}${size}`))
  const { elapsed } = bench(strategyArg, updates, STRATEGIES[strategyArg as keyof typeof STRATEGIES])

  console.log(JSON.stringify({ appends: updates.length, elapsed }))
} else {
  // Orchestrator mode: one child process per strategy/size.
  const sizes = [32, 128, 512]

  const run = (strategy: string, size: number): { appends: number; elapsed: number } => {
    const out = execFileSync('npx', ['tsx', 'scripts/bench-streaming-md.tsx', strategy, String(size)], {
      encoding: 'utf8',
      env: { ...process.env, NODE_OPTIONS: '--max-old-space-size=8192' },
      timeout: 3_600_000
    })

    return JSON.parse(out.trim().split('\n').at(-1)!)
  }

  const fmt = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(1)} ms`)

  console.log(
    '| Blocks | Append calls | naive (full Md) | monolithic prefix | per-block (new) | new vs naive | new vs monolithic |'
  )
  console.log(
    '|--------|--------------|-----------------|-------------------|-----------------|--------------|-------------------|'
  )

  for (const size of sizes) {
    const naive = run('naive', size)
    const mono = run('monolithic', size)
    const perBlock = run('per-block', size)

    console.log(
      `| ${size} | ${perBlock.appends} | ${fmt(naive.elapsed)} | ${fmt(mono.elapsed)} | ${fmt(perBlock.elapsed)} | ${(
        naive.elapsed / perBlock.elapsed
      ).toFixed(1)}x | ${(mono.elapsed / perBlock.elapsed).toFixed(1)}x |`
    )
  }
}
