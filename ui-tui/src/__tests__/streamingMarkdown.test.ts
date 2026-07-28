import { PassThrough } from 'stream'

import { Box, renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { Md } from '../components/markdown.js'
import { advanceScan, createScanState, findStableBoundary } from '../components/streamingMarkdown.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const BEL = String.fromCharCode(7)
const ESC = String.fromCharCode(27)
const CSI_RE = new RegExp(`${ESC}\\[[0-?]*[ -/]*[@-~]`, 'g')
const OSC_RE = new RegExp(`${ESC}\\][\\s\\S]*?(?:${BEL}|${ESC}\\\\)`, 'g')

const renderPlain = (node: React.ReactNode) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(node, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()

  return output
    .replace(OSC_RE, '')
    .split('\n')
    .map(line => stripAnsi(line).replace(CSI_RE, '').trimEnd())
}

describe('findStableBoundary', () => {
  it('returns -1 when no blank line exists yet', () => {
    expect(findStableBoundary('partial line with no newline yet')).toBe(-1)
  })

  it('returns -1 when only single newlines exist', () => {
    expect(findStableBoundary('line one\nline two\nline three')).toBe(-1)
  })

  it('splits after the last blank line separator', () => {
    // 'first\n\nsecond\n\nthird' → last blank = before 'third'
    const text = 'first paragraph\n\nsecond paragraph\n\nthird'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('first paragraph\n\nsecond paragraph\n\n')
    expect(text.slice(idx)).toBe('third')
  })

  it('refuses to split inside an open fenced block', () => {
    // Fence opens, contains a blank line inside the code, no close yet.
    const text = '```ts\nfn();\n\nmore code here'

    expect(findStableBoundary(text)).toBe(-1)
  })

  it('splits before an open fenced block but not inside', () => {
    const text = 'intro paragraph\n\n```ts\nfn();\n\nmore code'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('intro paragraph\n\n')
    expect(text.slice(idx).startsWith('```ts')).toBe(true)
  })

  it('allows splitting after a fenced block closes', () => {
    const text = '```ts\nfn();\n```\n\nnarration continues'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('```ts\nfn();\n```\n\n')
    expect(text.slice(idx)).toBe('narration continues')
  })

  it('walks backwards through nested fence boundaries safely', () => {
    // Two closed fences + narration + one new open fence. The only legal
    // split is before the open fence, not between the closed ones.
    const text = '```js\na\n```\n\nmid text\n\n```python\nstill open'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('```js\na\n```\n\nmid text\n\n')
  })

  it('handles empty input', () => {
    expect(findStableBoundary('')).toBe(-1)
  })

  it('refuses to split inside an open $$ math block', () => {
    // Display math has been opened but not closed; the only blank line
    // sits inside the open block, so there's no safe boundary yet.
    const text = '$$\nx + y\n\nmore math'

    expect(findStableBoundary(text)).toBe(-1)
  })

  it('allows splitting after a $$ math block closes', () => {
    const text = '$$\nx + y = z\n$$\n\nnarration continues'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('$$\nx + y = z\n$$\n\n')
    expect(text.slice(idx)).toBe('narration continues')
  })

  it('splits before an open $$ block but not inside', () => {
    // Mirror of the existing fenced-code test: prose, then an unclosed
    // math block. The only safe boundary is the blank line BEFORE `$$`.
    const text = 'intro paragraph\n\n$$\nx + y\n\nmore'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('intro paragraph\n\n')
    expect(text.slice(idx).startsWith('$$')).toBe(true)
  })

  it('treats single-line $$x$$ as zero net toggle', () => {
    // `$$x = y$$` opens AND closes on one line, so the stable boundary
    // after it is allowed.
    const text = 'intro\n\n$$x = y$$\n\nnarration'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('intro\n\n$$x = y$$\n\n')
    expect(text.slice(idx)).toBe('narration')
  })

  it('refuses to split inside an open \\[ math block', () => {
    const text = '\\[\nx + y\n\nmore'

    expect(findStableBoundary(text)).toBe(-1)
  })
})

// A corpus exercising every construct the boundary scanner must respect:
// paragraphs, fenced code (with blank lines and $$ bait inside), display
// math ($$ and \[), setext headings, tables, lists, quotes, headings.
const CORPUS = [
  'Intro paragraph explaining the plan in some detail.\n',
  '\nSection Title\n=============\n',
  '\nA paragraph before code.\n',
  '\n```ts\nconst a = 1\n\nconst b = 2\n// $$ not math $$\n```\n',
  '\nBetween-blocks narration.\n',
  '\n$$\nE = mc^2\n\n\\sum_i x_i\n$$\n',
  '\n- item one\n- item two\n\n1. first\n2. second\n',
  '\n| a | b |\n|---|---|\n| 1 | 2 |\n',
  '\n> quoted wisdom\n> second line\n',
  '\n\\[\nx^2 + y^2 = z^2\n\\]\n',
  '\n## Closing heading\n',
  '\nFinal paragraph without a trailing newline'
].join('')

const mulberry32 = (seed: number) => () => {
  seed |= 0
  seed = (seed + 0x6d2b79f5) | 0

  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)

  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t

  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}

describe('advanceScan (incremental scanner)', () => {
  it('reconstructs the input exactly: blocks + tail === text', () => {
    const state = createScanState()

    advanceScan(CORPUS, state)

    expect(state.blocks.join('')).toHaveLength(state.settledLen)
    expect(state.blocks.join('') + CORPUS.slice(state.settledLen)).toBe(CORPUS)
    expect(state.blocks.length).toBeGreaterThan(5)
  })

  it('produces identical blocks fed incrementally at arbitrary cut points', () => {
    const oneShot = createScanState()

    advanceScan(CORPUS, oneShot)

    for (let seed = 1; seed <= 8; seed++) {
      const rand = mulberry32(seed)
      const state = createScanState()

      let pos = 0

      while (pos < CORPUS.length) {
        pos = Math.min(CORPUS.length, pos + 1 + Math.floor(rand() * 7))

        const prevBlocks = state.blocks.length

        advanceScan(CORPUS.slice(0, pos), state)

        // Append-only: previously committed blocks never change.
        expect(state.blocks.length).toBeGreaterThanOrEqual(prevBlocks)
      }

      expect(state.blocks).toEqual(oneShot.blocks)
      expect(state.settledLen).toBe(oneShot.settledLen)
    }
  })

  it('is idempotent when called again with the same text', () => {
    const state = createScanState()

    advanceScan(CORPUS, state)

    const blocks = [...state.blocks]

    advanceScan(CORPUS, state)

    expect(state.blocks).toEqual(blocks)
  })

  it('holds a partial trailing line in the tail even if it looks fence-like', () => {
    // "``" could grow into "```ts" — the scanner must not judge the line
    // until its newline arrives.
    const state = createScanState()

    advanceScan('para\n\n``', state)

    expect(state.blocks).toEqual(['para\n\n'])
    expect(state.codeOpen).toBe(false)

    advanceScan('para\n\n```ts\ncode\n\nstill code', state)

    // The blank line inside the now-open fence must not commit a block.
    expect(state.blocks).toEqual(['para\n\n'])
    expect(state.codeOpen).toBe(true)

    advanceScan('para\n\n```ts\ncode\n\nstill code\n```\n\nafter\n\n', state)

    expect(state.blocks).toEqual(['para\n\n', '```ts\ncode\n\nstill code\n```\n\n', 'after\n\n'])
    expect(state.codeOpen).toBe(false)
  })

  it('does not commit whitespace-only blocks on 3+ newline runs', () => {
    const state = createScanState()

    advanceScan('alpha\n\n\n\nbeta\n\n', state)

    expect(state.blocks).toEqual(['alpha\n\n', '\n\nbeta\n\n'])
    expect(state.blocks.join('')).toBe('alpha\n\n\n\nbeta\n\n')
  })

  it('keeps a setext heading contiguous with its paragraph', () => {
    // A setext underline attaches to the line above; the only committed
    // boundary is the blank line, so the pair can never be torn apart or
    // retroactively merged (the desktop splitter needed a fix for this,
    // #67176 — blank-line boundaries are immune by construction).
    const state = createScanState()

    advanceScan('Title\n', state)
    advanceScan('Title\n====\n', state)
    advanceScan('Title\n====\n\nbody\n\n', state)

    expect(state.blocks).toEqual(['Title\n====\n\n', 'body\n\n'])
  })
})

describe('StreamingMd rendering equivalence', () => {
  it('settled blocks + tail render identically to one combined Md', () => {
    const state = createScanState()

    advanceScan(CORPUS, state)

    const tail = CORPUS.slice(state.settledLen)
    const t = DEFAULT_THEME

    const split = renderPlain(
      React.createElement(
        Box,
        { flexDirection: 'column' },
        ...state.blocks.map((block, i) => React.createElement(Md, { key: i, t, text: block })),
        tail ? React.createElement(Md, { key: 'tail', t, text: tail }) : null
      )
    )

    const combined = renderPlain(React.createElement(Md, { t, text: CORPUS }))

    expect(split).toEqual(combined)
  })

  it('renders split/combined identically at every streamed step', () => {
    const rand = mulberry32(42)
    const state = createScanState()
    const t = DEFAULT_THEME

    let pos = 0

    while (pos < CORPUS.length) {
      pos = Math.min(CORPUS.length, pos + 24 + Math.floor(rand() * 200))

      const text = CORPUS.slice(0, pos)

      advanceScan(text, state)

      const tail = text.slice(state.settledLen)

      const split = renderPlain(
        React.createElement(
          Box,
          { flexDirection: 'column' },
          ...state.blocks.map((block, i) => React.createElement(Md, { key: i, t, text: block })),
          tail ? React.createElement(Md, { key: 'tail', t, text: tail }) : null
        )
      )

      const combined = renderPlain(React.createElement(Md, { t, text }))

      expect(split).toEqual(combined)
    }
  })
})
