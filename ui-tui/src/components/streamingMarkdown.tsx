// StreamingMd — incremental markdown renderer for in-flight assistant text.
//
// Rendering <Md text={full}/> per delta re-tokenizes the whole message every
// time (O(total) × deltas). The prior stable-prefix split fixed the per-delta
// cost but not the per-block cliff: each advanced boundary re-tokenized the
// entire prefix from scratch — O(blocks²) — plus an O(total) fence rescan.
//
// This is fully incremental. A forward scanner keeps fence/math state + scan
// position in a ref across deltas, so each delta touches only newly-arrived
// complete lines. Settled top-level blocks (split on "\n\n" outside a fence)
// are frozen into an append-only array, each its own <Md> memoized on text
// that never changes — every block tokenizes exactly once. Only the in-flight
// tail re-parses per delta (O(tail)).
//
// Invariants that keep it correct:
//   · Only newline-terminated lines are scanned; a partial trailing line may
//     yet become a fence opener, so it stays in the tail.
//   · Blank-line boundaries can't be retroactively merged (a setext underline
//     only binds the contiguous line above it, never across a committed "\n\n").
//   · An unmatched `$$` / `\[` opener is treated as open forever — more
//     conservative than markdown.tsx's full-text fallback, because a committed
//     block is frozen and can't be un-decided once the closer streams in.
//   · State only advances (idempotent under StrictMode). The component
//     unmounts between turns; if `text` stops extending `scanned` (turn reuse,
//     or boundedLiveRenderText front-trimming a huge reply) the scanner resets
//     and <Md>'s LRU absorbs the re-parse.
//
// Layout: the <Md> subtrees MUST stack in a column — the messageLine.tsx
// parent is a default row Box, so bare siblings render side-by-side.

import { Box } from '@hermes/ink'
import { memo, useRef } from 'react'

import type { Theme } from '../theme.js'

import { Md } from './markdown.js'

export interface StreamScanState {
  /** Settled top-level block strings, in order. Append-only. */
  blocks: string[]
  /** Inside an unclosed ``` / ~~~ fence at the scan position. */
  codeOpen: boolean
  /** Non-null inside an unclosed display-math block at the scan position. */
  mathOpener: '$$' | '\\[' | null
  /** Prefix whose complete lines have been scanned (kept as text so the reset
   *  guard can confirm the scanned region — and thus fence state — still holds). */
  scanned: string
  /** Length of the committed prefix (blocks.join('').length). */
  settledLen: number
}

export const createScanState = (): StreamScanState => ({
  blocks: [],
  codeOpen: false,
  mathOpener: null,
  scanned: '',
  settledLen: 0
})

// Fold one complete line into the fence/math state: ``` / ~~~ toggle the code
// fence; `$$` / `\[` open display math unless the line also closes it; closers
// count only against a pending opener; math inside an open code fence is inert.
const applyLine = (state: StreamScanState, line: string) => {
  if (/^(?:`{3,}|~{3,})/.test(line)) {
    state.codeOpen = !state.codeOpen

    return
  }

  if (state.codeOpen) {
    return
  }

  if (!state.mathOpener) {
    if (/^\$\$/.test(line) && !(line.length >= 4 && /\$\$$/.test(line))) {
      state.mathOpener = '$$'
    } else if (/^\\\[/.test(line) && !/\\\]$/.test(line)) {
      state.mathOpener = '\\['
    }
  } else if (state.mathOpener === '$$' && /\$\$$/.test(line)) {
    state.mathOpener = null
  } else if (state.mathOpener === '\\[' && /\\\]$/.test(line)) {
    state.mathOpener = null
  }
}

// Consume newly-arrived complete lines, committing a settled block at every
// "\n\n" outside a fence. Whitespace-only runs stay with the next block, never
// committed as empty <Md>s. Mutates `state`; re-calling with the same text is
// a no-op (idempotent).
export const advanceScan = (text: string, state: StreamScanState) => {
  const start = state.scanned.length

  let i = start

  while (i < text.length) {
    const nl = text.indexOf('\n', i)

    if (nl < 0) {
      break // partial trailing line — could still open a fence; keep in tail
    }

    if (nl === i) {
      // Second half of a "\n\n" outside any fence → prior text is a block.
      if (i > 0 && !state.codeOpen && !state.mathOpener) {
        const block = text.slice(state.settledLen, nl + 1)

        if (/\S/.test(block)) {
          state.blocks.push(block)
          state.settledLen = nl + 1
        }
      }
    } else {
      applyLine(state, text.slice(i, nl).trim())
    }

    i = nl + 1
  }

  if (i > start) {
    state.scanned += text.slice(start, i)
  }
}

// Index just past the last committed boundary, or -1 if nothing has settled.
// Thin wrapper over the scanner for boundary-semantics tests.
export const findStableBoundary = (text: string) => {
  const state = createScanState()

  advanceScan(text, state)

  return state.settledLen > 0 ? state.settledLen : -1
}

export const StreamingMd = memo(function StreamingMd({ cols, compact, t, text }: StreamingMdProps) {
  const scanRef = useRef<StreamScanState>(createScanState())

  let state = scanRef.current

  // Reset if `text` no longer extends the scanned prefix (turn reuse, or a
  // front-trim), which would invalidate the persisted fence state.
  if (!text.startsWith(state.scanned)) {
    state = scanRef.current = createScanState()
  }

  advanceScan(text, state)

  const tail = text.slice(state.settledLen)

  return (
    <Box flexDirection="column">
      {state.blocks.map((block, i) => (
        <Md cols={cols} compact={compact} key={i} t={t} text={block} />
      ))}

      {tail ? <Md cols={cols} compact={compact} t={t} text={tail} /> : null}
    </Box>
  )
})

interface StreamingMdProps {
  cols?: number
  compact?: boolean
  t: Theme
  text: string
}
