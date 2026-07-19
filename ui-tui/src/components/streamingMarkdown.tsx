// StreamingMd — incremental markdown renderer for in-flight assistant text.
//
// Naive approach (render <Md text={full}/>) re-tokenizes the entire message
// on every stream delta. At 20-char batches over a 3 KB response that's 150
// full re-parses.
//
// The previous incarnation split `text` at the last stable top-level block
// boundary into ONE memoized stable-prefix <Md> plus a re-parsed tail <Md>.
// That removed the per-delta cost but kept a per-block cliff: every time the
// boundary advanced, the prefix string changed, its memo key missed, and the
// ENTIRE prefix re-tokenized from scratch — O(blocks²) work across a long
// reply, plus an O(total) fence scan on every delta to find the boundary.
//
// This version is fully incremental:
//   settled blocks — an append-only array of top-level block strings. Each
//                    block renders as its own <Md>, memoized on its exact
//                    text, which never changes once committed. Every block
//                    is tokenized exactly once for the life of the stream.
//   unstable tail — the in-flight block(s) after the last committed
//                   boundary. A single <Md> re-parses just this tail on
//                   every delta (O(tail) vs. O(total)).
//   scanner state — fence/math open-state and scan position persist across
//                   deltas in a ref, so each delta only scans the newly
//                   arrived complete lines instead of the whole text.
//
// A block boundary is a literal "\n\n" outside any fenced code or display
// math block. Only complete lines (terminated by "\n") are scanned — a
// partial trailing line could still grow into a fence opener, so it stays
// in the tail until its newline arrives. Blank-line boundaries also make
// the split immune to retroactive block merges (e.g. a setext underline
// only attaches to the contiguous paragraph above it — it cannot reach
// across a committed "\n\n").
//
// The scanner treats an unmatched `$$` / `\[` opener as open forever. This
// is INTENTIONALLY more conservative than `markdown.tsx`'s parser, which
// falls back to paragraph rendering when an opener has no closer. The
// renderer can do that safely because it always sees the full text on
// every call. The streaming chunker cannot — once a block is committed it
// is frozen, so prematurely deciding "this `$$` is just prose" would
// permanently commit a paragraph rendering that becomes wrong the instant
// the closer streams in. Parking the boundary keeps everything from the
// opener onward in the mutable tail until the closer arrives (or the
// stream ends and the non-streaming <Md> takes over, at which point the
// renderer's fallback kicks in correctly).
//
// State is stored in a ref and only ever advances — idempotent under
// StrictMode double-render. The component unmounts between turns
// (isStreaming flips off → message moves to history and renders via a
// single <Md>), so the ref resets naturally. If `text` stops extending the
// scanned prefix anyway (turn reuse, or the bounded live-render window
// trimming the front of a very long reply), the scanner resets and the
// per-text LRU inside <Md> still absorbs most of the re-parse.
//
// Layout: the <Md> subtrees MUST render stacked (column). The parent
// container in messageLine.tsx is a default `flexDirection: 'row'` Box
// (Ink's default), so returning a bare Fragment of sibling <Md>s would lay
// them out side-by-side — the "two jumbled columns while streaming" bug.

import { Box } from '@hermes/ink'
import { memo, useRef } from 'react'

import type { Theme } from '../theme.js'

import { Md } from './markdown.js'

export interface StreamScanState {
  /** Settled top-level block strings, in order. Append-only. */
  blocks: string[]
  /** True while inside an unclosed ``` / ~~~ fence at the scan position. */
  codeOpen: boolean
  /** Non-null while inside an unclosed display-math block at the scan position. */
  mathOpener: '$$' | '\\[' | null
  /**
   * The prefix of the text whose complete lines have been scanned —
   * text.slice(0, scanPos), with scanPos === scanned.length. Kept as a
   * string (not just an index) so the reset guard can verify the scanned
   * region still matches, fence state included.
   */
  scanned: string
  /** Length of the committed prefix — blocks.join('').length. */
  settledLen: number
}

export const createScanState = (): StreamScanState => ({
  blocks: [],
  codeOpen: false,
  mathOpener: null,
  scanned: '',
  settledLen: 0
})

// Apply one complete (newline-terminated) line to the fence/math state.
// Mirrors the toggle rules the old fenceOpenAt() used: ``` / ~~~ toggle the
// code fence; `$$` / `\[` open display math only when the line doesn't also
// close it; closers only count when the matching opener is pending. Math
// markers inside an open code fence are ignored (a `$$` example in a code
// block must not open math).
const applyLine = (state: StreamScanState, line: string) => {
  if (/^(?:`{3,}|~{3,})/.test(line)) {
    state.codeOpen = !state.codeOpen

    return
  }

  if (state.codeOpen) {
    return
  }

  if (!state.mathOpener) {
    if (/^\$\$/.test(line)) {
      const singleLine = line.length >= 4 && /\$\$$/.test(line)

      if (!singleLine) {
        state.mathOpener = '$$'
      }
    } else if (/^\\\[/.test(line)) {
      const singleLine = /\\\]$/.test(line)

      if (!singleLine) {
        state.mathOpener = '\\['
      }
    }
  } else if (state.mathOpener === '$$' && /\$\$$/.test(line)) {
    state.mathOpener = null
  } else if (state.mathOpener === '\\[' && /\\\]$/.test(line)) {
    state.mathOpener = null
  }
}

// Consume newly arrived COMPLETE lines from the scan position, committing a
// block at every "\n\n" boundary reached outside fences. Whitespace-only
// candidates (runs of 3+ newlines) are left for the next block rather than
// committed as empty <Md>s. Mutates `state`; calling again with the same
// text is a no-op (idempotent).
export const advanceScan = (text: string, state: StreamScanState) => {
  const start = state.scanned.length

  let i = start

  while (i < text.length) {
    const nl = text.indexOf('\n', i)

    if (nl < 0) {
      // Partial trailing line — could still become a fence opener. Leave it
      // (and its whole block) in the unstable tail.
      break
    }

    if (nl === i) {
      // Empty line. If it's the second half of a "\n\n" pair (i.e. not the
      // very first character) and no fence is open, the text before it is a
      // settled top-level block.
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

// Compatibility shim over the incremental scanner: index just past the last
// committed block boundary in `text`, or -1 when nothing has settled yet.
// Kept for boundary-semantics tests; StreamingMd itself keeps scanner state
// across deltas instead of recomputing from scratch.
export const findStableBoundary = (text: string) => {
  const state = createScanState()

  advanceScan(text, state)

  return state.settledLen > 0 ? state.settledLen : -1
}

export const StreamingMd = memo(function StreamingMd({ cols, compact, t, text }: StreamingMdProps) {
  const scanRef = useRef<StreamScanState>(createScanState())

  let state = scanRef.current

  // Reset when the text no longer extends the scanned prefix: a new turn
  // reusing the mounted component, or boundedLiveRenderText trimming the
  // front of a very long reply. Comparing the full scanned region (not just
  // the settled blocks) guarantees the persisted fence state is still valid.
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
