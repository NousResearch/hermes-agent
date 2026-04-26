// StreamingMd — incremental markdown renderer for in-flight assistant text.
//
// Naive approach (render <Md text={full}/>) re-tokenizes the entire message
// on every stream delta. At 20-char batches over a 3 KB response that's 150
// full re-parses.
//
// This splits `text` at the last stable top-level block boundary (blank
// line outside a fenced code span) into:
//   stablePrefix — passed to an inner <Md>, memoized on its exact text
//                  value. During the turn, the prefix only grows monotonically,
//                  so its memo key matches the previous render and React
//                  reuses the cached subtree — zero re-tokenization.
//   unstableSuffix — the in-flight block(s). A separate <Md> re-parses just
//                    this tail on every delta (O(unstable length) vs.
//                    O(total length)).
//
// The boundary is stored in a ref so it only advances — idempotent under
// StrictMode double-render. Component unmounts between turns (isStreaming
// flips off → message moves to history and renders via <Md> directly), so
// the ref resets naturally.
//
// Layout: the two <Md> subtrees MUST render stacked (column). The parent
// container in messageLine.tsx is a default `flexDirection: 'row'` Box
// (Ink's default), so returning a bare Fragment of two <Md> siblings
// laid them out side-by-side — producing the "two jumbled columns while
// streaming" rendering bug. Wrapping in a flexDirection="column" Box
// here localizes the fix to the streaming path; the non-streaming <Md>
// already returns its own column Box, so its single-child case was never
// affected.

import { Box } from '@hermes/ink'
import { memo, useRef } from 'react'

import type { Theme } from '../theme.js'

import { Md } from './markdown.js'

// Count ``` or ~~~ fence toggles in `s` up to `end`. Odd = currently inside
// a fenced block; we can't split the prefix there or we'd orphan the fence.
const fenceOpenAt = (s: string, end: number) => {
  let open = false
  let i = 0

  while (i < end) {
    const nl = s.indexOf('\n', i)
    const lineEnd = nl < 0 || nl > end ? end : nl
    const line = s.slice(i, lineEnd)

    if (/^\s*(?:`{3,}|~{3,})/.test(line)) {
      open = !open
    }

    if (nl < 0 || nl >= end) {
      break
    }

    i = nl + 1
  }

  return open
}

// Find the last "\n\n" boundary before `end` that is OUTSIDE a fenced code
// block. Returns the index AFTER the second newline (start of the next
// block), or -1 if no safe boundary exists yet.
export const findStableBoundary = (text: string) => {
  let idx = text.length

  while (idx > 0) {
    const boundary = text.lastIndexOf('\n\n', idx - 1)

    if (boundary < 0) {
      return -1
    }

    // Boundary candidate: end of stable prefix is boundary + 2 (start of
    // next block). Check fence balance up to that point.
    const splitAt = boundary + 2

    if (!fenceOpenAt(text, splitAt)) {
      return splitAt
    }

    idx = boundary
  }

  return -1
}

export const StreamingMd = memo(function StreamingMd({ compact, t, text }: StreamingMdProps) {
  const stablePrefixRef = useRef('')

  // Reset if the text no longer starts with our recorded prefix (defensive;
  // normally the component unmounts between turns so this shouldn't trigger).
  if (!text.startsWith(stablePrefixRef.current)) {
    stablePrefixRef.current = ''
  }

  const boundary = findStableBoundary(text)

  // Only advance the prefix — never retreat. The boundary math looks at the
  // FULL text each call; if it returns a larger index than before, we grow
  // the cached prefix. Monotonic growth makes the memo key stable across
  // deltas (identical string → same <Md> subtree → no re-render).
  if (boundary > stablePrefixRef.current.length) {
    stablePrefixRef.current = text.slice(0, boundary)
  }

  const stablePrefix = stablePrefixRef.current
  const unstableSuffix = text.slice(stablePrefix.length)

  if (!stablePrefix) {
    return <Md compact={compact} t={t} text={unstableSuffix} />
  }

  if (!unstableSuffix) {
    return <Md compact={compact} t={t} text={stablePrefix} />
  }

  return (
    <Box flexDirection="column">
      <Md compact={compact} t={t} text={stablePrefix} />
      <Md compact={compact} t={t} text={unstableSuffix} />
    </Box>
  )
})

interface StreamingMdProps {
  compact?: boolean
  t: Theme
  text: string
}
