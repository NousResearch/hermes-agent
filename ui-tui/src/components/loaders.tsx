import { Box, Text } from '@hermes/ink'
import { useEffect, useState } from 'react'

import { mix } from '../lib/color.js'

/**
 * Animated ASCII loaders — THE loading-state primitives (session panel
 * skeleton, widget apps via the SDK). A highlight band sweeps across block
 * runs; rows offset their phase for a diagonal shimmer. One interval per
 * composition (the parent ticks, rows are pure), colors are caller-owned
 * theme tones — never hardcoded.
 */

const BAND = 7

/** Pure band math: [pre, band, post] cell widths for a sweep at `phase`.
 *  The band enters from off-left and exits off-right, wrapping. */
export function shimmerSegments(width: number, phase: number, band = BAND): [number, number, number] {
  const cycle = width + band
  const start = (((phase % cycle) + cycle) % cycle) - band
  const from = Math.max(0, start)
  const to = Math.min(width, start + band)

  return to <= from ? [width, 0, 0] : [from, to - from, width - to]
}

/** One shimmering run. Controlled: the parent owns `phase` so sibling rows
 *  stay in lockstep (offset it per row for the diagonal). */
export function Shimmer({
  char = '▁',
  color,
  highlight,
  phase,
  width
}: {
  char?: string
  color: string
  highlight: string
  phase: number
  width: number
}) {
  const [pre, band, post] = shimmerSegments(width, phase)

  return (
    <Text>
      {pre > 0 && <Text color={color}>{char.repeat(pre)}</Text>}
      {band > 0 && <Text color={highlight}>{char.repeat(band)}</Text>}
      {post > 0 && <Text color={color}>{char.repeat(post)}</Text>}
    </Text>
  )
}

// ── Shared shimmer clock ─────────────────────────────────────────────
//
// ONE interval drives every mounted shimmer composition (review on #20379,
// finding 5: the session panel could mount independent 90 ms intervals for
// lazy skills AND lazy tools — ~22 state updates/sec on an otherwise-idle
// TUI). Subscribers share the tick; the interval exists only while
// subscribers do, and all updates land in one timer callback so React
// batches them into a single render pass.

const TICK_MS = 90

/** Animation budget per mount. A lazy watch session can stay lazy
 *  indefinitely — after the budget the skeleton freezes in place (still
 *  reads as "loading") instead of repainting forever. */
export const SHIMMER_ANIMATE_MS = 30_000

const clockListeners = new Set<(phase: number) => void>()
let clockId: NodeJS.Timeout | null = null
let clockPhase = 0

/** Subscribe to the shared clock (exported for tests). Returns unsubscribe;
 *  the interval stops with the last subscriber. */
export function subscribeShimmerClock(fn: (phase: number) => void): () => void {
  clockListeners.add(fn)

  if (!clockId) {
    clockId = setInterval(() => {
      clockPhase += 1

      for (const listener of clockListeners) {
        listener(clockPhase)
      }
    }, TICK_MS)

    clockId.unref?.()
  }

  return () => {
    clockListeners.delete(fn)

    if (!clockListeners.size && clockId) {
      clearInterval(clockId)
      clockId = null
    }
  }
}

/** Phase from the shared clock, bounded: stops advancing (and stops costing
 *  renders) after `animateMs`. */
export function useShimmerPhase(animateMs = SHIMMER_ANIMATE_MS): number {
  const [phase, setPhase] = useState(clockPhase)

  useEffect(() => {
    const startedAt = Date.now()

    let unsubscribe: (() => void) | null = subscribeShimmerClock(next => {
      if (Date.now() - startedAt >= animateMs) {
        unsubscribe?.()
        unsubscribe = null

        return
      }

      setPhase(next)
    })

    return () => {
      unsubscribe?.()
      unsubscribe = null
    }
  }, [animateMs])

  return phase
}

/** Skeleton rows shaped like `label: value` content, diagonal shimmer.
 *
 *  Ergonomic for generated code (the primary author is an agent):
 *  - `rows` — explicit `[labelWidth, valueWidth][]` mirroring real layout,
 *    OR just a count (row widths derive from `width`, staggered).
 *  - colors — explicit `color`/`highlight`, OR pass a theme `t` and they
 *    derive (muted-toward-surface base, label highlight). */
export function ShimmerRows({
  color,
  highlight,
  rows,
  t,
  width = 24
}: {
  color?: string
  highlight?: string
  rows: number | readonly (readonly [number, number])[]
  t?: { color: { completionBg: string; label: string; muted: string } }
  width?: number
}) {
  const phase = useShimmerPhase()
  const base = color ?? (t ? mix(t.color.muted, t.color.completionBg, 0.5) : '#808080')
  const glow = highlight ?? t?.color.label ?? '#a0a0a0'

  const spec: readonly (readonly [number, number])[] =
    typeof rows === 'number'
      ? Array.from({ length: Math.max(1, rows) }, (_, i) => {
          const label = Math.max(4, Math.round(width * 0.3) - (i % 3))

          return [label, Math.max(4, width - label - 1)] as const
        })
      : rows

  return (
    <Box flexDirection="column">
      {spec.map(([labelWidth, valueWidth], i) => (
        <Text key={i}>
          <Shimmer color={base} highlight={glow} phase={phase - i * 2} width={labelWidth} />
          <Text> </Text>
          <Shimmer color={base} highlight={glow} phase={phase - i * 2 - labelWidth} width={valueWidth} />
        </Text>
      ))}
    </Box>
  )
}
