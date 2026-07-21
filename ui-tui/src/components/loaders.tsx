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

/** Self-ticking phase for shimmer compositions. */
export function useShimmerPhase(tickMs = 90): number {
  const [phase, setPhase] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setPhase(p => p + 1), tickMs)

    id.unref?.()

    return () => clearInterval(id)
  }, [tickMs])

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
