// DeferredMd — renders a lightweight <Text> placeholder on first mount and
// upgrades to full <Md> markdown + syntax highlighting in a subsequent
// transition commit. Spreads the parse cost off the scroll critical path.
//
// Why: profiling shows the 63-112ms renderer spikes during hold-PageUp
// correlate with fresh MessageLine mounts running the markdown tokenizer
// + syntax highlighting synchronously. The new row is added by
// useVirtualHistory's slide step; React commits the tree; Ink lays out
// Yoga; stdout writes the result. All in one hitch frame.
//
// With this wrapper, the hitch frame lays out a pre-wrapped plain <Text>
// (Yoga only needs to wrap width-known strings — no tokenizer, no
// highlighter, no inline regex walk), then a follow-up commit re-renders
// the same row with full markdown. The follow-up is gated on a
// queueMicrotask so Ink has a chance to paint the placeholder before
// React starts the Md-heavy upgrade work.
//
// Upgrade cache: once a given (theme, text, compact) tuple has been
// rendered as full Md, we remember it so remounts (scroll-out then
// scroll-back) don't pay the placeholder round-trip again — they mount
// straight into the upgraded <Md> subtree, which Md internally memoizes
// on text identity, so there's no re-tokenization either.

import { Text } from '@hermes/ink'
import { memo, useEffect, useState } from 'react'

import type { Theme } from '../theme.js'

import { Md, stripInlineMarkup } from './markdown.js'

// Theme object is stable per-session; key upgrades under it so palette
// swaps naturally retrigger (colors differ → render changes).
const upgraded = new WeakMap<Theme, Set<string>>()

const cacheKey = (compact: boolean | undefined, text: string) => (compact ? `c:${text}` : `x:${text}`)

const hasUpgraded = (t: Theme, key: string) => upgraded.get(t)?.has(key) ?? false

const markUpgraded = (t: Theme, key: string) => {
  const bucket = upgraded.get(t) ?? new Set<string>()

  bucket.add(key)
  upgraded.set(t, bucket)
}

export const DeferredMd = memo(function DeferredMd({ color, compact, t, text }: DeferredMdProps) {
  const key = cacheKey(compact, text)
  const [ready, setReady] = useState(() => hasUpgraded(t, key) || !text)

  useEffect(() => {
    if (ready) {
      return
    }

    let cancelled = false

    queueMicrotask(() => {
      if (cancelled) {
        return
      }

      markUpgraded(t, key)
      setReady(true)
    })

    return () => {
      cancelled = true
    }
  }, [key, ready, t])

  if (ready) {
    return <Md compact={compact} t={t} text={text} />
  }

  // Placeholder: strip inline markup so the visible width approximately
  // matches the final Md layout (bold/italic/links are width-neutral or
  // collapse to anchor text). Line breaks preserved — Ink's wrap="wrap"
  // lays the plain text out as blocks at the right column count.
  // Using <Text> directly (no Box wrapper) so there's no column-flex
  // decision for Yoga — it just wraps a string.
  return <Text color={color ?? undefined}>{stripInlineMarkup(text)}</Text>
})

interface DeferredMdProps {
  /** Fallback color for the placeholder text (typically the role's body color). */
  color?: string
  compact?: boolean
  t: Theme
  text: string
}
