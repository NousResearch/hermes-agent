import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useMemo, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import { useT } from '../i18n/useT.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { OverlayHint, windowItems } from './overlayControls.js'
import { chipRowProps, clampOverlayWidth } from './overlayPrimitives.js'

const VISIBLE = 10
const MIN_WIDTH = 40
const MAX_WIDTH = 90

interface GalleryCompanion {
  slug: string
  displayName: string
  installed: boolean
  curated?: boolean
}

interface Gallery {
  enabled: boolean
  active: string
  companions: GalleryCompanion[]
}

/**
 * Interactive companiondex picker overlay. Pulls the gallery via `companion.gallery`,
 * filters as you type, and adopts the highlighted companion with `companion.select`
 * (install-on-demand). The mascot lights up live once `useCompanion` next polls —
 * no restart. This is the interactive sibling of the text `/companion <slug>` path.
 */
export function CompanionPicker({ gw, maxWidth, onClose, t }: CompanionPickerProps) {
  const [gallery, setGallery] = useState<Gallery | null>(null)
  const [query, setQuery] = useState('')
  const [idx, setIdx] = useState(0)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(true)
  const T = useT()
  const P = T.pickers.companion
  const C = T.pickers.common

  const { stdout } = useStdout()
  // Optional maxWidth lets grid layouts hand the picker its cell budget.
  const preferredWidth = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))
  const width = clampOverlayWidth(preferredWidth, maxWidth)

  useEffect(() => {
    gw.request<Gallery>('companion.gallery')
      .then(r => {
        setGallery(r)
        setErr('')
      })
      .catch((e: unknown) => setErr(rpcErrorMessage(e)))
      .finally(() => setLoading(false))
  }, [gw])

  const enabled = gallery?.enabled ?? false
  const active = gallery?.active ?? ''

  // Rank by the signals companiondex gives us — active, then installed, then curated
  // (its official set), then the rest — and hide the clawd placeholders.
  const view = useMemo(() => {
    const companions = (gallery?.companions ?? []).filter(p => !/^clawd(-|$)/i.test(p.slug))
    const needle = query.trim().toLowerCase()

    const matched = needle
      ? companions.filter(p => p.slug.toLowerCase().includes(needle) || p.displayName.toLowerCase().includes(needle))
      : companions

    const rank = (p: GalleryCompanion) => (enabled && p.slug === active ? 4 : 0) + (p.installed ? 2 : 0) + (p.curated ? 1 : 0)

    return [...matched].sort((a, b) => rank(b) - rank(a))
  }, [gallery, query, enabled, active])

  const adopt = (slug: string) => {
    setBusy(true)
    setErr('')
    gw.request('companion.select', { slug })
      .then(() => onClose())
      .catch((e: unknown) => {
        setErr(rpcErrorMessage(e))
        setBusy(false)
      })
  }

  useInput((input, key) => {
    if (busy) {
      return
    }

    if (key.escape) {
      return onClose()
    }

    if (key.upArrow) {
      return setIdx(i => Math.max(0, i - 1))
    }

    if (key.downArrow) {
      return setIdx(i => Math.min(view.length - 1, i + 1))
    }

    if (key.return) {
      const companion = view[idx]

      return companion ? adopt(companion.slug) : undefined
    }

    if (key.backspace || key.delete) {
      setQuery(q => q.slice(0, -1))

      return setIdx(0)
    }

    // Printable char → extend the filter (ignore control/chorded keys).
    if (input && input.length === 1 && input >= ' ' && !key.ctrl && !key.meta) {
      setQuery(q => q + input)
      setIdx(0)
    }
  })

  if (loading) {
    return <Text color={t.color.muted}>{P.loading}</Text>
  }

  if (err && !gallery) {
    return (
      <Box flexDirection="column" width={width}>
        <Text color={t.color.label}>{C.error(err)}</Text>
        <OverlayHint t={t}>{P.escCancelHint}</OverlayHint>
      </Box>
    )
  }

  const { items, offset } = windowItems(view, idx, VISIBLE)

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent}>
        {P.title}
      </Text>

      <Text color={t.color.muted} wrap="truncate-end">
        {query ? C.filter(query) : C.typeToFilter} ·{' '}
        {view.length === 1 ? P.companionCountOne(1) : P.companionCountOther(view.length)}
      </Text>

      {offset > 0 && <Text color={t.color.muted}>{C.moreAbove(offset)}</Text>}

      {view.length === 0 ? (
        <Text color={t.color.muted}>{query ? P.noMatches(query) : P.noneAvailable}</Text>
      ) : (
        items.map((companion, i) => {
          const at = offset + i === idx
          const isActive = enabled && companion.slug === active
          const mark = isActive ? '●' : companion.installed ? '✓' : ' '
          const tag = companion.installed ? '' : companion.curated ? P.officialTag : ''

          return (
            <Text color={t.color.muted} {...chipRowProps(t, at)} key={companion.slug} wrap="truncate-end">
              {at ? '▸ ' : '  '}
              {mark} {companion.displayName}
              <Text color={at ? t.color.accent : t.color.muted}>
                {' '}
                ({companion.slug}
                {tag})
              </Text>
            </Text>
          )
        })
      )}

      {offset + VISIBLE < view.length && (
        <Text color={t.color.muted}>{C.moreBelow(view.length - offset - VISIBLE)}</Text>
      )}

      {err ? <Text color={t.color.label}>{C.error(err)}</Text> : null}
      {busy ? <Text color={t.color.accent}>{P.adopting}</Text> : null}

      <OverlayHint t={t}>{P.hint}</OverlayHint>
    </Box>
  )
}

interface CompanionPickerProps {
  gw: GatewayClient
  maxWidth?: number
  onClose: () => void
  t: Theme
}
