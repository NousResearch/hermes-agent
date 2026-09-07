import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useMemo, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { OverlayHint, windowItems } from './overlayControls.js'
import { chipRowProps, clampOverlayWidth } from './overlayPrimitives.js'

const VISIBLE = 10
const MIN_WIDTH = 40
const MAX_WIDTH = 90

interface GalleryPet {
  slug: string
  displayName: string
  installed: boolean
  curated?: boolean
}

interface Gallery {
  enabled: boolean
  active: string
  pets: GalleryPet[]
}

/**
 * Interactive petdex picker overlay. Pulls the gallery via `pet.gallery`,
 * filters as you type, and adopts the highlighted pet with `pet.select`
 * (install-on-demand). The mascot lights up live once `usePet` next polls —
 * no restart. This is the interactive sibling of the text `/pet <slug>` path.
 */
export function PetPicker({ gw, maxWidth, onClose, t }: PetPickerProps) {
  const [gallery, setGallery] = useState<Gallery | null>(null)
  const [query, setQuery] = useState('')
  const [idx, setIdx] = useState(0)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(true)

  const { stdout } = useStdout()
  // Optional maxWidth lets grid layouts hand the picker its cell budget.
  const preferredWidth = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))
  const width = clampOverlayWidth(preferredWidth, maxWidth)

  useEffect(() => {
    gw.request<Gallery>('pet.gallery')
      .then(r => {
        setGallery(r)
        setErr('')
      })
      .catch((e: unknown) => setErr(rpcErrorMessage(e)))
      .finally(() => setLoading(false))
  }, [gw])

  const enabled = gallery?.enabled ?? false
  const active = gallery?.active ?? ''

  // Rank by the signals petdex gives us — active, then installed, then curated
  // (its official set), then the rest — and hide the clawd placeholders.
  const view = useMemo(() => {
    const pets = (gallery?.pets ?? []).filter(p => !/^clawd(-|$)/i.test(p.slug))
    const needle = query.trim().toLowerCase()

    const matched = needle
      ? pets.filter(p => p.slug.toLowerCase().includes(needle) || p.displayName.toLowerCase().includes(needle))
      : pets

    const rank = (p: GalleryPet) => (enabled && p.slug === active ? 4 : 0) + (p.installed ? 2 : 0) + (p.curated ? 1 : 0)

    return [...matched].sort((a, b) => rank(b) - rank(a))
  }, [gallery, query, enabled, active])

  const adopt = (slug: string) => {
    setBusy(true)
    setErr('')
    gw.request('pet.select', { slug })
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
      const pet = view[idx]

      return pet ? adopt(pet.slug) : undefined
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
    return <Text color={t.color.muted}>loading pets…</Text>
  }

  if (err && !gallery) {
    return (
      <Box flexDirection="column" width={width}>
        <Text color={t.color.label}>error: {err}</Text>
        <OverlayHint t={t}>Esc cancel</OverlayHint>
      </Box>
    )
  }

  const { items, offset } = windowItems(view, idx, VISIBLE)

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent}>
        Pets
      </Text>

      <Text color={t.color.muted} wrap="truncate-end">
        {query ? `filter: ${query}` : 'type to filter'} · {view.length} pet{view.length === 1 ? '' : 's'}
      </Text>

      {offset > 0 && <Text color={t.color.muted}> ↑ {offset} more</Text>}

      {view.length === 0 ? (
        <Text color={t.color.muted}>{query ? `no pets match "${query}"` : 'no pets available'}</Text>
      ) : (
        items.map((pet, i) => {
          const at = offset + i === idx
          const isActive = enabled && pet.slug === active
          const mark = isActive ? '●' : pet.installed ? '✓' : ' '
          const tag = pet.installed ? '' : pet.curated ? ' · official' : ''

          return (
            <Text color={t.color.muted} {...chipRowProps(t, at)} key={pet.slug} wrap="truncate-end">
              {at ? '▸ ' : '  '}
              {mark} {pet.displayName}
              <Text color={at ? t.color.accent : t.color.muted}>
                {' '}
                ({pet.slug}
                {tag})
              </Text>
            </Text>
          )
        })
      )}

      {offset + VISIBLE < view.length && <Text color={t.color.muted}> ↓ {view.length - offset - VISIBLE} more</Text>}

      {err ? <Text color={t.color.label}>error: {err}</Text> : null}
      {busy ? <Text color={t.color.accent}>adopting…</Text> : null}

      <OverlayHint t={t}>↑/↓ select · Enter adopt · type to filter · Esc cancel</OverlayHint>
    </Box>
  )
}

interface PetPickerProps {
  gw: GatewayClient
  maxWidth?: number
  onClose: () => void
  t: Theme
}
