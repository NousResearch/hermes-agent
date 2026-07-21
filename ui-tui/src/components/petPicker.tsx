import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useMemo, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import { fuzzyRank } from '../lib/fuzzy.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { clampIndex, OverlayHint, windowItems } from './overlayControls.js'

const VISIBLE = 10
const MIN_WIDTH = 40
const MAX_WIDTH = 90

export interface GalleryPet {
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

export function rankPets(pets: readonly GalleryPet[], query: string, enabled: boolean, active: string): GalleryPet[] {
  const available = pets.filter(pet => !/^clawd(-|$)/i.test(pet.slug))

  const statusRank = (pet: GalleryPet) =>
    (enabled && pet.slug === active ? 4 : 0) + (pet.installed ? 2 : 0) + (pet.curated ? 1 : 0)

  const ranked = [...available].sort((a, b) => statusRank(b) - statusRank(a))

  return query.trim()
    ? fuzzyRank(ranked, query, pet => `${pet.displayName} ${pet.slug}`).map(result => result.item)
    : ranked
}

/**
 * Interactive petdex picker overlay. Pulls the gallery via `pet.gallery`,
 * filters as you type, and adopts the highlighted pet with `pet.select`
 * (install-on-demand). The mascot lights up live once `usePet` next polls —
 * no restart. This is the interactive sibling of the text `/pet <slug>` path.
 */
export function PetPicker({ gw, onClose, t }: PetPickerProps) {
  const [gallery, setGallery] = useState<Gallery | null>(null)
  const [query, setQuery] = useState('')
  const [idx, setIdx] = useState(0)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(true)

  const { stdout } = useStdout()
  const width = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))

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

  // Rank by fuzzy match quality while retaining the petdex status priority
  // (active, installed, curated) as the stable tie-break order.
  const view = useMemo(() => rankPets(gallery?.pets ?? [], query, enabled, active), [gallery, query, enabled, active])
  const clampedIdx = clampIndex(idx, view.length)

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
      if (query) {
        setQuery('')
        setIdx(0)

        return
      }

      return onClose()
    }

    if (key.upArrow) {
      return setIdx(i => Math.max(0, i - 1))
    }

    if (key.downArrow) {
      return setIdx(i => clampIndex(i + 1, view.length))
    }

    if (key.return) {
      const pet = view[clampedIdx]

      return pet ? adopt(pet.slug) : undefined
    }

    if (key.backspace || key.delete) {
      setQuery(q => q.slice(0, -1))

      return setIdx(0)
    }

    if (key.ctrl && (input.toLowerCase() === 'u' || input === String.fromCharCode(21))) {
      setQuery('')

      return setIdx(0)
    }

    // Printable char → extend the filter (ignore control/chorded keys).
    if (input && input.length === 1 && input >= ' ' && !key.ctrl && !key.meta) {
      if (input === ' ' && !query) {
        return
      }

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

  const { items, offset } = windowItems(view, clampedIdx, VISIBLE)

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
          const at = offset + i === clampedIdx
          const isActive = enabled && pet.slug === active
          const mark = isActive ? '●' : pet.installed ? '✓' : ' '
          const tag = pet.installed ? '' : pet.curated ? ' · official' : ''

          return (
            <Text bold={at} color={at ? t.color.accent : t.color.muted} inverse={at} key={pet.slug} wrap="truncate-end">
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

      <OverlayHint t={t}>
        {query
          ? '↑/↓ select · Enter adopt · Backspace edit · Ctrl+U clear · Esc clear filter'
          : '↑/↓ select · Enter adopt · type to filter · Esc cancel'}
      </OverlayHint>
    </Box>
  )
}

interface PetPickerProps {
  gw: GatewayClient
  onClose: () => void
  t: Theme
}
