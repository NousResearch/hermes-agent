import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useMemo, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import { useI18n } from '../i18n/index.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { OverlayHint, windowItems } from './overlayControls.js'

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
export function PetPicker({ gw, onClose, t }: PetPickerProps) {
  const { t: ti } = useI18n()
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
    return <Text color={t.color.muted}>{ti('pet.loading')}</Text>
  }

  if (err && !gallery) {
    return (
      <Box flexDirection="column" width={width}>
        <Text color={t.color.label}>{ti('common.errorWithMessage', { message: err })}</Text>
        <OverlayHint t={t}>{ti('common.escCancel')}</OverlayHint>
      </Box>
    )
  }

  const { items, offset } = windowItems(view, idx, VISIBLE)

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent}>
        {ti('pet.title')}
      </Text>

      <Text color={t.color.muted} wrap="truncate-end">
        {query ? ti('pet.filter', { query }) : ti('pet.typeToFilter')} · {ti('pet.count', { count: view.length })}
      </Text>

      {offset > 0 && <Text color={t.color.muted}>{ti('sys.moreAbove', { count: offset })}</Text>}

      {view.length === 0 ? (
        <Text color={t.color.muted}>
          {query ? ti('pet.noMatch', { query }) : ti('pet.noneAvailable')}
        </Text>
      ) : (
        items.map((pet, i) => {
          const at = offset + i === idx
          const isActive = enabled && pet.slug === active
          const mark = isActive ? '●' : pet.installed ? '✓' : ' '
          const tag = pet.installed ? '' : pet.curated ? ti('pet.officialTag') : ''

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

      {offset + VISIBLE < view.length && (
        <Text color={t.color.muted}>{ti('sys.moreBelow', { count: view.length - offset - VISIBLE })}</Text>
      )}

      {err ? <Text color={t.color.label}>{ti('common.errorWithMessage', { message: err })}</Text> : null}
      {busy ? <Text color={t.color.accent}>{ti('pet.adopting')}</Text> : null}

      <OverlayHint t={t}>{ti('pet.hint')}</OverlayHint>
    </Box>
  )
}

interface PetPickerProps {
  gw: GatewayClient
  onClose: () => void
  t: Theme
}
