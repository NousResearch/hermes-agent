import { Box, Text, useInput, useStdout } from '@hermes/ink'
import type { SkinBranding } from '@hermes/shared/skin'
import { useEffect, useMemo, useRef, useState } from 'react'

import {
  applySkinPreview,
  getPersistedSkinRevision,
  restorePersistedSkin
} from '../app/createGatewayEventHandler.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { GatewaySkin } from '../gatewayTypes.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { OverlayHint, windowItems } from './overlayControls.js'
import { chipRowProps, clampOverlayWidth } from './overlayPrimitives.js'

const VISIBLE = 12
const MIN_WIDTH = 48
const MAX_WIDTH = 100
export const SKIN_PREVIEW_DEBOUNCE_MS = 120

export interface SkinOption {
  description?: string
  name: string
  source?: string
}

interface SkinOptionsResponse {
  active: string
  active_skin: GatewaySkin
  skins: SkinOption[]
}

export const filterSkinOptions = (skins: SkinOption[], query: string): SkinOption[] => {
  const needle = query.trim().toLowerCase()

  return needle
    ? skins.filter(
        skin => skin.name.toLowerCase().includes(needle) || (skin.description ?? '').toLowerCase().includes(needle)
      )
    : skins
}

export const skinAgentLabel = (branding: SkinBranding): string =>
  `agent: ${branding.agent_name?.trim() || 'Hermes Agent'}`

export function SkinPicker({ gw, maxWidth, onClose, t }: SkinPickerProps) {
  const [options, setOptions] = useState<SkinOptionsResponse | null>(null)
  const [query, setQuery] = useState('')
  const [idx, setIdx] = useState(0)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(true)
  const previewGeneration = useRef(0)

  const { stdout } = useStdout()
  const preferredWidth = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))
  const width = clampOverlayWidth(preferredWidth, maxWidth)

  useEffect(() => {
    let current = true

    gw.request<SkinOptionsResponse>('skin.options')
      .then(response => {
        if (!current) {
          return
        }

        setOptions(response)
        setIdx(Math.max(0, response.skins.findIndex(skin => skin.name === response.active)))
        setErr('')
      })
      .catch((error: unknown) => current && setErr(rpcErrorMessage(error)))
      .finally(() => current && setLoading(false))

    return () => {
      current = false
    }
  }, [gw])

  const view = useMemo(() => filterSkinOptions(options?.skins ?? [], query), [options, query])
  const selected = view[idx]

  useEffect(() => {
    if (!options || !selected) {
      return
    }

    const generation = ++previewGeneration.current
    const persistedRevision = getPersistedSkinRevision()

    const timer = setTimeout(() => {
      gw.request<GatewaySkin>('skin.preview', { name: selected.name })
        .then(
          skin =>
            generation === previewGeneration.current &&
            persistedRevision === getPersistedSkinRevision() &&
            applySkinPreview(skin)
        )
        .catch(
          (error: unknown) =>
            generation === previewGeneration.current &&
            persistedRevision === getPersistedSkinRevision() &&
            setErr(rpcErrorMessage(error))
        )
    }, SKIN_PREVIEW_DEBOUNCE_MS)

    return () => {
      previewGeneration.current += 1
      clearTimeout(timer)
    }
  }, [gw, options, selected])

  const cancel = () => {
    previewGeneration.current += 1
    restorePersistedSkin(options?.active_skin)
    onClose()
  }

  const select = (name: string) => {
    setBusy(true)
    setErr('')
    gw.request<{ value?: string }>('config.set', { key: 'skin', value: name })
      .then(response => {
        if (!response.value) {
          throw new Error('skin selection was not saved')
        }

        previewGeneration.current += 1
        onClose()
      })
      .catch((error: unknown) => {
        setErr(rpcErrorMessage(error))
        setBusy(false)
      })
  }

  useInput((input, key) => {
    if (busy) {
      return
    }

    if (key.escape || (key.ctrl && (input.toLowerCase() === 'c' || input === '\u0003'))) {
      return cancel()
    }

    if (key.upArrow) {
      return setIdx(current => Math.max(0, current - 1))
    }

    if (key.downArrow) {
      return setIdx(current => Math.min(Math.max(0, view.length - 1), current + 1))
    }

    if (key.return) {
      return selected ? select(selected.name) : undefined
    }

    if (key.backspace || key.delete) {
      setQuery(current => current.slice(0, -1))

      return setIdx(0)
    }

    if (input && input.length === 1 && input >= ' ' && !key.ctrl && !key.meta) {
      setQuery(current => current + input)
      setIdx(0)
    }
  })

  if (loading) {
    return <Text color={t.color.muted}>loading skins…</Text>
  }

  if (err && !options) {
    return (
      <Box flexDirection="column" width={width}>
        <Text color={t.color.label}>error: {err}</Text>
        <OverlayHint t={t}>Esc cancel</OverlayHint>
      </Box>
    )
  }

  const { items, offset } = windowItems(view, idx, VISIBLE)
  const total = options?.skins.length ?? 0

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent}>
        Skins
      </Text>
      <Text color={t.color.muted} wrap="truncate-end">
        {query ? `filter: ${query}` : 'type to filter'} · showing {view.length} of {total} ·{' '}
        {skinAgentLabel({ agent_name: t.brand.name })}
      </Text>

      {offset > 0 && <Text color={t.color.muted}> ↑ {offset} more</Text>}
      {view.length === 0 ? (
        <Text color={t.color.muted}>no skins match "{query}"</Text>
      ) : (
        items.map((skin, itemIndex) => {
          const at = offset + itemIndex === idx
          const active = skin.name === options?.active

          return (
            <Text color={t.color.muted} {...chipRowProps(t, at)} key={skin.name} wrap="truncate-end">
              {at ? '▸ ' : '  '}
              {active ? '●' : ' '} {skin.name}
              <Text color={at ? t.color.accent : t.color.muted}>
                {' '}
                · {skin.source ?? 'unknown'}
                {skin.description ? ` · ${skin.description}` : ''}
              </Text>
            </Text>
          )
        })
      )}
      {offset + VISIBLE < view.length && <Text color={t.color.muted}> ↓ {view.length - offset - VISIBLE} more</Text>}
      {err ? <Text color={t.color.label}>error: {err}</Text> : null}
      {busy ? <Text color={t.color.accent}>saving…</Text> : null}
      <OverlayHint t={t}>↑/↓ preview · Enter select · type to filter · Esc cancel</OverlayHint>
    </Box>
  )
}

interface SkinPickerProps {
  gw: GatewayClient
  maxWidth?: number
  onClose: () => void
  t: Theme
}
