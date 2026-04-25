import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useMemo, useState } from 'react'

import { providerDisplayNames } from '../domain/providers.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { ModelOptionProvider, ModelOptionsResponse } from '../gatewayTypes.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { OverlayControls, useOverlayKeys } from './overlayControls.js'

const VISIBLE = 12
const MIN_WIDTH = 40
const MAX_WIDTH = 90

const pageOffset = (count: number, sel: number) => Math.max(0, Math.min(sel - Math.floor(VISIBLE / 2), count - VISIBLE))

const visibleItems = (items: string[], sel: number) => {
  const off = pageOffset(items.length, sel)

  return { items: items.slice(off, off + VISIBLE), off }
}

export function ModelPicker({ gw, onCancel, onSelect, sessionId, t }: ModelPickerProps) {
  const [providers, setProviders] = useState<ModelOptionProvider[]>([])
  const [currentModel, setCurrentModel] = useState('')
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(true)
  const [persistGlobal, setPersistGlobal] = useState(false)
  const [providerIdx, setProviderIdx] = useState(0)
  const [modelIdx, setModelIdx] = useState(0)
  const [stage, setStage] = useState<'model' | 'provider'>('provider')

  const { stdout } = useStdout()
  // Pin the picker to a stable width so the FloatBox parent (which shrinks-
  // to-fit with alignSelf="flex-start") doesn't resize as long provider /
  // model names scroll into view, and so `wrap="truncate-end"` on each row
  // has an actual constraint to truncate against.
  const width = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))

  useEffect(() => {
    gw.request<ModelOptionsResponse>('model.options', sessionId ? { session_id: sessionId } : {})
      .then(raw => {
        const r = asRpcResult<ModelOptionsResponse>(raw)

        if (!r) {
          setErr('invalid response: model.options')
          setLoading(false)

          return
        }

        const next = r.providers ?? []
        setProviders(next)
        setCurrentModel(String(r.model ?? ''))
        setProviderIdx(
          Math.max(
            0,
            next.findIndex(p => p.is_current)
          )
        )
        setModelIdx(0)
        setErr('')
        setLoading(false)
      })
      .catch((e: unknown) => {
        setErr(rpcErrorMessage(e))
        setLoading(false)
      })
  }, [gw, sessionId])

  const provider = providers[providerIdx]
  const models = provider?.models ?? []
  const names = useMemo(() => providerDisplayNames(providers), [providers])

  const back = () => {
    if (stage === 'model') {
      setStage('provider')
      setModelIdx(0)

      return
    }

    onCancel()
  }

  useOverlayKeys({ onBack: back, onClose: onCancel })

  useInput((ch, key) => {
    const count = stage === 'provider' ? providers.length : models.length
    const sel = stage === 'provider' ? providerIdx : modelIdx
    const setSel = stage === 'provider' ? setProviderIdx : setModelIdx

    if (key.upArrow && sel > 0) {
      setSel(v => v - 1)

      return
    }

    if (key.downArrow && sel < count - 1) {
      setSel(v => v + 1)

      return
    }

    if (key.return) {
      if (stage === 'provider') {
        if (!provider) {
          return
        }

        setStage('model')
        setModelIdx(0)

        return
      }

      const model = models[modelIdx]

      if (provider && model) {
        onSelect(`${model} --provider ${provider.slug}${persistGlobal ? ' --global' : ''}`)
      } else {
        setStage('provider')
      }

      return
    }

    if (ch.toLowerCase() === 'g') {
      setPersistGlobal(v => !v)

      return
    }

    const n = ch === '0' ? 10 : parseInt(ch, 10)

    if (!Number.isNaN(n) && n >= 1 && n <= Math.min(10, count)) {
      const off = pageOffset(count, sel)

      if (stage === 'provider') {
        const next = off + n - 1

        if (providers[next]) {
          setProviderIdx(next)
        }
      } else if (provider && models[off + n - 1]) {
        onSelect(`${models[off + n - 1]} --provider ${provider.slug}${persistGlobal ? ' --global' : ''}`)
      }
    }
  })

  if (loading) {
    return <Text color={t.color.dim}>loading models…</Text>
  }

  if (err) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.label}>error: {err}</Text>
        <OverlayControls t={t}>Esc/q cancel</OverlayControls>
      </Box>
    )
  }

  if (!providers.length) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.dim}>no authenticated providers</Text>
        <OverlayControls t={t}>Esc/q cancel</OverlayControls>
      </Box>
    )
  }

  if (stage === 'provider') {
    const rows = providers.map(
      (p, i) => `${p.is_current ? '*' : ' '} ${names[i]} · ${p.total_models ?? p.models?.length ?? 0} models`
    )

    const { items, off } = visibleItems(rows, providerIdx)

    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.amber} wrap="truncate-end">
          Select Provider
        </Text>

        <Text color={t.color.dim} wrap="truncate-end">
          Current model: {currentModel || '(unknown)'}
        </Text>
        <Text color={t.color.label} wrap="truncate-end">
          {provider?.warning ? `warning: ${provider.warning}` : ' '}
        </Text>
        <Text color={t.color.dim} wrap="truncate-end">
          {off > 0 ? ` ↑ ${off} more` : ' '}
        </Text>

        {Array.from({ length: VISIBLE }, (_, i) => {
          const row = items[i]
          const idx = off + i

          return row ? (
            <Text
              bold={providerIdx === idx}
              color={providerIdx === idx ? t.color.amber : t.color.dim}
              inverse={providerIdx === idx}
              key={providers[idx]?.slug ?? `row-${idx}`}
              wrap="truncate-end"
            >
              {providerIdx === idx ? '▸ ' : '  '}
              {i + 1}. {row}
            </Text>
          ) : (
            <Text color={t.color.dim} key={`pad-${i}`} wrap="truncate-end">
              {' '}
            </Text>
          )
        })}

        <Text color={t.color.dim} wrap="truncate-end">
          {off + VISIBLE < rows.length ? ` ↓ ${rows.length - off - VISIBLE} more` : ' '}
        </Text>

        <Text color={t.color.dim} wrap="truncate-end">
          persist: {persistGlobal ? 'global' : 'session'} · g toggle
        </Text>
        <OverlayControls t={t}>↑/↓ select · Enter choose · 1-9,0 quick · Esc/q cancel</OverlayControls>
      </Box>
    )
  }

  const { items, off } = visibleItems(models, modelIdx)

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.amber} wrap="truncate-end">
        Select Model
      </Text>

      <Text color={t.color.dim} wrap="truncate-end">
        {names[providerIdx] || '(unknown provider)'}
      </Text>
      <Text color={t.color.label} wrap="truncate-end">
        {provider?.warning ? `warning: ${provider.warning}` : ' '}
      </Text>
      <Text color={t.color.dim} wrap="truncate-end">
        {off > 0 ? ` ↑ ${off} more` : ' '}
      </Text>

      {Array.from({ length: VISIBLE }, (_, i) => {
        const row = items[i]
        const idx = off + i

        if (!row) {
          return !models.length && i === 0 ? (
            <Text color={t.color.dim} key="empty" wrap="truncate-end">
              no models listed for this provider
            </Text>
          ) : (
            <Text color={t.color.dim} key={`pad-${i}`} wrap="truncate-end">
              {' '}
            </Text>
          )
        }

        return (
          <Text
            bold={modelIdx === idx}
            color={modelIdx === idx ? t.color.amber : t.color.dim}
            inverse={modelIdx === idx}
            key={`${provider?.slug ?? 'prov'}:${idx}:${row}`}
            wrap="truncate-end"
          >
            {modelIdx === idx ? '▸ ' : '  '}
            {i + 1}. {row}
          </Text>
        )
      })}

      <Text color={t.color.dim} wrap="truncate-end">
        {off + VISIBLE < models.length ? ` ↓ ${models.length - off - VISIBLE} more` : ' '}
      </Text>

      <Text color={t.color.dim} wrap="truncate-end">
        persist: {persistGlobal ? 'global' : 'session'} · g toggle
      </Text>
      <OverlayControls t={t}>
        {models.length ? '↑/↓ select · Enter switch · 1-9,0 quick · Esc back · q close' : 'Enter/Esc back · q close'}
      </OverlayControls>
    </Box>
  )
}

interface ModelPickerProps {
  gw: GatewayClient
  onCancel: () => void
  onSelect: (value: string) => void
  sessionId: string | null
  t: Theme
}
