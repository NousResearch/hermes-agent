import { Box, Text, useInput } from '@hermes/ink'
import { useEffect, useMemo, useState } from 'react'

import { providerDisplayNames } from '../domain/providers.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { ModelOptionProvider, ModelOptionsResponse } from '../gatewayTypes.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

const VISIBLE = 12

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

  useInput((ch, key) => {
    if (key.escape) {
      if (stage === 'model') {
        setStage('provider')
        setModelIdx(0)

        return
      }

      onCancel()

      return
    }

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
        <Text color={t.color.dim}>Esc to cancel</Text>
      </Box>
    )
  }

  if (!providers.length) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.dim}>no authenticated providers</Text>
        <Text color={t.color.dim}>Esc to cancel</Text>
      </Box>
    )
  }

  if (stage === 'provider') {
    const rows = providers.map(
      (p, i) => `${p.is_current ? '*' : ' '} ${names[i]} · ${p.total_models ?? p.models?.length ?? 0} models`
    )

    const { items, off } = visibleItems(rows, providerIdx)

    return (
      <Box flexDirection="column">
        <Text bold color={t.color.amber}>
          Select Provider
        </Text>

        <Text color={t.color.dim}>Current model: {currentModel || '(unknown)'}</Text>
        {provider?.warning ? <Text color={t.color.label}>warning: {provider.warning}</Text> : null}
        {off > 0 && <Text color={t.color.dim}> ↑ {off} more</Text>}

        {items.map((row, i) => {
          const idx = off + i

          return (
            <Text color={providerIdx === idx ? t.color.cornsilk : t.color.dim} key={row}>
              {providerIdx === idx ? '▸ ' : '  '}
              {i + 1}. {row}
            </Text>
          )
        })}

        {off + VISIBLE < rows.length && <Text color={t.color.dim}> ↓ {rows.length - off - VISIBLE} more</Text>}
        <Text color={t.color.dim}>persist: {persistGlobal ? 'global' : 'session'} · g toggle</Text>
        <Text color={t.color.dim}>↑/↓ select · Enter choose · 1-9,0 quick · Esc cancel</Text>
      </Box>
    )
  }

  const { items, off } = visibleItems(models, modelIdx)

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.amber}>
        Select Model
      </Text>

      <Text color={t.color.dim}>{names[providerIdx] || '(unknown provider)'}</Text>
      {!models.length ? <Text color={t.color.dim}>no models listed for this provider</Text> : null}
      {provider?.warning ? <Text color={t.color.label}>warning: {provider.warning}</Text> : null}
      {off > 0 && <Text color={t.color.dim}> ↑ {off} more</Text>}

      {items.map((row, i) => {
        const idx = off + i

        return (
          <Text color={modelIdx === idx ? t.color.cornsilk : t.color.dim} key={row}>
            {modelIdx === idx ? '▸ ' : '  '}
            {i + 1}. {row}
          </Text>
        )
      })}

      {off + VISIBLE < models.length && <Text color={t.color.dim}> ↓ {models.length - off - VISIBLE} more</Text>}
      <Text color={t.color.dim}>persist: {persistGlobal ? 'global' : 'session'} · g toggle</Text>
      <Text color={t.color.dim}>
        {models.length ? '↑/↓ select · Enter switch · 1-9,0 quick · Esc back' : 'Enter/Esc back'}
      </Text>
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
