import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { fuzzyRank } from '@hermes/shared/fuzzy'
import type { ModelOptionProvider, ModelOptionsResult } from '@hermes/shared/gateway-events'
import { modelSearchText } from '@hermes/shared/model-search-text'
import { useEffect, useMemo, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import {
  cachedModelOptions,
  invalidateModelOptions,
  rememberModelOptions
} from '../lib/modelOptionsCache.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { modelPickerCommand } from './modelPicker.js'
import { OverlayHint } from './overlayControls.js'
import { chipRowProps, clampOverlayWidth } from './overlayPrimitives.js'

const VISIBLE = 14
const MIN_WIDTH = 40
const MAX_WIDTH = 96

export interface SessionModelHopRow {
  currentProvider: boolean
  model: string
  providerName: string
  providerSlug: string
  selector: string
}

export const buildSessionModelHopRows = (providers: readonly ModelOptionProvider[]): SessionModelHopRow[] => {
  const rows: SessionModelHopRow[] = []

  for (const provider of providers) {
    if (provider.authenticated === false) {
      continue
    }

    const providerSlug = String(provider.slug || '').trim()
    const providerName = String(provider.name || providerSlug).trim() || providerSlug

    if (!providerSlug) {
      continue
    }

    for (const model of provider.models ?? []) {
      const id = String(model).trim()
      if (!id) {
        continue
      }

      rows.push({
        currentProvider: provider.is_current === true,
        model: id,
        providerName,
        providerSlug,
        selector: `${providerSlug}/${id}`
      })
    }
  }

  return rows
}

export const filterSessionModelHopRows = (
  rows: readonly SessionModelHopRow[],
  query: string
): SessionModelHopRow[] => {
  const trimmed = query.trim()

  if (!trimmed) {
    return [...rows]
  }

  return fuzzyRank(
    [...rows],
    trimmed,
    row => `${row.selector} ${row.providerName} ${modelSearchText(row.model)}`
  ).map(match => match.item)
}

export const sessionModelHopCurrentIndex = (
  rows: readonly SessionModelHopRow[],
  currentModel: string
): number => {
  const current = currentModel.trim()
  if (!current) {
    return -1
  }

  const exact = rows.findIndex(row => row.selector === current)
  if (exact >= 0) {
    return exact
  }

  const currentProviderMatch = rows.findIndex(row => row.currentProvider && row.model === current)
  if (currentProviderMatch >= 0) {
    return currentProviderMatch
  }

  const modelOnly = rows.findIndex(row => row.model === current)
  return modelOnly
}

export function SessionModelHop({
  gw,
  initialRefresh = false,
  maxWidth,
  onCancel,
  onOpenProviderPicker,
  onSelect,
  sessionId,
  t
}: SessionModelHopProps) {
  const [providers, setProviders] = useState<ModelOptionProvider[]>([])
  const [currentModel, setCurrentModel] = useState('')
  const [filter, setFilter] = useState('')
  const [sel, setSel] = useState(0)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(true)
  const { stdout } = useStdout()
  const preferredWidth = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))
  const width = clampOverlayWidth(preferredWidth, maxWidth)

  useEffect(() => {
    let active = true

    if (initialRefresh) {
      invalidateModelOptions(sessionId)
    }

    const cached = initialRefresh ? null : cachedModelOptions(sessionId)

    const apply = (result: ModelOptionsResult) => {
      if (!active) {
        return
      }

      setProviders(result.providers ?? [])
      setCurrentModel(String(result.model ?? ''))
      setErr('')
      setLoading(false)
    }

    if (cached) {
      apply(cached)

      return () => {
        active = false
      }
    }

    gw.request<ModelOptionsResult>('model.options', {
      ...(sessionId ? { session_id: sessionId } : {}),
      ...(initialRefresh ? { refresh: true } : {}),
      include_unconfigured: false
    })
      .then(raw => {
        const result = asRpcResult<ModelOptionsResult>(raw)

        if (!result) {
          if (active) {
            setErr('invalid response: model.options')
            setLoading(false)
          }
          return
        }

        rememberModelOptions(sessionId, result)
        apply(result)
      })
      .catch((e: unknown) => {
        if (active) {
          setErr(rpcErrorMessage(e))
          setLoading(false)
        }
      })

    return () => {
      active = false
    }
  }, [gw, initialRefresh, sessionId])

  const rows = useMemo(() => buildSessionModelHopRows(providers), [providers])
  const filtered = useMemo(() => filterSessionModelHopRows(rows, filter), [rows, filter])

  useEffect(() => {
    if (!filter && rows.length) {
      setSel(Math.max(0, sessionModelHopCurrentIndex(rows, currentModel)))
    }
  }, [currentModel, filter, rows])

  useEffect(() => {
    if (sel >= filtered.length && filtered.length > 0) {
      setSel(0)
    }
  }, [filtered.length, sel])

  useInput((ch, key) => {
    if (loading) {
      if (key.escape) {
        onCancel()
        return
      }

      if (key.backspace || key.delete) {
        setFilter(v => v.slice(0, -1))
        return
      }

      if (key.ctrl && ch === 'u') {
        setFilter('')
        return
      }

      if (ch && !key.ctrl && !key.meta && ch.length === 1 && ch >= ' ') {
        setFilter(v => v + ch)
      }
      return
    }

    if (err) {
      if (key.escape || ch === 'q') {
        onCancel()
      }
      return
    }

    if (rows.length === 0) {
      if (key.escape || ch === 'q') {
        onCancel()
      } else if (key.return) {
        onOpenProviderPicker()
      }
      return
    }

    if (key.escape) {
      if (filter) {
        setFilter('')
        setSel(Math.max(0, sessionModelHopCurrentIndex(rows, currentModel)))
      } else {
        onCancel()
      }
      return
    }


    if (key.upArrow && sel > 0) {
      setSel(v => v - 1)
      return
    }

    if (key.downArrow && sel < filtered.length - 1) {
      setSel(v => v + 1)
      return
    }

    if (key.return) {
      const row = filtered[sel]

      if (row) {
        invalidateModelOptions(sessionId)
        onSelect(modelPickerCommand(row.model, row.providerSlug, false))
      }
      return
    }

    if (key.backspace || key.delete) {
      setFilter(v => v.slice(0, -1))
      setSel(0)
      return
    }

    if (key.ctrl && ch === 'u') {
      setFilter('')
      setSel(0)
      return
    }

    if (ch && !key.ctrl && !key.meta && ch.length === 1 && ch >= ' ') {
      setFilter(v => v + ch)
      setSel(0)
    }
  })

  if (loading) {
    return <Text color={t.color.muted}>loading session models…</Text>
  }

  if (err) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.label}>error: {err}</Text>
        <OverlayHint t={t}>Esc/q cancel</OverlayHint>
      </Box>
    )
  }

  if (!rows.length) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.muted}>no configured models available</Text>
        <OverlayHint t={t}>Enter provider setup · Esc/q cancel</OverlayHint>
      </Box>
    )
  }

  const start = Math.max(0, Math.min(sel - Math.floor(VISIBLE / 2), Math.max(0, filtered.length - VISIBLE)))
  const shown = filtered.slice(start, start + VISIBLE)
  const currentIndex = sessionModelHopCurrentIndex(rows, currentModel)
  const currentSelector = currentIndex >= 0 ? rows[currentIndex]?.selector : undefined

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent} wrap="truncate-end">
        Switch model · this session
      </Text>
      <Text color={t.color.muted} wrap="truncate-end">
        Current: {currentModel || 'unknown'} · type to filter provider/model
      </Text>
      <Text color={t.color.muted} wrap="truncate-end">
        Filter: {filter || '—'}
      </Text>

      <Box flexDirection="column" marginTop={1}>
        {shown.map((row, i) => {
          const index = start + i
          const active = index === sel
          const current = row.selector === currentSelector

          return (
            <Text key={row.selector} {...chipRowProps(t, active)} wrap="truncate-end">
              {active ? '▸ ' : '  '}
              {current ? '* ' : '  '}
              {row.selector}
              <Text color={active ? undefined : t.color.muted}> · {row.providerName}</Text>
            </Text>
          )
        })}
        {filtered.length === 0 ? <Text color={t.color.muted}>  no matches</Text> : null}
      </Box>

      <OverlayHint t={t}>
        ↑↓ move · Enter switch session · type filter · Ctrl+U clear · Esc {filter ? 'clear' : 'cancel'}
      </OverlayHint>
    </Box>
  )
}

interface SessionModelHopProps {
  gw: GatewayClient
  initialRefresh?: boolean
  maxWidth?: number
  onCancel: () => void
  onOpenProviderPicker: () => void
  onSelect: (value: string) => void
  sessionId: null | string
  t: Theme
}
