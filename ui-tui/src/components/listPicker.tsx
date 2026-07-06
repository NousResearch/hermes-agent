import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import type { ListPickerConfig, ListPickerItem } from '../app/interfaces.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { OverlayHint, windowItems } from './overlayControls.js'

const VISIBLE = 12
const MIN_WIDTH = 44
const MAX_WIDTH = 96

export interface ListPickerProps {
  config: ListPickerConfig
  gw: GatewayClient
  onCancel: () => void
  onSelect?: (item: ListPickerItem) => void
  t: Theme
}

export function ListPicker({ config, gw, onCancel, onSelect, t }: ListPickerProps) {
  const [items, setItems] = useState<ListPickerItem[]>([])
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(true)
  const [sel, setSel] = useState(0)
  const [actionMsg, setActionMsg] = useState('')
  const [acting, setActing] = useState(false)

  const { stdout } = useStdout()
  const width = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))

  useEffect(() => {
    let cancelled = false

    setLoading(true)
    gw
      .request<Record<string, unknown>>(config.fetchMethod, config.fetchParams ?? {})
      .then(raw => {
        if (cancelled) return
        const r = asRpcResult<Record<string, unknown>>(raw)

        if (!r) {
          setErr('invalid response')
          setLoading(false)
          return
        }

        const mapped = config.mapResponse(r)

        setItems(mapped)
        setErr('')
        setLoading(false)
      })
      .catch((e: unknown) => {
        if (cancelled) return
        setErr(rpcErrorMessage(e))
        setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [config.fetchMethod, config.fetchParams, config.mapResponse, gw])

  useInput((ch, key) => {
    if (acting) return

    if (key.escape || ch === 'q') {
      return onCancel()
    }

    if (key.upArrow && sel > 0) {
      setSel(s => Math.max(0, s - 1))
      return
    }

    if (key.downArrow && sel < items.length - 1) {
      setSel(s => Math.min(items.length - 1, s + 1))
      return
    }

    if (key.return && items[sel]) {
      const item = items[sel]!

      if (config.viewOnly) {
        return onSelect?.(item)
      }

      if (!config.actionMethod) {
        return onSelect?.(item)
      }

      setActing(true)
      setActionMsg('')
      const params = { ...(config.actionParams?.(item) ?? {}), ...{ name: item.id, job_id: item.id } }

      gw
        .request<Record<string, unknown>>(config.actionMethod, params)
        .then(raw => {
          const r = asRpcResult<Record<string, unknown>>(raw)

          setActing(false)

          if (!r) {
            setActionMsg('error: invalid response')
            return
          }

          setActionMsg(config.actionLabel?.(item) ?? `✓ ${item.id}`)
          // Refresh the list after action
          setLoading(true)
          gw
            .request<Record<string, unknown>>(config.fetchMethod, config.fetchParams ?? {})
            .then(refreshRaw => {
              const rr = asRpcResult<Record<string, unknown>>(refreshRaw)
              if (rr) {
                setItems(config.mapResponse(rr))
              }
              setSel(s => Math.min(s, items.length - 1))
              setLoading(false)
            })
            .catch(() => setLoading(false))
        })
        .catch((e: unknown) => {
          setActing(false)
          setActionMsg(`error: ${rpcErrorMessage(e)}`)
        })
    }
  })

  if (loading) {
    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent}>
          {config.title}
        </Text>
        <Text color={t.color.muted}>loading…</Text>
        <OverlayHint t={t}>Esc/q close</OverlayHint>
      </Box>
    )
  }

  if (err) {
    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent}>
          {config.title}
        </Text>
        <Text color={t.color.label}>error: {err}</Text>
        <OverlayHint t={t}>Esc/q close</OverlayHint>
      </Box>
    )
  }

  if (!items.length) {
    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent}>
          {config.title}
        </Text>
        <Text color={t.color.muted}>no items</Text>
        <OverlayHint t={t}>Esc/q close</OverlayHint>
      </Box>
    )
  }

  const { items: visible, offset } = windowItems(items, sel, VISIBLE)

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent} wrap="truncate-end">
        {config.title}
      </Text>
      <Text color={t.color.muted} wrap="truncate-end">
        {items.length} {items.length === 1 ? 'item' : 'items'}
      </Text>

      {offset > 0 && (
        <Text color={t.color.muted} wrap="truncate-end">
          {' '}
          ↑ {offset} more
        </Text>
      )}

      {visible.map((item, i) => {
        const idx = offset + i
        const selected = sel === idx

        return (
          <Box
            backgroundColor={selected ? t.color.selectionBg : undefined}
            flexDirection="row"
            key={item.id ?? `row-${idx}`}
            width="100%"
          >
            <Text bold={selected} color={selected ? t.color.accent : t.color.muted}>
              {selected ? '▸ ' : '  '}
            </Text>
            <Box flexShrink={0} width={4}>
              <Text bold={selected} color={selected ? t.color.accent : t.color.muted}>
                {String(idx + 1).padStart(2)}
              </Text>
            </Box>
            <Box flexShrink={0} width={20}>
              <Text bold={selected} color={selected ? t.color.accent : t.color.text} wrap="truncate-end">
                {item.label}
              </Text>
            </Box>
            {item.meta ? (
              <Box flexShrink={0} width={14}>
                <Text color={selected ? t.color.accent : t.color.muted} wrap="truncate-end">
                  {item.meta}
                </Text>
              </Box>
            ) : null}
            <Box flexGrow={1} flexShrink={1} minWidth={0}>
              <Text color={selected ? t.color.accent : t.color.muted} wrap="truncate-end">
                {item.description ?? ''}
              </Text>
            </Box>
          </Box>
        )
      })}

      {offset + VISIBLE < items.length && (
        <Text color={t.color.muted} wrap="truncate-end">
          {' '}
          ↓ {items.length - offset - VISIBLE} more
        </Text>
      )}

      {actionMsg && (
        <Text color={actionMsg.startsWith('error') ? t.color.label : t.color.ok} wrap="truncate-end">
          {actionMsg}
        </Text>
      )}

      {acting && (
        <Text color={t.color.muted} wrap="truncate-end">
          acting…
        </Text>
      )}

      <OverlayHint t={t}>{config.hint ?? '↑/↓ select · Esc/q close'}</OverlayHint>
    </Box>
  )
}