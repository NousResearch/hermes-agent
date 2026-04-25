import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import type { SessionListItem, SessionListResponse } from '../gatewayTypes.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { OverlayHint, useOverlayKeys, windowOffset } from './overlayControls.js'

const VISIBLE = 15
const MIN_WIDTH = 60
const MAX_WIDTH = 120

const age = (ts: number) => {
  const d = (Date.now() / 1000 - ts) / 86400

  if (d < 1) {
    return 'today'
  }

  if (d < 2) {
    return 'yesterday'
  }

  return `${Math.floor(d)}d ago`
}

export function SessionPicker({ gw, onCancel, onSelect, t }: SessionPickerProps) {
  const [items, setItems] = useState<SessionListItem[]>([])
  const [err, setErr] = useState('')
  const [sel, setSel] = useState(0)
  const [loading, setLoading] = useState(true)

  const { stdout } = useStdout()
  const width = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))

  useOverlayKeys({ onClose: onCancel })

  useEffect(() => {
    gw.request<SessionListResponse>('session.list', { limit: 20 })
      .then(raw => {
        const r = asRpcResult<SessionListResponse>(raw)

        if (!r) {
          setErr('invalid response: session.list')
          setLoading(false)

          return
        }

        setItems(r.sessions ?? [])
        setErr('')
        setLoading(false)
      })
      .catch((e: unknown) => {
        setErr(rpcErrorMessage(e))
        setLoading(false)
      })
  }, [gw])

  useInput((ch, key) => {
    if (key.upArrow && sel > 0) {
      setSel(s => s - 1)
    }

    if (key.downArrow && sel < items.length - 1) {
      setSel(s => s + 1)
    }

    if (key.return && items[sel]) {
      onSelect(items[sel]!.id)
    }

    const n = parseInt(ch)

    if (n >= 1 && n <= Math.min(9, items.length)) {
      onSelect(items[n - 1]!.id)
    }
  })

  if (loading) {
    return <Text color={t.color.dim}>loading sessions…</Text>
  }

  if (err) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.label}>error: {err}</Text>
        <OverlayHint t={t}>Esc/q cancel</OverlayHint>
      </Box>
    )
  }

  if (!items.length) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.dim}>no previous sessions</Text>
        <OverlayHint t={t}>Esc/q cancel</OverlayHint>
      </Box>
    )
  }

  const offset = windowOffset(items.length, sel, VISIBLE)

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.amber}>
        Resume Session
      </Text>

      {offset > 0 && <Text color={t.color.dim}> ↑ {offset} more</Text>}

      {items.slice(offset, offset + VISIBLE).map((s, vi) => {
        const i = offset + vi
        const selected = sel === i

        return (
          <Box key={s.id}>
            <Text bold={selected} color={selected ? t.color.amber : t.color.dim} inverse={selected}>
              {selected ? '▸ ' : '  '}
            </Text>

            <Box width={30}>
              <Text bold={selected} color={selected ? t.color.amber : t.color.dim} inverse={selected}>
                {String(i + 1).padStart(2)}. [{s.id}]
              </Text>
            </Box>

            <Box width={30}>
              <Text bold={selected} color={selected ? t.color.amber : t.color.dim} inverse={selected}>
                ({s.message_count} msgs, {age(s.started_at)}, {s.source || 'tui'})
              </Text>
            </Box>

            <Text bold={selected} color={selected ? t.color.amber : t.color.dim} inverse={selected} wrap="truncate-end">
              {s.title || s.preview || '(untitled)'}
            </Text>
          </Box>
        )
      })}

      {offset + VISIBLE < items.length && <Text color={t.color.dim}> ↓ {items.length - offset - VISIBLE} more</Text>}
      <OverlayHint t={t}>↑/↓ select · Enter resume · 1-9 quick · Esc/q cancel</OverlayHint>
    </Box>
  )
}

interface SessionPickerProps {
  gw: GatewayClient
  onCancel: () => void
  onSelect: (id: string) => void
  t: Theme
}
