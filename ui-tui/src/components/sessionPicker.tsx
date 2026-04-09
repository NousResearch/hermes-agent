import { Box, Text, useInput } from 'ink'
import { useEffect, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import type { Theme } from '../theme.js'

interface SessionItem {
  id: string
  title: string
  preview: string
  started_at: number
  message_count: number
  source?: string
}

function age(ts: number): string {
  const d = (Date.now() / 1000 - ts) / 86400

  if (d < 1) {
    return 'today'
  }

  if (d < 2) {
    return 'yesterday'
  }

  return `${Math.floor(d)}d ago`
}

const VISIBLE = 15

export function SessionPicker({
  gw,
  onCancel,
  onSelect,
  t
}: {
  gw: GatewayClient
  onCancel: () => void
  onSelect: (id: string) => void
  t: Theme
}) {
  const [items, setItems] = useState<SessionItem[]>([])
  const [sel, setSel] = useState(0)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    gw.request('session.list', { limit: 20 })
      .then((r: any) => {
        setItems(r.sessions ?? [])
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [gw])

  useInput((ch, key) => {
    if (key.escape) {
      return onCancel()
    }

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

  if (!items.length) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.dim}>no previous sessions</Text>
        <Text color={t.color.dim}>Esc to cancel</Text>
      </Box>
    )
  }

  const off = Math.max(0, Math.min(sel - Math.floor(VISIBLE / 2), items.length - VISIBLE))
  const visible = items.slice(off, off + VISIBLE)

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.amber}>
        Resume Session
      </Text>
      {off > 0 && <Text color={t.color.dim}> ↑ {off} more</Text>}
      {visible.map((s, vi) => {
        const i = off + vi

        return (
          <Text key={s.id}>
            <Text color={sel === i ? t.color.label : t.color.dim}>{sel === i ? '▸ ' : '  '}</Text>
            <Text color={sel === i ? t.color.cornsilk : t.color.dim}>
              {i + 1}. {s.title || s.preview || s.id.slice(0, 8)}
            </Text>
            <Text color={t.color.dim}>
              {' '}
              ({s.message_count} msgs, {age(s.started_at)}{s.source && s.source !== 'tui' ? `, ${s.source}` : ''})
            </Text>
          </Text>
        )
      })}
      {off + VISIBLE < items.length && <Text color={t.color.dim}> ↓ {items.length - off - VISIBLE} more</Text>}
      <Text color={t.color.dim}>↑/↓ select · Enter resume · 1-9 quick · Esc cancel</Text>
    </Box>
  )
}
