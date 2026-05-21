import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import type { SessionDeleteResponse, SessionListItem, SessionListResponse } from '../gatewayTypes.js'
import { useI18n } from '../i18n/index.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { OverlayHint, useOverlayKeys, windowOffset } from './overlayControls.js'

const VISIBLE = 15
const MIN_WIDTH = 60
const MAX_WIDTH = 120

const age = (ts: number, ti: ReturnType<typeof useI18n>['t']) => {
  const d = (Date.now() / 1000 - ts) / 86400

  if (d < 1) {
    return ti('time.today')
  }

  if (d < 2) {
    return ti('time.yesterday')
  }

  return ti('time.daysAgo', { count: String(Math.floor(d)) })
}

export function SessionPicker({ gw, onCancel, onSelect, t }: SessionPickerProps) {
  const [items, setItems] = useState<SessionListItem[]>([])
  const [err, setErr] = useState('')
  const [sel, setSel] = useState(0)
  const [loading, setLoading] = useState(true)
  // When non-null, the user pressed `d` on this index and we're waiting for
  // a second `d`/`D` to confirm deletion.  Any other key cancels the prompt.
  const [confirmDelete, setConfirmDelete] = useState<null | number>(null)
  const [deleting, setDeleting] = useState(false)
  const { t: ti } = useI18n()

  const { stdout } = useStdout()
  const width = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))

  useOverlayKeys({ onClose: onCancel })

  useEffect(() => {
    gw.request<SessionListResponse>('session.list', { limit: 200 })
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

  const performDelete = (index: number) => {
    const target = items[index]

    if (!target || deleting) {
      return
    }

    setDeleting(true)
    gw.request<SessionDeleteResponse>('session.delete', { session_id: target.id })
      .then(raw => {
        const r = asRpcResult<SessionDeleteResponse>(raw)

        if (!r || r.deleted !== target.id) {
          setErr('invalid response: session.delete')
          setDeleting(false)

          return
        }

        setItems(prev => {
          const next = prev.filter((_, i) => i !== index)
          setSel(s => Math.max(0, Math.min(s, next.length - 1)))

          return next
        })
        setErr('')
        setDeleting(false)
      })
      .catch((e: unknown) => {
        setErr(rpcErrorMessage(e))
        setDeleting(false)
      })
  }

  useInput((ch, key) => {
    if (deleting) {
      return
    }

    if (confirmDelete !== null) {
      if (ch?.toLowerCase() === 'd') {
        const idx = confirmDelete
        setConfirmDelete(null)
        performDelete(idx)
      } else {
        setConfirmDelete(null)
      }

      return
    }

    if (key.upArrow && sel > 0) {
      setSel(s => s - 1)
    }

    if (key.downArrow && sel < items.length - 1) {
      setSel(s => s + 1)
    }

    if (key.return && items[sel]) {
      onSelect(items[sel]!.id)

      return
    }

    if (ch?.toLowerCase() === 'd' && items[sel]) {
      setConfirmDelete(sel)

      return
    }

    const n = parseInt(ch)

    if (n >= 1 && n <= Math.min(9, items.length)) {
      onSelect(items[n - 1]!.id)
    }
  })

  if (loading) {
    return <Text color={t.color.muted}>{ti('session.loading')}</Text>
  }

  if (err && !items.length) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.label}>{ti('sys.error', { message: err })}</Text>
        <OverlayHint t={t}>{ti('picker.cancel')}</OverlayHint>
      </Box>
    )
  }

  if (!items.length) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.muted}>{ti('session.nonePrevious')}</Text>
        <OverlayHint t={t}>{ti('picker.cancel')}</OverlayHint>
      </Box>
    )
  }

  const offset = windowOffset(items.length, sel, VISIBLE)

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent}>
        {ti('session.resumeTitle')}
      </Text>

      {offset > 0 && <Text color={t.color.muted}>  {ti('sys.moreAbove', { count: String(offset) })}</Text>}

      {items.slice(offset, offset + VISIBLE).map((s, vi) => {
        const i = offset + vi
        const selected = sel === i
        const pendingDelete = confirmDelete === i

        return (
          <Box key={s.id}>
            <Text bold={selected} color={selected ? t.color.accent : t.color.muted} inverse={selected}>
              {selected ? '▸ ' : '  '}
            </Text>

            <Box width={30}>
              <Text bold={selected} color={selected ? t.color.accent : t.color.muted} inverse={selected}>
                {String(i + 1).padStart(2)}. [{s.id}]
              </Text>
            </Box>

            <Box width={30}>
              <Text bold={selected} color={selected ? t.color.accent : t.color.muted} inverse={selected}>
                ({ti('session.messagesShort', { count: String(s.message_count) })}, {age(s.started_at, ti)}, {s.source || 'tui'})
              </Text>
            </Box>

            <Text
              bold={selected}
              color={pendingDelete ? t.color.label : selected ? t.color.accent : t.color.muted}
              inverse={selected}
              wrap="truncate-end"
            >
              {pendingDelete ? ti('session.deleteAgain') : s.title || s.preview || ti('session.untitled')}
            </Text>
          </Box>
        )
      })}

      {offset + VISIBLE < items.length && <Text color={t.color.muted}>  {ti('sys.moreBelow', { count: String(items.length - offset - VISIBLE) })}</Text>}
      {err && <Text color={t.color.label}>{ti('sys.error', { message: err })}</Text>}
      {deleting ? (
        <OverlayHint t={t}>{ti('session.deleting')}</OverlayHint>
      ) : (
        <OverlayHint t={t}>{ti('picker.sessionHint')}</OverlayHint>
      )}
    </Box>
  )
}

interface SessionPickerProps {
  gw: GatewayClient
  onCancel: () => void
  onSelect: (id: string) => void
  t: Theme
}
