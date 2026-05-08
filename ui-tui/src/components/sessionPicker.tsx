import { Box, Text, useInput, useStdout } from '@hermes/ink'
import type { Key } from '@hermes/ink'
import { useEffect, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import type { SessionDeleteResponse, SessionListItem, SessionListResponse } from '../gatewayTypes.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { OverlayHint, windowOffset } from './overlayControls.js'

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

export const sessionSearchText = (s: SessionListItem) =>
  [s.id, s.title, s.preview, s.source, String(s.message_count)].filter(Boolean).join(' ').toLowerCase()

export type SessionPickerMode = 'browse' | 'filter'

const isPrintableSearchChar = (ch: string | undefined) => !!ch && ch.length === 1 && /^[ -~]$/.test(ch)

const isBareQuestionMark = (ch: string | undefined, key: Key) =>
  ch === '?' && !key.ctrl && !key.meta

export const sessionPickerModeAfterKey = (
  mode: SessionPickerMode,
  ch: string | undefined,
  key: Key
): SessionPickerMode => {
  if (mode === 'browse' && isBareQuestionMark(ch, key)) {
    return 'filter'
  }

  if (mode === 'filter' && key.escape) {
    return 'browse'
  }

  return mode
}

export const shouldAppendSessionFilterChar = (
  ch: string | undefined,
  key: Key,
  mode: SessionPickerMode
) => mode === 'filter' && !key.ctrl && !key.meta && isPrintableSearchChar(ch)

export const sessionPickerHintText = ({ filterMode, hasQuery }: { filterMode: boolean; hasQuery: boolean }) =>
  filterMode
    ? `type to filter · ↑/↓ select · Enter resume · Backspace ${hasQuery ? 'clear' : 'exit'} · Esc exit filter`
    : '? filter · ↑/↓ select · Enter resume · 1-9 quick · Ctrl+D delete · Esc/q cancel'

export type SessionPickerKeyAction =
  | { type: 'append-filter'; ch: string }
  | { type: 'backspace-filter' }
  | { type: 'cancel' }
  | { type: 'delete' }
  | { type: 'enter-filter' }
  | { type: 'exit-filter' }
  | { type: 'quick-select'; index: number }
  | { type: 'resume-selected' }
  | { type: 'none' }

export const sessionPickerKeyAction = ({
  ch,
  key,
  mode,
  query,
  visibleCount
}: {
  ch: string | undefined
  key: Key
  mode: SessionPickerMode
  query: string
  visibleCount: number
}): SessionPickerKeyAction => {
  const filterMode = mode === 'filter'

  if (key.escape) {
    return filterMode ? { type: 'exit-filter' } : { type: 'cancel' }
  }

  if (key.backspace && filterMode) {
    return query ? { type: 'backspace-filter' } : { type: 'exit-filter' }
  }

  if (key.return && visibleCount > 0) {
    return { type: 'resume-selected' }
  }

  if (!filterMode && key.ctrl && ch?.toLowerCase() === 'd' && visibleCount > 0) {
    return { type: 'delete' }
  }

  if (!filterMode && isBareQuestionMark(ch, key)) {
    return { type: 'enter-filter' }
  }

  if (!filterMode && ch?.toLowerCase() === 'q') {
    return { type: 'cancel' }
  }

  const n = !filterMode ? parseInt(ch ?? '') : NaN

  if (n >= 1 && n <= Math.min(9, visibleCount)) {
    return { index: n - 1, type: 'quick-select' }
  }

  if (shouldAppendSessionFilterChar(ch, key, mode)) {
    return { ch: ch!, type: 'append-filter' }
  }

  return { type: 'none' }
}

export function SessionPicker({ gw, onCancel, onSelect, t }: SessionPickerProps) {
  const [items, setItems] = useState<SessionListItem[]>([])
  const [err, setErr] = useState('')
  const [sel, setSel] = useState(0)
  const [loading, setLoading] = useState(true)
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<SessionPickerMode>('browse')
  const filterMode = mode === 'filter'
  // When non-null, the user pressed `d` on this session and we're waiting for
  // a second `d`/`D` to confirm deletion.  Any other key cancels the prompt.
  const [confirmDelete, setConfirmDelete] = useState<null | string>(null)
  const [deleting, setDeleting] = useState(false)

  const { stdout } = useStdout()
  const width = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))
  const normalizedQuery = query.trim().toLowerCase()
  const visibleItems = normalizedQuery
    ? items.filter(s => sessionSearchText(s).includes(normalizedQuery))
    : items

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

  useEffect(() => {
    setSel(s => Math.max(0, Math.min(s, visibleItems.length - 1)))
  }, [visibleItems.length])

  const performDelete = (id: string) => {
    const target = items.find(item => item.id === id)

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
          const next = prev.filter(item => item.id !== id)
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

    const action = sessionPickerKeyAction({ ch, key, mode, query, visibleCount: visibleItems.length })

    if (action.type === 'exit-filter') {
      setQuery('')
      setMode('browse')

      return
    }

    if (action.type === 'cancel') {
      onCancel()

      return
    }

    if (confirmDelete !== null) {
      if (ch?.toLowerCase() === 'd') {
        const id = confirmDelete
        setConfirmDelete(null)
        performDelete(id)
      } else {
        setConfirmDelete(null)
      }

      return
    }

    if (key.upArrow && sel > 0) {
      setSel(s => s - 1)
    }

    if (key.downArrow && sel < visibleItems.length - 1) {
      setSel(s => s + 1)
    }

    if (action.type === 'backspace-filter') {
      setQuery(q => q.slice(0, -1))
      setConfirmDelete(null)

      return
    }

    if (action.type === 'resume-selected' && visibleItems[sel]) {
      onSelect(visibleItems[sel]!.id)

      return
    }

    if (action.type === 'delete' && visibleItems[sel]) {
      setConfirmDelete(visibleItems[sel]!.id)

      return
    }

    if (action.type === 'enter-filter') {
      setMode('filter')
      setConfirmDelete(null)

      return
    }

    if (action.type === 'quick-select' && visibleItems[action.index]) {
      onSelect(visibleItems[action.index]!.id)

      return
    }

    if (action.type === 'append-filter') {
      setQuery(q => q + action.ch)
      setConfirmDelete(null)
    }
  })

  if (loading) {
    return <Text color={t.color.muted}>loading sessions…</Text>
  }

  if (err && !items.length) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.label}>error: {err}</Text>
        <OverlayHint t={t}>Esc cancel</OverlayHint>
      </Box>
    )
  }

  if (!visibleItems.length) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.muted}>{query ? `no sessions matching "${query}"` : 'no previous sessions'}</Text>
        {filterMode && <Text color={t.color.muted}>filter: {query || 'type to filter'}</Text>}
        <OverlayHint t={t}>{sessionPickerHintText({ filterMode, hasQuery: !!query })}</OverlayHint>
      </Box>
    )
  }

  const offset = windowOffset(visibleItems.length, sel, VISIBLE)

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent}>
        Resume Session
      </Text>
      <Text color={t.color.muted}>{filterMode ? `filter: ${query || 'type to filter'}` : 'press ? to filter'}</Text>

      {offset > 0 && <Text color={t.color.muted}>  ↑ {offset} more</Text>}

      {visibleItems.slice(offset, offset + VISIBLE).map((s, vi) => {
        const i = offset + vi
        const selected = sel === i
        const pendingDelete = confirmDelete === s.id

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
                ({s.message_count} msgs, {age(s.started_at)}, {s.source || 'tui'})
              </Text>
            </Box>

            <Text
              bold={selected}
              color={pendingDelete ? t.color.label : selected ? t.color.accent : t.color.muted}
              inverse={selected}
              wrap="truncate-end"
            >
              {pendingDelete ? 'press d again to delete' : s.title || s.preview || '(untitled)'}
            </Text>
          </Box>
        )
      })}

      {offset + VISIBLE < visibleItems.length && (
        <Text color={t.color.muted}>  ↓ {visibleItems.length - offset - VISIBLE} more</Text>
      )}
      {err && <Text color={t.color.label}>error: {err}</Text>}
      {deleting ? (
        <OverlayHint t={t}>deleting…</OverlayHint>
      ) : (
        <OverlayHint t={t}>{sessionPickerHintText({ filterMode, hasQuery: !!query })}</OverlayHint>
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
