import { Box, NoSelect, ScrollBox, Text, useInput } from '@hermes/ink'
import type { SessionListResult, SessionListRow } from '@hermes/shared/gateway-events'
import type { SessionActiveItem, SessionActiveListResponse } from '@hermes/shared/gateway-types'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { $isBlocked, $overlayState } from '../app/overlayStore.js'
import { $sessionSidebarVisible } from '../app/sessionSidebarStore.js'
import type { GatewayClient } from '../gatewayClient.js'
import { messages } from '../i18n/runtime.js'
import { useT } from '../i18n/useT.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import {
  closeFallbackAfterClose,
  relativeSessionAge,
  resumableHistory,
  selectedSessionRowStyle,
  shortModel,
  STATUS_GLYPH
} from './activeSessionSwitcher.js'

const DEFAULT_WIDTH = 28

type SidebarHistoryRow = SessionListRow & { last_active_at?: number | null }
type SidebarSessionListResult = Omit<SessionListResult, 'sessions'> & { sessions: SidebarHistoryRow[] }
type CanonicalSessionRow = SidebarHistoryRow & { model?: null | string; running?: boolean }

export type SessionSidebarEntry =
  | { activityAt: number; dimmed: false; id: string; kind: 'live'; session: SessionActiveItem }
  | { activityAt: number; dimmed: true; id: string; kind: 'history'; session: SidebarHistoryRow }

/** Merge live and resumable sessions, keeping the live copy and sorting by activity. */
export function buildSessionSidebarEntries(
  history: readonly SidebarHistoryRow[],
  live: readonly SessionActiveItem[]
): SessionSidebarEntry[] {
  const liveById = new Map<string, SessionActiveItem>()

  for (const session of live) {
    if (!liveById.has(session.id)) {
      liveById.set(session.id, session)
    }
  }

  const uniqueLive = [...liveById.values()]

  // `last_active` is available on live items; stored rows use canonical `last_active_at` when present.
  // The legacy SessionListRow contract exposes only `started_at`, which is the fallback there.
  const entries: SessionSidebarEntry[] = [
    ...uniqueLive.map(session => ({
      activityAt: session.last_active ?? session.started_at ?? 0,
      dimmed: false as const,
      id: session.id,
      kind: 'live' as const,
      session
    })),
    ...resumableHistory(history, uniqueLive).map(session => ({
      activityAt: session.last_active_at ?? session.started_at ?? 0,
      dimmed: true as const,
      id: session.id,
      kind: 'history' as const,
      session
    }))
  ]

  return entries
    .map((entry, index) => ({ entry, index }))
    .sort((a, b) => b.entry.activityAt - a.entry.activityAt || a.index - b.index)
    .map(({ entry }) => entry)
}

export interface SessionSidebarProps {
  currentSessionId: null | string
  gw: GatewayClient
  onClose: (id: string) => Promise<null | { closed?: boolean; ok?: boolean }>
  onHide: () => void
  onNew: () => void
  onResume: (id: string) => void
  onSelect: (id: string) => void
  t: Theme
  width?: number
}

function SessionSidebarRow({
  entry,
  onActivate,
  selected,
  t,
  untitled,
  width
}: {
  entry: SessionSidebarEntry
  onActivate: () => void
  selected: boolean
  t: Theme
  untitled: string
  width: number
}) {
  const selectedStyle = selected ? selectedSessionRowStyle(t) : null
  const dimmedColor = t.color.muted
  const textColor = entry.dimmed ? dimmedColor : (selectedStyle?.color ?? t.color.text)

  const statusColor =
    entry.kind === 'history'
      ? dimmedColor
      : entry.session.status === 'working'
        ? t.color.ok
        : entry.session.status === 'waiting'
          ? t.color.label
          : t.color.muted

  const title = entry.session.title || entry.session.preview || untitled
  const glyph = entry.kind === 'live' ? (STATUS_GLYPH[entry.session.status] ?? '·') : '○'
  const detail = entry.kind === 'live' ? shortModel(entry.session.model) : relativeSessionAge(entry.session.started_at)
  const detailWidth = Math.min(8, Math.max(0, width - 12))

  return (
    <Box
      backgroundColor={selectedStyle?.backgroundColor}
      flexDirection="row"
      onClick={(event: { stopImmediatePropagation?: () => void }) => {
        event.stopImmediatePropagation?.()
        onActivate()
      }}
      width={width}
    >
      <Box flexShrink={0} width={2}>
        <Text bold={selected && !entry.dimmed} color={entry.dimmed ? dimmedColor : (selectedStyle?.color ?? t.color.muted)}>
          {selected ? '▸ ' : '  '}
        </Text>
      </Box>
      <Box flexShrink={0} width={2}>
        <Text bold={selected && !entry.dimmed} color={statusColor}>
          {glyph}
        </Text>
      </Box>
      <Box flexGrow={1} flexShrink={1} minWidth={0}>
        <Text bold={selected && !entry.dimmed} color={textColor} wrap="truncate-end">
          {title}
        </Text>
      </Box>
      <Box flexShrink={0} width={detailWidth}>
        <Text color={entry.dimmed ? dimmedColor : (selectedStyle?.color ?? t.color.muted)} wrap="truncate-end">
          {detail}
        </Text>
      </Box>
    </Box>
  )
}

export function SessionSidebar({
  currentSessionId,
  gw,
  onClose,
  onHide,
  onNew,
  onResume,
  onSelect,
  t,
  width = DEFAULT_WIDTH
}: SessionSidebarProps) {
  const S = useT().session.sidebar
  const visible = useStore($sessionSidebarVisible)
  const blocked = useStore($isBlocked)
  const overlay = useStore($overlayState)
  const active = visible && !blocked && !overlay.agents && !overlay.journey
  const [live, setLive] = useState<SessionActiveItem[]>([])
  const [history, setHistory] = useState<SidebarHistoryRow[]>([])
  const [selectedId, setSelectedId] = useState<null | string>(currentSessionId)
  const [error, setError] = useState('')
  const [closingId, setClosingId] = useState('')
  const liveRef = useRef<SessionActiveItem[]>([])
  const rows = useMemo(() => buildSessionSidebarEntries(history, live), [history, live])
  const rowIds = useMemo(() => new Set(rows.map(row => row.id)), [rows])

  const load = useCallback(
    async (includeHistory: boolean) => {
      const canonicalList = gw.isCanonical
        ? gw.request<{ sessions?: CanonicalSessionRow[] }>('session.list', { limit: 200 })
        : null

      const [liveResult, historyResult] = await Promise.allSettled([
        canonicalList
          ? canonicalList.then(value => ({
              sessions: value.sessions?.map(row => ({
                ...row,
                current: row.id === currentSessionId,
                last_active: row.last_active_at ?? row.started_at ?? 0,
                model: row.model ?? '',
                status: row.running ? ('working' as const) : ('idle' as const)
              }))
            }))
          : gw.request<SessionActiveListResponse>('session.active_list', { current_session_id: currentSessionId }),
        includeHistory
          ? gw.request<SidebarSessionListResult>('session.list', {
              limit: 200,
              ...(canonicalList ? { scope: 'stored' } : {})
            })
          : Promise.resolve(null)
      ])

      const response = liveResult.status === 'fulfilled' ? asRpcResult<SessionActiveListResponse>(liveResult.value) : null

      if (!response) {
        setError(messages().pickers.common.invalidResponse('session.active_list'))

        return liveRef.current
      }

      const nextLive = response.sessions ?? []
      liveRef.current = nextLive
      setLive(nextLive)
      setError('')

      if (includeHistory) {
        if (historyResult.status === 'fulfilled') {
          const parsedHistory = asRpcResult<SidebarSessionListResult>(historyResult.value)

          if (parsedHistory) {
            setHistory(parsedHistory.sessions ?? [])
          } else {
            setError(messages().pickers.common.invalidResponse('session.list'))
          }
        } else {
          setError(messages().pickers.session.errors.couldNotLoadResumable)
        }
      }

      return nextLive
    },
    [currentSessionId, gw]
  )

  useEffect(() => {
    if (!visible) {
      return
    }

    void load(true)
    const timer = setInterval(() => void load(false), 1500)

    return () => clearInterval(timer)
  }, [load, visible])

  useEffect(() => {
    setSelectedId(currentSessionId)
  }, [currentSessionId])

  useEffect(() => {
    if (selectedId && rowIds.has(selectedId)) {
      return
    }

    setSelectedId(currentSessionId && rowIds.has(currentSessionId) ? currentSessionId : null)
  }, [currentSessionId, rowIds, selectedId])

  const activate = useCallback(
    (entry: SessionSidebarEntry) => {
      setSelectedId(entry.id)

      if (entry.kind === 'live') {
        onSelect(entry.id)
      } else {
        onResume(entry.id)
      }
    },
    [onResume, onSelect]
  )

  const createSession = useCallback(() => {
    setSelectedId(null)
    onNew()
  }, [onNew])

  const closeSelected = useCallback(async () => {
    if (!selectedId || closingId || !live.some(session => session.id === selectedId)) {
      return
    }

    setClosingId(selectedId)
    setError('')

    try {
      const result = await onClose(selectedId)
      const closed = Boolean(result?.closed ?? result?.ok)

      if (!closed) {
        setError(
          gw.isCanonical && result === null
            ? messages().pickers.session.errors.notAttached
            : messages().pickers.session.errors.alreadyClosed
        )

        return
      }

      const remaining = await load(false)
      const fallback = closeFallbackAfterClose(selectedId, currentSessionId, remaining)

      if (fallback.action === 'activate') {
        setSelectedId(fallback.sessionId)
        onSelect(fallback.sessionId)
      } else if (fallback.action === 'new') {
        createSession()
      } else {
        setSelectedId(currentSessionId && remaining.some(session => session.id === currentSessionId) ? currentSessionId : null)
      }
    } catch (cause: unknown) {
      setError(rpcErrorMessage(cause))
    } finally {
      setClosingId('')
    }
  }, [closingId, createSession, currentSessionId, gw.isCanonical, live, load, onClose, onSelect, selectedId])

  useInput(
    (ch, key) => {
      const lower = ch.toLowerCase()
      const isCtrl = (letter: string) => key.ctrl && (lower === letter || ch === String.fromCharCode(letter.charCodeAt(0) - 96))

      if (key.escape || (isCtrl('y') && key.shift)) {
        onHide()

        return
      }

      if (isCtrl('n')) {
        createSession()

        return
      }

      if (isCtrl('d')) {
        void closeSelected()

        return
      }

      if (key.upArrow) {
        const index = rows.findIndex(row => row.id === selectedId)
        setSelectedId(index > 0 ? rows[index - 1]!.id : null)

        return
      }

      if (key.downArrow) {
        const index = rows.findIndex(row => row.id === selectedId)
        const nextIndex = selectedId === null ? 0 : Math.min(rows.length - 1, index + 1)
        setSelectedId(rows[nextIndex]?.id ?? null)

        return
      }

      if (key.return) {
        const selected = rows.find(row => row.id === selectedId)

        if (selected) {
          activate(selected)
        } else {
          createSession()
        }
      }
    },
    { isActive: active }
  )

  const newSelected = selectedId === null
  const newRowStyle = newSelected ? selectedSessionRowStyle(t) : null

  return (
    <NoSelect flexDirection="column" flexShrink={0} width={width}>
      <Box borderBottom borderColor={t.color.border} flexDirection="row" justifyContent="space-between" width={width}>
        <Text bold color={t.color.accent} wrap="truncate-end">
          {S.title}
        </Text>
        <Text color={t.color.muted}>{S.liveCount(live.length)}</Text>
      </Box>

      <Box
        backgroundColor={newRowStyle?.backgroundColor}
        flexDirection="row"
        onClick={(event: { stopImmediatePropagation?: () => void }) => {
          event.stopImmediatePropagation?.()
          createSession()
        }}
        width={width}
      >
        <Box flexShrink={0} width={2}>
          <Text bold={newSelected} color={newRowStyle?.color ?? t.color.label}>
            {newSelected ? '▸ ' : '  '}
          </Text>
        </Box>
        <Text bold={newSelected} color={newRowStyle?.color ?? t.color.label} wrap="truncate-end">
          {S.newSession}
        </Text>
      </Box>

      {error ? (
        <Text color={t.color.label} wrap="truncate-end">
          {error}
        </Text>
      ) : null}

      <ScrollBox flexDirection="column" flexGrow={1} flexShrink={1}>
        {rows.length ? (
          rows.map(entry => (
            <SessionSidebarRow
              entry={entry}
              key={entry.id}
              onActivate={() => activate(entry)}
              selected={selectedId === entry.id}
              t={t}
              untitled={S.untitled}
              width={width}
            />
          ))
        ) : (
          <Text color={t.color.muted} wrap="truncate-end">
            {S.noLiveSessions}
          </Text>
        )}
      </ScrollBox>
    </NoSelect>
  )
}
