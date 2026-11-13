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

import { closeFallbackAfterClose, selectedSessionRowStyle, shortModel, STATUS_GLYPH } from './activeSessionSwitcher.js'

const DEFAULT_WIDTH = 28

type CanonicalSessionRow = SessionListRow & { running?: boolean }

export interface SessionTreeNode {
  children: SessionTreeNode[]
  session: SessionActiveItem
}

/** Build an ordered live-session tree from the resumable compression lineage. */
export function buildSessionTree(rows: readonly SessionListRow[], live: readonly SessionActiveItem[]) {
  const liveById = new Map<string, SessionTreeNode>()

  for (const session of live) {
    if (!liveById.has(session.id)) {
      liveById.set(session.id, { children: [], session })
    }
  }

  const uniqueLive = [...liveById.values()].map(node => node.session)
  const historyById = new Map<string, SessionListRow>()

  for (const row of rows) {
    if (!historyById.has(row.id)) {
      historyById.set(row.id, row)
    }
  }

  const parentById = new Map<string, string>()

  for (const session of uniqueLive) {
    const parentId = historyById.get(session.id)?._lineage_root_id

    if (parentId && parentId !== session.id && liveById.has(parentId)) {
      parentById.set(session.id, parentId)
    }
  }

  // A malformed lineage cycle should not hide every member of the cycle.
  for (const id of parentById.keys()) {
    const seen = new Set([id])
    let parentId = parentById.get(id)

    while (parentId && !seen.has(parentId)) {
      seen.add(parentId)
      parentId = parentById.get(parentId)
    }

    if (parentId) {
      parentById.delete(id)
    }
  }

  const roots: SessionTreeNode[] = []

  for (const session of uniqueLive) {
    const node = liveById.get(session.id)!
    const parentId = parentById.get(session.id)
    const parent = parentId ? liveById.get(parentId) : undefined

    if (parent) {
      parent.children.push(node)
    } else {
      roots.push(node)
    }
  }

  return roots
}

interface FlatSessionRow {
  depth: number
  session: SessionActiveItem
}

function flattenSessionTree(
  nodes: readonly SessionTreeNode[],
  collapsed: ReadonlySet<string>,
  depth = 0
): FlatSessionRow[] {
  const rows: FlatSessionRow[] = []

  for (const node of nodes) {
    rows.push({ depth, session: node.session })

    if (node.children.length && !collapsed.has(node.session.id)) {
      rows.push(...flattenSessionTree(node.children, collapsed, depth + 1))
    }
  }

  return rows
}

export interface SessionSidebarProps {
  currentSessionId: null | string
  gw: GatewayClient
  onClose: (id: string) => Promise<null | { closed?: boolean; ok?: boolean }>
  onHide: () => void
  onNew: () => void
  onSelect: (id: string) => void
  t: Theme
  width?: number
}

export function SessionSidebar({
  currentSessionId,
  gw,
  onClose,
  onHide,
  onNew,
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
  const [history, setHistory] = useState<SessionListRow[]>([])
  const [selectedId, setSelectedId] = useState<null | string>(currentSessionId)
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(() => new Set())
  const [error, setError] = useState('')
  const [closingId, setClosingId] = useState('')
  const liveRef = useRef<SessionActiveItem[]>([])
  const tree = useMemo(() => buildSessionTree(history, live), [history, live])
  const rows = useMemo(() => flattenSessionTree(tree, collapsed), [tree, collapsed])
  const rowIds = useMemo(() => new Set(rows.map(row => row.session.id)), [rows])

  const parentIds = useMemo(() => {
    const ids = new Set<string>()

    const visit = (nodes: readonly SessionTreeNode[]) => {
      for (const node of nodes) {
        if (node.children.length) {
          ids.add(node.session.id)
          visit(node.children)
        }
      }
    }

    visit(tree)

    return ids
  }, [tree])

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
                status: row.running ? ('working' as const) : ('idle' as const)
              }))
            }))
          : gw.request<SessionActiveListResponse>('session.active_list', { current_session_id: currentSessionId }),
        includeHistory
          ? gw.request<SessionListResult>('session.list', {
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
          const parsedHistory = asRpcResult<SessionListResult>(historyResult.value)

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
    (id: string) => {
      setSelectedId(id)
      onSelect(id)
    },
    [onSelect]
  )

  const createSession = useCallback(() => {
    setSelectedId(null)
    onNew()
  }, [onNew])

  const toggleCollapsed = useCallback((id: string) => {
    setCollapsed(previous => {
      const next = new Set(previous)

      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }

      return next
    })
  }, [])

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
        activate(fallback.sessionId)
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
  }, [activate, closingId, createSession, currentSessionId, gw.isCanonical, live, load, onClose, selectedId])

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
        const index = rows.findIndex(row => row.session.id === selectedId)
        setSelectedId(index > 0 ? rows[index - 1]!.session.id : null)

        return
      }

      if (key.downArrow) {
        const index = rows.findIndex(row => row.session.id === selectedId)
        const nextIndex = selectedId === null ? 0 : Math.min(rows.length - 1, index + 1)
        setSelectedId(rows[nextIndex]?.session.id ?? null)

        return
      }

      if (key.return) {
        if (selectedId && live.some(session => session.id === selectedId)) {
          activate(selectedId)
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
          rows.map(({ depth, session }) => {
            const selected = selectedId === session.id
            const selectedStyle = selected ? selectedSessionRowStyle(t) : null
            const statusColor = session.status === 'working' ? t.color.ok : session.status === 'waiting' ? t.color.label : t.color.muted
            const title = session.title || session.preview || S.untitled
            const hasChildren = parentIds.has(session.id)

            return (
              <Box
                backgroundColor={selectedStyle?.backgroundColor}
                flexDirection="row"
                key={session.id}
                onClick={(event: { stopImmediatePropagation?: () => void }) => {
                  event.stopImmediatePropagation?.()
                  activate(session.id)
                }}
                width={width}
              >
                <Box flexShrink={0} width={Math.min(depth * 2, Math.max(0, width - 8))}>
                  <Text>{' '.repeat(Math.min(depth * 2, Math.max(0, width - 8)))}</Text>
                </Box>
                <Box
                  flexShrink={0}
                  onClick={(event: { stopImmediatePropagation?: () => void }) => {
                    event.stopImmediatePropagation?.()
                    toggleCollapsed(session.id)
                  }}
                  width={2}
                >
                  <Text bold={selected} color={selectedStyle?.color ?? t.color.muted}>
                    {hasChildren ? (collapsed.has(session.id) ? '▸ ' : '▾ ') : '  '}
                  </Text>
                </Box>
                <Box flexShrink={0} width={2}>
                  <Text bold={selected} color={statusColor}>
                    {STATUS_GLYPH[session.status] ?? '·'}
                  </Text>
                </Box>
                <Box flexGrow={1} flexShrink={1} minWidth={0}>
                  <Text bold={selected} color={selectedStyle?.color ?? t.color.text} wrap="truncate-end">
                    {title}
                  </Text>
                </Box>
                <Box flexShrink={0} width={Math.min(8, Math.max(0, width - 10))}>
                  <Text color={selectedStyle?.color ?? t.color.muted} wrap="truncate-end">
                    {shortModel(session.model)}
                  </Text>
                </Box>
              </Box>
            )
          })
        ) : (
          <Text color={t.color.muted} wrap="truncate-end">
            {S.noLiveSessions}
          </Text>
        )}
      </ScrollBox>
    </NoSelect>
  )
}
