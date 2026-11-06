import { Box, Text, useInput, useStdout } from '@hermes/ink'
import type { SessionListResult, SessionListRow } from '@hermes/shared/gateway-events'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { sessionScopedModelArg } from '../domain/slash.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { SessionActiveItem, SessionActiveListResponse, SessionCloseResponse } from '../gatewayTypes.js'
import { useSessionDelete } from '../hooks/useSessionDelete.js'
import { messages } from '../i18n/runtime.js'
import { useT } from '../i18n/useT.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { ModelPicker } from './modelPicker.js'
import { windowOffset } from './overlayControls.js'
import { clampOverlayWidth, listRowStyle } from './overlayPrimitives.js'
import {
  buildSwitcherWindowedRows,
  DEFAULT_PAGE_LIMIT,
  resolveLivePollSelection,
  resolvePageChangeSelection,
  type SwitcherWindowedRow,
  validateSessionPageResponse
} from './sessionPagination.js'
import { TextInput } from './textInput.js'

const VISIBLE = 12
const MIN_WIDTH = 64
const MAX_WIDTH = 128
const TITLE_MAX = 64

const STATUS_GLYPH: Record<string, string> = {
  idle: '✓',
  starting: '…',
  waiting: '?',
  working: '▶'
}

/** Human label for a live-session status value; unknown statuses render as-is.
 *  Resolved at call time so it follows the active language. */
export const sessionStatusLabel = (status: string): string => {
  const labels: Record<string, string | undefined> = messages().pickers.session.status

  return labels[status] ?? status
}

const CTRL_OFFSET = 96

const shortModel = (model = '') => model.replace(/^.*\//, '') || messages().pickers.session.modelUnknown
const ctrlChar = (letter: string) => String.fromCharCode(letter.charCodeAt(0) - CTRL_OFFSET)

export const fixedSessionColumnStyle = () => ({ flexShrink: 0 })

export const activeSessionCountLabel = (count: number) =>
  count === 1 ? messages().pickers.session.liveSessionsOne(1) : messages().pickers.session.liveSessionsOther(count)

export const sessionsCountLabel = (liveCount: number, resumableCount: number) =>
  messages().pickers.session.counts(liveCount, resumableCount)

export type SessionRowKind = 'history' | 'live' | 'new' | 'pagination-newer' | 'pagination-older'

/**
 * Map a flat row index into the merged Sessions list to its kind. Rows are
 * ordered [new][live…][history…] — the "+ new" row is pinned first so it is
 * always visible no matter how long the resumable history grows.
 */
export const sessionRowKindAt = (index: number, liveCount: number): SessionRowKind => {
  if (index <= 0) {
    return 'new'
  }

  return index - 1 < liveCount ? 'live' : 'history'
}

export const relativeSessionAge = (ts?: number) => {
  if (!ts) {
    return ''
  }

  const days = (Date.now() / 1000 - ts) / 86400

  const age = messages().pickers.session.age

  if (days < 1) {
    return age.today
  }

  if (days < 2) {
    return age.yesterday
  }

  return age.daysAgo(Math.floor(days))
}

/** Keep the first history row per ID, excluding live IDs; leave raw paging data untouched. */
export const resumableHistory = (history: readonly SessionListRow[], live: readonly SessionActiveItem[]) => {
  const seenIds = new Set(live.map(s => s.id))

  return history.filter(h => {
    if (seenIds.has(h.id)) {return false}
    seenIds.add(h.id)
    return true
  })
}

// Hint segment tables are functions (not module constants) so their text
// fragments resolve against the active language at render time.
export const resumeRowContextHintSegments = (): OrchestratorHintSegment[] => {
  const h = messages().pickers.session.hint

  return [
    { role: 'label', text: h.resumableLabel },
    { role: 'text', text: ' ' },
    { role: 'hotkey', text: 'Enter' },
    { role: 'text', text: h.resume },
    { role: 'hotkey', text: 'd' },
    { role: 'text', text: h.delete }
  ]
}

export const paginationRowContextHintSegments = (isRetry: boolean): OrchestratorHintSegment[] => {
  const h = messages().pickers.session.hint

  return [
    { role: 'label', text: h.pageLabel },
    { role: 'text', text: ' ' },
    { role: 'hotkey', text: 'Enter' },
    { role: 'text', text: isRetry ? h.pageRetry : h.pageNavigate }
  ]
}

export type OrchestratorHintRole = 'hotkey' | 'label' | 'text'

export interface OrchestratorHintSegment {
  role: OrchestratorHintRole
  text: string
}

export const orchestratorContextHintSegments = (newSelected: boolean): OrchestratorHintSegment[] => {
  const h = messages().pickers.session.hint

  return newSelected
    ? [
        { role: 'label', text: h.newRowLabel },
        { role: 'text', text: h.typePrompt },
        { role: 'hotkey', text: 'Enter' },
        { role: 'text', text: h.start },
        { role: 'hotkey', text: 'Tab' },
        { role: 'text', text: h.model }
      ]
    : [
        { role: 'label', text: h.sessionRowLabel },
        { role: 'text', text: ' ' },
        { role: 'hotkey', text: 'Enter' },
        { role: 'text', text: h.switch },
        { role: 'hotkey', text: 'Ctrl+D' },
        { role: 'text', text: h.close }
      ]
}

export const orchestratorGlobalHotkeyHintSegments = (): OrchestratorHintSegment[] => {
  const h = messages().pickers.session.hint

  return [
    { role: 'hotkey', text: '↑↓' },
    { role: 'text', text: h.move },
    { role: 'hotkey', text: 'Ctrl+N' },
    { role: 'text', text: h.new },
    { role: 'hotkey', text: 'Ctrl+R' },
    { role: 'text', text: h.refresh },
    { role: 'hotkey', text: 'Esc' },
    { role: 'text', text: h.close }
  ]
}

const hintText = (segments: readonly OrchestratorHintSegment[]) => segments.map(segment => segment.text).join('')

export const orchestratorContextHint = (newSelected: boolean) => hintText(orchestratorContextHintSegments(newSelected))

export const orchestratorGlobalHotkeyHint = () => hintText(orchestratorGlobalHotkeyHintSegments())

export const orchestratorHintSegmentColor = (t: Theme, role: OrchestratorHintRole) => {
  if (role === 'hotkey') {
    return t.color.accent
  }

  if (role === 'label') {
    return t.color.label
  }

  return t.color.muted
}

export const selectedSessionRowStyle = (t: Theme) => ({
  backgroundColor: listRowStyle(t, true).backgroundColor,
  color: listRowStyle(t, true).color
})

export const newSessionMarkerColor = (t: Theme, selected: boolean) =>
  selected ? selectedSessionRowStyle(t).color : t.color.label

export const newSessionRowIndex = (sessionCount: number) => Math.max(0, sessionCount)

export const isNewSessionRow = (index: number, sessionCount: number) => index >= newSessionRowIndex(sessionCount)

export const canTypeOrchestratorPrompt = (index: number, sessionCount: number) => isNewSessionRow(index, sessionCount)

export const clampOrchestratorSelection = (index: number, sessionCount: number) =>
  Math.max(0, Math.min(index, newSessionRowIndex(sessionCount)))

export const currentSessionSelectionIndex = (
  sessions: readonly SessionActiveItem[],
  currentSessionId: null | string
) => {
  const index = sessions.findIndex(s => Boolean(s.current) || (!!currentSessionId && s.id === currentSessionId))

  return index >= 0 ? index : 0
}

export const orchestratorVisibleRowIndexes = (sessionCount: number, selected: number, visible = VISIBLE) => {
  const total = Math.max(0, sessionCount) + 1
  const clamped = clampOrchestratorSelection(selected, sessionCount)
  const offset = windowOffset(total, clamped, visible)
  const count = Math.min(visible, total - offset)

  return Array.from({ length: count }, (_, i) => offset + i)
}

export type CloseFallback = { action: 'activate'; sessionId: string } | { action: 'new' } | { action: 'stay' }

export const closeFallbackAfterClose = (
  closedId: string,
  currentSessionId: null | string,
  remaining: readonly SessionActiveItem[]
): CloseFallback => {
  if (!currentSessionId || closedId !== currentSessionId) {
    return { action: 'stay' }
  }

  const next = remaining.find(s => s.id !== closedId)

  return next ? { action: 'activate', sessionId: next.id } : { action: 'new' }
}

export const draftModelArgFromPickerValue = (value: string) => {
  return sessionScopedModelArg(value)
}

export const draftModelNameFromArg = (value: string) => {
  const parts = draftModelArgFromPickerValue(value).split(/\s+/).filter(Boolean)
  const modelParts: string[] = []

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i]!

    if (part === '--provider' || part === '--reasoning') {
      i++

      continue
    }

    if (part.startsWith('--')) {
      continue
    }

    modelParts.push(part)
  }

  return modelParts.join(' ').trim()
}

export const draftModelDisplayLabel = (value: string) => {
  const modelName = draftModelNameFromArg(value)

  return modelName ? shortModel(modelName) : messages().pickers.session.currentOrDefault
}

export type OrchestratorRowClickAction = { action: 'activate'; sessionId: string } | { action: 'select-new' }

export const orchestratorRowClickAction = (
  index: number,
  sessions: readonly SessionActiveItem[]
): OrchestratorRowClickAction => {
  const target = sessions[index]

  return target && !isNewSessionRow(index, sessions.length)
    ? { action: 'activate', sessionId: target.id }
    : { action: 'select-new' }
}

export const draftTitleFromPrompt = (prompt: string, max = TITLE_MAX) => {
  const compact = prompt.replace(/\s+/g, ' ').trim()

  if (compact.length <= max) {
    return compact
  }

  return `${compact.slice(0, Math.max(0, max - 1)).trimEnd()}…`
}

function OrchestratorHintSegments({ segments, t }: OrchestratorHintTextProps) {
  return (
    <>
      {segments.map((segment, index) => (
        <Text color={orchestratorHintSegmentColor(t, segment.role)} key={`${segment.role}-${index}`}>
          {segment.text}
        </Text>
      ))}
    </>
  )
}

function OrchestratorHintText({ segments, t }: OrchestratorHintTextProps) {
  return (
    <Text color={orchestratorHintSegmentColor(t, 'text')} wrap="truncate-end">
      <OrchestratorHintSegments segments={segments} t={t} />
    </Text>
  )
}

export function handleSwitcherNavigation(
  key: { downArrow?: boolean; upArrow?: boolean },
  sel: number,
  total: number,
  setSel: (fn: (s: number) => number) => void
): boolean {
  if (key.upArrow && sel > 0) {
    setSel(s => Math.max(0, s - 1))

    return true
  }

  if (key.downArrow && sel < total - 1) {
    setSel(s => Math.min(total - 1, s + 1))

    return true
  }

  return false
}

export function handleSwitcherCtrlAction(
  ch: string | undefined,
  key: { ctrl?: boolean },
  selectedKind: SessionRowKind,
  actions: { onCloseLive: () => void; onNew: () => void; onRefresh: () => void }
): boolean {
  if (!key.ctrl) {
    return false
  }

  const lower = ch?.toLowerCase() ?? ''

  if (lower === 'n' || ch === ctrlChar('n')) {
    actions.onNew()

    return true
  }

  if (lower === 'r' || ch === ctrlChar('r')) {
    actions.onRefresh()

    return true
  }

  if (lower === 'd' || ch === ctrlChar('d')) {
    if (selectedKind === 'live') {
      actions.onCloseLive()
    }

    return true
  }

  return false
}

export function handleConfirmDeleteKey(
  ch: string | undefined,
  confirmId: string,
  history: readonly SessionListRow[],
  performDelete: (item?: SessionListRow) => void,
  clearConfirm: () => void
) {
  clearConfirm()
  if (ch?.toLowerCase() === 'd') {
    performDelete(history.find(h => h.id === confirmId))
  }
}

export function handleSwitcherRowActivation(
  row: SwitcherWindowedRow | undefined,
  onSelect: (id: string) => void,
  onResume: (id: string) => void,
  onNavigatePage: (offset: number) => void
) {
  if (!row) {
    return
  }

  if (row.kind === 'live') {
    onSelect(row.item.id)
  } else if (row.kind === 'history') {
    onResume(row.item.id)
  } else if (row.kind === 'pagination-newer' || row.kind === 'pagination-older') {
    onNavigatePage(row.targetOffset)
  }
}

interface SwitcherHeaderProps {
  countLabel: string
  errorMessage: string
  isOldGateway: boolean
  t: Theme
  title: string
}

function SwitcherHeader({ countLabel, errorMessage, isOldGateway, t, title }: SwitcherHeaderProps) {
  return (
    <>
      <Text bold color={t.color.accent}>
        {title}
      </Text>
      <Text color={t.color.muted}>{countLabel}</Text>
      {isOldGateway && <Text color={t.color.muted}>{messages().pickers.session.pagination.olderGatewayNotice}</Text>}
      {Boolean(errorMessage) && <Text color={t.color.label}>{messages().pickers.common.error(errorMessage)}</Text>}
    </>
  )
}

interface SwitcherDraftRowProps {
  draftModel: string
  onClick: (event: { stopImmediatePropagation?: () => void }) => void
  promptTitle: string
  selected: boolean
  t: Theme
}

function SwitcherDraftRow({ draftModel, onClick, promptTitle, selected, t }: SwitcherDraftRowProps) {
  const S = messages().pickers.session
  const rowStyle = selected ? selectedSessionRowStyle(t) : null
  const textColor = rowStyle?.color ?? t.color.muted
  const markerColor = newSessionMarkerColor(t, selected)

  return (
    <Box backgroundColor={rowStyle?.backgroundColor} flexDirection="row" onClick={onClick} width="100%">
      <Text bold={selected} color={textColor}>
        {selected ? '▸ ' : '  '}
      </Text>
      <Box {...fixedSessionColumnStyle()} width={5}>
        <Text bold={selected} color={markerColor}>
          {'+'.padStart(2)}
        </Text>
      </Box>
      <Box {...fixedSessionColumnStyle()} width={11}>
        <Text bold={selected} color={markerColor} wrap="truncate-end">
          {S.row.new}
        </Text>
      </Box>
      <Box {...fixedSessionColumnStyle()} width={11}>
        <Text color={textColor} wrap="truncate-end">
          {S.row.draft}
        </Text>
      </Box>
      <Box {...fixedSessionColumnStyle()} width={18}>
        <Text color={textColor} wrap="truncate-end">
          {draftModelDisplayLabel(draftModel)}
        </Text>
      </Box>
      <Box flexGrow={1} flexShrink={1} minWidth={0}>
        <Text bold={selected} color={textColor} wrap="truncate-end">
          {promptTitle}
        </Text>
      </Box>
    </Box>
  )
}

interface SwitcherPaginationRowProps {
  failedOffset: null | number
  index: number
  onClick: (event: { stopImmediatePropagation?: () => void }) => void
  pageError: string
  pageLoading: boolean
  row: Extract<SwitcherWindowedRow, { kind: 'pagination-newer' | 'pagination-older' }>
  selected: boolean
  t: Theme
}

function SwitcherPaginationRow({
  failedOffset,
  index,
  onClick,
  pageError,
  pageLoading,
  row,
  selected,
  t
}: SwitcherPaginationRowProps) {
  const S = messages().pickers.session
  const selectedStyle = selected ? selectedSessionRowStyle(t) : null
  const rowTextColor = selectedStyle?.color
  const isNewer = row.kind === 'pagination-newer'
  const isRetry = Boolean(pageError && failedOffset === row.targetOffset)
  const title = pageLoading
    ? S.pagination.loadingPage
    : isRetry
      ? `${isNewer ? S.pagination.newer : S.pagination.older} (${S.pagination.retry})`
      : isNewer
        ? S.pagination.newer
        : S.pagination.older

  return (
    <Box
      backgroundColor={selectedStyle?.backgroundColor}
      flexDirection="row"
      key={row.kind}
      onClick={onClick}
      width="100%"
    >
      <Text bold={selected} color={rowTextColor ?? t.color.muted}>
        {selected ? '▸ ' : '  '}
      </Text>
      <Box {...fixedSessionColumnStyle()} width={5}>
        <Text bold={selected} color={rowTextColor ?? t.color.muted}>
          {String(index).padStart(2)}.
        </Text>
      </Box>
      <Box {...fixedSessionColumnStyle()} width={11}>
        <Text
          bold={selected}
          color={selected ? (rowTextColor ?? t.color.accent) : t.color.muted}
          wrap="truncate-end"
        >
          {isNewer ? '▲ newer' : '▼ older'}
        </Text>
      </Box>
      <Box {...fixedSessionColumnStyle()} width={11}>
        <Text color={rowTextColor ?? t.color.muted} wrap="truncate-end">
          {''}
        </Text>
      </Box>
      <Box {...fixedSessionColumnStyle()} width={18}>
        <Text color={rowTextColor ?? t.color.muted} wrap="truncate-end">
          {''}
        </Text>
      </Box>
      <Box flexGrow={1} flexShrink={1} minWidth={0}>
        <Text
          bold={selected}
          color={isRetry ? t.color.label : (rowTextColor ?? (selected ? t.color.accent : t.color.muted))}
          wrap="truncate-end"
        >
          {title}
        </Text>
      </Box>
    </Box>
  )
}

interface SwitcherHistoryRowProps {
  index: number
  isDeleting: boolean
  isPendingDelete: boolean
  item: SessionListRow
  onClick: (event: { stopImmediatePropagation?: () => void }) => void
  selected: boolean
  t: Theme
}

function SwitcherHistoryRow({
  index,
  isDeleting,
  isPendingDelete,
  item,
  onClick,
  selected,
  t
}: SwitcherHistoryRowProps) {
  const S = messages().pickers.session
  const selectedStyle = selected ? selectedSessionRowStyle(t) : null
  const rowTextColor = selectedStyle?.color

  const title = isPendingDelete
    ? S.row.pressDAgain
    : isDeleting && selected
      ? S.row.deleting
      : item.title || item.preview || S.row.untitled

  return (
    <Box
      backgroundColor={selectedStyle?.backgroundColor}
      flexDirection="row"
      key={item.id}
      onClick={onClick}
      width="100%"
    >
      <Text bold={selected} color={rowTextColor ?? t.color.muted}>
        {selected ? '▸ ' : '  '}
      </Text>
      <Box {...fixedSessionColumnStyle()} width={5}>
        <Text bold={selected} color={rowTextColor ?? t.color.muted}>
          {String(index).padStart(2)}.
        </Text>
      </Box>
      <Box {...fixedSessionColumnStyle()} width={11}>
        <Text bold={selected} color={rowTextColor ?? t.color.muted} wrap="truncate-end">
          {item.id}
        </Text>
      </Box>
      <Box {...fixedSessionColumnStyle()} width={11}>
        <Text color={rowTextColor ?? t.color.muted} wrap="truncate-end">
          {relativeSessionAge(item.started_at)}
        </Text>
      </Box>
      <Box {...fixedSessionColumnStyle()} width={18}>
        <Text color={rowTextColor ?? t.color.muted} wrap="truncate-end">
          {S.row.messageCount(item.message_count ?? 0)}
        </Text>
      </Box>
      <Box flexGrow={1} flexShrink={1} minWidth={0}>
        <Text
          bold={selected}
          color={isPendingDelete ? t.color.label : (rowTextColor ?? t.color.muted)}
          wrap="truncate-end"
        >
          {title}
        </Text>
      </Box>
    </Box>
  )
}

interface SwitcherLiveRowProps {
  index: number
  isClosing: boolean
  isCurrent: boolean
  item: SessionActiveItem
  onClick: (event: { stopImmediatePropagation?: () => void }) => void
  selected: boolean
  t: Theme
}

function SwitcherLiveRow({
  index,
  isClosing,
  isCurrent,
  item,
  onClick,
  selected,
  t
}: SwitcherLiveRowProps) {
  const S = messages().pickers.session
  const selectedStyle = selected ? selectedSessionRowStyle(t) : null
  const rowTextColor = selectedStyle?.color
  const status = item.status ?? 'idle'
  const title = isClosing ? S.row.closing : item.title || item.preview || S.row.untitled

  return (
    <Box
      backgroundColor={selectedStyle?.backgroundColor}
      flexDirection="row"
      key={item.id}
      onClick={onClick}
      width="100%"
    >
      <Text bold={selected} color={rowTextColor ?? t.color.muted}>
        {selected ? '▸ ' : '  '}
      </Text>
      <Box {...fixedSessionColumnStyle()} width={5}>
        <Text bold={selected} color={rowTextColor ?? t.color.muted}>
          {String(index).padStart(2)}.
        </Text>
      </Box>
      <Box {...fixedSessionColumnStyle()} width={11}>
        <Text
          bold={selected}
          color={rowTextColor ?? (isCurrent ? t.color.label : t.color.muted)}
          wrap="truncate-end"
        >
          {isCurrent ? S.row.current : item.id}
        </Text>
      </Box>
      <Box {...fixedSessionColumnStyle()} width={11}>
        <Text
          color={
            rowTextColor ??
            (status === 'working' ? t.color.ok : status === 'waiting' ? t.color.label : t.color.muted)
          }
          wrap="truncate-end"
        >
          {STATUS_GLYPH[status] ?? '·'} {sessionStatusLabel(status)}
        </Text>
      </Box>
      <Box {...fixedSessionColumnStyle()} width={18}>
        <Text color={rowTextColor ?? t.color.muted} wrap="truncate-end">
          {shortModel(item.model)}
        </Text>
      </Box>
      <Box flexGrow={1} flexShrink={1} minWidth={0}>
        <Text bold={selected} color={rowTextColor ?? t.color.muted} wrap="truncate-end">
          {title}
        </Text>
      </Box>
    </Box>
  )
}

export function isGatewayConnectionChanged(
  prevGw: GatewayClient | Pick<GatewayClient, 'isCanonical' | 'request'>,
  nextGw: GatewayClient | Pick<GatewayClient, 'isCanonical' | 'request'>
): boolean {
  if (prevGw === nextGw) {
    return false
  }

  if (prevGw.isCanonical !== nextGw.isCanonical) {
    return true
  }

  if (prevGw.request !== nextGw.request) {
    return true
  }

  if ('channel' in prevGw || 'channel' in nextGw) {
    return prevGw !== nextGw
  }

  return false
}

export function ActiveSessionSwitcher({
  currentSessionId,
  filter,
  gw,
  maxWidth,
  onCancel,
  onClose,
  onNew,
  onNewPrompt,
  onResume,
  onSelect,
  profile,
  t
}: ActiveSessionSwitcherProps) {
  const [items, setItems] = useState<SessionActiveItem[]>([])
  const [history, setHistory] = useState<SessionListRow[]>([])
  const [historyOffset, setHistoryOffset] = useState(0)
  const [historyLimit, setHistoryLimit] = useState(DEFAULT_PAGE_LIMIT)
  const [hasMore, setHasMore] = useState(false)
  const historyOffsetRef = useRef(0)
  const historyLimitRef = useRef(DEFAULT_PAGE_LIMIT)
  const hasMoreRef = useRef(false)
  const [isOldGateway, setIsOldGateway] = useState(false)
  const [pageLoading, setPageLoading] = useState(false)
  const [pageError, setPageError] = useState('')
  const [failedOffset, setFailedOffset] = useState<null | number>(null)

  const [err, setErr] = useState('')
  const [actionErr, setActionErr] = useState('')
  const [sel, setSel] = useState(0)
  const [loading, setLoading] = useState(true)
  const [draft, setDraft] = useState('')
  const [draftModel, setDraftModel] = useState('')
  const [pickingModel, setPickingModel] = useState(false)
  const [closingId, setClosingId] = useState('')
  const [confirmDelete, setConfirmDelete] = useState<null | string>(null)

  const T = useT()
  const S = T.pickers.session
  const C = T.pickers.common
  const initialSelectionAppliedRef = useRef(false)
  const isMountedRef = useRef(true)

  // Lifecycle generation token: only increments on true context/connection changes,
  // never during background live polls.
  const contextGenRef = useRef(0)
  const liveSeqRef = useRef(0)
  const pendingRequestRef = useRef(false)
  const historyQueueRef = useRef<Promise<void>>(Promise.resolve())

  const rawHistoryRef = useRef<SessionListRow[]>([])
  const itemsRef = useRef<SessionActiveItem[]>([])
  const selectedKindRef = useRef<SessionRowKind>('new')
  const selectedIdRef = useRef<null | string>(null)

  const prevGwRef = useRef(gw)
  const prevProfileRef = useRef(profile)
  const prevFilterRef = useRef(filter)

  // Synchronous reset when gw connection, profile, or filter changes:
  // Immediately wipes old visible rows and raw history so old profile data is never exposed.
  const connectionChanged = isGatewayConnectionChanged(prevGwRef.current, gw)
  const profileChanged = prevProfileRef.current !== profile
  const filterChanged = prevFilterRef.current !== filter

  if (connectionChanged || profileChanged || filterChanged) {
    prevGwRef.current = gw
    prevProfileRef.current = profile
    prevFilterRef.current = filter
    contextGenRef.current++
    historyQueueRef.current = Promise.resolve()
    pendingRequestRef.current = false
    rawHistoryRef.current = []
    itemsRef.current = []
    historyOffsetRef.current = 0
    historyLimitRef.current = DEFAULT_PAGE_LIMIT
    hasMoreRef.current = false
    setLoading(true)
    setItems([])
    setHistory([])
    setHistoryOffset(0)
    setHistoryLimit(DEFAULT_PAGE_LIMIT)
    setHasMore(false)
    setIsOldGateway(false)
    setPageLoading(false)
    setPageError('')
    setFailedOffset(null)
    setErr('')
    setActionErr('')
    setSel(0)
    initialSelectionAppliedRef.current = false
  }

  const { stdout } = useStdout()
  const preferredWidth = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))
  const width = clampOverlayWidth(preferredWidth, maxWidth)
  const promptColumns = Math.max(20, width - 11)

  useEffect(() => {
    isMountedRef.current = true

    return () => {
      isMountedRef.current = false
    }
  }, [])

  const windowedRows: SwitcherWindowedRow[] = useMemo(
    () => buildSwitcherWindowedRows(items, history, historyOffset, historyLimit, hasMore),
    [items, history, historyOffset, historyLimit, hasMore]
  )

  const listLen = windowedRows.length
  const total = listLen + 1

  const rowKind = useCallback(
    (index: number): SessionRowKind => {
      if (index <= 0) {
        return 'new'
      }

      const row = windowedRows[index - 1]

      return row ? row.kind : 'history'
    },
    [windowedRows]
  )

  useEffect(() => {
    if (sel <= 0) {
      selectedKindRef.current = 'new'
      selectedIdRef.current = null

      return
    }

    const row = windowedRows[sel - 1]

    if (!row) {
      selectedKindRef.current = 'new'
      selectedIdRef.current = null

      return
    }

    selectedKindRef.current = row.kind

    if (row.kind === 'live' || row.kind === 'history') {
      selectedIdRef.current = row.item.id
    } else {
      selectedIdRef.current = null
    }
  }, [sel, windowedRows])

  // Fetches live sessions only (does NOT increment lifecycle generation, does not touch history)
  const fetchLiveSessions = useCallback(async () => {
    const curGen = contextGenRef.current
    const curSeq = ++liveSeqRef.current

    try {
      const canonicalList = gw.isCanonical
      const params: Record<string, unknown> = canonicalList
        ? { limit: 200 }
        : { current_session_id: currentSessionId }

      if (profile) {
        params.profile = profile
      }

      const res = canonicalList
        ? await gw.request<{ scope?: string; sessions?: (SessionListRow & { running?: boolean })[] }>('session.list', params)
        : await gw.request<SessionActiveListResponse>('session.active_list', params)

      if (!isMountedRef.current || curGen !== contextGenRef.current || curSeq !== liveSeqRef.current) {
        return null
      }

      const unwrapped = asRpcResult<SessionActiveListResponse>(res) ?? (res as SessionActiveListResponse)

      if (canonicalList) {
        const canonicalData = unwrapped as unknown as { sessions?: (SessionListRow & { running?: boolean })[] }

        return (
          canonicalData.sessions?.map(row => ({
            ...row,
            current: row.id === currentSessionId,
            status: row.running ? ('working' as const) : ('idle' as const)
          })) ?? []
        )
      }

      return unwrapped.sessions ?? []
    } catch {
      return null
    }
  }, [currentSessionId, filter, gw, profile])

  // Fetches a single history page with context check and validation
  const fetchHistoryPage = useCallback(
    async (targetOffset: number, isInitial: boolean) => {
      const curGen = contextGenRef.current
      const canonicalList = gw.isCanonical
      const params: Record<string, unknown> = {
        limit: historyLimitRef.current || DEFAULT_PAGE_LIMIT,
        ...(canonicalList ? { scope: 'stored' } : {})
      }

      if (!isInitial || targetOffset > 0) {
        params.offset = targetOffset
      }
      if (profile) {
        params.profile = profile
      }

      try {
        const raw = await gw.request<SessionListResult>('session.list', params)

        if (!isMountedRef.current || curGen !== contextGenRef.current) {
          return { kind: 'aborted' as const }
        }

        const unwrapped = asRpcResult<SessionListResult>(raw) ?? raw

        return validateSessionPageResponse(unwrapped, targetOffset, isInitial)
      } catch (e: unknown) {
        if (!isMountedRef.current || curGen !== contextGenRef.current) {
          return { kind: 'aborted' as const }
        }

        return { error: rpcErrorMessage(e), kind: 'error' as const }
      }
    },
    [filter, gw, profile]
  )

  // Serialized page navigation: does not clash with background live polls
  const handleNavigatePage = useCallback(
    async (targetOffset: number) => {
      if (pendingRequestRef.current) {
        return
      }

      pendingRequestRef.current = true
      setPageLoading(true)

      const curGen = contextGenRef.current

      await new Promise<void>(resolve => {
        historyQueueRef.current = historyQueueRef.current.then(async () => {
          try {
            if (!isMountedRef.current || curGen !== contextGenRef.current) {
              return
            }

            const result = await fetchHistoryPage(targetOffset, false)

            if (!isMountedRef.current || curGen !== contextGenRef.current || result.kind === 'aborted') {
              return
            }

            if (result.kind === 'valid') {
              // Finish projection before committing any of the last good page's state.
              const currentLive = itemsRef.current
              const nextHist = resumableHistory(result.sessions, currentLive)
              const nextRows = buildSwitcherWindowedRows(
                currentLive,
                nextHist,
                result.meta.offset,
                result.meta.limit,
                result.meta.has_more
              )

              historyOffsetRef.current = result.meta.offset
              historyLimitRef.current = result.meta.limit
              hasMoreRef.current = result.meta.has_more
              setHistoryOffset(result.meta.offset)
              setHistoryLimit(result.meta.limit)
              setHasMore(result.meta.has_more)
              setIsOldGateway(false)
              setFailedOffset(null)
              setPageError('')
              setErr('')

              rawHistoryRef.current = result.sessions
              setHistory(nextHist)

              setSel(() =>
                resolvePageChangeSelection(
                  selectedKindRef.current,
                  selectedIdRef.current,
                  currentLive,
                  nextHist,
                  nextRows
                )
              )
            } else {
              const errText =
                result.kind === 'malformed'
                  ? result.error
                  : result.kind === 'error'
                    ? result.error
                    : messages().pickers.session.errors.couldNotLoadResumable

              setPageError(errText)
              setErr(errText)
              setFailedOffset(targetOffset)
            }
          } catch {
            // A failed projection must not reject the serialized navigation queue.
            if (isMountedRef.current && curGen === contextGenRef.current) {
              const errText = messages().pickers.common.invalidResponse('session.list')
              setPageError(errText)
              setErr(errText)
              setFailedOffset(targetOffset)
            }
          } finally {
            if (isMountedRef.current && curGen === contextGenRef.current) {
              setPageLoading(false)
              pendingRequestRef.current = false
            }

            resolve()
          }
        })
      })
    },
    [fetchHistoryPage]
  )

  // Serialized initial / refresh load
  const loadInitialOrRefresh = useCallback(
    async (quiet = false) => {
      if (pendingRequestRef.current) {
        return
      }
      pendingRequestRef.current = true
      if (!quiet) {
        setLoading(true)
      }

      const curGen = contextGenRef.current

      await new Promise<void>(resolve => {
        historyQueueRef.current = historyQueueRef.current.then(async () => {
          try {
            if (!isMountedRef.current || curGen !== contextGenRef.current) {
              return
            }

            const [liveItems, histResult] = await Promise.all([
              fetchLiveSessions(),
              fetchHistoryPage(0, true)
            ])

            if (!isMountedRef.current || curGen !== contextGenRef.current) {
              return
            }

            const nextLive = liveItems ?? itemsRef.current
            itemsRef.current = nextLive
            setItems(nextLive)

            let histError = ''
            let newOffset = historyOffsetRef.current
            let newLimit = historyLimitRef.current
            let newHasMore = hasMoreRef.current

            if (histResult.kind === 'valid') {
              rawHistoryRef.current = histResult.sessions
              newOffset = histResult.meta.offset
              newLimit = histResult.meta.limit
              newHasMore = histResult.meta.has_more
              historyOffsetRef.current = newOffset
              historyLimitRef.current = newLimit
              hasMoreRef.current = newHasMore
              setHistoryOffset(newOffset)
              setHistoryLimit(newLimit)
              setHasMore(newHasMore)
              setIsOldGateway(false)
              setFailedOffset(null)
              setPageError('')
            } else if (histResult.kind === 'legacy-unpaged') {
              rawHistoryRef.current = histResult.sessions
              newOffset = 0
              newLimit = 200
              newHasMore = false
              historyOffsetRef.current = 0
              historyLimitRef.current = 200
              hasMoreRef.current = false
              setHistoryOffset(0)
              setHistoryLimit(200)
              setHasMore(false)
              setIsOldGateway(true)
              setFailedOffset(null)
              setPageError('')
            } else if (histResult.kind !== 'aborted') {
              histError = histResult.error
            }

            const nextHist = resumableHistory(rawHistoryRef.current, nextLive)
            setHistory(nextHist)

            const nextRows = buildSwitcherWindowedRows(nextLive, nextHist, newOffset, newLimit, newHasMore)
            const initializeSelection = !initialSelectionAppliedRef.current
            initialSelectionAppliedRef.current = true

            setSel(s => {
              if (initializeSelection) {
                return nextLive.length
                  ? Math.min(currentSessionSelectionIndex(nextLive, currentSessionId) + 1, nextRows.length)
                  : 0
              }

              // Failed history refreshes retain the committed layout, including paging selection.
              if (histResult.kind !== 'valid' && histResult.kind !== 'legacy-unpaged') {
                return resolveLivePollSelection(s, selectedKindRef.current, selectedIdRef.current, nextRows)
              }

              return resolvePageChangeSelection(
                selectedKindRef.current,
                selectedIdRef.current,
                nextLive,
                nextHist,
                nextRows
              )
            })

            setErr(histError)
          } finally {
            if (isMountedRef.current && curGen === contextGenRef.current) {
              setLoading(false)
              pendingRequestRef.current = false
            }

            resolve()
          }
        })
      })
    },
    [currentSessionId, fetchHistoryPage, fetchLiveSessions]
  )

  // Quiet background live poll: updates live status and reconciles selection without invalidating history
  const pollLive = useCallback(async () => {
    const nextLive = await fetchLiveSessions()

    if (!nextLive || !isMountedRef.current) {
      return
    }

    itemsRef.current = nextLive
    setItems(nextLive)

    const nextHist = resumableHistory(rawHistoryRef.current, nextLive)
    setHistory(nextHist)

    const nextRows = buildSwitcherWindowedRows(
      nextLive,
      nextHist,
      historyOffsetRef.current,
      historyLimitRef.current,
      hasMoreRef.current
    )

    setSel(s => resolveLivePollSelection(s, selectedKindRef.current, selectedIdRef.current, nextRows))
  }, [fetchLiveSessions])

  useEffect(() => {
    void loadInitialOrRefresh()
    const timer = setInterval(() => void pollLive(), 1500)

    return () => clearInterval(timer)
  }, [loadInitialOrRefresh, pollLive])

  // Connection identity hooks: subscribe to gateway client lifecycle events when supported
  useEffect(() => {
    const client = gw as unknown as {
      off?: (event: string, handler: (...args: unknown[]) => void) => void
      on?: (event: string, handler: (...args: unknown[]) => void) => void
    }

    if (typeof client?.on === 'function' && typeof client?.off === 'function') {
      const onGatewayEvent = (ev: unknown) => {
        const event = ev as { type?: string }

        if (event?.type === 'gateway.ready') {
          contextGenRef.current++
          historyQueueRef.current = Promise.resolve()
          pendingRequestRef.current = false
          rawHistoryRef.current = []
          itemsRef.current = []
          setItems([])
          setHistory([])
          setPageError('')
          setFailedOffset(null)
          initialSelectionAppliedRef.current = false
          void loadInitialOrRefresh()
        }
      }

      client.on('event', onGatewayEvent)

      return () => client.off?.('event', onGatewayEvent)
    }
  }, [gw, loadInitialOrRefresh])

  const submitDraft = useCallback(
    (value: string) => {
      const prompt = value.trim()

      if (!prompt) {
        return
      }

      setDraft('')
      onNewPrompt(prompt, draftModel || undefined)
    },
    [draftModel, onNewPrompt]
  )

  const closeSelected = useCallback(async () => {
    if (sel <= 0) {
      return
    }

    const row = windowedRows[sel - 1]

    if (!row || row.kind !== 'live' || closingId) {
      return
    }

    const target = row.item

    setActionErr('')
    setClosingId(target.id)

    try {
      const result = await onClose(target.id)
      const closed = Boolean(result?.closed ?? result?.ok)

      if (!closed) {
        setActionErr(
          gw.isCanonical && result === null
            ? messages().pickers.session.errors.notAttached
            : messages().pickers.session.errors.alreadyClosed
        )

        return
      }

      await pollLive()
      const remaining = itemsRef.current
      const fallback = closeFallbackAfterClose(target.id, currentSessionId, remaining)

      if (fallback.action === 'activate') {
        onSelect(fallback.sessionId)
      } else if (fallback.action === 'new') {
        onNew()
      } else {
        setSel(s => Math.max(0, Math.min(s, total - 1)))
      }
    } catch (e: unknown) {
      setActionErr(rpcErrorMessage(e))
    } finally {
      setClosingId('')
    }
  }, [closingId, currentSessionId, gw.isCanonical, onClose, onNew, onSelect, pollLive, sel, total, windowedRows])

  const removeDeleted = useCallback(
    (id: string) => {
      rawHistoryRef.current = rawHistoryRef.current.filter(h => h.id !== id)
      setHistory(prev => prev.filter(h => h.id !== id))
      setSel(s => Math.max(0, Math.min(s, total - 2)))
    },
    [total]
  )

  const keepView = useCallback((resolved: string) => resolved === currentSessionId, [currentSessionId])
  const { deletePending, deleting, performDelete } = useSessionDelete(gw, removeDeleted, setActionErr, keepView)

  const handleRowClick = useCallback(
    (index: number) => (event: { stopImmediatePropagation?: () => void }) => {
      event.stopImmediatePropagation?.()

      if (deletePending()) {
        return
      }

      const clamped = Math.max(0, Math.min(index, total - 1))
      setSel(clamped)

      if (clamped === 0) {
        return
      }

      handleSwitcherRowActivation(
        windowedRows[clamped - 1],
        onSelect,
        onResume,
        offset => void handleNavigatePage(offset)
      )
    },
    [deletePending, handleNavigatePage, onResume, onSelect, total, windowedRows]
  )

  const selectedKind = rowKind(sel)
  const newSelected = selectedKind === 'new'
  const draftHasText = Boolean(draft.trim())

  useInput((ch, key) => {
    if (pickingModel || deletePending()) {
      return
    }

    if (confirmDelete !== null) {
      handleConfirmDeleteKey(
        ch,
        confirmDelete,
        history,
        item => performDelete(item),
        () => setConfirmDelete(null)
      )

      return
    }

    if (key.escape) {
      return onCancel()
    }

    const handledCtrl = handleSwitcherCtrlAction(ch, key, selectedKind, {
      onCloseLive: () => void closeSelected(),
      onNew: () => onNew(),
      onRefresh: () => void loadInitialOrRefresh()
    })

    if (handledCtrl) {
      return
    }

    if (key.tab && newSelected) {
      setPickingModel(true)

      return
    }

    const lower = ch?.toLowerCase() ?? ''

    if (lower === 'd' && !key.ctrl && selectedKind === 'history') {
      const row = windowedRows[sel - 1]

      if (row?.kind === 'history') {
        setConfirmDelete(row.item.id)
      }

      return
    }

    if (newSelected && draftHasText) {
      return
    }

    if (handleSwitcherNavigation(key, sel, total, setSel)) {
      return
    }

    if (key.return) {
      if (newSelected) {
        if (!draftHasText) {
          onNew()
        }

        return
      }

      handleSwitcherRowActivation(
        windowedRows[sel - 1],
        onSelect,
        onResume,
        offset => void handleNavigatePage(offset)
      )
    }
  })

  if (pickingModel) {
    return (
      <ModelPicker
        allowPersistGlobal={false}
        gw={gw}
        onCancel={() => setPickingModel(false)}
        onSelect={value => {
          setDraftModel(draftModelArgFromPickerValue(value))
          setPickingModel(false)
        }}
        sessionId={currentSessionId}
        t={t}
      />
    )
  }

  if (loading) {
    return <Text color={t.color.muted}>{S.loading}</Text>
  }

  const listSel = sel > 0 ? sel - 1 : 0
  const viewOffset = windowOffset(listLen, listSel, VISIBLE)
  const visibleCount = Math.max(0, Math.min(VISIBLE, listLen - viewOffset))
  const visibleRows = Array.from({ length: visibleCount }, (_, k) => viewOffset + k + 1)
  const promptTitle = draftTitleFromPrompt(draft) || S.row.startNew

  return (
    <Box flexDirection="column" width={width}>
      <SwitcherHeader
        countLabel={sessionsCountLabel(items.length, history.length)}
        errorMessage={actionErr || pageError || err}
        isOldGateway={isOldGateway}
        t={t}
        title={S.title}
      />

      <SwitcherDraftRow
        draftModel={draftModel}
        onClick={handleRowClick(0)}
        promptTitle={promptTitle}
        selected={sel === 0}
        t={t}
      />

      {viewOffset > 0 && <Text color={t.color.muted}>{C.moreAbove(viewOffset)}</Text>}
      {!listLen && <Text color={t.color.muted}>{S.noOtherSessions}</Text>}

      {visibleRows.map(i => {
        const row = windowedRows[i - 1]

        if (!row) {
          return null
        }

        const selected = sel === i

        if (row.kind === 'pagination-newer' || row.kind === 'pagination-older') {
          return (
            <SwitcherPaginationRow
              failedOffset={failedOffset}
              index={i}
              key={row.kind}
              onClick={handleRowClick(i)}
              pageError={pageError}
              pageLoading={pageLoading}
              row={row}
              selected={selected}
              t={t}
            />
          )
        }

        if (row.kind === 'history') {
          return (
            <SwitcherHistoryRow
              index={i}
              isDeleting={Boolean(deleting && selected)}
              isPendingDelete={confirmDelete === row.item.id}
              item={row.item}
              key={row.item.id}
              onClick={handleRowClick(i)}
              selected={selected}
              t={t}
            />
          )
        }

        return (
          <SwitcherLiveRow
            index={i}
            isClosing={closingId === row.item.id}
            isCurrent={Boolean(row.item.current || row.item.id === currentSessionId)}
            item={row.item}
            key={row.item.id}
            onClick={handleRowClick(i)}
            selected={selected}
            t={t}
          />
        )
      })}

      {viewOffset + VISIBLE < listLen && (
        <Text color={t.color.muted}>{C.moreBelow(listLen - viewOffset - VISIBLE)}</Text>
      )}

      {newSelected ? (
        <>
          <Box marginTop={1}>
            <Text color={t.color.label}>{S.promptLabel}</Text>
            <TextInput
              color={t.color.text}
              columns={promptColumns}
              onChange={setDraft}
              onSubmit={submitDraft}
              value={draft}
            />
          </Box>
          <OrchestratorHintText segments={orchestratorContextHintSegments(true)} t={t} />
          <Text color={t.color.muted} wrap="truncate-end">
            {S.draftModel(draftModelDisplayLabel(draftModel))}
          </Text>
        </>
      ) : (
        <Box flexDirection="column" marginTop={1}>
          <OrchestratorHintText
            segments={
              selectedKind === 'history'
                ? resumeRowContextHintSegments()
                : selectedKind === 'pagination-newer' || selectedKind === 'pagination-older'
                  ? paginationRowContextHintSegments(Boolean(pageError && failedOffset !== null))
                  : orchestratorContextHintSegments(false)
            }
            t={t}
          />
          <Text color={t.color.muted} wrap="truncate-end">
            {S.selectNewPrefix}
            <Text color={newSessionMarkerColor(t, false)}>{S.newRowMarker}</Text>
            {S.selectNewSuffix}
          </Text>
        </Box>
      )}

      <OrchestratorHintText segments={orchestratorGlobalHotkeyHintSegments()} t={t} />
    </Box>
  )
}

interface OrchestratorHintTextProps {
  segments: readonly OrchestratorHintSegment[]
  t: Theme
}

export interface ActiveSessionSwitcherProps {
  currentSessionId: null | string
  filter?: string
  gw: GatewayClient
  maxWidth?: number
  onCancel: () => void
  onClose: (id: string) => Promise<null | SessionCloseResponse>
  onNew: () => void
  onNewPrompt: (prompt: string, modelArg?: string) => void
  onResume: (id: string) => void
  onSelect: (id: string) => void
  profile?: null | string
  t: Theme
}
