import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useCallback, useEffect, useRef, useState } from 'react'

import { TUI_SESSION_MODEL_FLAG } from '../domain/slash.js'
import type { GatewayClient } from '../gatewayClient.js'
import type {
  SessionActiveItem,
  SessionActiveListResponse,
  SessionCloseResponse,
  SessionDeleteResponse,
  SessionListItem,
  SessionListResponse
} from '../gatewayTypes.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { ModelPicker } from './modelPicker.js'
import { windowOffset } from './overlayControls.js'
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

const STATUS_LABEL: Record<string, string> = {
  idle: 'idle',
  starting: 'starting',
  waiting: 'waiting',
  working: 'working'
}

const CTRL_OFFSET = 96

const shortModel = (model = '') => model.replace(/^.*\//, '') || 'model?'
const ctrlChar = (letter: string) => String.fromCharCode(letter.charCodeAt(0) - CTRL_OFFSET)

export const fixedSessionColumnStyle = () => ({ flexShrink: 0 })

export const activeSessionCountLabel = (count: number) =>
  `${count} live ${count === 1 ? 'session' : 'sessions'}`

export const sessionsCountLabel = (liveCount: number, resumableCount: number) =>
  `${liveCount} live · ${resumableCount} resumable`

export type SessionRowKind = 'history' | 'live' | 'new'

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

  if (days < 1) {
    return 'today'
  }

  if (days < 2) {
    return 'yesterday'
  }

  return `${Math.floor(days)}d ago`
}

/** Drop already-live sessions from the resumable history list (dedupe by id). */
export const resumableHistory = (history: readonly SessionListItem[], live: readonly SessionActiveItem[]) => {
  const liveIds = new Set(live.map(s => s.id))

  return history.filter(h => !liveIds.has(h.id))
}

export const resumeRowContextHintSegments: OrchestratorHintSegment[] = [
  { role: 'label', text: 'Resumable:' },
  { role: 'text', text: ' ' },
  { role: 'hotkey', text: 'Enter' },
  { role: 'text', text: ' resume · ' },
  { role: 'hotkey', text: 'd' },
  { role: 'text', text: ' delete' }
]

export type OrchestratorHintRole = 'hotkey' | 'label' | 'text'

export interface OrchestratorHintSegment {
  role: OrchestratorHintRole
  text: string
}

export const orchestratorContextHintSegments = (newSelected: boolean): OrchestratorHintSegment[] =>
  newSelected
    ? [
        { role: 'label', text: 'New row:' },
        { role: 'text', text: ' type prompt · ' },
        { role: 'hotkey', text: 'Enter' },
        { role: 'text', text: ' start · ' },
        { role: 'hotkey', text: 'Tab' },
        { role: 'text', text: ' model' }
      ]
    : [
        { role: 'label', text: 'Session row:' },
        { role: 'text', text: ' ' },
        { role: 'hotkey', text: 'Enter' },
        { role: 'text', text: ' switch · ' },
        { role: 'hotkey', text: 'Ctrl+D' },
        { role: 'text', text: ' close' }
      ]

export const orchestratorGlobalHotkeyHintSegments: OrchestratorHintSegment[] = [
  { role: 'hotkey', text: '↑↓' },
  { role: 'text', text: ' move · ' },
  { role: 'hotkey', text: 'Ctrl+N' },
  { role: 'text', text: ' new · ' },
  { role: 'hotkey', text: 'Ctrl+R' },
  { role: 'text', text: ' refresh · ' },
  { role: 'hotkey', text: 'Esc' },
  { role: 'text', text: ' close' }
]

const hintText = (segments: readonly OrchestratorHintSegment[]) => segments.map(segment => segment.text).join('')

export const orchestratorContextHint = (newSelected: boolean) => hintText(orchestratorContextHintSegments(newSelected))

export const orchestratorGlobalHotkeyHint = hintText(orchestratorGlobalHotkeyHintSegments)

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
  backgroundColor: t.color.selectionBg,
  color: t.color.text
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
  const parts = value.trim().split(/\s+/).filter(Boolean)
  const kept: string[] = []

  for (const part of parts) {
    if (part === TUI_SESSION_MODEL_FLAG || part === '--global') {
      continue
    }

    kept.push(part)
  }

  return kept.join(' ')
}

export const draftModelNameFromArg = (value: string) => {
  const parts = draftModelArgFromPickerValue(value).split(/\s+/).filter(Boolean)
  const modelParts: string[] = []

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i]!

    if (part === '--provider') {
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

  return modelName ? shortModel(modelName) : 'current/default'
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

export function ActiveSessionSwitcher({
  currentSessionId,
  gw,
  onCancel,
  onClose,
  onNew,
  onNewPrompt,
  onResume,
  onSelect,
  t
}: ActiveSessionSwitcherProps) {
  const [items, setItems] = useState<SessionActiveItem[]>([])
  const [history, setHistory] = useState<SessionListItem[]>([])
  const [err, setErr] = useState('')
  const [sel, setSel] = useState(0)
  const [loading, setLoading] = useState(true)
  const [draft, setDraft] = useState('')
  const [draftModel, setDraftModel] = useState('')
  const [pickingModel, setPickingModel] = useState(false)
  const [closingId, setClosingId] = useState('')
  // When non-null, the user pressed `d` on this (history) session and we await
  // a second `d` to confirm deletion. Tracked by session id (not row index) so
  // the 1.5s live-status poll re-indexing rows can't redirect the delete to a
  // different session. Any other key cancels the prompt.
  const [confirmDelete, setConfirmDelete] = useState<null | string>(null)
  const [deleting, setDeleting] = useState(false)
  const initialSelectionAppliedRef = useRef(false)
  // Holds the RAW `session.list` results (pre-dedupe). The quiet 1.5s poll
  // re-derives the resumable list from this against the latest live set, so a
  // session that was hidden while live reappears in history once it closes —
  // without re-querying the DB. Only refreshed on a full (includeHistory) load.
  const rawHistoryRef = useRef<SessionListItem[]>([])
  // Mirror the displayed lists so the async poll can re-anchor the selection to
  // the *same* row (by session id) after live sessions appear/disappear, rather
  // than keeping a now-stale flat index.
  const itemsRef = useRef<SessionActiveItem[]>([])
  const historyDisplayRef = useRef<SessionListItem[]>([])
  const { stdout } = useStdout()
  const width = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))
  const promptColumns = Math.max(20, width - 11)

  // Rows are [new][live…][history…]: the "+ new" row is pinned first (index 0,
  // always rendered) and the live+history list is windowed below it. `total`
  // is the count of selectable rows (incl. the new row).
  const liveCount = items.length
  const histCount = history.length
  const listLen = liveCount + histCount
  const total = listLen + 1
  const rowKind = useCallback((index: number) => sessionRowKindAt(index, liveCount), [liveCount])

  const load = useCallback(
    // `quiet` skips the loading spinner (used by the live-status poll);
    // `includeHistory` re-queries the resumable DB list (skipped on the 1.5s
    // poll, which only needs fresh live-session status).
    async (quiet = false, includeHistory = true) => {
      if (!quiet) {
        setLoading(true)
      }

      try {
        // Fetch independently (allSettled) so a failing session.list can't
        // wipe the live-session list: live sessions still render and the
        // resumable history degrades on its own.
        const [liveRes, histRes] = await Promise.allSettled([
          gw.request<SessionActiveListResponse>('session.active_list', {
            current_session_id: currentSessionId
          }),
          includeHistory ? gw.request<SessionListResponse>('session.list', { limit: 200 }) : Promise.resolve(null)
        ])

        const r = liveRes.status === 'fulfilled' ? asRpcResult<SessionActiveListResponse>(liveRes.value) : null

        if (!r) {
          setErr('invalid response: session.active_list')
          setLoading(false)

          return []
        }

        const next = r.sessions ?? []

        // Surface a garbled/failed session.list rather than silently blanking
        // the resumable section; keep the last good raw history so a transient
        // failure doesn't wipe it.
        let histError = ''

        if (includeHistory) {
          if (histRes.status === 'fulfilled') {
            const parsedHist = asRpcResult<SessionListResponse>(histRes.value)

            if (parsedHist) {
              rawHistoryRef.current = parsedHist.sessions ?? []
            } else {
              histError = 'invalid response: session.list'
            }
          } else {
            histError = 'could not load resumable sessions'
          }
        }

        const hist = resumableHistory(rawHistoryRef.current, next)
        const initializeSelection = !initialSelectionAppliedRef.current
        initialSelectionAppliedRef.current = true
        const maxSel = next.length + hist.length // == total - 1 (new row is index 0)

        setItems(next)
        setHistory(hist)
        // Re-anchor selection to the same row by identity (the live list can
        // grow/shrink between polls, which would otherwise drift a flat index).
        setSel(s => {
          if (initializeSelection) {
            // Land on the current live session (shifted +1 past the pinned new
            // row); with no live sessions, start on the new row itself.
            return next.length ? Math.min(currentSessionSelectionIndex(next, currentSessionId) + 1, maxSel) : 0
          }

          if (s <= 0) {
            return 0 // "+ new" row
          }

          const prevItems = itemsRef.current
          const prevHist = historyDisplayRef.current
          const clamp = () => Math.max(0, Math.min(s, maxSel))

          if (s - 1 < prevItems.length) {
            const id = prevItems[s - 1]?.id
            const i = id ? next.findIndex(x => x.id === id) : -1

            return i >= 0 ? i + 1 : clamp()
          }

          const id = prevHist[s - 1 - prevItems.length]?.id
          const i = id ? hist.findIndex(x => x.id === id) : -1

          return i >= 0 ? 1 + next.length + i : clamp()
        })
        setErr(histError)
        setLoading(false)

        return next
      } catch (e: unknown) {
        setErr(rpcErrorMessage(e))
        setLoading(false)

        return []
      }
    },
    [currentSessionId, gw]
  )

  useEffect(() => {
    itemsRef.current = items
    historyDisplayRef.current = history
  }, [items, history])

  useEffect(() => {
    void load()
    const timer = setInterval(() => void load(true, false), 1500)

    return () => clearInterval(timer)
  }, [load])

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
    const target = items[sel - 1]

    if (!target || rowKind(sel) !== 'live' || closingId) {
      return
    }

    setErr('')
    setClosingId(target.id)

    try {
      const result = await onClose(target.id)
      const closed = Boolean(result?.closed ?? result?.ok)

      if (!closed) {
        setErr('session was already closed')

        return
      }

      const remaining = await load(true)
      const fallback = closeFallbackAfterClose(target.id, currentSessionId, remaining)

      if (fallback.action === 'activate') {
        onSelect(fallback.sessionId)
      } else if (fallback.action === 'new') {
        onNew()
      } else {
        setSel(s => Math.max(0, Math.min(s, remaining.length + history.length)))
      }
    } catch (e: unknown) {
      setErr(rpcErrorMessage(e))
    } finally {
      setClosingId('')
    }
  }, [closingId, currentSessionId, history.length, items, load, onClose, onNew, onSelect, rowKind, sel])

  const performDelete = useCallback(
    (id: string) => {
      const target = history.find(h => h.id === id)

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

          rawHistoryRef.current = rawHistoryRef.current.filter(h => h.id !== target.id)
          setHistory(prev => prev.filter(h => h.id !== target.id))
          setSel(s => Math.max(0, Math.min(s, items.length + history.length - 1)))
          setErr('')
          setDeleting(false)
        })
        .catch((e: unknown) => {
          setErr(rpcErrorMessage(e))
          setDeleting(false)
        })
    },
    [deleting, gw, history, items.length]
  )

  const handleRowClick = useCallback(
    (index: number) => (event: { stopImmediatePropagation?: () => void }) => {
      event.stopImmediatePropagation?.()
      const kind = rowKind(index)
      const clamped = Math.max(0, Math.min(index, total - 1))

      if (kind === 'live') {
        setSel(clamped)
        onSelect(items[index - 1]!.id)

        return
      }

      if (kind === 'history') {
        setSel(clamped)
        onResume(history[index - 1 - items.length]!.id)

        return
      }

      setSel(0)
    },
    [history, items, onResume, onSelect, rowKind, total]
  )

  const selectedKind = rowKind(sel)
  const newSelected = selectedKind === 'new'
  const draftHasText = Boolean(draft.trim())

  useInput((ch, key) => {
    if (pickingModel || deleting) {
      return
    }

    // Two-press history delete: once armed, only a second `d` deletes; any
    // other key cancels the prompt (mirrors the standalone resume picker).
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

    const lower = ch?.toLowerCase() ?? ''
    const isCtrl = (letter: string) => key.ctrl && (lower === letter || ch === ctrlChar(letter))

    if (key.escape) {
      return onCancel()
    }

    if (isCtrl('n')) {
      return onNew()
    }

    if (isCtrl('r')) {
      void load()

      return
    }

    if (key.tab) {
      if (newSelected) {
        setPickingModel(true)
      }

      return
    }

    if (isCtrl('d')) {
      if (selectedKind === 'live') {
        void closeSelected()
      }

      return
    }

    // `d` arms deletion on a resumable history row. (On the New row `d` is
    // captured by the prompt's TextInput, so it never reaches here.)
    if (lower === 'd' && !key.ctrl && selectedKind === 'history') {
      setConfirmDelete(history[sel - 1 - items.length]?.id ?? null)

      return
    }

    if (newSelected && draftHasText) {
      return
    }

    if (key.upArrow && sel > 0) {
      return setSel(s => Math.max(0, s - 1))
    }

    if (key.downArrow && sel < total - 1) {
      return setSel(s => Math.min(total - 1, s + 1))
    }

    if (key.return) {
      if (newSelected) {
        if (!draftHasText) {
          return onNew()
        }

        return
      }

      if (selectedKind === 'live' && items[sel - 1]) {
        return onSelect(items[sel - 1]!.id)
      }

      if (selectedKind === 'history' && history[sel - 1 - items.length]) {
        return onResume(history[sel - 1 - items.length]!.id)
      }
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
    return <Text color={t.color.muted}>loading sessions…</Text>
  }

  // The "+ new" row (sel 0) is pinned at the top so it's always visible; the
  // live + history list is windowed beneath it.
  const listSel = sel > 0 ? sel - 1 : 0
  const offset = windowOffset(listLen, listSel, VISIBLE)
  const visibleCount = Math.max(0, Math.min(VISIBLE, listLen - offset))
  const visibleRows = Array.from({ length: visibleCount }, (_, k) => offset + k + 1)

  const newSelectedRow = sel === 0
  const newRowStyle = newSelectedRow ? selectedSessionRowStyle(t) : null
  const newRowTextColor = newRowStyle?.color
  const newRowMarkerColor = newSessionMarkerColor(t, newSelectedRow)
  const promptTitle = draftTitleFromPrompt(draft) || 'Start a new live session'

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent}>
        Sessions
      </Text>
      <Text color={t.color.muted}>{sessionsCountLabel(items.length, history.length)}</Text>

      {err && <Text color={t.color.label}>error: {err}</Text>}

      <Box
        backgroundColor={newRowStyle?.backgroundColor}
        flexDirection="row"
        onClick={handleRowClick(0)}
        width="100%"
      >
        <Text bold={newSelectedRow} color={newRowTextColor ?? t.color.muted}>
          {newSelectedRow ? '▸ ' : '  '}
        </Text>

        <Box {...fixedSessionColumnStyle()} width={5}>
          <Text bold={newSelectedRow} color={newRowMarkerColor}>
            {'+'.padStart(2)}
          </Text>
        </Box>

        <Box {...fixedSessionColumnStyle()} width={11}>
          <Text bold={newSelectedRow} color={newRowMarkerColor} wrap="truncate-end">
            new
          </Text>
        </Box>

        <Box {...fixedSessionColumnStyle()} width={11}>
          <Text color={newRowTextColor ?? t.color.muted} wrap="truncate-end">
            ✎ draft
          </Text>
        </Box>

        <Box {...fixedSessionColumnStyle()} width={18}>
          <Text color={newRowTextColor ?? t.color.muted} wrap="truncate-end">
            {draftModelDisplayLabel(draftModel)}
          </Text>
        </Box>

        <Box flexGrow={1} flexShrink={1} minWidth={0}>
          <Text bold={newSelectedRow} color={newRowTextColor ?? t.color.muted} wrap="truncate-end">
            {promptTitle}
          </Text>
        </Box>
      </Box>

      {offset > 0 && <Text color={t.color.muted}> ↑ {offset} more</Text>}
      {!listLen && <Text color={t.color.muted}>no other sessions — Enter on +new to start one</Text>}

      {visibleRows.map(i => {
        const selected = sel === i
        const selectedStyle = selected ? selectedSessionRowStyle(t) : null
        const rowTextColor = selectedStyle?.color
        const kind = rowKind(i)

        if (kind === 'history') {
          const h = history[i - 1 - items.length]!
          const pendingDelete = confirmDelete === h.id

          const title = pendingDelete
            ? 'press d again to delete'
            : deleting && selected
              ? 'deleting…'
              : h.title || h.preview || '(untitled)'

          return (
            <Box
              backgroundColor={selectedStyle?.backgroundColor}
              flexDirection="row"
              key={h.id}
              onClick={handleRowClick(i)}
              width="100%"
            >
              <Text bold={selected} color={rowTextColor ?? t.color.muted}>
                {selected ? '▸ ' : '  '}
              </Text>

              <Box {...fixedSessionColumnStyle()} width={5}>
                <Text bold={selected} color={rowTextColor ?? t.color.muted}>
                  {String(i).padStart(2)}.
                </Text>
              </Box>

              <Box {...fixedSessionColumnStyle()} width={11}>
                <Text bold={selected} color={rowTextColor ?? t.color.muted} wrap="truncate-end">
                  {h.id}
                </Text>
              </Box>

              <Box {...fixedSessionColumnStyle()} width={11}>
                <Text color={rowTextColor ?? t.color.muted} wrap="truncate-end">
                  {relativeSessionAge(h.started_at)}
                </Text>
              </Box>

              <Box {...fixedSessionColumnStyle()} width={18}>
                <Text color={rowTextColor ?? t.color.muted} wrap="truncate-end">
                  {h.message_count} msgs
                </Text>
              </Box>

              <Box flexGrow={1} flexShrink={1} minWidth={0}>
                <Text
                  bold={selected}
                  color={pendingDelete ? t.color.label : rowTextColor ?? t.color.muted}
                  wrap="truncate-end"
                >
                  {title}
                </Text>
              </Box>
            </Box>
          )
        }

        const s = items[i - 1]!
        const status = s.status ?? 'idle'
        const current = s.current || s.id === currentSessionId
        const title = closingId === s.id ? 'closing…' : s.title || s.preview || '(untitled)'

        return (
          <Box
            backgroundColor={selectedStyle?.backgroundColor}
            flexDirection="row"
            key={s.id}
            onClick={handleRowClick(i)}
            width="100%"
          >
            <Text bold={selected} color={rowTextColor ?? t.color.muted}>
              {selected ? '▸ ' : '  '}
            </Text>

            <Box {...fixedSessionColumnStyle()} width={5}>
              <Text bold={selected} color={rowTextColor ?? t.color.muted}>
                {String(i).padStart(2)}.
              </Text>
            </Box>

            <Box {...fixedSessionColumnStyle()} width={11}>
              <Text
                bold={selected}
                color={rowTextColor ?? (current ? t.color.label : t.color.muted)}
                wrap="truncate-end"
              >
                {current ? 'current' : s.id}
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
                {STATUS_GLYPH[status] ?? '·'} {STATUS_LABEL[status] ?? status}
              </Text>
            </Box>

            <Box {...fixedSessionColumnStyle()} width={18}>
              <Text color={rowTextColor ?? t.color.muted} wrap="truncate-end">
                {shortModel(s.model)}
              </Text>
            </Box>

            <Box flexGrow={1} flexShrink={1} minWidth={0}>
              <Text bold={selected} color={rowTextColor ?? t.color.muted} wrap="truncate-end">
                {title}
              </Text>
            </Box>
          </Box>
        )
      })}

      {offset + VISIBLE < listLen && <Text color={t.color.muted}> ↓ {listLen - offset - VISIBLE} more</Text>}

      {newSelected ? (
        <>
          <Box marginTop={1}>
            <Text color={t.color.label}>prompt › </Text>
            <TextInput columns={promptColumns} onChange={setDraft} onSubmit={submitDraft} value={draft} />
          </Box>
          <OrchestratorHintText segments={orchestratorContextHintSegments(true)} t={t} />
          <Text color={t.color.muted} wrap="truncate-end">
            model: {draftModelDisplayLabel(draftModel)}
          </Text>
        </>
      ) : (
        <Box flexDirection="column" marginTop={1}>
          <OrchestratorHintText
            segments={selectedKind === 'history' ? resumeRowContextHintSegments : orchestratorContextHintSegments(false)}
            t={t}
          />
          <Text color={t.color.muted} wrap="truncate-end">
            Select <Text color={newSessionMarkerColor(t, false)}>+new</Text> to type a prompt
          </Text>
        </Box>
      )}

      <OrchestratorHintText segments={orchestratorGlobalHotkeyHintSegments} t={t} />
    </Box>
  )
}

interface OrchestratorHintTextProps {
  segments: readonly OrchestratorHintSegment[]
  t: Theme
}

interface ActiveSessionSwitcherProps {
  currentSessionId: null | string
  gw: GatewayClient
  onCancel: () => void
  onClose: (id: string) => Promise<null | SessionCloseResponse>
  onNew: () => void
  onNewPrompt: (prompt: string, modelArg?: string) => void
  onResume: (id: string) => void
  onSelect: (id: string) => void
  t: Theme
}
