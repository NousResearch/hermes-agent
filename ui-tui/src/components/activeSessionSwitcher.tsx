import { Box, Text, useInput, useStdout } from '@hermes/ink'
import type {
  ProjectsTreeResult,
  ProjectTreeNode,
  SessionListResult,
  SessionListRow
} from '@hermes/shared/gateway-events'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { mutateCanonical } from '../app/slash/canonicalSessionControls.js'
import {
  DEFAULT_GROUP_VIEW,
  deriveSwitcherRows,
  groupKey,
  groupSessionsByProject,
  type GroupView,
  nextGroupView,
  type SessionGroup,
  type SwitcherRow,
  switcherRowKey
} from '../domain/sessionGroups.js'
import { sessionScopedModelArg } from '../domain/slash.js'
import { type SwitcherIntent, switcherKeyIntent, type SwitcherRowKind } from '../domain/switcherKeys.js'
import type { GatewayClient } from '../gatewayClient.js'
import type {
  SessionActiveItem,
  SessionActiveListResponse,
  SessionCloseResponse,
  SessionDeleteResponse
} from '../gatewayTypes.js'
import { messages } from '../i18n/runtime.js'
import { useT } from '../i18n/useT.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { ModelPicker } from './modelPicker.js'
import { windowOffset } from './overlayControls.js'
import { clampOverlayWidth, listRowStyle } from './overlayPrimitives.js'
import { TextInput } from './textInput.js'

const VISIBLE = 12

type SwitcherListRow = SwitcherRow<SessionActiveItem, SessionListRow>
type SwitcherHeaderRow = Extract<SwitcherListRow, { kind: 'header' }>
// Enough to cover every row `session.list` can return (its own limit is 200).
const PROJECT_TREE_SESSION_LIMIT = 500
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

const shortModel = (model = '') => model.replace(/^.*\//, '') || messages().pickers.session.modelUnknown

export const fixedSessionColumnStyle = () => ({ flexShrink: 0 })

export const activeSessionCountLabel = (count: number) =>
  count === 1 ? messages().pickers.session.liveSessionsOne(1) : messages().pickers.session.liveSessionsOther(count)

export const sessionsCountLabel = (liveCount: number, resumableCount: number) =>
  messages().pickers.session.counts(liveCount, resumableCount)

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

  const age = messages().pickers.session.age

  if (days < 1) {
    return age.today
  }

  if (days < 2) {
    return age.yesterday
  }

  return age.daysAgo(Math.floor(days))
}

/** Drop already-live sessions from the resumable history list (dedupe by id). */
export const resumableHistory = (history: readonly SessionListRow[], live: readonly SessionActiveItem[]) => {
  const liveIds = new Set(live.map(s => s.id))

  return history.filter(h => !liveIds.has(h.id))
}

// Hint segment tables are functions (not module constants) so their text
// fragments resolve against the active language at render time.
export const resumeRowContextHintSegments = (toggle: 'group' | 'ungroup' | null = null): OrchestratorHintSegment[] => {
  const h = messages().pickers.session.hint

  return [
    { role: 'label', text: h.resumableLabel },
    { role: 'text', text: ' ' },
    { role: 'hotkey', text: 'Enter' },
    { role: 'text', text: h.resume },
    { role: 'hotkey', text: 'd' },
    { role: 'text', text: h.delete },
    ...(toggle
      ? [
          { role: 'text', text: ' · ' } as const,
          { role: 'hotkey', text: 'g' } as const,
          { role: 'text', text: toggle === 'group' ? h.group : h.ungroup } as const
        ]
      : [])
  ]
}

/** Hint for a selected project header row. */
export const groupRowContextHintSegments = (): OrchestratorHintSegment[] => {
  const h = messages().pickers.session.hint

  return [
    { role: 'label', text: h.groupRowLabel },
    { role: 'text', text: ' ' },
    { role: 'hotkey', text: 'Enter' },
    { role: 'text', text: h.cycleGroup },
    { role: 'hotkey', text: 'g' },
    { role: 'text', text: h.ungroup }
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

// Delegates to the shared list-row primitive so the session switcher and the
// completions popover cannot disagree about what "selected" looks like.
// (`selectionBg` remains the TEXT-selection highlight — a different semantic.)
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

export function ActiveSessionSwitcher({
  currentSessionId,
  gw,
  maxWidth,
  onCancel,
  onClose,
  onNew,
  onNewPrompt,
  onResume,
  onSelect,
  t
}: ActiveSessionSwitcherProps) {
  const [items, setItems] = useState<SessionActiveItem[]>([])
  const [history, setHistory] = useState<SessionListRow[]>([])
  // Project buckets over `history`; empty when `projects.tree` is unavailable (flat list).
  const [historyGroups, setHistoryGroups] = useState<SessionGroup<SessionListRow>[]>([])
  // `g` flips between project buckets and the flat most-recent-first list;
  // `groupViews` is each bucket's collapsed / first-3 / all state (default first-3).
  const [grouped, setGrouped] = useState(true)
  const [groupViews, setGroupViews] = useState<Record<string, GroupView>>({})
  // `/` opens the search box (`searching`: keys go to it); the `query` filter outlives it until Esc.
  const [query, setQuery] = useState('')
  const [searching, setSearching] = useState(false)
  const [err, setErr] = useState('')
  // A close/delete failure: kept apart from `err`, which every quiet poll rewrites.
  const [actionErr, setActionErr] = useState('')
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
  const T = useT()
  const S = T.pickers.session
  const C = T.pickers.common
  const initialSelectionAppliedRef = useRef(false)
  // Holds the RAW `session.list` results (pre-dedupe). The quiet 1.5s poll
  // re-derives the resumable list from this against the latest live set, so a
  // session that was hidden while live reappears in history once it closes —
  // without re-querying the DB. Only refreshed on a full (includeHistory) load.
  const rawHistoryRef = useRef<SessionListRow[]>([])
  // Last good `projects.tree` nodes, kept across polls like the raw history.
  const projectsRef = useRef<ProjectTreeNode[]>([])
  // Mirror the displayed rows (and the view settings that shaped them) so the
  // async poll can re-anchor the selection to the *same* row (by key) after
  // rows appear/disappear, rather than keeping a now-stale flat index.
  const rowsRef = useRef<SwitcherListRow[]>([])
  const groupedRef = useRef(grouped)
  const groupViewsRef = useRef(groupViews)
  const queryRef = useRef(query)
  const { stdout } = useStdout()
  // Optional maxWidth lets grid layouts hand the switcher its cell budget.
  const preferredWidth = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))
  const width = clampOverlayWidth(preferredWidth, maxWidth)
  const promptColumns = Math.max(20, width - 11)

  // Rows are [new][live…][headers + history…]: the "+ new" row is pinned first
  // (index 0, always rendered) and the rest is windowed below it. Project
  // headers are selectable rows (Enter / click cycles their view). `total` is
  // the count of selectable rows (incl. the new row).
  const rows = useMemo(
    () => deriveSwitcherRows(items, history, historyGroups, { grouped, query, views: groupViews }),
    [grouped, groupViews, historyGroups, history, items, query]
  )

  const listLen = rows.length
  const total = listLen + 1

  // Resumable rows (raw most-recent-first, for the flat list) plus their project
  // buckets; buckets stay empty when `projects.tree` is unavailable.
  const deriveHistory = useCallback((live: readonly SessionActiveItem[]) => {
    const hist = resumableHistory(rawHistoryRef.current, live)

    const groups = projectsRef.current.length
      ? groupSessionsByProject(hist, projectsRef.current, messages().pickers.session.otherProject)
      : []

    return { groups, hist }
  }, [])

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
        const canonicalList = gw.isCanonical
          ? gw.request<{ scope?: string; sessions?: (SessionListRow & { running?: boolean })[] }>('session.list', {
              limit: 200
            })
          : null

        const [liveRes, histRes, treeRes] = await Promise.allSettled([
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
            : Promise.resolve(null),
          // Grouping is decoration: a failing tree degrades to the flat list. The shared
          // (canonical) gateway has no `projects.tree`, so the picker stays flat there.
          includeHistory && !gw.isCanonical
            ? gw.request<ProjectsTreeResult>('projects.tree', {
                preview_limit: 1,
                session_limit: PROJECT_TREE_SESSION_LIMIT
              })
            : Promise.resolve(null)
        ])

        const r = liveRes.status === 'fulfilled' ? asRpcResult<SessionActiveListResponse>(liveRes.value) : null

        if (!r) {
          setErr(messages().pickers.common.invalidResponse('session.active_list'))
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
            const parsedHist = asRpcResult<SessionListResult>(histRes.value)

            if (parsedHist) {
              rawHistoryRef.current = parsedHist.sessions ?? []
            } else {
              histError = messages().pickers.common.invalidResponse('session.list')
            }
          } else {
            histError = messages().pickers.session.errors.couldNotLoadResumable
          }
        }

        if (includeHistory && treeRes.status === 'fulfilled') {
          const tree = asRpcResult<ProjectsTreeResult>(treeRes.value)

          if (tree) {
            projectsRef.current = tree.projects ?? []
          }
        }

        const { groups, hist } = deriveHistory(next)
        const initializeSelection = !initialSelectionAppliedRef.current
        initialSelectionAppliedRef.current = true

        const nextRows = deriveSwitcherRows(next, hist, groups, {
          grouped: groupedRef.current,
          query: queryRef.current,
          views: groupViewsRef.current
        })

        const maxSel = nextRows.length // == total - 1 (new row is index 0)

        setItems(next)
        setHistory(hist)
        setHistoryGroups(groups)
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

          const prev = rowsRef.current[s - 1]
          const i = prev ? nextRows.findIndex(r => switcherRowKey(r) === switcherRowKey(prev)) : -1

          return i >= 0 ? i + 1 : Math.max(0, Math.min(s, maxSel))
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
    [currentSessionId, deriveHistory, gw]
  )

  useEffect(() => {
    rowsRef.current = rows
    groupedRef.current = grouped
    groupViewsRef.current = groupViews
    queryRef.current = query
  }, [rows, grouped, groupViews, query])

  // Matches shrink as the query grows: keep the cursor on a real row.
  useEffect(() => {
    setSel(s => Math.min(s, listLen))
  }, [listLen])

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
    const selectedRow = rows[sel - 1]
    const target = selectedRow?.kind === 'live' ? selectedRow.item : null

    if (!target || closingId) {
      return
    }

    setActionErr('')
    setClosingId(target.id)

    try {
      const result = await onClose(target.id)
      const closed = Boolean(result?.closed ?? result?.ok)

      if (!closed) {
        setActionErr(messages().pickers.session.errors.alreadyClosed)

        return
      }

      const remaining = await load(true)
      const fallback = closeFallbackAfterClose(target.id, currentSessionId, remaining)

      if (fallback.action === 'activate') {
        onSelect(fallback.sessionId)
      } else if (fallback.action === 'new') {
        onNew()
      } else {
        setSel(s => Math.max(0, Math.min(s, rows.length - 1)))
      }
    } catch (e: unknown) {
      setActionErr(rpcErrorMessage(e))
    } finally {
      setClosingId('')
    }
  }, [closingId, currentSessionId, load, onClose, onNew, onSelect, rows, sel])

  const performDelete = useCallback(
    (id: string) => {
      const target = history.find(h => h.id === id)

      if (!target || deleting) {
        return
      }

      setDeleting(true)
      setActionErr('')

      // The shared gateway has no `session.delete`: its delete is the revision-fenced
      // `session.mutate operation=delete` (the whole lineage), as the classic CLI sends it.
      const deleted = gw.isCanonical
        ? mutateCanonical(gw, target.id, 'delete', {}).then(mutation => Boolean(mutation?.result))
        : gw
            .request<SessionDeleteResponse>('session.delete', { session_id: target.id })
            .then(raw => asRpcResult<SessionDeleteResponse>(raw)?.deleted === target.id)

      deleted
        .then(ok => {
          if (!ok) {
            setActionErr(
              messages().pickers.common.invalidResponse(gw.isCanonical ? 'session.mutate' : 'session.delete')
            )
            setDeleting(false)

            return
          }

          rawHistoryRef.current = rawHistoryRef.current.filter(h => h.id !== target.id)
          const derived = deriveHistory(items)
          setHistory(derived.hist)
          setHistoryGroups(derived.groups)
          setSel(s => Math.max(0, Math.min(s, rows.length - 1)))
          setDeleting(false)
        })
        .catch((e: unknown) => {
          setActionErr(rpcErrorMessage(e))
          setDeleting(false)
        })
    },
    [deleting, deriveHistory, gw, history, items, rows.length]
  )

  const cycleGroupView = useCallback((group: SessionGroup<SessionListRow>) => {
    const key = groupKey(group)

    setGroupViews(v => ({ ...v, [key]: nextGroupView(v[key] ?? DEFAULT_GROUP_VIEW, group.rows.length) }))
  }, [])

  // Re-anchor the selection by row key when the row set changes shape. A header
  // that disappears (ungrouping) hands the cursor to its first session.
  const toggleGrouped = useCallback(() => {
    if (!historyGroups.length) {
      return
    }

    const prev = rows[sel - 1]
    const nextGrouped = !grouped

    const nextRows = deriveSwitcherRows(items, history, historyGroups, {
      grouped: nextGrouped,
      query,
      views: groupViews
    })

    const anchor =
      prev?.kind === 'header' && !nextGrouped ? `hist:${prev.group.rows[0]?.id}` : prev && switcherRowKey(prev)

    const i = anchor ? nextRows.findIndex(r => switcherRowKey(r) === anchor) : -1

    setGrouped(nextGrouped)
    setSel(i >= 0 ? i + 1 : Math.min(sel, nextRows.length))
  }, [grouped, groupViews, history, historyGroups, items, query, rows, sel])

  const activateRow = useCallback(
    (row: SwitcherListRow) => {
      if (row.kind === 'live') {
        onSelect(row.item.id)
      } else if (row.kind === 'history') {
        onResume(row.row.id)
      } else {
        cycleGroupView(row.group)
      }
    },
    [cycleGroupView, onResume, onSelect]
  )

  const handleRowClick = useCallback(
    (index: number) => (event: { stopImmediatePropagation?: () => void }) => {
      event.stopImmediatePropagation?.()
      const row = rows[index - 1]

      setSel(row ? Math.max(0, Math.min(index, total - 1)) : 0)

      if (row) {
        activateRow(row)
      }
    },
    [activateRow, rows, total]
  )

  const onQueryChange = useCallback((value: string) => {
    setQuery(value)
    setSel(1)
  }, [])

  const selectedRow = rows[sel - 1]
  const selectedKind: SwitcherRowKind = sel === 0 || !selectedRow ? 'new' : selectedRow.kind
  const draftHasText = Boolean(draft.trim())

  // What each key meaning does; which key means what is `switcherKeyIntent` (domain/switcherKeys.ts).
  const keyIntents: Record<Exclude<SwitcherIntent, 'none'>, () => void> = {
    activate: () => (selectedRow ? activateRow(selectedRow) : onNew()),
    'arm-delete': () => setConfirmDelete(selectedRow?.kind === 'history' ? selectedRow.row.id : null),
    cancel: onCancel,
    'clear-query': () => setQuery(''),
    'close-live': () => void closeSelected(),
    'delete-confirm': () => {
      setConfirmDelete(null)

      if (confirmDelete !== null) {
        performDelete(confirmDelete)
      }
    },
    'delete-dismiss': () => setConfirmDelete(null),
    down: () => setSel(s => Math.min(total - 1, s + 1)),
    new: onNew,
    'open-search': () => setSearching(true),
    'pick-model': () => setPickingModel(true),
    refresh: () => void load(),
    'search-cancel': () => {
      setSearching(false)
      setQuery('')
    },
    'search-down': () => setSel(s => Math.min(total - 1, s + 1)),
    'search-up': () => setSel(s => Math.max(rows.length ? 1 : 0, s - 1)),
    'toggle-group': toggleGrouped,
    up: () => setSel(s => Math.max(0, s - 1))
  }

  useInput((ch, key) => {
    const intent = switcherKeyIntent(
      {
        busy: pickingModel || deleting,
        canMoveDown: sel < total - 1,
        canMoveUp: sel > 0,
        confirmingDelete: confirmDelete !== null,
        draftHasText,
        hasQuery: Boolean(query),
        kind: selectedKind,
        searching
      },
      ch,
      key
    )

    if (intent !== 'none') {
      keyIntents[intent]()
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

  // The "+ new" row (sel 0) is pinned at the top so it's always visible; the
  // live + history list is windowed beneath it.
  const listSel = sel > 0 ? sel - 1 : 0
  const offset = windowOffset(listLen, listSel, VISIBLE)
  const visibleCount = Math.max(0, Math.min(VISIBLE, listLen - offset))
  const visibleRows = Array.from({ length: visibleCount }, (_, k) => offset + k + 1)

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent}>
        {S.title}
      </Text>
      <Text color={t.color.muted}>{sessionsCountLabel(items.length, history.length)}</Text>

      {(actionErr || err) && <Text color={t.color.label}>{C.error(actionErr || err)}</Text>}

      {(searching || query) && (
        <SearchLine
          columns={promptColumns}
          onChange={onQueryChange}
          onSubmit={() => setSearching(false)}
          query={query}
          searching={searching}
          t={t}
        />
      )}

      <NewSessionRow draft={draft} draftModel={draftModel} onClick={handleRowClick(0)} selected={sel === 0} t={t} />

      {offset > 0 && <Text color={t.color.muted}>{C.moreAbove(offset)}</Text>}
      {!listLen && <Text color={t.color.muted}>{query ? S.noMatches : S.noOtherSessions}</Text>}

      {visibleRows.map(i => (
        <SwitcherListRowView
          closingId={closingId}
          confirmDelete={confirmDelete}
          currentSessionId={currentSessionId}
          deleting={deleting}
          index={i}
          key={switcherRowKey(rows[i - 1]!)}
          onClick={handleRowClick(i)}
          row={rows[i - 1]!}
          selected={sel === i}
          t={t}
        />
      ))}

      {offset + VISIBLE < listLen && <Text color={t.color.muted}>{C.moreBelow(listLen - offset - VISIBLE)}</Text>}

      <SwitcherFooter
        draft={draft}
        draftModel={draftModel}
        kind={selectedKind}
        onDraftChange={setDraft}
        onDraftSubmit={submitDraft}
        promptColumns={promptColumns}
        searching={searching}
        t={t}
        toggle={historyGroups.length ? (grouped ? 'ungroup' : 'group') : null}
      />

      <OrchestratorHintText segments={orchestratorGlobalHotkeyHintSegments()} t={t} />
    </Box>
  )
}

const rowSelection = (t: Theme, selected: boolean) => {
  const style = selected ? selectedSessionRowStyle(t) : null

  return { backgroundColor: style?.backgroundColor, color: style?.color }
}

interface SwitcherRowProps {
  onClick: (event: { stopImmediatePropagation?: () => void }) => void
  selected: boolean
  t: Theme
}

function SearchLine({
  columns,
  onChange,
  onSubmit,
  query,
  searching,
  t
}: {
  columns: number
  onChange: (value: string) => void
  onSubmit: () => void
  query: string
  searching: boolean
  t: Theme
}) {
  const S = useT().pickers.session

  return (
    <Box>
      <Text color={t.color.label}>{S.searchPrompt}</Text>
      {searching ? (
        <TextInput color={t.color.text} columns={columns} onChange={onChange} onSubmit={onSubmit} value={query} />
      ) : (
        <Text color={t.color.text}>
          {query}
          <Text color={t.color.muted}>{S.searchClear}</Text>
        </Text>
      )}
    </Box>
  )
}

function NewSessionRow({
  draft,
  draftModel,
  onClick,
  selected,
  t
}: SwitcherRowProps & { draft: string; draftModel: string }) {
  const S = useT().pickers.session
  const { backgroundColor, color } = rowSelection(t, selected)
  const markerColor = newSessionMarkerColor(t, selected)

  return (
    <Box backgroundColor={backgroundColor} flexDirection="row" onClick={onClick} width="100%">
      <Text bold={selected} color={color ?? t.color.muted}>
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
        <Text color={color ?? t.color.muted} wrap="truncate-end">
          {S.row.draft}
        </Text>
      </Box>

      <Box {...fixedSessionColumnStyle()} width={18}>
        <Text color={color ?? t.color.muted} wrap="truncate-end">
          {draftModelDisplayLabel(draftModel)}
        </Text>
      </Box>

      <Box flexGrow={1} flexShrink={1} minWidth={0}>
        <Text bold={selected} color={color ?? t.color.muted} wrap="truncate-end">
          {draftTitleFromPrompt(draft) || S.row.startNew}
        </Text>
      </Box>
    </Box>
  )
}

function GroupHeaderRow({ row, onClick, selected, t }: SwitcherRowProps & { row: SwitcherHeaderRow }) {
  const { group, shown, view } = row
  const { backgroundColor, color } = rowSelection(t, selected)
  const marker = view === 'collapsed' ? '▸' : view === 'all' ? '▼' : '▾'

  // A collapsed header shows no sessions by design, so `0/n` would read as an empty project.
  const count =
    view === 'collapsed' || shown === group.rows.length ? `${group.rows.length}` : `${shown}/${group.rows.length}`

  return (
    <Box backgroundColor={backgroundColor} flexDirection="row" onClick={onClick} width="100%">
      <Text bold={selected} color={color ?? t.color.muted}>
        {selected ? '▸ ' : '  '}
      </Text>
      <Text bold color={color ?? t.color.accent} wrap="truncate-end">
        {`${marker} ${group.label} (${count})`}
      </Text>
    </Box>
  )
}

function HistoryRow({
  deleting,
  h,
  index,
  onClick,
  pendingDelete,
  selected,
  t
}: SwitcherRowProps & { deleting: boolean; h: SessionListRow; index: number; pendingDelete: boolean }) {
  const S = useT().pickers.session
  const { backgroundColor, color } = rowSelection(t, selected)

  const title = pendingDelete
    ? S.row.pressDAgain
    : deleting && selected
      ? S.row.deleting
      : h.title || h.preview || S.row.untitled

  return (
    <Box backgroundColor={backgroundColor} flexDirection="row" onClick={onClick} width="100%">
      <Text bold={selected} color={color ?? t.color.muted}>
        {selected ? '▸ ' : '  '}
      </Text>

      <Box {...fixedSessionColumnStyle()} width={5}>
        <Text bold={selected} color={color ?? t.color.muted}>
          {String(index).padStart(2)}.
        </Text>
      </Box>

      <Box {...fixedSessionColumnStyle()} width={11}>
        <Text bold={selected} color={color ?? t.color.muted} wrap="truncate-end">
          {h.id}
        </Text>
      </Box>

      <Box {...fixedSessionColumnStyle()} width={11}>
        <Text color={color ?? t.color.muted} wrap="truncate-end">
          {relativeSessionAge(h.started_at)}
        </Text>
      </Box>

      <Box {...fixedSessionColumnStyle()} width={18}>
        <Text color={color ?? t.color.muted} wrap="truncate-end">
          {S.row.messageCount(h.message_count ?? 0)}
        </Text>
      </Box>

      <Box flexGrow={1} flexShrink={1} minWidth={0}>
        <Text bold={selected} color={pendingDelete ? t.color.label : (color ?? t.color.muted)} wrap="truncate-end">
          {title}
        </Text>
      </Box>
    </Box>
  )
}

const liveStatusColor = (t: Theme, status: string) =>
  status === 'working' ? t.color.ok : status === 'waiting' ? t.color.label : t.color.muted

function LiveRow({
  closing,
  current,
  index,
  onClick,
  s,
  selected,
  t
}: SwitcherRowProps & { closing: boolean; current: boolean; index: number; s: SessionActiveItem }) {
  const S = useT().pickers.session
  const { backgroundColor, color } = rowSelection(t, selected)
  const status = s.status ?? 'idle'
  const title = closing ? S.row.closing : s.title || s.preview || S.row.untitled

  return (
    <Box backgroundColor={backgroundColor} flexDirection="row" onClick={onClick} width="100%">
      <Text bold={selected} color={color ?? t.color.muted}>
        {selected ? '▸ ' : '  '}
      </Text>

      <Box {...fixedSessionColumnStyle()} width={5}>
        <Text bold={selected} color={color ?? t.color.muted}>
          {String(index).padStart(2)}.
        </Text>
      </Box>

      <Box {...fixedSessionColumnStyle()} width={11}>
        <Text bold={selected} color={color ?? (current ? t.color.label : t.color.muted)} wrap="truncate-end">
          {current ? S.row.current : s.id}
        </Text>
      </Box>

      <Box {...fixedSessionColumnStyle()} width={11}>
        <Text color={color ?? liveStatusColor(t, status)} wrap="truncate-end">
          {STATUS_GLYPH[status] ?? '·'} {sessionStatusLabel(status)}
        </Text>
      </Box>

      <Box {...fixedSessionColumnStyle()} width={18}>
        <Text color={color ?? t.color.muted} wrap="truncate-end">
          {shortModel(s.model)}
        </Text>
      </Box>

      <Box flexGrow={1} flexShrink={1} minWidth={0}>
        <Text bold={selected} color={color ?? t.color.muted} wrap="truncate-end">
          {title}
        </Text>
      </Box>
    </Box>
  )
}

function SwitcherListRowView({
  closingId,
  confirmDelete,
  currentSessionId,
  deleting,
  index,
  row,
  ...rest
}: SwitcherRowProps & {
  closingId: string
  confirmDelete: null | string
  currentSessionId: null | string
  deleting: boolean
  index: number
  row: SwitcherListRow
}) {
  if (row.kind === 'header') {
    return <GroupHeaderRow row={row} {...rest} />
  }

  if (row.kind === 'history') {
    return (
      <HistoryRow
        deleting={deleting}
        h={row.row}
        index={index}
        pendingDelete={confirmDelete === row.row.id}
        {...rest}
      />
    )
  }

  const s = row.item

  return (
    <LiveRow
      closing={closingId === s.id}
      current={Boolean(s.current) || s.id === currentSessionId}
      index={index}
      s={s}
      {...rest}
    />
  )
}

const contextHintSegments = (kind: SwitcherRowKind, toggle: 'group' | 'ungroup' | null) =>
  kind === 'header'
    ? groupRowContextHintSegments()
    : kind === 'history'
      ? resumeRowContextHintSegments(toggle)
      : orchestratorContextHintSegments(false)

function SwitcherFooter({
  draft,
  draftModel,
  kind,
  onDraftChange,
  onDraftSubmit,
  promptColumns,
  searching,
  t,
  toggle
}: {
  draft: string
  draftModel: string
  kind: SwitcherRowKind
  onDraftChange: (value: string) => void
  onDraftSubmit: (value: string) => void
  promptColumns: number
  searching: boolean
  t: Theme
  toggle: 'group' | 'ungroup' | null
}) {
  const S = useT().pickers.session

  if (kind === 'new' && !searching) {
    return (
      <>
        <Box marginTop={1}>
          <Text color={t.color.label}>{S.promptLabel}</Text>
          <TextInput
            color={t.color.text}
            columns={promptColumns}
            onChange={onDraftChange}
            onSubmit={onDraftSubmit}
            value={draft}
          />
        </Box>
        <OrchestratorHintText segments={orchestratorContextHintSegments(true)} t={t} />
        <Text color={t.color.muted} wrap="truncate-end">
          {S.draftModel(draftModelDisplayLabel(draftModel))}
        </Text>
      </>
    )
  }

  return (
    <Box flexDirection="column" marginTop={1}>
      <OrchestratorHintText segments={contextHintSegments(kind, toggle)} t={t} />
      <Text color={t.color.muted} wrap="truncate-end">
        {S.selectNewPrefix}
        <Text color={newSessionMarkerColor(t, false)}>{S.newRowMarker}</Text>
        {S.selectNewSuffix}
      </Text>
      {!searching && <Text color={t.color.muted}>{S.searchHint}</Text>}
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
  maxWidth?: number
  onCancel: () => void
  onClose: (id: string) => Promise<null | SessionCloseResponse>
  onNew: () => void
  onNewPrompt: (prompt: string, modelArg?: string) => void
  onResume: (id: string) => void
  onSelect: (id: string) => void
  t: Theme
}
