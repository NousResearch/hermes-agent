import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useCallback, useEffect, useRef, useState } from 'react'

import { TUI_SESSION_MODEL_FLAG } from '../domain/slash.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { SessionActiveItem, SessionActiveListResponse, SessionCloseResponse } from '../gatewayTypes.js'
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
  onSelect,
  t
}: ActiveSessionSwitcherProps) {
  const [items, setItems] = useState<SessionActiveItem[]>([])
  const [err, setErr] = useState('')
  const [sel, setSel] = useState(0)
  const [loading, setLoading] = useState(true)
  const [draft, setDraft] = useState('')
  const [draftModel, setDraftModel] = useState('')
  const [pickingModel, setPickingModel] = useState(false)
  const [closingId, setClosingId] = useState('')
  const initialSelectionAppliedRef = useRef(false)
  const { stdout } = useStdout()
  const width = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))
  const promptColumns = Math.max(20, width - 11)

  const load = useCallback(
    async (quiet = false) => {
      if (!quiet) {
        setLoading(true)
      }

      try {
        const raw = await gw.request<SessionActiveListResponse>('session.active_list', {
          current_session_id: currentSessionId
        })
        const r = asRpcResult<SessionActiveListResponse>(raw)

        if (!r) {
          setErr('invalid response: session.active_list')
          setLoading(false)

          return []
        }

        const next = r.sessions ?? []
        const initializeSelection = !initialSelectionAppliedRef.current
        initialSelectionAppliedRef.current = true
        setItems(next)
        setSel(s =>
          initializeSelection
            ? clampOrchestratorSelection(currentSessionSelectionIndex(next, currentSessionId), next.length)
            : clampOrchestratorSelection(s, next.length)
        )
        setErr('')
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
    void load()
    const timer = setInterval(() => void load(true), 1500)

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
    const target = items[sel]

    if (!target || isNewSessionRow(sel, items.length) || closingId) {
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
        setSel(s => clampOrchestratorSelection(s, remaining.length))
      }
    } catch (e: unknown) {
      setErr(rpcErrorMessage(e))
    } finally {
      setClosingId('')
    }
  }, [closingId, currentSessionId, items, load, onClose, onNew, onSelect, sel])

  const handleRowClick = useCallback(
    (index: number) => (event: { stopImmediatePropagation?: () => void }) => {
      event.stopImmediatePropagation?.()
      const action = orchestratorRowClickAction(index, items)

      if (action.action === 'activate') {
        setSel(clampOrchestratorSelection(index, items.length))
        onSelect(action.sessionId)

        return
      }

      setSel(newSessionRowIndex(items.length))
    },
    [items, onSelect]
  )

  const newSelected = isNewSessionRow(sel, items.length)
  const draftHasText = Boolean(draft.trim())

  useInput((ch, key) => {
    if (pickingModel) {
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
      if (!newSelected) {
        void closeSelected()
      }

      return
    }

    if (newSelected && draftHasText) {
      return
    }

    if (key.upArrow && sel > 0) {
      return setSel(s => clampOrchestratorSelection(s - 1, items.length))
    }

    if (key.downArrow && sel < newSessionRowIndex(items.length)) {
      return setSel(s => clampOrchestratorSelection(s + 1, items.length))
    }

    if (key.return) {
      if (newSelected) {
        if (!draftHasText) {
          return onNew()
        }

        return
      }

      if (items[sel]) {
        return onSelect(items[sel]!.id)
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
    return <Text color={t.color.muted}>loading session orchestrator…</Text>
  }

  const totalRows = items.length + 1
  const offset = windowOffset(totalRows, sel, VISIBLE)
  const visibleRows = orchestratorVisibleRowIndexes(items.length, sel, VISIBLE)

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent}>
        Session Orchestrator
      </Text>
      <Text color={t.color.muted}>{activeSessionCountLabel(items.length)}</Text>

      {err && <Text color={t.color.label}>error: {err}</Text>}
      {!items.length && (
        <Text color={t.color.muted}>no live sessions — closed TUIs only leave resumable transcripts</Text>
      )}
      {offset > 0 && <Text color={t.color.muted}> ↑ {offset} more</Text>}

      {visibleRows.map(i => {
        const selected = sel === i
        const selectedStyle = selected ? selectedSessionRowStyle(t) : null
        const rowTextColor = selectedStyle?.color

        if (isNewSessionRow(i, items.length)) {
          const promptTitle = draftTitleFromPrompt(draft) || 'Start a new live session'
          const markerColor = newSessionMarkerColor(t, selected)

          return (
            <Box
              backgroundColor={selectedStyle?.backgroundColor}
              flexDirection="row"
              key="new-session"
              onClick={handleRowClick(i)}
              width="100%"
            >
              <Text bold={selected} color={rowTextColor ?? t.color.muted}>
                {selected ? '▸ ' : '  '}
              </Text>

              <Box {...fixedSessionColumnStyle()} width={5}>
                <Text bold={selected} color={markerColor}>
                  +
                </Text>
              </Box>

              <Box {...fixedSessionColumnStyle()} width={11}>
                <Text bold={selected} color={markerColor} wrap="truncate-end">
                  new
                </Text>
              </Box>

              <Box {...fixedSessionColumnStyle()} width={11}>
                <Text color={rowTextColor ?? t.color.muted} wrap="truncate-end">
                  ✎ draft
                </Text>
              </Box>

              <Box {...fixedSessionColumnStyle()} width={18}>
                <Text color={rowTextColor ?? t.color.muted} wrap="truncate-end">
                  {draftModelDisplayLabel(draftModel)}
                </Text>
              </Box>

              <Box flexGrow={1} flexShrink={1} minWidth={0}>
                <Text bold={selected} color={rowTextColor ?? t.color.muted} wrap="truncate-end">
                  {promptTitle}
                </Text>
              </Box>
            </Box>
          )
        }

        const s = items[i]!
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
                {String(i + 1).padStart(2)}.
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

      {offset + VISIBLE < totalRows && <Text color={t.color.muted}> ↓ {totalRows - offset - VISIBLE} more</Text>}

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
        <Box marginTop={1} flexDirection="column">
          <OrchestratorHintText segments={orchestratorContextHintSegments(false)} t={t} />
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
  onSelect: (id: string) => void
  t: Theme
}
