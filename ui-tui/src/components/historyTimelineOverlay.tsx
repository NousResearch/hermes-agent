import { Box, Text } from '@hermes/ink'

import type { HistoryTimelineItem, HistoryTimelineOverlayState } from '../app/interfaces.js'
import type { Theme } from '../theme.js'
import type { Msg } from '../types.js'

import { OverlayHint } from './overlayControls.js'

const MAX_PREVIEW = 96

const fallbackText = (msg: Msg): string => {
  const text = msg.text.trim()

  if (text) {
    return text
  }

  if (msg.tools?.length) {
    return `(${msg.tools.length} tool calls)`
  }

  if (msg.thinking?.trim()) {
    return '(assistant reasoning)'
  }

  if (msg.role === 'assistant') {
    return '(empty assistant message)'
  }

  if (msg.role === 'tool') {
    return '(empty tool result)'
  }

  return msg.role === 'system' ? '(empty system message)' : '(empty user message)'
}

const oneLinePreview = (text: string): string => {
  const compact = text.replace(/\s+/g, ' ').trim()

  return compact.length > MAX_PREVIEW ? `${compact.slice(0, MAX_PREVIEW).trimEnd()}…` : compact
}

interface BuildHistoryTimelineOptions {
  preferLatestPersistedBranchable?: boolean
}

export const buildHistoryTimelineState = (
  messages: Msg[],
  options: BuildHistoryTimelineOptions = {}
): HistoryTimelineOverlayState => {
  const rawItems = messages.flatMap((msg, sourceIndex): HistoryTimelineItem[] => {
    if (msg.kind === 'intro' || (msg.role !== 'user' && msg.role !== 'assistant' && msg.role !== 'system')) {
      return []
    }

    const fullText = fallbackText(msg)
    const roleOrdinal = messages.slice(0, sourceIndex + 1).filter(m => m.role === msg.role).length
    const fallbackIdentity = `ordinal:${msg.role}:${roleOrdinal}:source:${sourceIndex}`

    return [
      {
        actionable: msg.role === 'user' || msg.role === 'assistant',
        ...(typeof msg.dbId === 'number' && { dbId: msg.dbId }),
        fullText,
        identity: typeof msg.dbId === 'number' ? `db:${msg.dbId}` : fallbackIdentity,
        ordinal: 0,
        preview: oneLinePreview(fullText),
        role: msg.role,
        sourceIndex
      }
    ]
  })

  const allItems = rawItems.map((item, index) => ({ ...item, ordinal: index + 1 }))
  const latestIndex = Math.max(0, allItems.length - 1)
  const latestPersistedBranchableIndex = options.preferLatestPersistedBranchable
    ? allItems.findLastIndex(item => item.actionable && typeof item.dbId === 'number')
    : -1
  const selected = latestPersistedBranchableIndex >= 0 ? latestPersistedBranchableIndex : latestIndex

  return {
    allItems,
    filterActive: false,
    items: allItems,
    query: '',
    selected,
    unfilteredSelected: selected
  }
}

export const historyTimelinePageSize = (terminalRows: number): number => Math.max(5, Math.floor((terminalRows - 8) / 2))

export const historyTimelineMaxVisibleItems = (pagerPageSize: number): number => Math.max(3, Math.min(6, Math.floor(pagerPageSize / 2)))

export const historyTimelineVisibleRange = (
  state: HistoryTimelineOverlayState,
  maxVisibleItems: number
): { end: number; start: number } => {
  const total = state.items.length

  if (!total) {
    return { end: 0, start: 0 }
  }

  const count = Math.max(1, Math.min(total, maxVisibleItems))
  const preferredStart = state.selected - Math.floor(count / 2)
  const start = Math.max(0, Math.min(preferredStart, total - count))

  return { end: start + count, start }
}

const itemMatchesQuery = (item: HistoryTimelineItem, query: string): boolean => {
  const needle = query.trim().toLowerCase()

  if (!needle) {
    return true
  }

  return `${item.role} ${item.fullText}`.toLowerCase().includes(needle)
}

const clampSelection = (selected: number, itemCount: number): number => Math.max(0, Math.min(selected, Math.max(0, itemCount - 1)))

export const updateHistoryTimelineFilter = (state: HistoryTimelineOverlayState, query: string): HistoryTimelineOverlayState => {
  const allItems = state.allItems.length ? state.allItems : state.items
  const enteringFilter = !state.filterActive
  const unfilteredSelected = enteringFilter ? state.selected : state.unfilteredSelected
  const normalizedQuery = query.trim()

  if (!normalizedQuery) {
    const selected = clampSelection(unfilteredSelected, allItems.length)

    return {
      ...state,
      allItems,
      filterActive: true,
      items: allItems,
      query,
      selected,
      unfilteredSelected: selected
    }
  }

  const previousIdentity = state.items[state.selected]?.identity
  const filteredItems = allItems.filter(item => itemMatchesQuery(item, normalizedQuery))
  const existingSelection = previousIdentity ? filteredItems.findIndex(item => item.identity === previousIdentity) : -1
  const selected = existingSelection >= 0 ? existingSelection : 0

  return {
    ...state,
    allItems,
    filterActive: true,
    items: filteredItems,
    query,
    selected,
    unfilteredSelected
  }
}

export const clearHistoryTimelineFilter = (state: HistoryTimelineOverlayState): HistoryTimelineOverlayState => {
  const allItems = state.allItems.length ? state.allItems : state.items
  const selected = clampSelection(state.unfilteredSelected, allItems.length)

  return {
    ...state,
    allItems,
    filterActive: false,
    items: allItems,
    query: '',
    selected,
    unfilteredSelected: selected
  }
}

export const moveHistoryTimelineSelection = (
  state: HistoryTimelineOverlayState,
  delta: number | 'bottom' | 'top'
): HistoryTimelineOverlayState => {
  if (!state.items.length) {
    return { ...state, selected: 0 }
  }

  const next = delta === 'top' ? 0 : delta === 'bottom' ? state.items.length - 1 : state.selected + delta
  const selected = Math.max(0, Math.min(next, state.items.length - 1))

  if (selected === state.selected) {
    return state
  }

  return { ...state, selected, ...(state.filterActive ? {} : { unfilteredSelected: selected }) }
}

const roleLabel = (role: HistoryTimelineItem['role']): string =>
  role === 'user' ? 'You' : role === 'assistant' ? 'Hermes' : role === 'tool' ? 'Tool' : 'System'

const matchParts = (text: string, query: string) => {
  const needle = query.trim().toLowerCase()

  if (!needle) {
    return [{ match: false, text }]
  }

  const parts: Array<{ match: boolean; text: string }> = []
  let cursor = 0
  let next = text.toLowerCase().indexOf(needle, cursor)

  while (next >= 0) {
    if (next > cursor) {
      parts.push({ match: false, text: text.slice(cursor, next) })
    }

    parts.push({ match: true, text: text.slice(next, next + needle.length) })
    cursor = next + needle.length
    next = text.toLowerCase().indexOf(needle, cursor)
  }

  if (cursor < text.length) {
    parts.push({ match: false, text: text.slice(cursor) })
  }

  return parts.length ? parts : [{ match: false, text }]
}

interface HistoryTimelineOverlayProps {
  maxVisibleItems: number
  state: HistoryTimelineOverlayState
  t: Theme
  width: number
}

export function HistoryTimelineOverlay({ maxVisibleItems, state, t, width }: HistoryTimelineOverlayProps) {
  const selected = state.items[state.selected]
  const listWidth = Math.max(24, Math.min(42, Math.floor(width * 0.36)))
  const total = state.filterActive ? state.allItems.length : state.items.length
  const matchCount = state.filterActive ? state.items.length : state.allItems.length
  const visibleRange = historyTimelineVisibleRange(state, maxVisibleItems)
  const visibleItems = state.items.slice(visibleRange.start, visibleRange.end)

  const filterLabel = state.filterActive
    ? `filter /${state.query}${state.query ? '' : '…'} · ${matchCount}/${total} current-session matches`
    : 'type / to filter role + message text in this session only'

  if (!selected) {
    return (
      <Box flexDirection="column" paddingX={1} paddingY={1} width={width}>
        <Text bold color={t.color.primary}>
          History Timeline
        </Text>
        <Text color={t.color.muted}>
          {state.allItems.length ? `no matches for /${state.query}` : 'no conversation yet — send a message first'}
        </Text>
        <Text color={t.color.label}>{filterLabel}</Text>
        <OverlayHint t={t}>{state.filterActive ? 'Esc clear filter · type to refine · current session only' : 'Esc/q close'}</OverlayHint>
      </Box>
    )
  }

  return (
    <Box flexDirection="column" paddingX={1} paddingY={1} width={width}>
      <Box justifyContent="center" marginBottom={1}>
        <Text bold color={t.color.primary}>
          History Timeline
        </Text>
      </Box>
      <Box marginBottom={1}>
        <Text color={state.filterActive ? t.color.accent : t.color.muted}>{filterLabel}</Text>
      </Box>

      <Box flexDirection="row" gap={2}>
        <Box flexDirection="column" flexShrink={0} width={listWidth}>
          {visibleRange.start > 0 && <Text color={t.color.muted}>  ↑ {visibleRange.start} earlier</Text>}
          {visibleItems.map((item, visibleIndex) => {
            const index = visibleRange.start + visibleIndex
            const active = index === state.selected
            const marker = active ? '›' : ' '
            const disabled = !item.actionable
            const label = `${marker} #${item.ordinal} ${roleLabel(item.role)}${disabled ? ' (view)' : ''} — `

            return (
              <Box key={item.identity}>
                <Text bold={active} color={active ? t.color.accent : disabled ? t.color.muted : t.color.label} wrap="truncate-end">
                  {label}
                  {matchParts(item.preview, state.query).map((part, partIndex) =>
                    part.match ? (
                      <Text bold color={t.color.accent} key={partIndex}>
                        {part.text}
                      </Text>
                    ) : (
                      part.text
                    )
                  )}
                </Text>
              </Box>
            )
          })}
          {visibleRange.end < state.items.length && <Text color={t.color.muted}>  ↓ {state.items.length - visibleRange.end} later</Text>}
        </Box>

        <Box flexDirection="column" flexGrow={1}>
          <Text color={t.color.label}>
            #{selected.ordinal} {roleLabel(selected.role)} · {selected.dbId ? `db id ${selected.dbId}` : selected.identity} · source index{' '}
            {selected.sourceIndex}
          </Text>
          <Text color={t.color.text} wrap="wrap">
            {matchParts(selected.fullText, state.query).map((part, partIndex) =>
              part.match ? (
                <Text bold color={t.color.accent} key={partIndex}>
                  {part.text}
                </Text>
              ) : (
                part.text
              )
            )}
          </Text>
        </Box>
      </Box>

      <Box marginTop={1}>
        <OverlayHint t={t}>
          {`↑↓/j/k move · PgUp/PgDn page · / filter · n/N next/prev match · Enter branch/edit · c copy · b/e branch/edit · r retry user branch · Esc clear/close (${state.selected + 1}/${state.items.length})`}
        </OverlayHint>
      </Box>
    </Box>
  )
}
