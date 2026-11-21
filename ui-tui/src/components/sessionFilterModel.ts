// Pure rules for the session switcher's `/` fuzzy filter: what text a row is
// matched on, how the live and resumable groups are ranked, where the
// selection lands when the query changes, and what each key does.
//
// Rows stay grouped as [new][live…][resumable…] (the switcher's row-index
// math depends on it); each group is ranked on its own with the shared
// `fuzzyRank`, so a query orders rows exactly like the model picker does.

import type { Key } from '@hermes/ink'
import { fuzzyRank } from '@hermes/shared/fuzzy'
import type { SessionListRow } from '@hermes/shared/gateway-events'
import type { SessionActiveItem } from '@hermes/shared/gateway-types'

import { messages } from '../i18n/runtime.js'

const joinText = (parts: readonly (null | string | undefined)[]) => parts.filter(Boolean).join(' ')

/** Live rows match on title, first prompt, id and model. */
export const liveSearchText = (s: SessionActiveItem) => joinText([s.title, s.preview, s.id, s.model])

/** Resumable rows match on title, first prompt, id and source (cli, telegram, …). */
export const historySearchText = (h: SessionListRow) => joinText([h.title, h.preview, h.id, h.source])

export const filterLiveSessions = (items: readonly SessionActiveItem[], query: string) =>
  fuzzyRank(items, query, liveSearchText).map(r => r.item)

export const filterHistorySessions = (history: readonly SessionListRow[], query: string) =>
  fuzzyRank(history, query, historySearchText).map(r => r.item)

/** Session id at a flat row index ([new][live…][history…]); null for the new row or past the end. */
export function rowIdAt(index: number, live: readonly { id: string }[], history: readonly { id: string }[]) {
  if (index <= 0) {
    return null
  }

  return (index - 1 < live.length ? live[index - 1]?.id : history[index - 1 - live.length]?.id) ?? null
}

/** Flat row index of a session id, or null when it is not shown. */
export function rowIndexForId(
  id: null | string,
  live: readonly { id: string }[],
  history: readonly { id: string }[]
): null | number {
  if (!id) {
    return null
  }

  const liveIndex = live.findIndex(s => s.id === id)

  if (liveIndex >= 0) {
    return liveIndex + 1
  }

  const histIndex = history.findIndex(h => h.id === id)

  return histIndex >= 0 ? 1 + live.length + histIndex : null
}

/**
 * Selection after the query changed. A non-empty query jumps to the best
 * match (the first list row; the pinned new row when nothing matches). An
 * emptied query keeps the session that was highlighted, now in the full list.
 */
export function selectionForQuery(
  current: number,
  query: string,
  anchorId: null | string,
  live: readonly { id: string }[],
  history: readonly { id: string }[]
) {
  const listLen = live.length + history.length

  if (query.trim()) {
    return listLen ? 1 : 0
  }

  return rowIndexForId(anchorId, live, history) ?? Math.min(current, listLen)
}

/** ↑/↓ while typing a filter: stay on the list rows (never the pinned new row unless the list is empty). */
export const moveFilteredSelection = (current: number, delta: number, listLen: number) =>
  Math.max(Math.min(1, listLen), Math.min(listLen, current + delta))

export const emptySessionsLabel = (query: string) => {
  const S = messages().pickers.session

  return query.trim() ? S.noMatches(query.trim()) : S.noOtherSessions
}

export type FilterKeyAction =
  | { type: 'append'; text: string }
  | { type: 'backspace' }
  | { type: 'cancel' }
  | { type: 'clear' }
  | { type: 'move'; delta: -1 | 1 }
  | { type: 'open' }
  | { type: 'reset' }
  /** Stop typing and let the switcher's own Enter open the highlighted row. */
  | { type: 'submit' }
  | { type: 'swallow' }

export interface FilterKeyState {
  active: boolean
  /** `/` may open the filter: a session row is selected (the new row's prompt owns typing) and no delete is armed. */
  canOpen: boolean
  /** At least one session row matches, so Enter has something to open. */
  hasMatches: boolean
  query: string
}

/** Printable text from one key event (a paste arrives as one chunk); control characters are dropped. */
const printable = (ch: string, key: Key) =>
  key.ctrl || key.meta ? '' : [...ch].filter(c => c >= ' ' && c !== '\x7f').join('')

function activeFilterKey(ch: string, key: Key, hasMatches: boolean): FilterKeyAction {
  if (key.escape) {
    return { type: 'cancel' }
  }

  // With no match the highlight sits on the pinned new row; Enter there would start a blank session.
  if (key.return) {
    return hasMatches ? { type: 'submit' } : { type: 'swallow' }
  }

  if (key.upArrow || key.downArrow) {
    return { type: 'move', delta: key.upArrow ? -1 : 1 }
  }

  if (key.backspace || key.delete) {
    return { type: 'backspace' }
  }

  if (key.ctrl && ch === 'u') {
    return { type: 'clear' }
  }

  const text = printable(ch, key)

  return text ? { type: 'append', text } : { type: 'swallow' }
}

/**
 * What a key does to the filter, or null when the switcher's own handler
 * should get it. While the filter is being typed it owns every key except a
 * submitting Enter; once typing ends it only claims `/` (edit again) and Esc
 * (clear the filter before Esc closes the switcher).
 */
export function filterKeyAction(ch: string, key: Key, state: FilterKeyState): FilterKeyAction | null {
  if (state.active) {
    return activeFilterKey(ch, key, state.hasMatches)
  }

  if (ch === '/' && state.canOpen && !key.ctrl && !key.meta) {
    return { type: 'open' }
  }

  return key.escape && state.query ? { type: 'reset' } : null
}
