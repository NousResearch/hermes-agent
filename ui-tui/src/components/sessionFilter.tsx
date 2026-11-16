// State, keys and status line for the session switcher's `/` fuzzy filter.
// Key rules live in sessionFilterModel.ts; this file wires them to Ink.

import { Text, useInput } from '@hermes/ink'
import type { SessionListRow } from '@hermes/shared/gateway-events'
import type { SessionActiveItem } from '@hermes/shared/gateway-types'
import { type Dispatch, type SetStateAction, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { $sessionFilterHoldsEsc } from '../app/sessionFilterStore.js'
import { useT } from '../i18n/useT.js'
import type { Theme } from '../theme.js'

import {
  filterHistorySessions,
  type FilterKeyAction,
  filterKeyAction,
  filterLiveSessions,
  moveFilteredSelection,
  rowIdAt,
  selectionForQuery
} from './sessionFilterModel.js'

export function useSessionFilter() {
  const [active, setActive] = useState(false)
  const [query, setQueryState] = useState('')
  // The switcher's async poll re-anchors the selection inside the filtered
  // lists, so it reads the query through a ref rather than a stale closure.
  const queryRef = useRef('')

  const setQuery = useCallback((next: string) => {
    queryRef.current = next
    setQueryState(next)
  }, [])

  return { active, query, queryRef, setActive, setQuery }
}

export type SessionFilter = ReturnType<typeof useSessionFilter>

/**
 * The live and resumable lists as shown (filtered and ranked by the query),
 * plus the selection moves the filter keys need. Row indexes everywhere in
 * the switcher refer to these shown lists.
 */
export function useFilteredSessions(
  filter: SessionFilter,
  allItems: readonly SessionActiveItem[],
  allHistory: readonly SessionListRow[],
  sel: number,
  setSel: Dispatch<SetStateAction<number>>
) {
  const items = useMemo(() => filterLiveSessions(allItems, filter.query), [allItems, filter.query])
  const history = useMemo(() => filterHistorySessions(allHistory, filter.query), [allHistory, filter.query])
  const listLen = items.length + history.length

  const onQuery = useCallback(
    (next: string) => {
      const anchorId = rowIdAt(sel, items, history)

      filter.setQuery(next)
      setSel(s =>
        selectionForQuery(
          s,
          next,
          anchorId,
          filterLiveSessions(allItems, next),
          filterHistorySessions(allHistory, next)
        )
      )
    },
    [allHistory, allItems, filter, history, items, sel, setSel]
  )

  const onMove = useCallback(
    (delta: -1 | 1) => setSel(s => moveFilteredSelection(s, delta, listLen)),
    [listLen, setSel]
  )

  return { history, items, onMove, onQuery }
}

export interface SessionFilterKeyOptions {
  deleteArmed: boolean
  deletePending: () => boolean
  /** At least one session row matches the query. */
  hasMatches: boolean
  newSelected: boolean
  onMove: (delta: -1 | 1) => void
  /** Set the query and move the selection to fit it. */
  onQuery: (query: string) => void
  pickingModel: boolean
}

function applyFilterAction(action: FilterKeyAction, filter: SessionFilter, o: SessionFilterKeyOptions) {
  const query = filter.queryRef.current

  switch (action.type) {
    case 'open':
      return filter.setActive(true)

    case 'submit':
      return filter.setActive(false)

    case 'cancel':

    case 'reset':
      filter.setActive(false)

      return o.onQuery('')

    case 'clear':
      return o.onQuery('')

    case 'backspace':
      return o.onQuery([...query].slice(0, -1).join(''))

    case 'append':
      return o.onQuery(query + action.text)

    case 'move':
      return o.onMove(action.delta)

    case 'swallow':
      return
  }
}

/**
 * Register the filter's key handler. Call it BEFORE the switcher's own
 * `useInput` so it runs first: a key the filter claims stops there.
 */
export function useSessionFilterKeys(filter: SessionFilter, o: SessionFilterKeyOptions) {
  const holdsEsc = !o.pickingModel && (filter.active || Boolean(filter.query))

  useEffect(() => {
    $sessionFilterHoldsEsc.set(holdsEsc)
  }, [holdsEsc])

  useEffect(() => () => $sessionFilterHoldsEsc.set(false), [])

  useInput((ch, key, event) => {
    if (o.pickingModel || o.deletePending()) {
      return
    }

    const action = filterKeyAction(ch, key, {
      active: filter.active,
      canOpen: !o.newSelected && !o.deleteArmed,
      hasMatches: o.hasMatches,
      query: filter.queryRef.current
    })

    if (!action) {
      return
    }

    // A submitting Enter ends typing and still reaches the switcher, which opens the highlighted row.
    if (action.type !== 'submit') {
      event.stopImmediatePropagation()
    }

    applyFilterAction(action, filter, o)
  })
}

/** `filter: <query>▎` plus the keys that apply; nothing while no filter is open or applied. */
export function SessionFilterLine({ filter, t }: { filter: SessionFilter; t: Theme }) {
  const T = useT()

  if (!filter.active && !filter.query) {
    return null
  }

  const hint = filter.active ? T.pickers.session.filterLine.typing : T.pickers.session.filterLine.applied

  return (
    <Text color={t.color.accent} wrap="truncate-end">
      {T.pickers.common.filter(filter.query)}
      {filter.active ? '▎' : ''}
      <Text color={t.color.muted}>{hint}</Text>
    </Text>
  )
}
