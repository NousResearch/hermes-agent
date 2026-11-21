import type { Key } from '@hermes/ink'
import type { SessionListRow } from '@hermes/shared/gateway-events'
import type { SessionActiveItem } from '@hermes/shared/gateway-types'
import { describe, expect, it } from 'vitest'

import {
  emptySessionsLabel,
  filterHistorySessions,
  filterKeyAction,
  filterLiveSessions,
  historySearchText,
  liveSearchText,
  moveFilteredSelection,
  rowIdAt,
  rowIndexForId,
  selectionForQuery
} from '../components/sessionFilterModel.js'

const NO_KEY = {
  backspace: false,
  ctrl: false,
  delete: false,
  downArrow: false,
  end: false,
  escape: false,
  fn: false,
  home: false,
  leftArrow: false,
  meta: false,
  pageDown: false,
  pageUp: false,
  return: false,
  rightArrow: false,
  shift: false,
  super: false,
  tab: false,
  upArrow: false,
  wheelDown: false,
  wheelUp: false
} as unknown as Key

const k = (over: Partial<Record<string, boolean>>) => ({ ...NO_KEY, ...over }) as Key

const live: SessionActiveItem[] = [
  { id: 'a1', status: 'idle', title: 'Fix wpa_supplicant build', model: 'anthropic/claude-opus' },
  { id: 'b2', status: 'working', title: 'Draft TGbn comment', preview: 'review the 802.11bn draft' }
]

const history: SessionListRow[] = [
  { id: 'h1', title: 'Quarterly travel budget', source: 'cli' },
  { id: 'h2', title: 'Session picker fuzzy search', source: 'telegram' },
  { id: 'h3', preview: 'what is the weather in portland' }
]

const idle = { active: false, canOpen: true, hasMatches: true, query: '' }
const typing = { ...idle, active: true }

describe('session filter text', () => {
  it('matches live rows on title, prompt, id and model', () => {
    expect(liveSearchText(live[0]!)).toBe('Fix wpa_supplicant build a1 anthropic/claude-opus')
    expect(liveSearchText(live[1]!)).toBe('Draft TGbn comment review the 802.11bn draft b2')
  })

  it('matches resumable rows on title, prompt, id and source', () => {
    expect(historySearchText(history[1]!)).toBe('Session picker fuzzy search h2 telegram')
    expect(historySearchText(history[2]!)).toBe('what is the weather in portland h3')
  })
})

describe('session filtering', () => {
  it('returns every row in order for an empty query', () => {
    expect(filterLiveSessions(live, '')).toEqual(live)
    expect(filterHistorySessions(history, '  ')).toEqual(history)
  })

  it('matches subsequences, ANDs tokens and ranks each group on its own', () => {
    expect(filterHistorySessions(history, 'fzy').map(h => h.id)).toEqual(['h2'])
    expect(filterHistorySessions(history, 'sess tele').map(h => h.id)).toEqual(['h2'])
    expect(filterHistorySessions(history, 'portland').map(h => h.id)).toEqual(['h3'])
    expect(filterLiveSessions(live, 'opus').map(s => s.id)).toEqual(['a1'])
    expect(filterLiveSessions(live, 'tgbn').map(s => s.id)).toEqual(['b2'])
    expect(filterLiveSessions(live, 'zzzz')).toEqual([])
  })

  it('says when nothing matches instead of "no other sessions"', () => {
    expect(emptySessionsLabel(' fzy ')).toBe('no sessions match "fzy"')
    expect(emptySessionsLabel('')).toBe('no other sessions — Enter on +new to start one')
  })
})

describe('session filter selection', () => {
  const liveIds = [{ id: 'a1' }, { id: 'b2' }]
  const histIds = [{ id: 'h1' }, { id: 'h2' }]

  it('maps flat rows ([new][live…][history…]) to ids and back', () => {
    expect(rowIdAt(0, liveIds, histIds)).toBeNull()
    expect(rowIdAt(2, liveIds, histIds)).toBe('b2')
    expect(rowIdAt(4, liveIds, histIds)).toBe('h2')
    expect(rowIdAt(5, liveIds, histIds)).toBeNull()
    expect(rowIndexForId('h1', liveIds, histIds)).toBe(3)
    expect(rowIndexForId('gone', liveIds, histIds)).toBeNull()
  })

  it('jumps to the best match while a query is typed', () => {
    expect(selectionForQuery(4, 'x', 'h2', liveIds, histIds)).toBe(1)
    // Nothing matches: only the pinned new row is left.
    expect(selectionForQuery(4, 'x', 'h2', [], [])).toBe(0)
  })

  it('keeps the highlighted session when the query is cleared', () => {
    expect(selectionForQuery(1, '', 'h2', liveIds, histIds)).toBe(4)
    expect(selectionForQuery(9, '', null, liveIds, histIds)).toBe(4)
  })

  it('moves only among list rows while filtering', () => {
    expect(moveFilteredSelection(1, -1, 3)).toBe(1)
    expect(moveFilteredSelection(1, 1, 3)).toBe(2)
    expect(moveFilteredSelection(3, 1, 3)).toBe(3)
    expect(moveFilteredSelection(0, 1, 0)).toBe(0)
  })
})

describe('session filter keys', () => {
  it('opens with / on a session row only, and leaves other keys to the switcher', () => {
    expect(filterKeyAction('/', NO_KEY, idle)).toEqual({ type: 'open' })
    expect(filterKeyAction('/', NO_KEY, { ...idle, canOpen: false })).toBeNull()
    expect(filterKeyAction('/', k({ ctrl: true }), idle)).toBeNull()
    expect(filterKeyAction('d', NO_KEY, idle)).toBeNull()
    expect(filterKeyAction('', k({ return: true }), idle)).toBeNull()
    expect(filterKeyAction('', k({ escape: true }), idle)).toBeNull()
  })

  it('claims Esc to clear an applied filter before Esc closes the switcher', () => {
    expect(filterKeyAction('', k({ escape: true }), { ...idle, query: 'fzy' })).toEqual({ type: 'reset' })
    expect(filterKeyAction('/', NO_KEY, { ...idle, query: 'fzy' })).toEqual({ type: 'open' })
  })

  it('types every printable key, including d, q and pasted text', () => {
    expect(filterKeyAction('d', NO_KEY, typing)).toEqual({ type: 'append', text: 'd' })
    expect(filterKeyAction('q', NO_KEY, typing)).toEqual({ type: 'append', text: 'q' })
    expect(filterKeyAction('/', NO_KEY, typing)).toEqual({ type: 'append', text: '/' })
    expect(filterKeyAction('two words', NO_KEY, typing)).toEqual({ type: 'append', text: 'two words' })
  })

  it('edits, moves, cancels and submits while typing', () => {
    expect(filterKeyAction('', k({ backspace: true }), typing)).toEqual({ type: 'backspace' })
    expect(filterKeyAction('u', k({ ctrl: true }), typing)).toEqual({ type: 'clear' })
    expect(filterKeyAction('', k({ downArrow: true }), typing)).toEqual({ type: 'move', delta: 1 })
    expect(filterKeyAction('', k({ upArrow: true }), typing)).toEqual({ type: 'move', delta: -1 })
    expect(filterKeyAction('', k({ escape: true }), typing)).toEqual({ type: 'cancel' })
    expect(filterKeyAction('', k({ return: true }), typing)).toEqual({ type: 'submit' })
  })

  it('swallows Enter with no match (it would start a blank session) and other control keys', () => {
    expect(filterKeyAction('', k({ return: true }), { ...typing, hasMatches: false })).toEqual({ type: 'swallow' })
    expect(filterKeyAction('n', k({ ctrl: true }), typing)).toEqual({ type: 'swallow' })
    expect(filterKeyAction('', k({ tab: true }), typing)).toEqual({ type: 'swallow' })
  })
})
