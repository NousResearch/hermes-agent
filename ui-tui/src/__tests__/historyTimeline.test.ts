import { describe, expect, it } from 'vitest'

import {
  buildHistoryTimelineState,
  clearHistoryTimelineFilter,
  historyTimelineMaxVisibleItems,
  historyTimelinePageSize,
  historyTimelineVisibleRange,
  moveHistoryTimelineSelection,
  updateHistoryTimelineFilter
} from '../components/historyTimelineOverlay.js'
import { toTranscriptMessages } from '../domain/messages.js'
import type { Msg } from '../types.js'

const msg = (role: Msg['role'], text: string, tools?: string[]): Msg => ({ role, text, tools })

describe('history timeline overlay model', () => {
  it('keeps navigable transcript rows with stable identities and defaults to the latest item', () => {
    const state = buildHistoryTimelineState([
      msg('system', 'intro'),
      { ...msg('user', 'first prompt'), dbId: 101 },
      msg('tool', 'tool result'),
      msg('assistant', 'first answer'),
      msg('user', 'second prompt')
    ])

    expect(state.items).toMatchObject([
      { actionable: false, ordinal: 1, role: 'system', sourceIndex: 0 },
      { actionable: true, dbId: 101, identity: 'db:101', ordinal: 2, role: 'user', sourceIndex: 1 },
      { actionable: true, ordinal: 3, role: 'assistant', sourceIndex: 3 },
      { actionable: true, ordinal: 4, role: 'user', sourceIndex: 4 }
    ])
    expect(state.items[2]?.identity).toBe('ordinal:assistant:1:source:3')
    expect(state.selected).toBe(3)
  })

  it('can default selection to the latest persisted branchable message when mixed with unpersisted local tail rows', () => {
    const state = buildHistoryTimelineState(
      [
        { ...msg('user', 'persisted prompt'), dbId: 101 },
        { ...msg('assistant', 'persisted answer'), dbId: 102 },
        msg('user', 'optimistic retry draft'),
        msg('assistant', 'optimistic streaming answer')
      ],
      { preferLatestPersistedBranchable: true }
    )

    expect(state.items).toMatchObject([
      { dbId: 101, role: 'user' },
      { dbId: 102, role: 'assistant' },
      { role: 'user' },
      { role: 'assistant' }
    ])
    expect(state.selected).toBe(1)
    expect(state.items[state.selected]).toMatchObject({ dbId: 102, role: 'assistant' })
  })

  it('drops gateway tool rows from the timeline while keeping system context read-only', () => {
    const transcript = toTranscriptMessages([
      { db_id: 201, role: 'system', text: 'system prompt' },
      { db_id: 202, role: 'user', text: 'inspect repo' },
      { db_id: 203, role: 'tool', context: 'src', name: 'search_files' },
      { db_id: 204, role: 'assistant', text: 'done' }
    ])

    const state = buildHistoryTimelineState(transcript)

    expect(transcript[2]).toMatchObject({ dbId: 203, role: 'tool' })
    expect(transcript[3]?.tools?.[0]).toContain('Search Files')
    expect(state.items).toMatchObject([
      { actionable: false, dbId: 201, role: 'system' },
      { actionable: true, dbId: 202, role: 'user' },
      { actionable: true, dbId: 204, role: 'assistant' }
    ])
    expect(state.items.some(item => item.role === 'tool')).toBe(false)
  })

  it('uses readable previews for empty assistant tool-call messages', () => {
    const state = buildHistoryTimelineState([msg('assistant', '   ', ['search', 'read_file'])])

    expect(state.items[0]?.preview).toBe('(2 tool calls)')
    expect(state.items[0]?.fullText).toBe('(2 tool calls)')
  })

  it('moves selection with arrows, pages, and top/bottom clamps', () => {
    const state = buildHistoryTimelineState([
      msg('user', 'one'),
      msg('assistant', 'two'),
      msg('user', 'three'),
      msg('assistant', 'four'),
      msg('user', 'five')
    ])

    expect(moveHistoryTimelineSelection(state, -1).selected).toBe(3)
    expect(moveHistoryTimelineSelection(state, -99).selected).toBe(0)
    expect(moveHistoryTimelineSelection(state, 99).selected).toBe(4)
    expect(moveHistoryTimelineSelection(state, 'top').selected).toBe(0)
    expect(moveHistoryTimelineSelection(state, 'bottom').selected).toBe(4)
  })

  it('filters by role and current-session text while preserving stable ids and restoring selection on clear', () => {
    const state = buildHistoryTimelineState([
      msg('system', 'intro'),
      { ...msg('user', 'first prompt'), dbId: 101 },
      msg('tool', 'tool result'),
      msg('assistant', 'first answer'),
      msg('user', 'second prompt about search')
    ])

    const filtered = updateHistoryTimelineFilter(state, 'search')
    expect(filtered.filterActive).toBe(true)
    expect(filtered.items).toHaveLength(1)
    expect(filtered.items[0]).toMatchObject({ identity: 'ordinal:user:2:source:4', role: 'user', sourceIndex: 4 })
    expect(filtered.selected).toBe(0)

    const moved = moveHistoryTimelineSelection(filtered, 1)
    expect(moved.selected).toBe(0)

    const cleared = clearHistoryTimelineFilter({ ...filtered, selected: 0 })
    expect(cleared.filterActive).toBe(false)
    expect(cleared.items).toHaveLength(4)
    expect(cleared.selected).toBe(3)
  })

  it('keeps original timeline identity through empty filter results and role filters', () => {
    const state = buildHistoryTimelineState([
      { ...msg('user', 'open file'), dbId: 101 },
      { ...msg('assistant', 'read complete'), dbId: 102 },
      { ...msg('tool', 'terminal output'), dbId: 103 }
    ])

    const noMatches = updateHistoryTimelineFilter(state, 'does-not-exist')
    expect(noMatches.filterActive).toBe(true)
    expect(noMatches.items).toHaveLength(0)
    expect(noMatches.allItems).toMatchObject([
      { dbId: 101, identity: 'db:101', role: 'user', sourceIndex: 0 },
      { dbId: 102, identity: 'db:102', role: 'assistant', sourceIndex: 1 }
    ])

    const roleMatches = updateHistoryTimelineFilter(state, 'assistant')
    expect(roleMatches.items).toMatchObject([{ dbId: 102, identity: 'db:102', role: 'assistant', sourceIndex: 1 }])

    const cleared = clearHistoryTimelineFilter(noMatches)
    expect(cleared.items.map(item => item.identity)).toEqual(['db:101', 'db:102'])
    expect(cleared.selected).toBe(1)
  })

  it('derives a useful page step from terminal height', () => {
    expect(historyTimelinePageSize(24)).toBeGreaterThanOrEqual(5)
    expect(historyTimelinePageSize(80)).toBeGreaterThan(historyTimelinePageSize(24))
    expect(historyTimelineMaxVisibleItems(historyTimelinePageSize(24))).toBeLessThanOrEqual(6)
    expect(historyTimelineMaxVisibleItems(historyTimelinePageSize(80))).toBe(6)
  })

  it('windows long timelines around the selected row instead of rendering every history item', () => {
    const state = buildHistoryTimelineState(Array.from({ length: 30 }, (_, i) => msg(i % 2 ? 'assistant' : 'user', `message ${i}`)))

    expect(historyTimelineVisibleRange(state, 7)).toEqual({ end: 30, start: 23 })

    const middle = moveHistoryTimelineSelection(state, -14)
    expect(middle.selected).toBe(15)
    expect(historyTimelineVisibleRange(middle, 7)).toEqual({ end: 19, start: 12 })

    const top = moveHistoryTimelineSelection(state, 'top')
    expect(historyTimelineVisibleRange(top, 7)).toEqual({ end: 7, start: 0 })
  })
})
