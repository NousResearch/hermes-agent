import type { SessionListRow } from '@hermes/shared/gateway-events'
import type { SessionActiveItem } from '@hermes/shared/gateway-types'
import { describe, expect, it } from 'vitest'

import { buildSessionSidebarEntries } from '../components/sessionSidebar.js'

type HistoryRow = SessionListRow & { last_active_at?: number | null }

const live = (id: string, lastActive: number): SessionActiveItem => ({
  id,
  last_active: lastActive,
  status: 'idle',
  title: id
})

const history = (id: string, lastActive: number | undefined, startedAt: number): HistoryRow => ({
  id,
  last_active_at: lastActive,
  started_at: startedAt,
  title: id
})

describe('buildSessionSidebarEntries', () => {
  it('merges live and history into one master list sorted by activity', () => {
    const rows = buildSessionSidebarEntries(
      [history('history-old', undefined, 100), history('history-new', 300, 50)],
      [live('live-middle', 200), live('live-new', 400)]
    )

    expect(rows.map(row => row.id)).toEqual(['live-new', 'history-new', 'live-middle', 'history-old'])
  })

  it('deduplicates ids with the live entry winning', () => {
    const rows = buildSessionSidebarEntries(
      [history('shared', 900, 1), history('inactive', 20, 1)],
      [live('shared', 50)]
    )

    expect(rows.filter(row => row.id === 'shared')).toEqual([
      { activityAt: 50, dimmed: false, id: 'shared', kind: 'live', session: live('shared', 50) }
    ])
    expect(rows.map(row => row.id)).toEqual(['shared', 'inactive'])
  })

  it('marks only inactive history entries as dimmed', () => {
    const rows = buildSessionSidebarEntries([history('inactive', 5, 1)], [live('active', 10)])

    expect(rows.map(({ kind, dimmed }) => [kind, dimmed])).toEqual([
      ['live', false],
      ['history', true]
    ])
  })

  it('falls back to started_at when history has no last-activity timestamp', () => {
    const rows = buildSessionSidebarEntries([history('older', undefined, 10)], [live('newer', 20)])

    expect(rows.map(row => [row.id, row.activityAt])).toEqual([
      ['newer', 20],
      ['older', 10]
    ])
  })

  it('returns live entries when history is empty and an empty list for no sessions', () => {
    expect(buildSessionSidebarEntries([], [live('live', 10)]).map(row => row.id)).toEqual(['live'])
    expect(buildSessionSidebarEntries([], [])).toEqual([])
  })
})
