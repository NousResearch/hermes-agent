/**
 * Resume-picker pure logic (docs/plans/opentui-resume-picker.md §B item 5):
 * source→tab classification, per-tab `session.list` params, the client-side
 * search chain (title/preview/cwd/id), the key-routing decision table, the
 * relative-time formatter, row-meta composition (cwd tail-truncation), the
 * `/sessions <tab>` arg parser and the `/resume <id|name>` resolver.
 */
import { describe, expect, test } from 'vitest'

import {
  classifySource,
  filterSessions,
  INTERACTIVE_SOURCES,
  listParamsFor,
  mapSessionRows,
  parseSessionTabArg,
  PLATFORM_SOURCES,
  relativeTime,
  resolveSessionArg,
  routeSessionPickerKey,
  rowMeta,
  SESSION_TABS,
  tabAccepts,
  tailTruncate,
  type SessionRow,
  normalizeCwd,
  orderRowsForCwd
} from '../logic/sessionPicker.ts'

const row = (over: Partial<SessionRow>): SessionRow => ({
  id: 'sid',
  lastActive: 0,
  messageCount: 0,
  preview: '',
  source: 'tui',
  startedAt: 0,
  title: '',
  ...over
})

describe('tabs + classification', () => {
  test('tab strip order: Recent (default) · Cron · Gateways · All', () => {
    expect(SESSION_TABS.map(t => t.id)).toEqual(['recent', 'cron', 'gateways', 'all'])
  })

  test('every source classifies to its home tab', () => {
    // interactive (+ unknown/custom + empty) → Recent
    for (const s of ['cli', 'tui', 'acp', '', undefined, 'my-custom-source', 'TUI ']) {
      expect(classifySource(s)).toBe('recent')
    }
    expect(classifySource('cron')).toBe('cron')
    // every known platform → Gateways
    for (const s of PLATFORM_SOURCES) expect(classifySource(s)).toBe('gateways')
    expect(classifySource('Telegram')).toBe('gateways') // case-insensitive
    // tool is deny-listed everywhere
    expect(classifySource('tool')).toBe('tool')
  })

  test('tabAccepts: All = everything minus tool; others match their class', () => {
    expect(tabAccepts('all', 'cli')).toBe(true)
    expect(tabAccepts('all', 'cron')).toBe(true)
    expect(tabAccepts('all', 'telegram')).toBe(true)
    expect(tabAccepts('all', 'tool')).toBe(false)
    expect(tabAccepts('recent', 'tui')).toBe(true)
    expect(tabAccepts('recent', 'cron')).toBe(false)
    expect(tabAccepts('cron', 'cron')).toBe(true)
    expect(tabAccepts('gateways', 'discord')).toBe(true)
    expect(tabAccepts('gateways', 'cli')).toBe(false)
  })

  test('listParamsFor: each tab queries with its sources allow-list', () => {
    expect(listParamsFor('recent', 0, 100)).toEqual({ limit: 100, offset: 0, sources: [...INTERACTIVE_SOURCES] })
    expect(listParamsFor('cron', 200, 100)).toEqual({ limit: 100, offset: 200, sources: ['cron'] })
    expect(listParamsFor('gateways', 0, 50)).toEqual({ limit: 50, offset: 0, sources: [...PLATFORM_SOURCES] })
    expect(listParamsFor('all', 100, 100)).toEqual({ limit: 100, offset: 100 }) // gateway deny-lists tool itself
  })
})

describe('mapSessionRows (widened session.list projection)', () => {
  test('maps rich fields; truncated flag; tolerates absent optionals + garbage', () => {
    const { rows, truncated } = mapSessionRows({
      sessions: [
        {
          cwd: '/home/u/proj',
          ended_at: 30,
          id: 's1',
          last_active: 20,
          message_count: 7,
          model: 'hermes-4',
          preview: 'hello',
          source: 'tui',
          started_at: 10,
          title: 'First'
        },
        { id: 's2', started_at: 5 }, // minimal legacy row
        { title: 'no id — dropped' },
        'garbage'
      ],
      truncated: true
    })
    expect(truncated).toBe(true)
    expect(rows).toHaveLength(2)
    expect(rows[0]).toEqual({
      cwd: '/home/u/proj',
      endedAt: 30,
      id: 's1',
      lastActive: 20,
      messageCount: 7,
      model: 'hermes-4',
      preview: 'hello',
      source: 'tui',
      startedAt: 10,
      title: 'First'
    })
    // last_active falls back to started_at
    expect(rows[1]).toMatchObject({ id: 's2', lastActive: 5, startedAt: 5 })
    expect(mapSessionRows(null)).toEqual({ rows: [], truncated: false })
    expect(mapSessionRows({ sessions: 'nope' })).toEqual({ rows: [], truncated: false })
  })
})

describe('search filter chain (client-side, title/preview/cwd/id)', () => {
  const rows = [
    row({ id: 'aaa-1', preview: 'fix the scrollbox', title: 'Adopt OpenTUI paradigm' }),
    row({ cwd: '/home/u/worktrees/lively-thrush', id: 'bbb-2', title: 'goal-v4' }),
    row({ id: 'ccc-3', preview: 'cron run output', title: '' })
  ]

  test('empty query keeps fetch order; title/preview/cwd/id all match', () => {
    expect(filterSessions('', rows)).toEqual(rows)
    expect(filterSessions('opentui', rows).map(r => r.id)).toEqual(['aaa-1'])
    expect(filterSessions('scrollbox', rows).map(r => r.id)).toEqual(['aaa-1']) // preview
    expect(filterSessions('thrush', rows).map(r => r.id)).toEqual(['bbb-2']) // cwd
    expect(filterSessions('ccc-3', rows).map(r => r.id)).toEqual(['ccc-3']) // id
    expect(filterSessions('zzzz', rows)).toEqual([])
  })
})

describe('key routing decision table', () => {
  const base = { queryEmpty: true, renaming: false }

  test('Esc/Ctrl+C close; Enter resumes; ↑↓ (and Ctrl+P/N) move', () => {
    expect(routeSessionPickerKey('escape', {}, base)).toEqual({ kind: 'close' })
    expect(routeSessionPickerKey('c', { ctrl: true }, base)).toEqual({ kind: 'close' })
    expect(routeSessionPickerKey('return', {}, base)).toEqual({ kind: 'resume' })
    expect(routeSessionPickerKey('up', {}, base)).toEqual({ dir: -1, kind: 'move' })
    expect(routeSessionPickerKey('down', {}, base)).toEqual({ dir: 1, kind: 'move' })
    expect(routeSessionPickerKey('p', { ctrl: true }, base)).toEqual({ dir: -1, kind: 'move' })
    expect(routeSessionPickerKey('n', { ctrl: true }, base)).toEqual({ dir: 1, kind: 'move' })
  })

  test('Tab/Shift+Tab cycle; ←/→ cycle ONLY on an empty query', () => {
    expect(routeSessionPickerKey('tab', {}, base)).toEqual({ dir: 1, kind: 'cycle-tab' })
    expect(routeSessionPickerKey('tab', { shift: true }, base)).toEqual({ dir: -1, kind: 'cycle-tab' })
    expect(routeSessionPickerKey('left', {}, base)).toEqual({ dir: -1, kind: 'cycle-tab' })
    expect(routeSessionPickerKey('right', {}, base)).toEqual({ dir: 1, kind: 'cycle-tab' })
    const typing = { ...base, queryEmpty: false }
    expect(routeSessionPickerKey('left', {}, typing)).toEqual({ kind: 'pass' }) // stays a cursor move
    expect(routeSessionPickerKey('right', {}, typing)).toEqual({ kind: 'pass' })
    expect(routeSessionPickerKey('tab', {}, typing)).toEqual({ dir: 1, kind: 'cycle-tab' }) // Tab always cycles
  })

  test('Space toggles preview; Ctrl+R starts rename; printables pass to the input', () => {
    expect(routeSessionPickerKey('space', {}, base)).toEqual({ kind: 'preview' })
    expect(routeSessionPickerKey('space', {}, { ...base, queryEmpty: false })).toEqual({ kind: 'preview' })
    expect(routeSessionPickerKey('r', { ctrl: true }, base)).toEqual({ kind: 'rename' })
    expect(routeSessionPickerKey('r', {}, base)).toEqual({ kind: 'pass' })
    expect(routeSessionPickerKey('a', {}, base)).toEqual({ kind: 'pass' })
    expect(routeSessionPickerKey('backspace', {}, base)).toEqual({ kind: 'pass' })
  })

  test('while RENAMING the rename input owns everything but Enter/Esc', () => {
    const renaming = { queryEmpty: true, renaming: true }
    expect(routeSessionPickerKey('return', {}, renaming)).toEqual({ kind: 'commit-rename' })
    expect(routeSessionPickerKey('escape', {}, renaming)).toEqual({ kind: 'cancel-rename' })
    expect(routeSessionPickerKey('c', { ctrl: true }, renaming)).toEqual({ kind: 'cancel-rename' })
    expect(routeSessionPickerKey('up', {}, renaming)).toEqual({ kind: 'pass' })
    expect(routeSessionPickerKey('tab', {}, renaming)).toEqual({ kind: 'pass' })
    expect(routeSessionPickerKey('space', {}, renaming)).toEqual({ kind: 'pass' })
    expect(routeSessionPickerKey('a', {}, renaming)).toEqual({ kind: 'pass' })
  })
})

describe('relative time + row meta', () => {
  const now = 1_760_000_000_000 // ms

  test('relativeTime: steps, singular/plural, seconds vs ms epochs', () => {
    const sec = now / 1000
    expect(relativeTime(sec - 10, now)).toBe('just now')
    expect(relativeTime(sec - 60, now)).toBe('1 minute ago')
    expect(relativeTime(sec - 5 * 60, now)).toBe('5 minutes ago')
    expect(relativeTime(sec - 3600, now)).toBe('1 hour ago')
    expect(relativeTime(sec - 26 * 3600, now)).toBe('1 day ago')
    expect(relativeTime(sec - 8 * 86_400, now)).toBe('1 week ago')
    expect(relativeTime(sec - 40 * 86_400, now)).toBe('1 month ago')
    expect(relativeTime(sec - 800 * 86_400, now)).toBe('2 years ago')
    expect(relativeTime(now - 3_600_000, now)).toBe('1 hour ago') // ms epoch tolerated
    expect(relativeTime(0, now)).toBe('unknown')
    expect(relativeTime(undefined, now)).toBe('unknown')
  })

  test('tailTruncate keeps the path TAIL under the cap', () => {
    expect(tailTruncate('short', 10)).toBe('short')
    const long = '/home/daimon/github/worktrees/hermes-agent/lively-thrush/hermes-agent'
    const cut = tailTruncate(long, 30)
    expect(cut).toHaveLength(30)
    expect(cut.startsWith('…')).toBe(true)
    expect(cut.endsWith('hermes-agent')).toBe(true)
  })

  test('rowMeta: time · source · msgs · truncated cwd (cwd omitted when absent)', () => {
    const sec = now / 1000
    const meta = rowMeta(
      row({
        cwd: '/home/daimon/github/worktrees/hermes-agent/lively-thrush/hermes-agent',
        lastActive: sec - 3600,
        messageCount: 142,
        source: 'tui'
      }),
      now
    )
    expect(meta).toContain('1 hour ago · tui · 142 msgs · …')
    expect(meta).toContain('lively-thrush/hermes-agent')
    const bare = rowMeta(row({ lastActive: sec - 60, messageCount: 9, source: 'cli' }), now)
    expect(bare).toBe('1 minute ago · cli · 9 msgs')
    // lastActive falls back to startedAt; empty source reads "unknown"
    expect(rowMeta(row({ lastActive: 0, messageCount: 1, source: '', startedAt: sec - 60 }), now)).toBe(
      '1 minute ago · unknown · 1 msgs'
    )
  })
})

describe('slash entry points', () => {
  test('parseSessionTabArg: bare → recent; names ci; singular gateway; garbage → undefined', () => {
    expect(parseSessionTabArg('')).toBe('recent')
    expect(parseSessionTabArg('recent')).toBe('recent')
    expect(parseSessionTabArg('CRON')).toBe('cron')
    expect(parseSessionTabArg('gateway')).toBe('gateways')
    expect(parseSessionTabArg('gateways')).toBe('gateways')
    expect(parseSessionTabArg('all')).toBe('all')
    expect(parseSessionTabArg('bogus')).toBeUndefined()
  })

  test('resolveSessionArg: exact id → unique id prefix → exact title → unique substring', () => {
    const rows = [
      row({ id: 'abc-123', title: 'Adopt OpenTUI' }),
      row({ id: 'abd-456', title: 'goal-v4' }),
      row({ id: 'xyz-789', title: 'Goal-V4 retry' })
    ]
    expect(resolveSessionArg(rows, 'abc-123')?.id).toBe('abc-123') // exact id
    expect(resolveSessionArg(rows, 'xyz')?.id).toBe('xyz-789') // unique prefix
    expect(resolveSessionArg(rows, 'ab')).toBeUndefined() // ambiguous prefix
    expect(resolveSessionArg(rows, 'goal-v4')?.id).toBe('abd-456') // exact title beats substring
    expect(resolveSessionArg(rows, 'adopt')?.id).toBe('abc-123') // unique substring
    expect(resolveSessionArg(rows, 'goal')).toBeUndefined() // ambiguous substring
    expect(resolveSessionArg(rows, '')).toBeUndefined()
  })
})

describe('orderRowsForCwd — pure edges', () => {
  const row = (id: string, cwd?: string) =>
    ({ cwd, id, lastActive: 0, messageCount: 0, preview: '', source: 'tui', startedAt: 0, title: id }) as never

  test('normalizeCwd trims and strips trailing slashes; empty-safe', () => {
    expect(normalizeCwd(' /a/b// ')).toBe('/a/b')
    expect(normalizeCwd(undefined)).toBe('')
    expect(normalizeCwd('   ')).toBe('')
  })

  test('identity passthrough when nothing matches or no cwd known', () => {
    const rows = [row('a', '/x'), row('b')]
    expect(orderRowsForCwd(rows, undefined, '')).toEqual({ hereCount: 0, rows })
    expect(orderRowsForCwd(rows, '/nowhere', '')).toEqual({ hereCount: 0, rows })
    expect(orderRowsForCwd(rows, '/x', 'query')).toEqual({ hereCount: 0, rows })
  })

  test('stable partition: here first, recency kept within groups', () => {
    const rows = [row('a', '/y'), row('b', '/x'), row('c'), row('d', '/x/')]
    const out = orderRowsForCwd(rows, '/x', '')
    expect(out.hereCount).toBe(2)
    expect(out.rows.map(r => (r as { id: string }).id)).toEqual(['b', 'd', 'a', 'c'])
  })
})
