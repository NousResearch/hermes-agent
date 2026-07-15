import { stringWidth } from '@hermes/ink'
import { describe, expect, it } from 'vitest'

import type { KanbanActivityResponse, KanbanActivityTask } from '../gatewayTypes.js'
import { activityPresentation, aggregateKanbanActivity, buildKanbanActivityRows, collapsedActivityLabel, collapsedActivitySegments, composeActivityLabel, normalizeKanbanActivity, truncateActivityLabel } from '../lib/kanbanActivity.js'

const NOW = 1_721_000_000
const task = (overrides: Partial<KanbanActivityTask> = {}): KanbanActivityTask => ({ assignee: 'implementer', block_reason: null, children: [], parents: [], status: 'running', task_id: 'task-a', title: 'Implement the execution spine', run: { ended_at: null, last_heartbeat_at: NOW - 10, max_runtime_seconds: 3600, outcome: null, profile: 'implementer', run_id: 12, started_at: NOW - 100 }, ...overrides })
const payload = (roots: KanbanActivityTask[]): KanbanActivityResponse => ({ active_count: 0, attention_count: 0, boards: [{ board: 'default', checked_at: NOW, roots }], checked_at: NOW })
const firstRoot = (t: KanbanActivityTask) => normalizeKanbanActivity(payload([t]), NOW).boards[0]!.roots[0]!
const taskRows = (model: ReturnType<typeof normalizeKanbanActivity>, options: Parameters<typeof buildKanbanActivityRows>[1] = {}) => buildKanbanActivityRows(model, { now: NOW, ...options }).filter(row => row.kind === 'task')

describe('kanban activity model', () => {
  it('preserves backend order across status changes', () => {
    const roots = [task({ task_id: 'b' }), task({ task_id: 'a' })]
    const before = normalizeKanbanActivity(payload(roots), NOW).boards[0]!.roots.map(root => root.id)
    roots[0]!.status = 'done'; roots[1]!.status = 'blocked'
    expect(normalizeKanbanActivity(payload(roots), NOW).boards[0]!.roots.map(root => root.id)).toEqual(before)
  })
  it('handles every backend status and scopes heartbeat health to running', () => {
    for (const [status, label] of [['review', 'review queued'], ['scheduled', 'scheduled'], ['ready', 'ready']] as const) {
      const node = firstRoot(task({ run: null, status }))
      expect(node.status).toBe(status); expect(activityPresentation(node, NOW).stateLabel).toBe(label)
    }

    const done = firstRoot(task({ run: null, status: 'done' }))
    expect(activityPresentation(done, NOW)).toMatchObject({ glyph: '●', stateLabel: 'completed' })
    const missing = firstRoot(task({ run: null, status: 'running' }))
    expect(activityPresentation(missing, NOW)).toMatchObject({ glyph: '!', stateLabel: 'heartbeat unavailable' })
  })
  it('reserves accent for live work and keeps queued states neutral', () => {
    expect(activityPresentation(firstRoot(task()), NOW).tone).toBe('accent')
    expect(activityPresentation(firstRoot(task({ run: null, status: 'ready' })), NOW).tone).toBe('neutral')
    expect(activityPresentation(firstRoot(task({ run: null, status: 'review' })), NOW).tone).toBe('neutral')
    expect(activityPresentation(firstRoot(task({ run: null, status: 'scheduled' })), NOW).tone).toBe('muted')
  })
  it('prioritizes current attention and ignores stale terminal history', () => {
    const old = task({ run: { ...task().run!, ended_at: NOW - 1000, outcome: 'failed' }, status: 'done', task_id: 'old' })
    const model = normalizeKanbanActivity(payload([old, task({ task_id: 'active', title: 'Current work' }), task({ block_reason: 'review', status: 'blocked', task_id: 'blocked', title: 'Needs Henry' })]), NOW)
    expect(aggregateKanbanActivity(model, NOW)).toMatchObject({ active: 1, attention: 1, completed: 0 }); expect(collapsedActivityLabel(model, 100, NOW)).toContain('Needs Henry')
  })
  it('shows every parent without duplicating task counts', () => {
    const shared = task({ parents: ['parent-a', 'parent-b'], task_id: 'shared', title: 'Shared child' })
    const model = normalizeKanbanActivity(payload([task({ children: [shared], task_id: 'parent-a', title: 'Parent A' }), task({ task_id: 'parent-b', title: 'Parent B' })]), NOW)
    const rows = buildKanbanActivityRows(model, { now: NOW }); expect(rows.some(row => row.kind === 'summary' && row.label === '↳ also linked from Parent B')).toBe(true); expect(aggregateKanbanActivity(model, NOW).total).toBe(3)
  })
  it('surfaces graph truncation', () => {
    const wire = payload([task()]); wire.boards[0]!.truncated = true
    expect(buildKanbanActivityRows(normalizeKanbanActivity(wire, NOW), { now: NOW }).some(row => row.kind === 'summary' && row.label === '+ more activity not shown' && row.prefix === '└── ')).toBe(true)
  })
  it('compresses branches and bounds cycles', () => {
    const children = Array.from({ length: 7 }, (_, i) => task({ task_id: `child-${i}` })); const rows = buildKanbanActivityRows(normalizeKanbanActivity(payload([task({ children, task_id: 'root' })]), NOW), { maxChildren: 3, now: NOW }); expect(rows.some(row => row.kind === 'summary' && row.label === '+4 more tasks')).toBe(true)
    const cyclic = task({ task_id: 'cycle' }); cyclic.children = [cyclic]; const model = normalizeKanbanActivity(payload([cyclic]), NOW); expect(model.diagnostics).toContain('cycle:cycle'); expect(model.boards[0]!.roots[0]!.children).toEqual([])
  })
  it('never fabricates progress and truncates without wrapping', () => {
    const rows = buildKanbanActivityRows(normalizeKanbanActivity(payload([task()]), NOW), { now: NOW }); expect(rows.map(row => row.kind === 'task' ? row.label : '').join(' ')).not.toMatch(/%|percent|phase/i); expect(truncateActivityLabel('Implement a very long execution spine', 12)).toBe('Implement a…')
  })
})

describe('segmented labels and truncation priority', () => {
  const blocked = () => firstRoot(task({ assignee: 'implementer', block_reason: 'Needs Henry visual choice', run: null, status: 'blocked', task_id: 'visual', title: 'Visual approval' }))

  it('keeps the full blocked reason when it fits', () => {
    expect(composeActivityLabel(blocked(), 80, NOW)).toEqual({ owner: 'implementer', state: 'blocked: Needs Henry visual choice', title: 'Visual approval' })
  })
  it('trims the blocked reason before anything else', () => {
    expect(composeActivityLabel(blocked(), 50, NOW)).toEqual({ owner: 'implementer', state: 'blocked', title: 'Visual approval' })
  })
  it('drops the owner before the state', () => {
    expect(composeActivityLabel(blocked(), 30, NOW)).toEqual({ owner: null, state: 'blocked', title: 'Visual approval' })
  })
  it('truncates the title while the state survives', () => {
    expect(composeActivityLabel(blocked(), 20, NOW)).toEqual({ owner: null, state: 'blocked', title: 'Visual ap…' })
  })
  it('drops the state word only as a last resort', () => {
    expect(composeActivityLabel(blocked(), 12, NOW)).toEqual({ owner: null, state: null, title: 'Visual appr…' })
  })
  it('never renders an unassigned placeholder', () => {
    const node = firstRoot(task({ assignee: null, run: null, status: 'ready' }))
    expect(composeActivityLabel(node, 80, NOW).owner).toBeNull()
  })
})

describe('topology prefixes', () => {
  it('renders ancestor continuation rails at depth 2', () => {
    const grandchildren = [task({ task_id: 'g1', title: 'Grandchild 1' }), task({ run: null, status: 'ready', task_id: 'g2', title: 'Grandchild 2' })]
    const roots = [task({ children: [task({ children: grandchildren, task_id: 'a', title: 'Child A' }), task({ run: null, status: 'ready', task_id: 'b', title: 'Child B' })], task_id: 'root', title: 'Ship feature' })]
    expect(taskRows(normalizeKanbanActivity(payload(roots), NOW)).map(row => row.prefix)).toEqual(['│ ', '├── ', '│   ├── ', '│   └── ', '└── '])
  })
  it('uses blank continuation under a last child', () => {
    const roots = [task({ children: [task({ children: [task({ task_id: 'g' })], task_id: 'a' })], task_id: 'root' })]
    expect(taskRows(normalizeKanbanActivity(payload(roots), NOW)).map(row => row.prefix)).toEqual(['│ ', '└── ', '    └── '])
  })
  it('keeps a branch connector on the last visible child when siblings are omitted', () => {
    const children = Array.from({ length: 3 }, (_, i) => task({ task_id: `child-${i}` }))
    const rows = buildKanbanActivityRows(normalizeKanbanActivity(payload([task({ children, task_id: 'root' })]), NOW), { maxChildren: 2, now: NOW })
    const prefixes = rows.filter(row => row.kind !== 'board').map(row => row.prefix)
    expect(prefixes).toEqual(['│ ', '├── ', '├── ', '└── '])
    expect(rows.at(-1)).toMatchObject({ kind: 'summary', label: '+1 more tasks' })
  })
  it('closes the linked-parent row when nothing follows it', () => {
    const shared = task({ parents: ['parent-a', 'parent-b'], task_id: 'shared', title: 'Shared child' })
    const model = normalizeKanbanActivity(payload([task({ children: [shared], task_id: 'parent-a', title: 'Parent A' }), task({ task_id: 'parent-b', title: 'Parent B' })]), NOW)
    const linked = buildKanbanActivityRows(model, { now: NOW }).find(row => row.kind === 'summary')!
    expect(linked.prefix).toBe('    └── ')
  })
  it('keeps a branch connector on the linked-parent row when children follow', () => {
    const shared = task({ children: [task({ task_id: 'nested' })], parents: ['parent-a', 'parent-b'], task_id: 'shared', title: 'Shared child' })
    const model = normalizeKanbanActivity(payload([task({ children: [shared], task_id: 'parent-a', title: 'Parent A' }), task({ task_id: 'parent-b', title: 'Parent B' })]), NOW)
    const linked = buildKanbanActivityRows(model, { now: NOW }).find(row => row.kind === 'summary')!
    expect(linked.prefix).toBe('    ├── ')
  })
  it('budgets labels from the exact prefix width', () => {
    const long = 'A deliberately very long task title that cannot fit in a narrow spine'
    const roots = [task({ children: [task({ task_id: 'child', title: long })], task_id: 'root', title: long })]

    for (const row of taskRows(normalizeKanbanActivity(payload(roots), NOW), { width: 30 })) {
      expect(stringWidth(`${row.prefix}${row.glyph} ${row.label}`)).toBeLessThanOrEqual(30)
    }
  })
})

describe('board rows', () => {
  const erroredWire = () => { const wire = payload([task()]); wire.boards[0]!.error = 'db locked';

 return wire }

  const boardRow = (width: number) => buildKanbanActivityRows(normalizeKanbanActivity(erroredWire(), NOW), { now: NOW, width })[0]!

  it('labels boards by bare name and flags per-board errors', () => {
    expect(boardRow(120)).toMatchObject({ error: true, kind: 'board', label: 'default', suffix: ' · unavailable' })
  })
  it('never repeats the Kanban brand inside board headers', () => {
    const wire: KanbanActivityResponse = { active_count: 0, attention_count: 0, boards: [{ board: 'alpha', checked_at: NOW, roots: [task({ task_id: 'a' })] }, { board: 'zeta', checked_at: NOW, roots: [task({ task_id: 'z' })] }], checked_at: NOW }
    const boards = buildKanbanActivityRows(normalizeKanbanActivity(wire, NOW), { now: NOW }).filter(row => row.kind === 'board')
    expect(boards.map(row => row.label)).toEqual(['alpha', 'zeta'])
    expect(boards.map(row => row.suffix)).toEqual([null, null])
  })
  it('budgets the name and error suffix together at every width', () => {
    expect(boardRow(20)).toMatchObject({ label: 'defau…', suffix: ' · unavailable' })

    for (const width of [1, 2, 4, 8, 12, 16, 20, 40]) {
      const row = boardRow(width)

      expect(row.kind).toBe('board')

      if (row.kind === 'board') {
        expect(stringWidth(`${row.label}${row.suffix ?? ''}`)).toBeLessThanOrEqual(width)
      }
    }
  })
  it('keeps the error signal over identity at extreme widths', () => {
    expect(boardRow(1)).toMatchObject({ label: '', suffix: '…' })
    expect(boardRow(12)).toMatchObject({ label: '', suffix: 'unavailable' })
    expect(boardRow(8)).toMatchObject({ label: '', suffix: 'unavail…' })
  })
})

describe('resting dock ladder', () => {
  const attentionModel = () => normalizeKanbanActivity(payload([task({ block_reason: 'review', run: null, status: 'blocked', task_id: 'blocked', title: 'Needs Henry' }), task({ task_id: 'active', title: 'Current work' })]), NOW)

  it('uses plural copy and appends the headline state at wide widths', () => {
    expect(collapsedActivityLabel(attentionModel(), 100, NOW)).toBe('Kanban · 1 need attention · 1 active · Needs Henry blocked')
  })
  it('ranks failed above blocked above stale for the headline', () => {
    const stale = task({ run: { ...task().run!, last_heartbeat_at: NOW - 1000 }, task_id: 'stale', title: 'Long job' })
    const failed = task({ run: { ...task().run!, ended_at: NOW, outcome: 'failed' }, status: 'done', task_id: 'failed', title: 'Tests' })
    const model = normalizeKanbanActivity(payload([stale, failed]), NOW)
    expect(collapsedActivityLabel(model, 100, NOW)).toContain('Tests failed')
  })
  it('falls back to counts instead of a headline fragment', () => {
    const blockedTasks = Array.from({ length: 3 }, (_, i) => task({ block_reason: 'review', run: null, status: 'blocked', task_id: `blocked-${i}`, title: `Blocked task ${i}` }))
    const model = normalizeKanbanActivity(payload([...blockedTasks, task({ task_id: 'active' })]), NOW)
    expect(collapsedActivityLabel(model, 50, NOW)).toBe('Kanban · 3 need attention · 1 active')
    expect(collapsedActivityLabel(model, 60, NOW).startsWith('Kanban · 3 need attention · 1 active · Blocked')).toBe(true)
  })
  it('never shows a zero-count badge and reports completed-only activity truthfully', () => {
    const completedOnly = normalizeKanbanActivity(payload([task({ run: { ...task().run!, ended_at: NOW - 10, outcome: 'completed' }, status: 'done' })]), NOW)
    expect(collapsedActivityLabel(completedOnly, 20, NOW)).toBe('K ●1')
    expect(collapsedActivityLabel(completedOnly, 13, NOW)).toBe('●1')
    expect(collapsedActivityLabel(completedOnly, 40, NOW)).toBe('Kanban · 1 completed')
    expect(collapsedActivityLabel(completedOnly, 80, NOW)).toBe('Kanban · 1 completed · Implement the execution spine')

    const attentionOnly = normalizeKanbanActivity(payload([task({ block_reason: 'review', run: null, status: 'blocked', task_id: 'blocked' })]), NOW)
    expect(collapsedActivityLabel(attentionOnly, 20, NOW)).toBe('K !1')
    expect(collapsedActivityLabel(attentionOnly, 13, NOW)).toBe('!1')
  })
  it('exposes narrow shorthand as tone-tagged segments', () => {
    expect(collapsedActivitySegments(attentionModel(), 18, NOW)).toEqual([
      { text: 'K', tone: 'neutral' },
      { text: ' !1', tone: 'warning' },
      { text: ' ◉1', tone: 'accent' }
    ])
    expect(collapsedActivitySegments(attentionModel(), 13, NOW)).toEqual([
      { text: '!1', tone: 'warning' },
      { text: '◉1', tone: 'accent' }
    ])
    const completedOnly = normalizeKanbanActivity(payload([task({ run: { ...task().run!, ended_at: NOW - 10, outcome: 'completed' }, status: 'done' })]), NOW)
    expect(collapsedActivitySegments(completedOnly, 18, NOW)).toEqual([
      { text: 'K', tone: 'neutral' },
      { text: ' ●1', tone: 'success' }
    ])
    expect(collapsedActivitySegments(attentionModel(), 40, NOW)).toEqual([
      { text: 'Kanban · ', tone: 'neutral' },
      { text: '1', tone: 'warning' },
      { text: ' need attention · ', tone: 'neutral' },
      { text: '1', tone: 'accent' },
      { text: ' active', tone: 'neutral' }
    ])
    expect(collapsedActivitySegments(completedOnly, 40, NOW)).toEqual([
      { text: 'Kanban · ', tone: 'neutral' },
      { text: '1', tone: 'success' },
      { text: ' completed', tone: 'neutral' }
    ])
  })
  it('speaks glyph shorthand at narrow widths without duplicate bangs', () => {
    expect(collapsedActivityLabel(attentionModel(), 20, NOW)).toBe('K !1 ◉1')
    expect(collapsedActivityLabel(attentionModel(), 13, NOW)).toBe('!1◉1')
    const calm = normalizeKanbanActivity(payload([task()]), NOW)
    expect(collapsedActivityLabel(calm, 20, NOW)).toBe('K ◉1')
    expect(collapsedActivityLabel(calm, 13, NOW)).toBe('◉1')
  })
})
