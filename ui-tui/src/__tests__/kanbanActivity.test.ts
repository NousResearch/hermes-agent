import { describe, expect, it } from 'vitest'

import type { KanbanActivityResponse, KanbanActivityTask } from '../gatewayTypes.js'
import { activityPresentation, aggregateKanbanActivity, buildKanbanActivityRows, collapsedActivityLabel, normalizeKanbanActivity, truncateActivityLabel } from '../lib/kanbanActivity.js'

const NOW = 1_721_000_000
const task = (overrides: Partial<KanbanActivityTask> = {}): KanbanActivityTask => ({ assignee: 'implementer', block_reason: null, children: [], parents: [], status: 'running', task_id: 'task-a', title: 'Implement the execution spine', run: { ended_at: null, last_heartbeat_at: NOW - 10, max_runtime_seconds: 3600, outcome: null, profile: 'implementer', run_id: 12, started_at: NOW - 100 }, ...overrides })
const payload = (roots: KanbanActivityTask[]): KanbanActivityResponse => ({ active_count: 0, attention_count: 0, boards: [{ board: 'default', checked_at: NOW, roots }], checked_at: NOW })

describe('kanban activity model', () => {
  it('preserves backend order across status changes', () => {
    const roots = [task({ task_id: 'b' }), task({ task_id: 'a' })]
    const before = normalizeKanbanActivity(payload(roots), NOW).boards[0]!.roots.map(root => root.id)
    roots[0]!.status = 'done'; roots[1]!.status = 'blocked'
    expect(normalizeKanbanActivity(payload(roots), NOW).boards[0]!.roots.map(root => root.id)).toEqual(before)
  })
  it('handles every backend status and scopes heartbeat health to running', () => {
    for (const [status, label] of [['review', 'review queued'], ['scheduled', 'scheduled'], ['ready', 'ready']] as const) {
      const node = normalizeKanbanActivity(payload([task({ run: null, status })]), NOW).boards[0]!.roots[0]!
      expect(node.status).toBe(status); expect(activityPresentation(node, NOW).stateLabel).toBe(label)
    }

    const done = normalizeKanbanActivity(payload([task({ run: null, status: 'done' })]), NOW).boards[0]!.roots[0]!
    expect(activityPresentation(done, NOW)).toMatchObject({ glyph: '●', stateLabel: 'completed' })
    const missing = normalizeKanbanActivity(payload([task({ run: null, status: 'running' })]), NOW).boards[0]!.roots[0]!
    expect(activityPresentation(missing, NOW)).toMatchObject({ glyph: '!', stateLabel: 'heartbeat unavailable' })
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
    expect(buildKanbanActivityRows(normalizeKanbanActivity(wire, NOW), { now: NOW }).some(row => row.kind === 'summary' && row.label === '+ more activity not shown')).toBe(true)
  })
  it('compresses branches and bounds cycles', () => {
    const children = Array.from({ length: 7 }, (_, i) => task({ task_id: `child-${i}` })); const rows = buildKanbanActivityRows(normalizeKanbanActivity(payload([task({ children, task_id: 'root' })]), NOW), { maxChildren: 3, now: NOW }); expect(rows.some(row => row.kind === 'summary' && row.label === '+4 more tasks')).toBe(true)
    const cyclic = task({ task_id: 'cycle' }); cyclic.children = [cyclic]; const model = normalizeKanbanActivity(payload([cyclic]), NOW); expect(model.diagnostics).toContain('cycle:cycle'); expect(model.boards[0]!.roots[0]!.children).toEqual([])
  })
  it('never fabricates progress and truncates without wrapping', () => {
    const rows = buildKanbanActivityRows(normalizeKanbanActivity(payload([task()]), NOW), { now: NOW }); expect(rows.map(row => row.kind === 'task' ? row.label : '').join(' ')).not.toMatch(/%|percent|phase/i); expect(truncateActivityLabel('Implement a very long execution spine', 12)).toBe('Implement a…')
  })
})
