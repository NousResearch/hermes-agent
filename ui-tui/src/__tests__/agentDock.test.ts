import { describe, expect, it } from 'vitest'

import { DOCK_MAX_ROWS, DOCK_NARROW_WIDTH, projectAgentDock } from '../lib/agentDock.js'
import { formatToolCall } from '../lib/text.js'
import type { SubagentProgress } from '../types.js'

const NOW = 1_750_000_000_000

const makeItem = (overrides: Partial<SubagentProgress> & Pick<SubagentProgress, 'id' | 'index'>): SubagentProgress => ({
  depth: 0,
  goal: overrides.id,
  notes: [],
  parentId: null,
  status: 'running',
  taskCount: 1,
  thinking: [],
  toolCount: 0,
  tools: [],
  ...overrides
})

const project = (items: SubagentProgress[], opts: Partial<Parameters<typeof projectAgentDock>[1]> = {}) =>
  projectAgentDock(items, { nowMs: NOW, width: 100, ...opts })

describe('projectAgentDock: visibility', () => {
  it('hides for an empty subagent list', () => {
    const view = project([])

    expect(view.hidden).toBe(true)
    expect(view.rows).toEqual([])
    expect(view.totalCount).toBe(0)
  })

  it('shows a single running row with a safe callsign, plain activity, and live elapsed time', () => {
    const view = project([makeItem({ id: 'a', index: 0, goal: 'scan repo', startedAt: NOW - 5000 })])

    expect(view.hidden).toBe(false)
    expect(view.activeCount).toBe(1)
    expect(view.totalCount).toBe(1)
    expect(view.rows).toHaveLength(1)

    const row = view.rows[0]
    expect(row.callsign).toBe('scan')
    expect(row.activity).toBe('working')
    expect(row.glyph).toBe('●')
    expect(row.tone).toBe('accent')
    expect(row.elapsed).toBe('5s')
    expect(row.live).toBe(true)
  })
})

describe('projectAgentDock: status metadata', () => {
  it('distinguishes every status with glyph + tone, never color alone', () => {
    const view = project(
      [
        makeItem({ id: 'run', index: 0, status: 'running' }),
        makeItem({ id: 'que', index: 1, status: 'queued' }),
        makeItem({ id: 'com', index: 2, status: 'completed' }),
        makeItem({ id: 'int', index: 3, status: 'interrupted' }),
        makeItem({ id: 'fai', index: 4, status: 'failed' }),
        makeItem({ id: 'tim', index: 5, status: 'timeout' }),
        makeItem({ id: 'err', index: 6, status: 'error' })
      ],
      { maxRows: 7 }
    )

    const meta = Object.fromEntries(view.rows.map(r => [r.id, { glyph: r.glyph, tone: r.tone }]))

    expect(meta.run).toEqual({ glyph: '●', tone: 'accent' })
    expect(meta.que).toEqual({ glyph: '○', tone: 'muted' })
    expect(meta.com).toEqual({ glyph: '✓', tone: 'statusGood' })
    expect(meta.int).toEqual({ glyph: '■', tone: 'warn' })
    expect(meta.fai).toEqual({ glyph: '✗', tone: 'error' })
    expect(meta.tim).toEqual({ glyph: 'T', tone: 'warn' })
    expect(meta.err).toEqual({ glyph: '!', tone: 'error' })

    // Glyphs are pairwise distinct so color is never the only signal.
    const glyphs = view.rows.map(r => r.glyph)
    expect(new Set(glyphs).size).toBe(glyphs.length)
  })

  it('counts running + queued as active', () => {
    const view = project([
      makeItem({ id: 'a', index: 0, status: 'running' }),
      makeItem({ id: 'b', index: 1, status: 'queued' }),
      makeItem({ id: 'c', index: 2, status: 'completed' })
    ])

    expect(view.activeCount).toBe(2)
    expect(view.totalCount).toBe(3)
  })
})

describe('projectAgentDock: ordering', () => {
  it('preserves spawn/tree order with children after their parent', () => {
    const view = project([
      makeItem({ id: 'q', index: 1 }),
      makeItem({ depth: 1, id: 'c1', index: 0, parentId: 'p' }),
      makeItem({ id: 'p', index: 0 })
    ])

    expect(view.rows.map(r => r.id)).toEqual(['p', 'c1', 'q'])
    expect(view.rows.map(r => r.depth)).toEqual([0, 1, 0])
  })
})

describe('projectAgentDock: durations', () => {
  it('prefers terminal durationSeconds over live elapsed', () => {
    const view = project([
      makeItem({ durationSeconds: 90, id: 'a', index: 0, startedAt: NOW - 5000, status: 'completed' })
    ])

    expect(view.rows[0].elapsed).toBe('1m 30s')
    expect(view.rows[0].live).toBe(false)
  })

  it('preserves an explicit zero-second terminal duration', () => {
    const view = project([
      makeItem({ durationSeconds: 0, id: 'a', index: 0, startedAt: NOW - 5000, status: 'completed' })
    ])

    expect(view.rows[0].elapsed).toBe('0s')
    expect(view.rows[0].live).toBe(false)
  })

  it('reports no duration when nothing is known', () => {
    const view = project([makeItem({ id: 'a', index: 0, status: 'queued' })])

    expect(view.rows[0].elapsed).toBe('')
    expect(view.rows[0].live).toBe(false)
  })
})

describe('projectAgentDock: labels', () => {
  it('derives a bounded ASCII callsign and falls back truthfully for non-ASCII goals', () => {
    const view = project(
      [
        makeItem({ goal: 'Patch token-bucket refill race', id: 'a', index: 0 }),
        makeItem({ goal: '分析レポート 🔎', id: 'b', index: 2 }),
        makeItem({ goal: '/Users/person/private/file.txt', id: 'c', index: 4 }),
        makeItem({ goal: './private/file.ts', id: 'd', index: 5 }),
        makeItem({ goal: '../private/file.ts', id: 'e', index: 6 }),
        makeItem({ goal: 'src/private/file.ts', id: 'f', index: 7 }),
        makeItem({ goal: 'file:///private/file.ts', id: 'g', index: 8 }),
        makeItem({ goal: 'ssh://private.example/repo', id: 'h', index: 9 }),
        makeItem({ goal: 'C:\\private\\file.ts', id: 'i', index: 10 }),
        makeItem({ goal: '', id: 'j', index: 11 }),
        makeItem({ goal: 'abcdefghijklmnop', id: 'k', index: 12 })
      ],
      { maxRows: 11 }
    )

    expect(view.rows[0].callsign).toBe('patch')
    expect(view.rows[1].callsign).toBe('agent 3')
    expect(view.rows[2].callsign).toBe('agent 5')
    expect(view.rows.slice(3, 9).map(row => row.callsign)).toEqual([
      'agent 6',
      'agent 7',
      'agent 8',
      'agent 9',
      'agent 10',
      'agent 11'
    ])
    expect(view.rows[9].callsign).toBe('agent 12')
    expect(view.rows[10].callsign).toBe('agent 13')
    expect(view.rows.every(row => /^[\x20-\x7e]{1,12}$/.test(row.callsign))).toBe(true)
  })

  it('never exposes arbitrary names, projects, medical topics, emails, or token-like goal words', () => {
    const view = project(
      [
        makeItem({ goal: 'Acme client handoff', id: 'a', index: 0 }),
        makeItem({ goal: 'dana.reyes@example.com followup', id: 'b', index: 1 }),
        makeItem({ goal: 'oncology review for patient', id: 'c', index: 2 }),
        makeItem({ goal: 'sk_live_ABC123 rotate credentials', id: 'd', index: 3 })
      ],
      { maxRows: 4 }
    )

    expect(view.rows.map(row => row.callsign)).toEqual(['agent 1', 'agent 2', 'review', 'agent 4'])
    expect(JSON.stringify(view)).not.toMatch(/Acme|reyes|oncology|patient|sk_live|ABC123/i)
  })

  it('maps the latest tool to a plain-language activity without its raw preview', () => {
    const view = project([
      makeItem({ id: 'a', index: 0, tools: ['Read', 'Terminal("npm test -- --run private/path")'] })
    ])

    expect(view.rows[0].activity).toBe('running commands')
    expect(view.rows[0].activity).not.toContain('private/path')
  })

  it.each([
    ['Read File("private/path")', 'reading files'],
    ['Search Files("private/query")', 'searching'],
    ['Web Search("private/query")', 'searching web'],
    ['Write File("private/path")', 'writing files']
  ])('maps real gateway label %s without exposing its private preview', (tool, activity) => {
    const view = project([makeItem({ id: 'a', index: 0, tools: [tool] })])

    expect(view.rows[0].activity).toBe(activity)
    expect(view.rows[0].activity).not.toContain('private')
  })

  it.each([
    ['terminal', 'running commands'],
    ['read_file', 'reading files'],
    ['write_file', 'writing files'],
    ['patch', 'editing files'],
    ['search_files', 'searching'],
    ['web_search', 'searching web'],
    ['web_extract', 'reading web'],
    ['execute_code', 'running code'],
    ['delegate_task', 'delegating'],
    ['browser_navigate', 'browsing'],
    ['browser_click', 'browsing'],
    ['memory', 'updating memory'],
    ['todo', 'updating todos'],
    ['process', 'managing process'],
    ['session_search', 'searching sessions'],
    ['computer_use', 'using computer'],
    ['skill_view', 'reading skills'],
    ['kanban_show', 'reading task'],
    ['ha_get_state', 'checking home']
  ])('maps formatToolCall(%s) into a specific plain activity without leaking the preview', (name, activity) => {
    const line = formatToolCall(name, 'private/value')
    const view = project([makeItem({ id: 'a', index: 0, tools: [line] })])

    expect(line).toContain('private/value')
    expect(view.rows[0].activity).toBe(activity)
    expect(view.rows[0].activity).not.toContain('private')
    expect(view.rows[0].activity).not.toBe('working')
  })

  it('keeps unknown tool names generic without leaking preview text', () => {
    const line = formatToolCall('totally_unknown_tool', 'private/secret')
    const view = project([makeItem({ id: 'a', index: 0, tools: [line] })])

    expect(view.rows[0].activity).toBe('working')
    expect(view.rows[0].activity).not.toContain('private')
    expect(view.rows[0].activity).not.toContain('secret')
  })

  it('uses truthful terminal activity labels rather than stale tool details', () => {
    const view = project([
      makeItem({
        id: 'failed',
        index: 0,
        status: 'failed',
        tools: ['Terminal("npm test -- --run private/path")']
      })
    ])

    expect(view.rows[0].activity).toBe('failed')
    expect(view.rows[0].activity).not.toContain('private/path')
  })

  it('marks completed output as result-ready without rendering its summary', () => {
    const view = project([makeItem({ id: 'a', index: 0, status: 'completed', summary: 'wrote 3 tests' })])

    expect(view.rows[0].activity).toBe('result ready')
    expect(JSON.stringify(view)).not.toContain('wrote 3 tests')
  })

  it('labels every terminal and queued branch truthfully', () => {
    const view = project(
      [
        makeItem({ id: 'done', index: 0, status: 'completed' }),
        makeItem({ id: 'queued', index: 1, status: 'queued' }),
        makeItem({ id: 'error', index: 2, status: 'error' }),
        makeItem({ id: 'interrupted', index: 3, status: 'interrupted' }),
        makeItem({ id: 'timeout', index: 4, status: 'timeout' })
      ],
      { maxRows: 5 }
    )

    expect(Object.fromEntries(view.rows.map(row => [row.id, row.activity]))).toEqual({
      done: 'done',
      error: 'failed',
      interrupted: 'interrupted',
      queued: 'queued',
      timeout: 'timed out'
    })
  })
})

describe('projectAgentDock: header counts', () => {
  it('separates running, queued, ready, and blocked counts', () => {
    const view = project([
      makeItem({ id: 'run', index: 0, status: 'running' }),
      makeItem({ id: 'queue', index: 1, status: 'queued' }),
      makeItem({ id: 'ready', index: 2, status: 'completed', summary: 'result' }),
      makeItem({ id: 'failed', index: 3, status: 'failed' })
    ])

    expect(view.header).toBe('1 running · 1 queued · 1 ready · 1 blocked')
  })

  it('uses terminal counts instead of zero-running language when all work is done', () => {
    const view = project([
      makeItem({ id: 'done-0', index: 0, status: 'completed' }),
      makeItem({ id: 'done-1', index: 1, status: 'completed', summary: 'result' }),
      makeItem({ id: 'done-2', index: 2, status: 'completed' }),
      makeItem({ id: 'done-3', index: 3, status: 'completed' }),
      makeItem({ id: 'failed', index: 4, status: 'failed' })
    ])

    expect(view.header).toBe('4 done · 1 blocked')
    expect(view.header).not.toContain('0 running')
  })
})

describe('projectAgentDock: bounds', () => {
  it('caps visible rows at three and reports overflow', () => {
    const items = Array.from({ length: 8 }, (_, i) =>
      makeItem({ id: `a${i}`, index: i, status: i >= 6 ? 'running' : 'completed' })
    )

    const view = project(items)

    expect(DOCK_MAX_ROWS).toBe(3)
    expect(view.rows).toHaveLength(3)
    expect(view.rows.map(row => row.id)).toEqual(['a0', 'a6', 'a7'])
    expect(view.overflow).toBe(5)
    expect(view.overflowActive).toBe(0)
    expect(view.overflowSummary).toBe('5 more · 5 done')
  })

  it('reports active overflow when more than three agents are live', () => {
    const view = project(Array.from({ length: 6 }, (_, index) => makeItem({ id: `live-${index}`, index })))

    expect(view.overflow).toBe(3)
    expect(view.overflowActive).toBe(3)
    expect(view.overflowSummary).toBe('3 more · 3 running')
  })

  it('honors a smaller maxRows override', () => {
    const items = Array.from({ length: 4 }, (_, i) => makeItem({ id: `a${i}`, index: i }))
    const view = project(items, { maxRows: 2 })

    expect(view.rows).toHaveLength(2)
    expect(view.overflow).toBe(2)
  })

  it('keeps later active and failed work visible ahead of completed rows', () => {
    const view = project(
      [
        makeItem({ id: 'done-0', index: 0, status: 'completed' }),
        makeItem({ id: 'done-1', index: 1, status: 'completed' }),
        makeItem({ id: 'done-2', index: 2, status: 'completed' }),
        makeItem({ id: 'done-3', index: 3, status: 'completed' }),
        makeItem({ id: 'done-4', index: 4, status: 'completed' }),
        makeItem({ id: 'live', index: 5, status: 'running' }),
        makeItem({ id: 'failed', index: 6, status: 'failed' })
      ],
      { maxRows: 3 }
    )

    expect(view.rows.map(row => row.id)).toEqual(['done-0', 'live', 'failed'])
    expect(view.overflowActive).toBe(0)
  })

  it('keeps queued work and every blocked status ahead of completed rows under pressure', () => {
    const view = project(
      [
        makeItem({ id: 'done-0', index: 0, status: 'completed' }),
        makeItem({ id: 'queued', index: 1, status: 'queued' }),
        makeItem({ id: 'error', index: 2, status: 'error' }),
        makeItem({ id: 'interrupted', index: 3, status: 'interrupted' }),
        makeItem({ id: 'timeout', index: 4, status: 'timeout' }),
        makeItem({ id: 'failed', index: 5, status: 'failed' }),
        makeItem({ id: 'done-1', index: 6, status: 'completed' })
      ],
      { maxRows: 5 }
    )

    expect(view.rows.map(row => row.id)).toEqual(['queued', 'error', 'interrupted', 'timeout', 'failed'])
  })

  it('keeps a selected completed parent above its live child', () => {
    const view = project([
      makeItem({ id: 'parent', index: 0, status: 'completed' }),
      makeItem({ depth: 1, id: 'child', index: 0, parentId: 'parent', status: 'running' })
    ])

    expect(view.rows.map(row => row.id)).toEqual(['parent', 'child'])
    expect(view.rows.map(row => row.depth)).toEqual([0, 1])
  })

  it('rebases a live child to depth zero when its parent is overflowed', () => {
    const view = project(
      [
        makeItem({ id: 'parent', index: 0, status: 'completed' }),
        makeItem({ depth: 1, id: 'child', index: 0, parentId: 'parent', status: 'running' })
      ],
      { maxRows: 1 }
    )

    expect(view.rows.map(row => row.id)).toEqual(['child'])
    expect(view.rows[0].depth).toBe(0)
  })
})

describe('projectAgentDock: narrow terminals', () => {
  it('collapses to a one-line summary below the narrow threshold', () => {
    const items = [
      makeItem({ id: 'a', index: 0, startedAt: NOW - 65_000 }),
      makeItem({ id: 'b', index: 1, status: 'completed' }),
      makeItem({ id: 'c', index: 2, status: 'failed' })
    ]

    const view = project(items, { width: DOCK_NARROW_WIDTH - 1 })

    expect(view.summaryOnly).toBe(true)
    expect(view.rows).toEqual([])
    expect(view.summary).toContain('1/3 active')
    expect(view.summary).toContain('1 issue')
  })

  it('stays expanded at 80 columns', () => {
    const view = project([makeItem({ id: 'a', index: 0 })], { width: 80 })

    expect(view.summaryOnly).toBe(false)
    expect(view.rows).toHaveLength(1)
  })

  it('stays expanded at the exact 60-column boundary', () => {
    const view = project([makeItem({ id: 'a', index: 0 })], { width: DOCK_NARROW_WIDTH })

    expect(view.summaryOnly).toBe(false)
    expect(view.rows).toHaveLength(1)
  })
})

describe('projectAgentDock: purity', () => {
  it('does not mutate its input', () => {
    const item = Object.freeze(makeItem({ id: 'a', index: 0, startedAt: NOW - 1000 }))
    const items = Object.freeze([item]) as unknown as SubagentProgress[]

    expect(() => project(items)).not.toThrow()

    const view = project(items)
    expect(view.rows[0].id).toBe('a')
  })
})
