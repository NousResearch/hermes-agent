import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { KanbanActivityDock, KanbanExecutionSpine } from '../components/kanbanActivity.js'
import type { KanbanActivityResponse, KanbanActivityTask } from '../gatewayTypes.js'
import { normalizeKanbanActivity } from '../lib/kanbanActivity.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const NOW = 1_721_000_000

function task(overrides: Partial<KanbanActivityTask> = {}): KanbanActivityTask {
  return {
    assignee: 'implementer',
    block_reason: null,
    children: [],
    parents: [],
    run: {
      ended_at: null,
      last_heartbeat_at: NOW - 5,
      max_runtime_seconds: 3600,
      outcome: null,
      profile: 'implementer',
      run_id: 1,
      started_at: NOW - 50
    },
    status: 'running',
    task_id: 'task',
    title: 'Build execution spine',
    ...overrides
  }
}

function activity(boards: { board: string; roots: KanbanActivityTask[] }[]) {
  const payload: KanbanActivityResponse = {
    active_count: 0,
    attention_count: 0,
    boards: boards.map(board => ({ ...board, checked_at: NOW })),
    checked_at: NOW
  }

  return normalizeKanbanActivity(payload, NOW)
}

function renderFrame(element: React.ReactElement, columns = 80): string {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''
  Object.assign(stdout, { columns, isTTY: false, rows: 30 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(element, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()

  return stripAnsi(output)
    .split('\n')
    .map(line => line.replace(/\s+$/, ''))
    .filter(Boolean)
    .join('\n')
}

describe('Kanban activity static Ink components', () => {
  it('renders an active collapsed dock on exactly one line', () => {
    const frame = renderFrame(
      <KanbanActivityDock
        activity={activity([{ board: 'default', roots: [task()] }])}
        now={NOW}
        t={DEFAULT_THEME}
        width={70}
      />
    )

    expect(frame.split('\n')).toHaveLength(1)
    expect(frame).toContain('Kanban')
    expect(frame).toContain('1 active')
    expect(frame).toContain('Build execution spine')
  })

  it('renders branching children with legible connectors', () => {
    const children = ['Research', 'Implement', 'Review'].map((title, index) =>
      task({ status: index === 2 ? 'ready' : 'running', task_id: `child-${index}`, title })
    )

    const frame = renderFrame(
      <KanbanExecutionSpine
        activity={activity([{ board: 'default', roots: [task({ children, task_id: 'root', title: 'Ship feature' })] }])}
        now={NOW}
        t={DEFAULT_THEME}
        width={70}
      />
    )

    expect(frame).toContain('├──')
    expect(frame).toContain('└──')
    expect(frame).toContain('○ Review')
  })

  it('renders a completed/current/upcoming pipeline with heavy and light rails', () => {
    const frame = renderFrame(
      <KanbanExecutionSpine
        activity={activity([
          {
            board: 'release',
            roots: [
              task({ status: 'done', task_id: 'plan', title: 'Plan' }),
              task({ task_id: 'build', title: 'Build' }),
              task({ status: 'ready', task_id: 'verify', title: 'Verify' })
            ]
          }
        ])}
        now={NOW}
        t={DEFAULT_THEME}
        width={70}
      />
    )

    expect(frame).toMatch(/┃ ● Plan/)
    expect(frame).toMatch(/│ ◉ Build/)
    expect(frame).toMatch(/│ ○ Verify/)
  })

  it('renders blocked, stale, and failed states truthfully', () => {
    const frame = renderFrame(
      <KanbanExecutionSpine
        activity={activity([
          {
            board: 'default',
            roots: [
              task({ block_reason: 'Human review required', status: 'blocked', task_id: 'blocked', title: 'Publish' }),
              task({ run: { ...task().run!, last_heartbeat_at: NOW - 1000 }, task_id: 'stale', title: 'Long job' }),
              task({
                run: { ...task().run!, ended_at: NOW, outcome: 'failed' },
                status: 'done',
                task_id: 'failed',
                title: 'Tests'
              })
            ]
          }
        ])}
        now={NOW}
        staleAfterSeconds={300}
        t={DEFAULT_THEME}
        width={80}
      />
    )

    expect(frame).toContain('! Publish — implementer · blocked: Human review required')
    expect(frame).toContain('! Long job — implementer · heartbeat stale')
    expect(frame).toContain('× Tests — implementer · failed')
    expect(frame).not.toMatch(/%|phase/i)
  })

  it('groups multiple boards deterministically and hides empty activity', () => {
    const multi = renderFrame(
      <KanbanExecutionSpine
        activity={activity([
          { board: 'zeta', roots: [task({ task_id: 'z', title: 'Z task' })] },
          { board: 'alpha', roots: [task({ task_id: 'a', title: 'A task' })] }
        ])}
        now={NOW}
        t={DEFAULT_THEME}
        width={60}
      />
    )

    expect(multi.indexOf('alpha')).toBeLessThan(multi.indexOf('zeta'))
    expect(renderFrame(<KanbanExecutionSpine activity={activity([])} now={NOW} t={DEFAULT_THEME} width={60} />)).toBe(
      ''
    )
    expect(renderFrame(<KanbanActivityDock activity={activity([])} now={NOW} t={DEFAULT_THEME} width={60} />)).toBe('')
  })

  it('hides backlog-only and expired-completion activity', () => {
    const queued = activity([
      {
        board: 'default',
        roots: [
          task({ run: null, status: 'triage', task_id: 'triage' }),
          task({ run: null, status: 'ready', task_id: 'ready' }),
          task({ run: null, status: 'review', task_id: 'review' }),
          task({ run: null, status: 'scheduled', task_id: 'scheduled' })
        ]
      }
    ])

    const expired = activity([
      {
        board: 'default',
        roots: [task({ run: { ...task().run!, ended_at: NOW - 1000, outcome: 'completed' }, status: 'done' })]
      }
    ])

    expect(renderFrame(<KanbanActivityDock activity={queued} now={NOW} t={DEFAULT_THEME} width={60} />)).toBe('')
    expect(renderFrame(<KanbanExecutionSpine activity={queued} now={NOW} t={DEFAULT_THEME} width={60} />)).toBe('')
    expect(renderFrame(<KanbanActivityDock activity={expired} now={NOW} t={DEFAULT_THEME} width={60} />)).toBe('')
  })

  it('progressively collapses the narrow dock without wrapping', () => {
    const frame = renderFrame(
      <KanbanActivityDock
        activity={activity([
          {
            board: 'default',
            roots: [task(), task({ block_reason: 'Review', status: 'blocked', task_id: 'blocked' })]
          }
        ])}
        now={NOW}
        t={DEFAULT_THEME}
        width={18}
      />,
      18
    )

    expect(frame.split('\n')).toHaveLength(1)
    expect(frame).toContain('!1')
    expect(frame.length).toBeLessThanOrEqual(18)
    expect(frame).not.toContain('task-')
  })

  it('compresses many children into a bounded summary row', () => {
    const children = Array.from({ length: 9 }, (_, index) =>
      task({ task_id: `child-${index}`, title: `Child ${index}` })
    )

    const frame = renderFrame(
      <KanbanExecutionSpine
        activity={activity([
          { board: 'default', roots: [task({ children, task_id: 'root', title: 'Parallel work' })] }
        ])}
        maxChildren={3}
        now={NOW}
        t={DEFAULT_THEME}
        width={70}
      />
    )

    expect(frame).toContain('+6 more tasks')
    expect(frame).not.toContain('Child 8')
  })
})
