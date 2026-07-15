import { PassThrough } from 'stream'

import { renderSync, stringWidth, Text } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { KanbanActivityDock, KanbanActivityRow, KanbanExecutionSpine } from '../components/kanbanActivity.js'
import type { KanbanActivityResponse, KanbanActivityTask } from '../gatewayTypes.js'
import { buildKanbanActivityRows, normalizeKanbanActivity } from '../lib/kanbanActivity.js'
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

function activity(boards: { board: string; error?: string; roots: KanbanActivityTask[] }[]) {
  const payload: KanbanActivityResponse = {
    active_count: 0,
    attention_count: 0,
    boards: boards.map(board => ({ ...board, checked_at: NOW })),
    checked_at: NOW
  }

  return normalizeKanbanActivity(payload, NOW)
}

// Each stdout.write CALL from @hermes/ink is one complete frame (verified by
// probing: non-TTY renders write the full frame per render pass, then a bare
// newline on cleanup). Intercept the write method itself — stream 'data'
// events may coalesce or split chunks, write-call boundaries cannot.
function captureFrames(element: React.ReactElement, columns: number): string[] {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  const frames: string[] = []
  const originalWrite = stdout.write.bind(stdout)

  stdout.write = ((
    chunk: Buffer | string,
    encoding?: ((error?: Error | null) => void) | BufferEncoding,
    callback?: (error?: Error | null) => void
  ): boolean => {
    frames.push(chunk.toString())

    return typeof encoding === 'function' ? originalWrite(chunk, encoding) : originalWrite(chunk, encoding, callback)
  }) as typeof stdout.write
  Object.assign(stdout, { columns, isTTY: false, rows: 30 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })

  const instance = renderSync(element, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()

  return frames
}

function renderFrame(element: React.ReactElement, columns = 80): string {
  const lastFrame =
    captureFrames(element, columns)
      .map(frame => stripAnsi(frame))
      .filter(frame => frame.trim().length > 0)
      .at(-1) ?? ''

  return lastFrame
    .split('\n')
    .map(line => line.replace(/\s+$/, ''))
    .filter(Boolean)
    .join('\n')
}

interface TextSegment {
  color: string | undefined
  dim: boolean
  text: string
}

// The non-TTY test renderer strips all color/dim codes from output, so the
// paint-channel contract is asserted on the element tree instead. These
// components are hook-free, which makes their memo-wrapped render functions
// directly callable.
function renderElement<Props>(component: unknown, props: Props): React.ReactElement {
  return (component as { type: (componentProps: Props) => React.ReactElement }).type(props)
}

function collectTextSegments(node: unknown, segments: TextSegment[] = []): TextSegment[] {
  if (!React.isValidElement(node)) {
    return segments
  }

  const props = node.props as { children?: React.ReactNode; color?: string; dim?: boolean }

  if (node.type === Text) {
    const text = React.Children.toArray(props.children)
      .filter(child => typeof child === 'string' || typeof child === 'number')
      .join('')

    segments.push({ color: props.color, dim: Boolean(props.dim), text })

    return segments
  }

  React.Children.forEach(props.children, child => {
    collectTextSegments(child, segments)
  })

  return segments
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

  it('draws ancestor continuation rails at depth 2', () => {
    const grandchildren = [
      task({ task_id: 'g1', title: 'Grandchild 1' }),
      task({ run: null, status: 'ready', task_id: 'g2', title: 'Grandchild 2' })
    ]

    const roots = [
      task({
        children: [
          task({ children: grandchildren, task_id: 'a', title: 'Child A' }),
          task({ run: null, status: 'ready', task_id: 'b', title: 'Child B' })
        ],
        task_id: 'root',
        title: 'Ship feature'
      })
    ]

    const frame = renderFrame(
      <KanbanExecutionSpine activity={activity([{ board: 'default', roots }])} now={NOW} t={DEFAULT_THEME} width={70} />
    )

    expect(frame).toContain('│   ├── ◉ Grandchild 1')
    expect(frame).toContain('│   └── ○ Grandchild 2')
    expect(frame).toContain('└── ○ Child B')
  })

  it('renders a completed/current/upcoming pipeline on one light rail', () => {
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

    expect(frame).toMatch(/│ ● Plan/)
    expect(frame).toMatch(/│ ◉ Build/)
    expect(frame).toMatch(/│ ○ Verify/)
    expect(frame).not.toContain('┃')
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

  it('keeps the state label visible when width squeezes the row', () => {
    const frame = renderFrame(
      <KanbanExecutionSpine
        activity={activity([
          {
            board: 'default',
            roots: [
              task({ block_reason: 'Needs Henry', run: null, status: 'blocked', task_id: 'visual', title: 'Visual approval' })
            ]
          }
        ])}
        now={NOW}
        t={DEFAULT_THEME}
        width={32}
      />,
      40
    )

    expect(frame).toContain('Visual approval · blocked')
    expect(frame).not.toContain('implementer')
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
    expect(multi).not.toContain('Kanban ·')
    expect(renderFrame(<KanbanExecutionSpine activity={activity([])} now={NOW} t={DEFAULT_THEME} width={60} />)).toBe(
      ''
    )
    expect(renderFrame(<KanbanActivityDock activity={activity([])} now={NOW} t={DEFAULT_THEME} width={60} />)).toBe('')
  })

  it('marks an unavailable board in place', () => {
    const frame = renderFrame(
      <KanbanExecutionSpine
        activity={activity([{ board: 'default', error: 'db locked', roots: [task()] }])}
        now={NOW}
        t={DEFAULT_THEME}
        width={60}
      />
    )

    expect(frame).toContain('default · unavailable')
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

  it('speaks glyph shorthand in the narrow dock without a duplicate bang', () => {
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

    expect(frame).toBe('K !1 ◉1')
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

    expect(frame).toContain('├── ◉ Child 2')
    expect(frame).toContain('└── +6 more tasks')
    expect(frame).not.toContain('Child 8')
  })

  it('captures every line of a multi-row frame, including repeated labels', () => {
    const frame = renderFrame(
      <KanbanExecutionSpine
        activity={activity([
          {
            board: 'default',
            roots: [
              task({
                children: [task({ task_id: 'c1', title: 'Repeat' }), task({ task_id: 'c2', title: 'Repeat' })],
                task_id: 'root'
              })
            ]
          }
        ])}
        now={NOW}
        t={DEFAULT_THEME}
        width={70}
      />
    )

    expect(frame).toBe(
      [
        '│ ◉ Build execution spine — implementer · running',
        '├── ◉ Repeat — implementer · running',
        '└── ◉ Repeat — implementer · running'
      ].join('\n')
    )
  })

  it('bounds the unavailable-board row at extreme widths', () => {
    const frame = renderFrame(
      <KanbanExecutionSpine
        activity={activity([{ board: 'default', error: 'db locked', roots: [task()] }])}
        now={NOW}
        t={DEFAULT_THEME}
        width={12}
      />,
      40
    )

    expect(frame).toContain('unavailable')
    expect(frame).not.toContain('· unavailable')

    for (const line of frame.split('\n')) {
      expect(stringWidth(line)).toBeLessThanOrEqual(12)
    }
  })

  it('reports completed-only recent activity truthfully in the narrow dock', () => {
    const frame = renderFrame(
      <KanbanActivityDock
        activity={activity([
          {
            board: 'default',
            roots: [task({ run: { ...task().run!, ended_at: NOW - 10, outcome: 'completed' }, status: 'done' })]
          }
        ])}
        now={NOW}
        t={DEFAULT_THEME}
        width={18}
      />,
      18
    )

    expect(frame).toBe('K ●1')
  })
})

describe('segment paint contract', () => {
  const themedRows = (roots: KanbanActivityTask[], width = 60) =>
    buildKanbanActivityRows(activity([{ board: 'default', roots }]), { now: NOW, width })

  it('never dims or tints a task title, even on muted rows', () => {
    const rows = themedRows([
      task({ children: [task({ run: null, status: 'scheduled', task_id: 'c1', title: 'Scheduled thing' })], task_id: 'root' })
    ])

    const muted = rows.find(row => row.kind === 'task' && row.task.id === 'c1')!
    const segments = collectTextSegments(renderElement(KanbanActivityRow, { row: muted, t: DEFAULT_THEME, width: 60 }))
    const title = segments.find(segment => segment.text === 'Scheduled thing')

    expect(title).toBeDefined()
    expect(title!.dim).toBe(false)
    expect(title!.color).toBeUndefined()

    const state = segments.find(segment => segment.text === ' · scheduled')

    expect(state!.color).toBe(DEFAULT_THEME.color.muted)
    expect(state!.dim).toBe(false)
  })

  it('keeps warning color on the state word while the title stays default', () => {
    const rows = themedRows([
      task({ block_reason: 'Needs Henry', run: null, status: 'blocked', task_id: 'blocked', title: 'Visual approval' })
    ])

    const blocked = rows.find(row => row.kind === 'task')!
    const segments = collectTextSegments(renderElement(KanbanActivityRow, { row: blocked, t: DEFAULT_THEME, width: 60 }))

    expect(segments.find(segment => segment.text === 'Visual approval')!.color).toBeUndefined()
    expect(segments.find(segment => segment.text.includes('blocked'))!.color).toBe(DEFAULT_THEME.color.warn)
  })

  it('splits the narrow dock into neutral, warning, and accent segments', () => {
    const model = activity([
      { board: 'default', roots: [task(), task({ block_reason: 'Review', status: 'blocked', task_id: 'blocked' })] }
    ])

    const segments = collectTextSegments(
      renderElement(KanbanActivityDock, { activity: model, now: NOW, t: DEFAULT_THEME, width: 18 })
    )

    expect(segments.map(segment => ({ color: segment.color, text: segment.text }))).toEqual([
      { color: undefined, text: 'K' },
      { color: DEFAULT_THEME.color.warn, text: ' !1' },
      { color: DEFAULT_THEME.color.accent, text: ' ◉1' }
    ])
  })

  it('colors medium dock lamps and counts without tinting the copy', () => {
    const model = activity([
      { board: 'default', roots: [task(), task({ block_reason: 'Review', status: 'blocked', task_id: 'blocked' })] }
    ])

    const segments = collectTextSegments(
      renderElement(KanbanActivityDock, { activity: model, now: NOW, t: DEFAULT_THEME, width: 40 })
    )

    expect(segments.map(segment => ({ color: segment.color, text: segment.text }))).toEqual([
      { color: DEFAULT_THEME.color.warn, text: '! ' },
      { color: undefined, text: 'Kanban · ' },
      { color: DEFAULT_THEME.color.warn, text: '1' },
      { color: undefined, text: ' need attention · ' },
      { color: DEFAULT_THEME.color.accent, text: '1' },
      { color: undefined, text: ' active' }
    ])
  })

  it('applies dock ladder breakpoints to the full component width', () => {
    const model = activity([{ board: 'default', roots: [task({ title: 'Current work' })] }])

    for (const width of [28, 29]) {
      const frame = renderFrame(
        <KanbanActivityDock activity={model} now={NOW} t={DEFAULT_THEME} width={width} />,
        width
      )

      expect(frame).toContain('Kanban · 1 active')
      expect(frame).not.toContain('K ◉1')
    }

    const belowWide = renderFrame(
      <KanbanActivityDock activity={model} now={NOW} t={DEFAULT_THEME} width={47} />,
      47
    )

    expect(belowWide).toBe('◉ Kanban · 1 active')

    for (const width of [48, 49]) {
      const frame = renderFrame(
        <KanbanActivityDock activity={model} now={NOW} t={DEFAULT_THEME} width={width} />,
        width
      )

      expect(frame).toContain('Kanban · 1 active · Current work')
    }
  })

  it('colors the completed-only narrow dock with the success tone', () => {
    const model = activity([
      { board: 'default', roots: [task({ run: { ...task().run!, ended_at: NOW - 10, outcome: 'completed' }, status: 'done' })] }
    ])

    const segments = collectTextSegments(
      renderElement(KanbanActivityDock, { activity: model, now: NOW, t: DEFAULT_THEME, width: 18 })
    )

    expect(segments.map(segment => ({ color: segment.color, text: segment.text }))).toEqual([
      { color: undefined, text: 'K' },
      { color: DEFAULT_THEME.color.ok, text: ' ●1' }
    ])
  })
})
