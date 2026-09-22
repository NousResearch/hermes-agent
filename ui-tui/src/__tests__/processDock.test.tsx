import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { expect, it } from 'vitest'

import { buildProcessRows, PROCESS_RETAIN_SECONDS, type ProcessEntry } from '../app/processRoster.js'
import { AgentsPanelView, buildProcessBlock, splitDockBudget } from '../components/agentsPanel.js'
import { buildAgentRows, dockRowLimit } from '../lib/agentRows.js'
import { DEFAULT_THEME } from '../theme.js'
import type { SubagentProgress } from '../types.js'

const NOW = 1_000_000_000

const agent = (id: string): SubagentProgress => ({
  id,
  goal: 'Investigate authentication handshake',
  depth: 0,
  index: 0,
  parentId: null,
  notes: [],
  tools: ['read_file auth.ts'],
  thinking: [],
  toolCount: 1,
  taskCount: 1,
  startedAt: NOW * 1000 - 5000,
  status: 'running'
})

const running: ProcessEntry = {
  session_id: 'proc_run',
  command: 'npm run build\n',
  status: 'running',
  uptime_seconds: 42,
  output_preview: 'compiling…\nbundled 300 modules\n\n'
}

const exited = (id: string, exitedAgoSeconds: number, exitCode = 0): ProcessEntry => ({
  session_id: id,
  command: 'pytest tests/',
  status: 'exited',
  uptime_seconds: 12 + exitedAgoSeconds,
  exit_code: exitCode,
  exited_at: NOW - exitedAgoSeconds,
  completion_reason: 'exited'
})

const paint = (element: React.ReactElement, rows = 30): string => {
  const stdout = Object.assign(new PassThrough(), { columns: 80, rows })
  const frames: string[] = []
  stdout.on('data', chunk => frames.push(chunk.toString()))

  const view = renderSync(element, {
    stdout: stdout as unknown as NodeJS.WriteStream,
    stdin: new PassThrough() as unknown as NodeJS.ReadStream
  })

  view.unmount()
  view.cleanup()

  return stripAnsi(frames.join(''))
}

it('keeps running processes, ages finished ones out after the retention window, and carries the exit verdict', () => {
  const rows = buildProcessRows(
    [exited('old', PROCESS_RETAIN_SECONDS + 1), exited('fresh', 5, 1), running, exited('killed', 2)],
    NOW * 1000
  )

  expect(rows.map(r => r.id)).toEqual(['proc_run', 'killed', 'fresh'])
  expect(rows[0]).toMatchObject({ status: 'running', detail: 'last: bundled 300 modules', elapsedSeconds: 42 })
  expect(rows[2]).toMatchObject({ status: 'failed', detail: 'exit 1 · 5s ago', elapsedSeconds: 12 })
  // A checkpoint-recovered exit without a timestamp never lingers as a phantom row.
  expect(buildProcessRows([{ ...exited('x', 1), exited_at: null }], NOW * 1000)).toEqual([])
})

it('paints a Processes block under the agents without letting either block hide the other', () => {
  const agents = Array.from({ length: 6 }, (_, i) => agent(`child-${i}`))
  const processes = buildProcessRows([running, exited('fresh', 5)], NOW * 1000)
  const limit = dockRowLimit(30)
  const budget = splitDockBudget(limit, agents.length, processes.length)
  expect(budget.agents).toBeGreaterThanOrEqual(1)
  expect(budget.processes).toBeGreaterThanOrEqual(1)
  expect(budget.agents + budget.processes).toBeLessThanOrEqual(Math.max(limit, 2))

  const agentRows = buildAgentRows(agents, [], NOW * 1000, budget.agents)
  const block = buildProcessBlock(processes, budget.processes)
  const text = paint(<AgentsPanelView cols={80} {...agentRows} processes={block} t={DEFAULT_THEME} />)

  expect(text).toContain('6 live agents')
  expect(text).toContain('Processes · 1 running · 1 done')
  expect(text).toContain('⚙ npm run build')
  expect(text.indexOf('live agents')).toBeLessThan(text.indexOf('Processes ·'))

  // Processes alone still surface the dock, with the expand/collapse controls on their heading.
  const alone = paint(
    <AgentsPanelView
      cols={80}
      {...buildAgentRows([], [], NOW * 1000)}
      processes={buildProcessBlock(processes, limit)}
      t={DEFAULT_THEME}
    />
  )

  expect(alone).toContain('Processes · 1 running · 1 done · Ctrl+T expand')
  expect(alone).toContain('✔ pytest tests/ · exit 0 · 5s ago')

  const collapsed = paint(
    <AgentsPanelView collapsed cols={80} {...agentRows} processes={block} t={DEFAULT_THEME} />
  )

  expect(collapsed.trim().split('\n')).toHaveLength(1)
  expect(collapsed).toContain('6 live agents · 1 procs')
})
