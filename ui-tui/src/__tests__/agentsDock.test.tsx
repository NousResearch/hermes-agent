import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import chalk from 'chalk'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { afterEach, beforeEach, expect, it } from 'vitest'

import { renderToScreen } from '../../packages/hermes-ink/src/ink/render-to-screen.js'
import { cellAtIndex } from '../../packages/hermes-ink/src/ink/screen.js'
import { AgentsPanelView } from '../components/agentsPanel.js'
import { buildAgentRows, dockRowLimit } from '../lib/agentRows.js'
import { DARK_THEME, DEFAULT_THEME, LIGHT_THEME } from '../theme.js'
import type { SubagentProgress } from '../types.js'

const colorLevel = chalk.level

beforeEach(() => {
  chalk.level = 3
})
afterEach(() => {
  chalk.level = colorLevel
})

const agent = (id: string): SubagentProgress => ({
  id,
  goal: 'Investigate authentication handshake '.repeat(8),
  depth: 0,
  index: 0,
  parentId: null,
  notes: [],
  tools: ['read_file auth.ts'],
  thinking: [],
  toolCount: 1,
  taskCount: 1,
  startedAt: 1000,
  status: 'running'
})

it('bounds painted chrome by viewport while retaining true live count and activity', () => {
  const agents = Array.from({ length: 12 }, (_, i) => agent(`child-${i}`))

  for (const height of [14, 24, 40]) {
    const rows = buildAgentRows(agents, [], 45000, dockRowLimit(height))
    expect(rows.running).toBe(agents.length)
    expect(rows.hidden + rows.rows.length).toBe(agents.length)
    const stdout = Object.assign(new PassThrough(), { columns: 72, rows: height })
    const frames: string[] = []
    stdout.on('data', chunk => {
      frames.push(chunk.toString())
    })

    const view = renderSync(<AgentsPanelView cols={72} {...rows} t={DEFAULT_THEME} />, {
      stdout: stdout as unknown as NodeJS.WriteStream,
      stdin: new PassThrough() as NodeJS.ReadStream
    })

    view.unmount()
    view.cleanup()

    const lines = stripAnsi(frames.filter(frame => stripAnsi(frame).trim()).at(-1) ?? '')
      .trim()
      .split('\n')

    expect(lines.length).toBeLessThanOrEqual(dockRowLimit(height) * 2 + 1)
    expect(lines.join('\n')).toContain('12 live')
    expect(lines.join('\n')).toContain('read_file')
  }

  expect(dockRowLimit(14)).toBeLessThan(dockRowLimit(40))

  for (const t of [DARK_THEME, LIGHT_THEME]) {
    const rows = buildAgentRows(agents, [], 45000, dockRowLimit(20))
    const { screen, height } = renderToScreen(<AgentsPanelView cols={80} {...rows} t={t} />, 80)
    // Ink marks fills in the low style bit: even blank edge cells must paint.
    const filled = Array.from({ length: height * 80 }, (_, i) => cellAtIndex(screen, i).styleId & 1)

    expect(filled.every(Boolean)).toBe(true)
  }
})

it('deduplicates async batches and hides settled history without dropping live work', () => {
  const a = agent('child-1')

  const rows = buildAgentRows(
    [a, { ...agent('done'), status: 'completed' }],
    [{ delegation_id: 'batch', status: 'running', subagent_ids: [a.id] }],
    50000
  )

  expect(rows.running).toBe(1)
  expect(rows.rows.map(row => row.key)).toEqual(['live:child-1'])
})
