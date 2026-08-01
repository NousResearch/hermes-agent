import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@hermes/ink', async importOriginal => {
  const mod = await importOriginal()

  return { ...mod, useInput: () => {} }
})

import { clearDiffPair, clearSpawnHistory } from '../app/spawnHistoryStore.js'
import { patchTurnState, resetTurnState } from '../app/turnStore.js'
import { AgentsOverlay } from '../components/agentsOverlay.js'
import type { GatewayClient } from '../gatewayClient.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'
import type { SubagentOutcome, SubagentProgress, SubagentStatus } from '../types.js'

const subagent = (id: string, status: SubagentStatus, outcome?: SubagentOutcome): SubagentProgress => ({
  depth: 0,
  goal: `${id} worker`,
  id,
  index: 0,
  notes: [],
  outcome,
  parentId: null,
  status,
  taskCount: 1,
  thinking: [],
  toolCount: 0,
  tools: []
})

function renderAgents(items: SubagentProgress[]): string {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 120, isTTY: false, rows: 40 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  patchTurnState({ subagents: items })

  const gw = {
    request: vi.fn(async () => ({ active: [], paused: false }))
  } as unknown as GatewayClient

  const instance = renderSync(
    React.createElement(AgentsOverlay, {
      gw,
      onClose: () => undefined,
      t: DEFAULT_THEME
    }),
    {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    }
  )

  instance.unmount()
  instance.cleanup()

  return stripAnsi(output)
}

afterEach(() => {
  resetTurnState()
  clearSpawnHistory()
  clearDiffPair()
})

describe('AgentsOverlay logical outcome labels', () => {
  it('renders logical terminal outcomes instead of lifecycle completion labels', () => {
    const output = renderAgents([
      subagent('partial', 'interrupted', 'partial'),
      subagent('unverified', 'completed', 'unverified'),
      subagent('unknown', 'completed', 'unknown'),
      subagent('failed', 'interrupted', 'failed')
    ])

    expect(output).toContain('partial worker · partial')
    expect(output).toContain('unverified worker · verification required')
    expect(output).toContain('unknown worker · verification required')
    expect(output).toContain('failed worker · failed')
    expect(output).not.toContain('unverified worker · completed')
    expect(output).not.toContain('failed worker · interrupted')
  })
})
