import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { patchDelegationState, resetDelegationState } from '../app/delegationStore.js'
import type { AppLayoutStatusProps } from '../app/interfaces.js'
import { patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { patchTurnState, resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { DevContextRail } from '../components/devContextRail.js'
import {
  DEV_CONTEXT_MIN_TERMINAL_COLS,
  DEV_CONTEXT_RAIL_WIDTH,
  devContextRailVisible,
  devContextRailWidth
} from '../domain/devContext.js'
import { DEFAULT_THEME } from '../theme.js'

const makeStreams = (columns = 140, rows = 40) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  Object.assign(stdout, { columns, isTTY: false, rows })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })

  let captured = ''
  stdout.on('data', chunk => {
    captured += chunk.toString()
  })

  return { capture: () => captured, stderr, stdin, stdout }
}

const status: AppLayoutStatusProps = {
  cwdLabel: '~/work/hermes-agent',
  goodVibesTick: 0,
  lastTurnEndedAt: null,
  sessionStartedAt: null,
  sessionTitle: 'HUD work',
  showStickyPrompt: false,
  statusColor: DEFAULT_THEME.color.ok,
  stickyPrompt: '',
  turnStartedAt: null,
  voiceLabel: '',
  workspace: {
    branch: 'feature/dev-context',
    dirty: true,
    gitRoot: '/Users/example/work/hermes-agent',
    github: { fullName: 'NousResearch/hermes-agent', owner: 'NousResearch', repo: 'hermes-agent' },
    projectName: 'Hermes Agent',
    pullRequest: { number: 42, state: 'open', title: 'Add developer context rail' },
    upstream: { ahead: 2, behind: 1 }
  }
}

const renderRail = async (columns = 140, rows = 40): Promise<string> => {
  const streams = makeStreams(columns, rows)

  const instance = renderSync(React.createElement(DevContextRail, { cols: columns, queuedCount: 2, status }), {
    patchConsole: false,
    stderr: streams.stderr as unknown as NodeJS.WriteStream,
    stdin: streams.stdin as unknown as NodeJS.ReadStream,
    stdout: streams.stdout as unknown as NodeJS.WriteStream
  })

  await new Promise(resolve => setTimeout(resolve, 20))

  try {
    // Strip SGR styling; the assertions target the user-visible labels.
    // eslint-disable-next-line no-control-regex
    return streams.capture().replace(/\u001b\[[0-9;]*m/g, '')
  } finally {
    instance.unmount()
    instance.cleanup()
  }
}

describe('developer context rail layout', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
    resetOverlayState()
    resetDelegationState()
  })

  afterEach(() => {
    resetUiState()
    resetTurnState()
    resetOverlayState()
    resetDelegationState()
  })

  it('keeps the built-in rail enabled by default', () => {
    expect(getUiState().devContext).toBe(true)
  })

  it('reserves a fixed rail only when the transcript still has room', () => {
    expect(devContextRailVisible(true, DEV_CONTEXT_MIN_TERMINAL_COLS, 0)).toBe(true)
    expect(devContextRailWidth(true, DEV_CONTEXT_MIN_TERMINAL_COLS, 0)).toBe(DEV_CONTEXT_RAIL_WIDTH)
    expect(devContextRailVisible(true, DEV_CONTEXT_MIN_TERMINAL_COLS - 1, 0)).toBe(false)
    expect(devContextRailVisible(true, 140, 44)).toBe(true)
    expect(devContextRailVisible(true, 130, 44)).toBe(false)
    expect(devContextRailWidth(false, 160, 0)).toBe(0)
  })

  it('renders repository, runtime, plan, work, safety, and activity state without secrets', async () => {
    patchUiState({
      busy: true,
      destructiveSlashConfirm: true,
      devContext: true,
      focusView: true,
      info: {
        mcp_servers: [
          { connected: true, name: 'support', status: 'connected', tools: 3, transport: 'http' },
          { connected: false, name: 'broken', status: 'failed', tools: 0, transport: 'stdio' }
        ],
        model: 'test-model',
        provider: 'test-provider',
        reasoning_effort: 'high',
        service_tier: 'priority',
        skills: { core: ['one', 'two'] },
        tools: { file: ['read_file', 'write_file'] }
      },
      status: 'working',
      usage: {
        avg_latency_s: 1.2,
        avg_tps: 40,
        cache_hit_pct: 87,
        calls: 4,
        compressions: 2,
        context_max: 200_000,
        context_percent: 42,
        context_used: 84_000,
        cost_usd: 0.12,
        input: 12_000,
        output: 3_000,
        total: 15_000
      }
    })
    patchTurnState({
      activity: [{ id: 1, text: 'checked token=hidden at https://example.invalid/private', tone: 'info' }],
      subagents: [
        {
          depth: 0,
          goal: 'Inspect the changed files',
          id: 'agent-1',
          index: 0,
          notes: [],
          parentId: null,
          status: 'running',
          taskCount: 1,
          thinking: [],
          toolCount: 1,
          tools: ['read_file']
        }
      ],
      todos: [
        { content: 'Inspect the changed files', id: 'todo-1', status: 'in_progress' },
        { content: 'Run focused tests', id: 'todo-2', status: 'pending' }
      ],
      tools: [{ id: 'tool-1', name: 'read_file' }]
    })
    patchDelegationState({ paused: true })
    patchOverlayState({ approval: { choices: ['allow', 'deny'], command: 'git status', description: 'inspect workspace' } })

    const frame = await renderRail()

    expect(frame).toContain('DEV CONTEXT')
    expect(frame).toContain('Hermes Agent')
    expect(frame).toContain('feature/dev-context')
    expect(frame).toContain('GH NousResearch/hermes-agent')
    expect(frame).toContain('PR')
    expect(frame).toContain('test-model')
    expect(frame).toContain('42%')
    expect(frame).toContain('agents 1/1')
    expect(frame).toContain('0/2 done')
    expect(frame).toContain('approval required')
    expect(frame).not.toContain('token=hidden')
    expect(frame).not.toContain('https://example.invalid')
  })

  it('renders nothing when the terminal is too narrow', async () => {
    patchUiState({ devContext: true })

    expect(await renderRail(DEV_CONTEXT_MIN_TERMINAL_COLS - 1)).toBe('')
  })
})
