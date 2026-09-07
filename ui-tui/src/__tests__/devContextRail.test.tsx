import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { patchTurnState, resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { DevContextBottomDock, DevContextRail } from '../components/devContextRail.js'
import {
  DEV_CONTEXT_MIN_TERMINAL_COLS,
  DEV_CONTEXT_MIN_TRANSCRIPT_COLS,
  DEV_CONTEXT_RAIL_WIDTH,
  devContextHasActivity,
  devContextPlacement,
  devContextRailVisible,
  devContextRailWidth
} from '../domain/devContext.js'
import type { RailInputs } from '../domain/railInputs.js'
import type { RailFlowStatus } from '../hooks/useContextRailInputs.js'

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

const renderRail = async (
  columns = 140,
  rows = 40,
  queuedCount = 2,
  railInputs?: RailInputs | null,
  flowStatus?: RailFlowStatus | null
): Promise<string> => {
  const streams = makeStreams(columns, rows)

  const instance = renderSync(
    React.createElement(DevContextRail, { cols: columns, flowStatus, queuedCount, railInputs }),
    {
      patchConsole: false,
      stderr: streams.stderr as unknown as NodeJS.WriteStream,
      stdin: streams.stdin as unknown as NodeJS.ReadStream,
      stdout: streams.stdout as unknown as NodeJS.WriteStream
    }
  )

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

const renderDock = async (columns = 80, rows = 40): Promise<string> => {
  const streams = makeStreams(columns, rows)

  const instance = renderSync(React.createElement(DevContextBottomDock, { cols: columns, queuedCount: 0 }), {
    patchConsole: false,
    stderr: streams.stderr as unknown as NodeJS.WriteStream,
    stdin: streams.stdin as unknown as NodeJS.ReadStream,
    stdout: streams.stdout as unknown as NodeJS.WriteStream
  })

  await new Promise(resolve => setTimeout(resolve, 20))

  try {
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
  })

  afterEach(() => {
    resetUiState()
    resetTurnState()
    resetOverlayState()
  })

  it('keeps the built-in rail enabled by default', () => {
    expect(getUiState().devContext).toBe(true)
  })

  it('uses the side rail only when there is content and the transcript has room', () => {
    expect(devContextHasActivity(0, 0, 0, 0, 0)).toBe(false)
    expect(devContextHasActivity(1, 0, 0, 0, 0)).toBe(true)
    expect(devContextHasActivity(0, 0, 0, 0, 0, true)).toBe(true)
    expect(devContextPlacement(true, DEV_CONTEXT_MIN_TERMINAL_COLS, 0, true)).toBe('side')
    expect(devContextPlacement(true, DEV_CONTEXT_MIN_TERMINAL_COLS - 1, 0, true)).toBe('bottom')
    expect(devContextPlacement(true, 140, 44, true)).toBe('side')
    expect(devContextPlacement(true, 130, 44, true)).toBe('side')
    expect(devContextPlacement(true, 140, 0, false)).toBe('hidden')
    expect(devContextRailVisible(true, DEV_CONTEXT_MIN_TERMINAL_COLS, 0, true)).toBe(true)
    expect(devContextRailWidth(true, DEV_CONTEXT_MIN_TERMINAL_COLS, 0, true)).toBe(
      DEV_CONTEXT_MIN_TERMINAL_COLS - DEV_CONTEXT_MIN_TRANSCRIPT_COLS
    )
    expect(devContextRailWidth(true, 160, 0, true)).toBe(DEV_CONTEXT_RAIL_WIDTH)
    expect(devContextRailWidth(false, 160, 0, true)).toBe(0)
    expect(devContextRailWidth(true, 160, 0, false)).toBe(0)
  })

  it('renders only plan and work, with every todo and redacted display text', async () => {
    patchUiState({ devContext: true })
    patchTurnState({
      subagents: [
        {
          depth: 0,
          goal: 'Inspect token=hidden at https://example.invalid/private',
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
        { content: 'Run focused tests', id: 'todo-2', status: 'pending' },
        { content: 'Verify the narrow dock', id: 'todo-3', status: 'completed' },
        { content: 'Archive the plan', id: 'todo-4', status: 'cancelled' }
      ],
      tools: [{ id: 'tool-1', name: 'read_file' }]
    })

    const frame = await renderRail()

    expect(frame).toContain('DEV CONTEXT')
    expect(frame).toContain('PLAN')
    expect(frame).toContain('WORK')
    expect(frame).toContain('2/4 done')
    expect(frame).toContain('Inspect the changed files')
    expect(frame).toContain('Run focused tests')
    expect(frame).toContain('Verify the narrow dock')
    expect(frame).toContain('Archive the plan')
    expect(frame).toContain('▶ Inspect the changed files')
    expect(frame).toContain('· Run focused tests')
    expect(frame).toContain('✓ Verify the narrow dock')
    expect(frame).toContain('× Archive the plan')
    expect(frame).toContain('agents 1')
    expect(frame).toContain('tools 1')
    expect(frame).toContain('queue 2')
    expect(frame).not.toContain('REPO')
    expect(frame).not.toContain('RUNTIME')
    expect(frame).not.toContain('GUARDRAILS')
    expect(frame).not.toContain('ACTIVITY')
    expect(frame).not.toContain('token=hidden')
    expect(frame).not.toContain('https://example.invalid')
  })

  it('renders configured repo context with source labels, timestamps, and redaction', async () => {
    patchUiState({ devContext: true })
    patchTurnState({ todos: [{ content: 'Review the rail', id: 'todo-rail', status: 'in_progress' }] })

    const railInputs: RailInputs = {
      checks: 'npm test token=check-secret https://checks.invalid/run',
      decisions: ['2026-09-05: keep Flow as the work authority'],
      evidence: 'docs/ops-log.md',
      flow: '.flow',
      mtimeMs: Date.now(),
      product: 'Reader token=product-secret https://product.invalid',
      sourceFile: '/repo/AGENTS.md'
    }

    const frame = await renderRail(140, 40, 0, railInputs, { at: Date.now(), text: '2 open · next ready task' })

    expect(frame).toContain('FOCUS')
    expect(frame).toContain('PLAN')
    expect(frame).toContain('DECISIONS')
    expect(frame).toContain('EVIDENCE')
    expect(frame).toContain('WORK')
    expect(frame).toContain('product: Reader')
    expect(frame).toContain('tree: .flow')
    expect(frame).toContain('progress: 2 open')
    expect(frame).toContain('0/1 done')
    expect(frame).toContain('decision: 2026-09-05')
    expect(frame).toContain('durable: docs/ops-log.md')
    expect(frame).toContain('as of')
    expect(frame).not.toContain('token=product-secret')
    expect(frame).not.toContain('token=check-secret')
    expect(frame).not.toContain('https://product.invalid')
    expect(frame).not.toContain('https://checks.invalid')
  })

  it('renders at most one explicit attention item', async () => {
    patchUiState({ devContext: true })
    patchOverlayState({
      approval: { command: 'git status', description: 'inspect workspace' },
      clarify: { choices: null, question: 'Which path?', requestId: 'clarify-1' }
    })

    const frame = await renderRail(140, 40, 0)

    expect(frame.match(/NEEDS ME/g)).toHaveLength(1)
    expect(frame).toContain('approval: git status')
    expect(frame).not.toContain('Which path?')
  })

  it('renders every todo in the bottom dock when the terminal is narrow', async () => {
    patchUiState({ devContext: true })
    patchTurnState({
      todos: [
        { content: 'First narrow todo', id: 'todo-1', status: 'pending' },
        { content: 'Second narrow todo', id: 'todo-2', status: 'in_progress' },
        { content: 'Third narrow todo', id: 'todo-3', status: 'completed' },
        { content: 'Fourth narrow todo', id: 'todo-4', status: 'cancelled' }
      ]
    })

    expect(await renderRail(DEV_CONTEXT_MIN_TERMINAL_COLS, 40, 0)).toContain('DEV CONTEXT')
    expect(await renderRail(DEV_CONTEXT_MIN_TERMINAL_COLS - 1, 40, 0)).toBe('')

    const frame = await renderDock(DEV_CONTEXT_MIN_TERMINAL_COLS - 1)

    expect(frame).toContain('DEV CONTEXT')
    expect(frame).toContain('PLAN')
    expect(frame).toContain('WORK')
    expect(frame).toContain('First narrow todo')
    expect(frame).toContain('Second narrow todo')
    expect(frame).toContain('Third narrow todo')
    expect(frame).toContain('Fourth narrow todo')
    expect(frame).not.toContain('REPO')
  })

  it('shows work without an empty plan section', async () => {
    patchUiState({ devContext: true })
    patchTurnState({
      subagents: [
        {
          depth: 0,
          goal: 'Run the active worker',
          id: 'agent-1',
          index: 0,
          notes: [],
          parentId: null,
          status: 'running',
          taskCount: 1,
          thinking: [],
          toolCount: 0,
          tools: []
        }
      ]
    })

    const frame = await renderRail(140, 40, 0)

    expect(frame).toContain('WORK')
    expect(frame).toContain('Run the active worker')
    expect(frame).not.toContain('PLAN')
  })

  it('renders nothing when there is no plan or active work', async () => {
    patchUiState({ devContext: true })

    expect(await renderRail(140, 40, 0)).toBe('')
    expect(await renderDock(80)).toBe('')
  })
})
