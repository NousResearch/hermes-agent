import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'
import { resetUiState } from '../app/uiStore.js'

describe('createGatewayEventHandler swarm boundary', () => {
  beforeEach(() => {
    resetOverlayState()
    resetTurnState()
    resetUiState()
  })

  it('records delegated subagent progress inline in turn state without opening a dedicated overlay', () => {
    const handler = createGatewayEventHandler(buildCtx())

    handler({ type: 'message.start' })
    handler({
      type: 'subagent.start',
      payload: { goal: 'Audit slash routes', task_count: 2, task_index: 0 }
    })
    handler({
      type: 'subagent.thinking',
      payload: { goal: 'Audit slash routes', task_count: 2, task_index: 0, text: 'Checking authoritative route owners' }
    })
    handler({
      type: 'subagent.tool',
      payload: { goal: 'Audit slash routes', task_count: 2, task_index: 0, tool_name: 'read_file', tool_preview: 'createSlashHandler.ts' }
    })
    handler({
      type: 'subagent.progress',
      payload: { goal: 'Audit slash routes', task_count: 2, task_index: 0, text: 'Confirmed no swarm live route' }
    })
    handler({
      type: 'subagent.complete',
      payload: {
        duration_seconds: 1.5,
        goal: 'Audit slash routes',
        status: 'completed',
        summary: 'No dedicated swarm command found',
        task_count: 2,
        task_index: 0
      }
    })

    expect(getTurnState().subagents).toEqual([
      {
        durationSeconds: 1.5,
        goal: 'Audit slash routes',
        id: 'sa:0:Audit slash routes',
        index: 0,
        notes: ['Confirmed no swarm live route'],
        status: 'completed',
        summary: 'No dedicated swarm command found',
        taskCount: 2,
        thinking: ['Checking authoritative route owners'],
        tools: ['Read File("createSlashHandler.ts")']
      }
    ])
    expect(getOverlayState()).toEqual({
      approval: null,
      clarify: null,
      modelPicker: false,
      pager: null,
      picker: false,
      secret: null,
      setupWizard: false,
      sudo: null,
      swarm: false
    })
  })

  it('keeps completed subagents completed when later progress arrives', () => {
    const handler = createGatewayEventHandler(buildCtx())

    handler({ type: 'message.start' })
    handler({ type: 'subagent.start', payload: { goal: 'Review T7', task_index: 0 } })
    handler({
      type: 'subagent.complete',
      payload: { goal: 'Review T7', status: 'completed', summary: 'done', task_index: 0 }
    })
    handler({
      type: 'subagent.progress',
      payload: { goal: 'Review T7', task_index: 0, text: 'late note still visible' }
    })

    expect(getTurnState().subagents[0]).toMatchObject({
      goal: 'Review T7',
      notes: ['late note still visible'],
      status: 'completed',
      summary: 'done'
    })
  })
})

const buildCtx = () => ({
  composer: {
    dequeue: vi.fn(() => undefined),
    queueEditRef: { current: null },
    sendQueued: vi.fn()
  },
  gateway: {
    rpc: vi.fn(() => Promise.resolve(null))
  },
  session: {
    STARTUP_RESUME_ID: '',
    colsRef: { current: 80 },
    newSession: vi.fn(),
    resetSession: vi.fn(),
    resumeById: vi.fn(),
    setCatalog: vi.fn()
  },
  system: {
    bellOnComplete: false,
    stdout: undefined,
    sys: vi.fn()
  },
  transcript: {
    appendMessage: vi.fn(),
    panel: vi.fn(),
    setHistoryItems: vi.fn()
  }
}) satisfies Parameters<typeof createGatewayEventHandler>[0]
