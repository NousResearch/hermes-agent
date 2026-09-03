import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import type { Msg } from '../types.js'

const ref = <T>(current: T) => ({ current })

const buildCtx = (appended: Msg[]) =>
  ({
    composer: {
      dequeue: () => undefined,
      queueEditRef: ref<null | number>(null),
      sendQueued: vi.fn(),
      setInput: vi.fn()
    },
    gateway: {
      gw: { request: vi.fn() },
      rpc: vi.fn(async () => null)
    },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: ref(80),
      newSession: vi.fn(),
      resetSession: vi.fn(),
      resumeById: vi.fn(),
      setCatalog: vi.fn()
    },
    submission: {
      submitRef: { current: vi.fn() }
    },
    system: {
      bellOnComplete: false,
      sys: vi.fn()
    },
    transcript: {
      appendMessage: (msg: Msg) => appended.push(msg),
      panel: (title: string, sections: any[]) =>
        appended.push({ kind: 'panel', panelData: { sections, title }, role: 'system', text: '' }),
      setHistoryItems: vi.fn()
    },
    voice: {
      setProcessing: vi.fn(),
      setRecording: vi.fn(),
      setVoiceEnabled: vi.fn()
    }
  }) as any

describe('createGatewayEventHandler delegation outcome', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
    resetTurnState()
    turnController.fullReset()
    patchUiState({ showReasoning: true })
  })

  it('preserves logical outcome and schema evidence on subagent.complete', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({
      payload: { goal: 'verify me', subagent_id: 'sa-unverified', task_index: 3 },
      type: 'subagent.start'
    } as any)
    onEvent({
      payload: {
        goal: 'verify me',
        error: 'Final answer does not satisfy the declared output_schema.',
        error_authoritative: true,
        outcome: 'failed',
        schema_errors: ["'city' is a required property"],
        schema_retries: 1,
        schema_valid: false,
        status: 'completed',
        subagent_id: 'sa-unverified',
        task_index: 3
      },
      type: 'subagent.complete'
    } as any)

    expect(getTurnState().subagents.find(s => s.id === 'sa-unverified')).toMatchObject({
      errorAuthoritative: true,
      outcome: 'failed',
      schemaErrors: ["'city' is a required property"],
      schemaRetries: 1,
      schemaValid: false,
      status: 'completed'
    })
  })
})
