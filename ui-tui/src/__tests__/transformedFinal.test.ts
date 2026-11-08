import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import type { Msg } from '../types.js'

// A transform_llm_output plugin rewrites the final after the raw reply streamed: the completion
// carries the rewrite (response_transformed) and the transcript must show only that.
const RAW = 'Your api key is SECRET-123 and the backup is SECRET-456.'
const REWRITE = 'Your api key is [REDACTED] and the backup is [REDACTED].'

const ref = <T>(current: T) => ({ current })

const buildCtx = (appended: Msg[]) =>
  ({
    composer: { dequeue: () => undefined, queueEditRef: ref<null | number>(null), sendQueued: vi.fn(), setInput: vi.fn() },
    gateway: { gw: { request: vi.fn(async () => null) }, rpc: vi.fn(async () => null) },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: ref(80),
      newSession: vi.fn(),
      resetSession: vi.fn(),
      resumeById: vi.fn(),
      setCatalog: vi.fn()
    },
    submission: { submitRef: { current: vi.fn() } },
    system: { bellOnComplete: false, sys: vi.fn() },
    transcript: { appendMessage: (msg: Msg) => appended.push(msg), panel: vi.fn(), setHistoryItems: vi.fn() },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
  }) as any

const streamThenComplete = (complete: Record<string, unknown>) => {
  const appended: Msg[] = []
  const onEvent = createGatewayEventHandler(buildCtx(appended))

  onEvent({ payload: {}, type: 'message.start' } as any)

  for (const word of RAW.split(' ')) {
    onEvent({ payload: { text: `${word} ` }, type: 'message.delta' } as any)
  }

  onEvent({ payload: complete, type: 'message.complete' } as any)

  return appended.filter(m => m.role === 'assistant' && m.text).map(m => m.text)
}

describe('transformed final settlement', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
    resetTurnState()
    turnController.fullReset()
    patchUiState({ showReasoning: true })
  })

  it('replaces the streamed raw reply with the rewrite', () => {
    expect(streamThenComplete({ response_transformed: true, status: 'complete', text: REWRITE })).toEqual([REWRITE])
  })

  it('keeps a streamed tail the final does not carry when nothing was rewritten', () => {
    expect(streamThenComplete({ status: 'complete', text: 'A different closing line.' })).toEqual([
      `${RAW} `,
      'A different closing line.'
    ])
  })
})
