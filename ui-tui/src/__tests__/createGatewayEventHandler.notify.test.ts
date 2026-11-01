import { beforeEach, describe, expect, it, vi } from 'vitest'

import { resetConnectionOperationsForTests } from '../app/connectionOperationStore.js'
import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { resetServerRequestsForTests } from '../app/serverRequestStore.js'
import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import type { Msg } from '../types.js'

// display.bell_on_* desktop notifications. Kept out of createGatewayEventHandler.test.ts
// (file-size budget); the mock and ctx builder are duplicated from there unchanged.
const openExternalUrlMock = vi.fn((_url: string) => true)
vi.mock('../lib/openExternalUrl.js', () => ({
  openExternalUrl: (url: string) => openExternalUrlMock(url)
}))

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
      gw: { request: vi.fn(async () => null) },
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

describe('createGatewayEventHandler desktop notifications', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
    resetTurnState()
    resetServerRequestsForTests()
    resetConnectionOperationsForTests()
    turnController.fullReset()
  })

  it('sends a desktop notification with the completion bell', () => {
    const ctx = buildCtx([])
    const notifyAttention = vi.fn()

    ctx.system.bellOnComplete = true
    ctx.system.notifyAttention = notifyAttention
    ctx.system.stdout = { isTTY: true, write: vi.fn() }

    const onEvent = createGatewayEventHandler(ctx)
    onEvent({ payload: { text: 'Finished the task.' }, type: 'message.complete' } as any)

    expect(notifyAttention).toHaveBeenCalledExactlyOnceWith('complete')
  })

  it('sends a desktop notification with the prompt bell when an approval opens', () => {
    const ctx = buildCtx([])
    const notifyAttention = vi.fn()

    ctx.system.bellOnPrompt = true
    ctx.system.notifyAttention = notifyAttention
    ctx.system.stdout = { isTTY: true, write: vi.fn() }

    const onEvent = createGatewayEventHandler(ctx)
    patchUiState({ sid: 'sess-1' })
    onEvent({
      type: 'approval.request',
      session_id: 'sess-1',
      payload: { prompt_id: 'p-1', command: 'rm -rf build', execution_generation: 1 }
    } as any)

    expect(notifyAttention).toHaveBeenCalledExactlyOnceWith('prompt')
    expect(ctx.system.stdout!.write).not.toHaveBeenCalled()
  })
})
