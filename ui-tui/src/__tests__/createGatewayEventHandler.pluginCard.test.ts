import { beforeEach, describe, expect, it, vi } from 'vitest'

import { resetConnectionOperationsForTests } from '../app/connectionOperationStore.js'
import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { resetServerRequestsForTests } from '../app/serverRequestStore.js'
import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import type { Msg } from '../types.js'

// Split out of createGatewayEventHandler.test.ts (file-size budget); the mock,
// ctx builder and beforeEach are duplicated from there unchanged.
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

describe('createGatewayEventHandler', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
    resetTurnState()
    resetServerRequestsForTests()
    resetConnectionOperationsForTests()
    turnController.fullReset()
    patchUiState({ showReasoning: true })
  })

  it('prints a plugin card as a system line', () => {
    const ctx = buildCtx([])
    const onEvent = createGatewayEventHandler(ctx)

    onEvent({
      payload: { card_id: 'share-1', plugin_id: 'cwplug', text: 'Share "deploy-checklist"?' },
      type: 'plugin.card'
    } as any)

    expect(ctx.system.sys).toHaveBeenCalledWith('Share "deploy-checklist"?')
  })

  it('prints nothing for a plugin card without text', () => {
    const ctx = buildCtx([])

    createGatewayEventHandler(ctx)({
      payload: { card_id: 'c', plugin_id: 'p', text: '  ' },
      type: 'plugin.card'
    } as any)

    expect(ctx.system.sys).not.toHaveBeenCalled()
  })
})
