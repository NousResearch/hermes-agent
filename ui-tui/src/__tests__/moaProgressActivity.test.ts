import { beforeEach, describe, expect, it } from 'vitest'

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
      sendQueued: () => undefined,
      setInput: () => undefined
    },
    gateway: {
      gw: { request: () => undefined },
      rpc: async () => null
    },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: ref(80),
      newSession: () => undefined,
      resetSession: () => undefined,
      resumeById: () => undefined,
      setCatalog: () => undefined
    },
    submission: {
      submitRef: { current: () => undefined }
    },
    system: {
      bellOnComplete: false,
      sys: () => undefined
    },
    transcript: {
      appendMessage: (msg: Msg) => appended.push(msg),
      panel: () => undefined,
      setHistoryItems: () => undefined
    },
    voice: {
      setProcessing: () => undefined,
      setRecording: () => undefined,
      setVoiceEnabled: () => undefined
    }
  }) as any

const activityTexts = () => getTurnState().activity.map(item => item.text)

describe('moa.progress / moa.phase activity surface', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
    resetTurnState()
    turnController.fullReset()
    patchUiState({ showReasoning: true })
  })

  it('shows "MoA: refs k/n" as each reference completes, replacing in place', () => {
    const onEvent = createGatewayEventHandler(buildCtx([]))

    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { label: 'model-a', refs_done: 1, refs_total: 3 }, type: 'moa.progress' } as any)

    expect(activityTexts()).toContain('MoA: refs 1/3')

    onEvent({ payload: { label: 'model-b', refs_done: 2, refs_total: 3 }, type: 'moa.progress' } as any)

    const texts = activityTexts()
    expect(texts).toContain('MoA: refs 2/3')
    // Replaced in place — the stale 1/3 line must not linger alongside 2/3.
    expect(texts).not.toContain('MoA: refs 1/3')
  })

  it('swaps the progress line for aggregator copy on moa.phase', () => {
    const onEvent = createGatewayEventHandler(buildCtx([]))

    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { label: 'model-a', refs_done: 3, refs_total: 3 }, type: 'moa.progress' } as any)
    onEvent({ payload: { phase: 'aggregator', refs_done: 3, refs_total: 3 }, type: 'moa.phase' } as any)

    const texts = activityTexts()
    expect(texts).toContain('MoA: aggregating…')
    expect(texts).not.toContain('MoA: refs 3/3')
  })

  it('ignores malformed payloads (missing counters / unknown phase)', () => {
    const onEvent = createGatewayEventHandler(buildCtx([]))

    onEvent({ payload: {}, type: 'message.start' } as any)
    const before = activityTexts().length
    onEvent({ payload: { label: 'model-a' }, type: 'moa.progress' } as any)
    onEvent({ payload: { phase: 'reference' }, type: 'moa.phase' } as any)

    expect(activityTexts().length).toBe(before)
  })
})
