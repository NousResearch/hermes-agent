// Regression tests for issue #59423 — TUI: assistant message renders twice.
//
// The double-render surfaced when a `message.complete` arrived for a turn
// whose transcript snapshot had already been written (most commonly via
// interruptTurn → user submits a new prompt → `startMessage` resets
// `interrupted = false` → stale `message.complete` for the OLD turn sees
// `interrupted === false` and re-appends the assistant's final text).
//
// Fix lives in turnController.messagesPersisted + the message.complete
// branch in createGatewayEventHandler that snapshots the latch BEFORE
// recordMessageComplete runs.

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
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

const assistantTexts = (msgs: Msg[]) =>
  msgs.filter(m => m.role === 'assistant' && m.text).map(m => m.text as string)

describe('assistant message does not render twice (#59423)', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
    resetTurnState()
    turnController.fullReset()
    patchUiState({ showReasoning: true, sid: 'sess-1' })
  })

  it('renders a single assistant message exactly once', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { text: '29页更新完成。' }, type: 'message.delta' } as any)
    onEvent({ payload: { text: '29页更新完成。' }, type: 'message.complete' } as any)

    expect(assistantTexts(appended)).toEqual(['29页更新完成。'])
  })

  it('still renders only one assistant message when streaming has many chunks', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: {}, type: 'message.start' } as any)
    // Many small deltas — the kind of multi-token stream that used to race.
    const chunks = ['29页更新完成。', '到店三种模式、', '到家双轨制、', '供应链三层专供、', '青峰……']
    for (const chunk of chunks) {
      onEvent({ payload: { text: chunk }, type: 'message.delta' } as any)
    }
    onEvent({ payload: { text: chunks.join('') }, type: 'message.complete' } as any)

    const texts = assistantTexts(appended)
    // Exactly one assistant message, with the joined text.
    expect(texts).toHaveLength(1)
    expect(texts[0]).toBe(chunks.join(''))
  })

  it('renders each assistant message exactly once across a multi-message conversation', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    // Turn 1.
    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { text: 'first reply' }, type: 'message.delta' } as any)
    onEvent({ payload: { text: 'first reply' }, type: 'message.complete' } as any)

    // Turn 2.
    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { text: 'second reply' }, type: 'message.delta' } as any)
    onEvent({ payload: { text: 'second reply' }, type: 'message.complete' } as any)

    // Turn 3.
    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { text: 'third reply' }, type: 'message.delta' } as any)
    onEvent({ payload: { text: 'third reply' }, type: 'message.complete' } as any)

    expect(assistantTexts(appended)).toEqual(['first reply', 'second reply', 'third reply'])
  })

  it('does not re-append when a stale message.complete arrives after interrupt+new turn', () => {
    const appended: Msg[] = []
    const ctx = buildCtx(appended)
    ctx.gateway.gw.request = vi.fn(async () => ({ status: 'interrupted' }))
    const onEvent = createGatewayEventHandler(ctx)

    // Turn 1 — interrupted mid-flight, partial answer captured.
    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { text: 'partial draft…' }, type: 'message.delta' } as any)
    turnController.interruptTurn(
      { appendMessage: (msg: Msg) => appended.push(msg), gw: ctx.gateway.gw, sid: 'sess-1', sys: ctx.system.sys },
      { keepBusy: true }
    )

    const snapshotAfterInterrupt = appended.length

    // Realistic ordering: the gateway's `message.complete` for the
    // cancelled turn lands FIRST (turnController is still on generation 1
    // because the user hasn't started turn 2 yet) — and must be dropped,
    // not re-appended.
    onEvent({
      payload: { text: 'partial draft…' },
      type: 'message.complete'
    } as any)

    expect(appended.length).toBe(snapshotAfterInterrupt)

    // User submits a new prompt — the new turn begins.
    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { text: 'new turn reply' }, type: 'message.delta' } as any)
    onEvent({ payload: { text: 'new turn reply' }, type: 'message.complete' } as any)

    // Exactly one assistant message for turn 2 — the stale turn 1 reply
    // did not slip back in.
    const newTurnTexts = appended
      .slice(snapshotAfterInterrupt)
      .filter(m => m.role === 'assistant' && m.text)
      .map(m => m.text)

    expect(newTurnTexts).toEqual(['new turn reply'])
  })

  it('suppresses a duplicate message.complete arriving back-to-back', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { text: 'only once' }, type: 'message.delta' } as any)
    onEvent({ payload: { text: 'only once' }, type: 'message.complete' } as any)
    // Gateway somehow emits the same complete twice.
    onEvent({ payload: { text: 'only once' }, type: 'message.complete' } as any)

    expect(assistantTexts(appended)).toEqual(['only once'])
  })
})