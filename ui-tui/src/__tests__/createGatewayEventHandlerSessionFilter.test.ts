import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { isForeignSessionEvent } from '../app/sessionEventFilter.js'
import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import type { GatewayEvent } from '../gatewayTypes.js'
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

describe('isForeignSessionEvent', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
    turnController.fullReset()
  })

  it('classifies the switch-window cases directly', () => {
    patchUiState({ sid: 'sess-active' })

    // foreign session
    expect(
      isForeignSessionEvent({ payload: {}, session_id: 'sess-other', type: 'message.delta' } as any, 'sess-active')
    ).toBe(true)
    // foreign session during the null-sid switch window
    expect(isForeignSessionEvent({ payload: {}, session_id: 'sess-other', type: 'message.delta' } as any, null)).toBe(
      true
    )
    // explicit empty session id
    expect(isForeignSessionEvent({ payload: {}, session_id: '', type: 'message.delta' } as any, 'sess-active')).toBe(
      true
    )
    // own session
    expect(
      isForeignSessionEvent({ payload: {}, session_id: 'sess-active', type: 'message.delta' } as any, 'sess-active')
    ).toBe(false)
    // keyless (unscoped by design)
    expect(isForeignSessionEvent({ payload: {}, type: 'message.delta' } as any, 'sess-active')).toBe(false)
    // global prefixes always pass, even with an explicit empty id
    expect(isForeignSessionEvent({ payload: {}, session_id: '', type: 'skin.changed' } as any, 'sess-active')).toBe(
      false
    )
    expect(isForeignSessionEvent({ payload: {}, session_id: '', type: 'gateway.ready' } as any, null)).toBe(false)
  })
})

describe('cross-session event filtering', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
    turnController.fullReset()
  })

  it.each([
    ['foreign session', 'sess-active', 'sess-other'],
    ['foreign session during the null-sid switch window', null, 'sess-other'],
    ['explicit empty session id', 'sess-active', '']
  ])('drops the %s transcript sequence', (_case, activeSid, eventSid) => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    patchUiState({ sid: activeSid })
    onEvent({ payload: { text: 'leaked delta' }, session_id: eventSid, type: 'message.delta' } satisfies GatewayEvent)
    onEvent({
      payload: { text: 'leaked answer' },
      session_id: eventSid,
      type: 'message.complete'
    } satisfies GatewayEvent)

    expect(appended).toEqual([])
  })

  it('accepts the transcript sequence matching the active session', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    patchUiState({ sid: 'sess-active' })
    // The wire contract: message.complete carries the streamed reply, so the
    // delta text is contained in the completion text (mirrors the sibling
    // delta/complete case in createGatewayEventHandler.test.ts) and the
    // transcript holds the final once.
    onEvent({
      payload: { text: 'current answer' },
      session_id: 'sess-active',
      type: 'message.delta'
    } satisfies GatewayEvent)
    onEvent({
      payload: { text: 'current answer' },
      session_id: 'sess-active',
      type: 'message.complete'
    } satisfies GatewayEvent)

    expect(appended).toEqual([{ role: 'assistant', text: 'current answer' }])
  })

  it('accepts a truly unscoped transcript sequence', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    patchUiState({ sid: 'sess-active' })
    onEvent({ payload: { text: 'unscoped answer' }, type: 'message.delta' } satisfies GatewayEvent)
    onEvent({ payload: { text: 'unscoped answer' }, type: 'message.complete' } satisfies GatewayEvent)

    expect(appended).toEqual([{ role: 'assistant', text: 'unscoped answer' }])
  })

  it('does not buffer a foreign session delta in the streaming segment', () => {
    // message.delta only accumulates into the streaming buffer (never appends a
    // message), so the appended[] assertion above cannot see a leaked delta that
    // lands in turnController.bufRef and would surface on the NEXT flush.
    const onEvent = createGatewayEventHandler(buildCtx([]))

    patchUiState({ sid: 'sess-active' })
    onEvent({
      payload: { text: 'leaked delta' },
      session_id: 'sess-other',
      type: 'message.delta'
    } satisfies GatewayEvent)
    expect(turnController.bufRef).toBe('')

    onEvent({ payload: { text: 'leaked delta 2' }, session_id: '', type: 'message.delta' } satisfies GatewayEvent)
    expect(turnController.bufRef).toBe('')

    // and inside the null-sid switch window, every session-scoped event drops
    patchUiState({ sid: null })
    onEvent({
      payload: { text: 'leaked during switch' },
      session_id: 'sess-other',
      type: 'message.delta'
    } satisfies GatewayEvent)
    expect(turnController.bufRef).toBe('')

    // An event with NO session_id key is unscoped by design (CLI-direct or
    // global), not a leak — it still streams. Only a present-but-mismatched
    // or empty one is filtered.
    patchUiState({ sid: 'sess-active' })
    onEvent({ payload: { text: ' unscoped ok' }, type: 'message.delta' } satisfies GatewayEvent)
    expect(turnController.bufRef).toBe(' unscoped ok')
  })

  it('drops every session-scoped delta inside the null-sid switch window, including the target session', () => {
    // The null-sid window is deliberately conservative: with no active session
    // there is nothing to route a session-scoped event to, so all of them drop
    // (own-session deltas included) rather than guess. The window is transient
    // and the streaming buffer is flushed on switch, so nothing is lost that
    // the resumed session will not re-send. Pinned here so a future "accept
    // own-session deltas during the window" change is a conscious decision.
    const onEvent = createGatewayEventHandler(buildCtx([]))

    patchUiState({ sid: 'sess-active' })
    onEvent({ payload: { text: 'mine' }, session_id: 'sess-active', type: 'message.delta' } satisfies GatewayEvent)
    expect(turnController.bufRef).toBe('mine')

    patchUiState({ sid: null })
    onEvent({ payload: { text: ' stale' }, session_id: 'sess-active', type: 'message.delta' } satisfies GatewayEvent)
    expect(turnController.bufRef).toBe('mine')
  })

  it('accepts an explicit empty session id for a global skin event', () => {
    const onEvent = createGatewayEventHandler(buildCtx([]))

    patchUiState({ sid: 'sess-active' })
    onEvent({
      payload: { branding: { agent_name: 'Event Contract Skin' } },
      session_id: '',
      type: 'skin.changed'
    } satisfies GatewayEvent)

    expect(getUiState().theme.brand.name).toBe('Event Contract Skin')
  })
})
