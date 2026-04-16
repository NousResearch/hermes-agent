import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { resetUiState } from '../app/uiStore.js'
import { estimateTokensRough } from '../lib/text.js'
import type { Msg } from '../types.js'

const ref = <T>(current: T) => ({ current })

const buildCtx = (appended: Msg[]) =>
  ({
    composer: {
      dequeue: () => undefined,
      queueEditRef: ref<null | number>(null),
      sendQueued: vi.fn()
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
      setCatalog: vi.fn()
    },
    system: {
      bellOnComplete: false,
      sys: vi.fn()
    },
    transcript: {
      appendMessage: (msg: Msg) => appended.push(msg),
      setHistoryItems: vi.fn()
    }
  }) as any

describe('createGatewayEventHandler', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
    resetTurnState()
    turnController.fullReset()
  })

  it('persists completed tool rows when message.complete lands immediately after tool.complete', () => {
    const appended: Msg[] = []

    turnController.reasoningText = 'mapped the page'
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({
      payload: { context: 'home page', name: 'search', tool_id: 'tool-1' },
      type: 'tool.start'
    } as any)
    onEvent({
      payload: { name: 'search', preview: 'hero cards' },
      type: 'tool.progress'
    } as any)
    onEvent({
      payload: { summary: 'done', tool_id: 'tool-1' },
      type: 'tool.complete'
    } as any)
    onEvent({
      payload: { text: 'final answer' },
      type: 'message.complete'
    } as any)

    expect(appended).toHaveLength(1)
    expect(appended[0]).toMatchObject({
      role: 'assistant',
      text: 'final answer',
      thinking: 'mapped the page'
    })
    expect(appended[0]?.tools).toHaveLength(1)
    expect(appended[0]?.tools?.[0]).toContain('hero cards')
    expect(appended[0]?.toolTokens).toBeGreaterThan(0)
  })

  it('keeps tool tokens across handler recreation mid-turn', () => {
    const appended: Msg[] = []

    turnController.reasoningText = 'mapped the page'

    createGatewayEventHandler(buildCtx(appended))({
      payload: { context: 'home page', name: 'search', tool_id: 'tool-1' },
      type: 'tool.start'
    } as any)

    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({
      payload: { name: 'search', preview: 'hero cards' },
      type: 'tool.progress'
    } as any)
    onEvent({
      payload: { summary: 'done', tool_id: 'tool-1' },
      type: 'tool.complete'
    } as any)
    onEvent({
      payload: { text: 'final answer' },
      type: 'message.complete'
    } as any)

    expect(appended).toHaveLength(1)
    expect(appended[0]?.tools).toHaveLength(1)
    expect(appended[0]?.toolTokens).toBeGreaterThan(0)
  })

  it('ignores fallback reasoning.available when streamed reasoning already exists', () => {
    const appended: Msg[] = []
    const streamed = 'short streamed reasoning'
    const fallback = 'x'.repeat(400)

    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { text: streamed }, type: 'reasoning.delta' } as any)
    onEvent({ payload: { text: fallback }, type: 'reasoning.available' } as any)
    onEvent({ payload: { text: 'final answer' }, type: 'message.complete' } as any)

    expect(appended).toHaveLength(1)
    expect(appended[0]?.thinking).toBe(streamed)
    expect(appended[0]?.thinkingTokens).toBe(estimateTokensRough(streamed))
  })

  it('uses message.complete reasoning when no streamed reasoning ref', () => {
    const appended: Msg[] = []
    const fromServer = 'recovered from last_reasoning'

    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { reasoning: fromServer, text: 'final answer' }, type: 'message.complete' } as any)

    expect(appended).toHaveLength(1)
    expect(appended[0]?.thinking).toBe(fromServer)
    expect(appended[0]?.thinkingTokens).toBe(estimateTokensRough(fromServer))
  })
})
