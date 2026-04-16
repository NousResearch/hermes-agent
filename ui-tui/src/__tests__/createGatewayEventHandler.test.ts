import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { estimateTokensRough } from '../lib/text.js'
import type { Msg } from '../types.js'

const ref = <T>(current: T) => ({ current })

describe('createGatewayEventHandler', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
  })

  it('persists completed tool rows when message.complete lands immediately after tool.complete', () => {
    const appended: Msg[] = []

    const state = {
      activity: [] as unknown[],
      reasoningTokens: 0,
      streaming: '',
      toolTokens: 0,
      tools: [] as unknown[],
      turnTrail: [] as string[]
    }

    const setTools = vi.fn((next: unknown) => {
      if (typeof next !== 'function') {
        state.tools = next as unknown[]
      }
    })

    const setTurnTrail = vi.fn((next: unknown) => {
      if (typeof next !== 'function') {
        state.turnTrail = next as string[]
      }
    })

    const refs = {
      activeToolsRef: ref([] as { context?: string; id: string; name: string; startedAt?: number }[]),
      bufRef: ref(''),
      interruptedRef: ref(false),
      lastStatusNoteRef: ref(''),
      persistedToolLabelsRef: ref(new Set<string>()),
      protocolWarnedRef: ref(false),
      reasoningRef: ref('mapped the page'),
      statusTimerRef: ref<ReturnType<typeof setTimeout> | null>(null),
      toolTokenAccRef: ref(0),
      toolCompleteRibbonRef: ref(null),
      turnToolsRef: ref([] as string[])
    }

    const onEvent = createGatewayEventHandler({
      composer: {
        dequeue: () => undefined,
        queueEditRef: ref<number | null>(null),
        sendQueued: vi.fn()
      },
      gateway: {
        gw: { request: vi.fn() } as any,
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
      },
      turn: {
        actions: {
          clearReasoning: vi.fn(() => {
            refs.reasoningRef.current = ''
            refs.toolTokenAccRef.current = 0
            state.toolTokens = 0
          }),
          endReasoningPhase: vi.fn(),
          idle: vi.fn(() => {
            refs.activeToolsRef.current = []
            state.tools = []
          }),
          pruneTransient: vi.fn(),
          pulseReasoningStreaming: vi.fn(),
          pushActivity: vi.fn(),
          pushTrail: vi.fn(),
          scheduleReasoning: vi.fn(),
          scheduleStreaming: vi.fn(),
          setActivity: vi.fn(),
          setReasoningTokens: vi.fn((next: number) => {
            state.reasoningTokens = next
          }),
          setStreaming: vi.fn((next: string) => {
            state.streaming = next
          }),
          setToolTokens: vi.fn((next: number) => {
            state.toolTokens = next
          }),
          setTools,
          setTurnTrail
        },
        refs
      }
    } as any)

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

    const state = {
      activity: [] as unknown[],
      reasoningTokens: 0,
      streaming: '',
      toolTokens: 0,
      tools: [] as unknown[],
      turnTrail: [] as string[]
    }

    const refs = {
      activeToolsRef: ref([] as { context?: string; id: string; name: string; startedAt?: number }[]),
      bufRef: ref(''),
      interruptedRef: ref(false),
      lastStatusNoteRef: ref(''),
      persistedToolLabelsRef: ref(new Set<string>()),
      protocolWarnedRef: ref(false),
      reasoningRef: ref('mapped the page'),
      statusTimerRef: ref<ReturnType<typeof setTimeout> | null>(null),
      toolTokenAccRef: ref(0),
      toolCompleteRibbonRef: ref(null),
      turnToolsRef: ref([] as string[])
    }

    const buildHandler = () =>
      createGatewayEventHandler({
        composer: {
          dequeue: () => undefined,
          queueEditRef: ref<number | null>(null),
          sendQueued: vi.fn()
        },
        gateway: {
          gw: { request: vi.fn() } as any,
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
        },
        turn: {
          actions: {
            clearReasoning: vi.fn(() => {
              refs.reasoningRef.current = ''
              refs.toolTokenAccRef.current = 0
              state.toolTokens = 0
            }),
            endReasoningPhase: vi.fn(),
            idle: vi.fn(() => {
              refs.activeToolsRef.current = []
              state.tools = []
            }),
            pruneTransient: vi.fn(),
            pulseReasoningStreaming: vi.fn(),
            pushActivity: vi.fn(),
            pushTrail: vi.fn(),
            scheduleReasoning: vi.fn(),
            scheduleStreaming: vi.fn(),
            setActivity: vi.fn(),
            setReasoningTokens: vi.fn((next: number) => {
              state.reasoningTokens = next
            }),
            setStreaming: vi.fn((next: string) => {
              state.streaming = next
            }),
            setToolTokens: vi.fn((next: number) => {
              state.toolTokens = next
            }),
            setTools: vi.fn((next: unknown) => {
              if (typeof next !== 'function') {
                state.tools = next as unknown[]
              }
            }),
            setTurnTrail: vi.fn((next: unknown) => {
              if (typeof next !== 'function') {
                state.turnTrail = next as string[]
              }
            })
          },
          refs
        }
      } as any)

    buildHandler()({
      payload: { context: 'home page', name: 'search', tool_id: 'tool-1' },
      type: 'tool.start'
    } as any)

    const onEvent = buildHandler()

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

    const refs = {
      activeToolsRef: ref([] as { context?: string; id: string; name: string; startedAt?: number }[]),
      bufRef: ref(''),
      interruptedRef: ref(false),
      lastStatusNoteRef: ref(''),
      persistedToolLabelsRef: ref(new Set<string>()),
      protocolWarnedRef: ref(false),
      reasoningRef: ref(''),
      statusTimerRef: ref<ReturnType<typeof setTimeout> | null>(null),
      toolTokenAccRef: ref(0),
      toolCompleteRibbonRef: ref(null),
      turnToolsRef: ref([] as string[])
    }

    const onEvent = createGatewayEventHandler({
      composer: {
        dequeue: () => undefined,
        queueEditRef: ref<number | null>(null),
        sendQueued: vi.fn()
      },
      gateway: {
        gw: { request: vi.fn() } as any,
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
      },
      turn: {
        actions: {
          clearReasoning: vi.fn(() => {
            refs.reasoningRef.current = ''
            refs.toolTokenAccRef.current = 0
          }),
          endReasoningPhase: vi.fn(),
          idle: vi.fn(() => {
            refs.activeToolsRef.current = []
          }),
          pruneTransient: vi.fn(),
          pulseReasoningStreaming: vi.fn(),
          pushActivity: vi.fn(),
          pushTrail: vi.fn(),
          scheduleReasoning: vi.fn(),
          scheduleStreaming: vi.fn(),
          setActivity: vi.fn(),
          setReasoningTokens: vi.fn(),
          setStreaming: vi.fn(),
          setToolTokens: vi.fn(),
          setTools: vi.fn(),
          setTurnTrail: vi.fn()
        },
        refs
      }
    } as any)

    onEvent({
      payload: { text: streamed },
      type: 'reasoning.delta'
    } as any)
    onEvent({
      payload: { text: fallback },
      type: 'reasoning.available'
    } as any)
    onEvent({
      payload: { text: 'final answer' },
      type: 'message.complete'
    } as any)

    expect(appended).toHaveLength(1)
    expect(appended[0]?.thinking).toBe(streamed)
    expect(appended[0]?.thinkingTokens).toBe(estimateTokensRough(streamed))
  })

  it('uses message.complete reasoning when no streamed reasoning ref', () => {
    const appended: Msg[] = []
    const fromServer = 'recovered from last_reasoning'

    const refs = {
      activeToolsRef: ref([] as { context?: string; id: string; name: string; startedAt?: number }[]),
      bufRef: ref(''),
      interruptedRef: ref(false),
      lastStatusNoteRef: ref(''),
      persistedToolLabelsRef: ref(new Set<string>()),
      protocolWarnedRef: ref(false),
      reasoningRef: ref(''),
      statusTimerRef: ref<ReturnType<typeof setTimeout> | null>(null),
      toolTokenAccRef: ref(0),
      toolCompleteRibbonRef: ref(null),
      turnToolsRef: ref([] as string[])
    }

    const onEvent = createGatewayEventHandler({
      composer: {
        dequeue: () => undefined,
        queueEditRef: ref<number | null>(null),
        sendQueued: vi.fn()
      },
      gateway: {
        gw: { request: vi.fn() } as any,
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
      },
      turn: {
        actions: {
          clearReasoning: vi.fn(() => {
            refs.reasoningRef.current = ''
            refs.toolTokenAccRef.current = 0
          }),
          endReasoningPhase: vi.fn(),
          idle: vi.fn(() => {
            refs.activeToolsRef.current = []
          }),
          pruneTransient: vi.fn(),
          pulseReasoningStreaming: vi.fn(),
          pushActivity: vi.fn(),
          pushTrail: vi.fn(),
          scheduleReasoning: vi.fn(),
          scheduleStreaming: vi.fn(),
          setActivity: vi.fn(),
          setReasoningTokens: vi.fn(),
          setStreaming: vi.fn(),
          setToolTokens: vi.fn(),
          setTools: vi.fn(),
          setTurnTrail: vi.fn()
        },
        refs
      }
    } as any)

    onEvent({
      payload: { reasoning: fromServer, text: 'final answer' },
      type: 'message.complete'
    } as any)

    expect(appended).toHaveLength(1)
    expect(appended[0]?.thinking).toBe(fromServer)
    expect(appended[0]?.thinkingTokens).toBe(estimateTokensRough(fromServer))
  })

  it('merges message.complete usage into existing context fields', () => {
    const appended: Msg[] = []

    patchUiState({
      usage: {
        calls: 1,
        context_max: 100_000,
        context_percent: 12,
        context_used: 12_000,
        input: 10,
        output: 20,
        total: 30
      }
    })

    const refs = {
      activeToolsRef: ref([] as { context?: string; id: string; name: string; startedAt?: number }[]),
      bufRef: ref(''),
      interruptedRef: ref(false),
      lastStatusNoteRef: ref(''),
      persistedToolLabelsRef: ref(new Set<string>()),
      protocolWarnedRef: ref(false),
      reasoningRef: ref(''),
      statusTimerRef: ref<ReturnType<typeof setTimeout> | null>(null),
      toolTokenAccRef: ref(0),
      toolCompleteRibbonRef: ref(null),
      turnToolsRef: ref([] as string[])
    }

    const onEvent = createGatewayEventHandler({
      composer: {
        dequeue: () => undefined,
        queueEditRef: ref<number | null>(null),
        sendQueued: vi.fn()
      },
      gateway: {
        gw: { request: vi.fn() } as any,
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
      },
      turn: {
        actions: {
          clearReasoning: vi.fn(() => {
            refs.reasoningRef.current = ''
          }),
          endReasoningPhase: vi.fn(),
          idle: vi.fn(),
          pruneTransient: vi.fn(),
          pulseReasoningStreaming: vi.fn(),
          pushActivity: vi.fn(),
          pushTrail: vi.fn(),
          scheduleReasoning: vi.fn(),
          scheduleStreaming: vi.fn(),
          setActivity: vi.fn(),
          setReasoningTokens: vi.fn(),
          setStreaming: vi.fn(),
          setToolTokens: vi.fn(),
          setTools: vi.fn(),
          setTurnTrail: vi.fn()
        },
        refs
      }
    } as any)

    onEvent({
      payload: {
        text: 'ok',
        usage: { calls: 2, input: 50, output: 60, total: 110 }
      },
      type: 'message.complete'
    } as any)

    const u = getUiState().usage
    expect(u.input).toBe(50)
    expect(u.total).toBe(110)
    expect(u.context_max).toBe(100_000)
    expect(u.context_used).toBe(12_000)
    expect(u.context_percent).toBe(12)
  })
})
