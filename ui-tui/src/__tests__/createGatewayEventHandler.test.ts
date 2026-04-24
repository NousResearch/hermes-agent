import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { estimateTokensRough } from '../lib/text.js'
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

describe('createGatewayEventHandler', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
    resetTurnState()
    turnController.fullReset()
    patchUiState({ showReasoning: true })
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

  it('anchors inline_diff as its own segment where the edit happened', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    const diff = '\u001b[31m--- a/foo.ts\u001b[0m\n\u001b[32m+++ b/foo.ts\u001b[0m\n@@\n-old\n+new'
    const cleaned = '--- a/foo.ts\n+++ b/foo.ts\n@@\n-old\n+new'
    const block = `\`\`\`diff\n${cleaned}\n\`\`\``

    // Narration → tool → tool-complete → more narration → message-complete.
    // The diff MUST land between the two narration segments, not tacked
    // onto the final one.
    onEvent({ payload: { text: 'Editing the file' }, type: 'message.delta' } as any)
    onEvent({ payload: { context: 'foo.ts', name: 'patch', tool_id: 'tool-1' }, type: 'tool.start' } as any)
    onEvent({ payload: { inline_diff: diff, summary: 'patched', tool_id: 'tool-1' }, type: 'tool.complete' } as any)

    // Diff is already committed to segmentMessages as its own segment.
    expect(appended).toHaveLength(0)
    expect(turnController.segmentMessages).toEqual([
      { role: 'assistant', text: 'Editing the file' },
      { kind: 'diff', role: 'assistant', text: block }
    ])

    onEvent({ payload: { text: 'patch applied' }, type: 'message.complete' } as any)

    // Three transcript messages: pre-tool narration → diff (kind='diff',
    // so MessageLine gives it blank-line breathing room) → post-tool
    // narration. The final message does NOT contain a diff.
    expect(appended).toHaveLength(3)
    expect(appended[0]?.text).toBe('Editing the file')
    expect(appended[1]).toMatchObject({ kind: 'diff', text: block })
    expect(appended[2]?.text).toBe('patch applied')
    expect(appended[2]?.text).not.toContain('```diff')
  })

  it('drops the diff segment when the final assistant text narrates the same diff', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    const cleaned = '--- a/foo.ts\n+++ b/foo.ts\n@@\n-old\n+new'
    const assistantText = `Done. Here's the inline diff:\n\n\`\`\`diff\n${cleaned}\n\`\`\``

    onEvent({ payload: { inline_diff: cleaned, summary: 'patched', tool_id: 'tool-1' }, type: 'tool.complete' } as any)
    onEvent({ payload: { text: assistantText }, type: 'message.complete' } as any)

    // Only the final message — diff-only segment dropped so we don't
    // render two stacked copies of the same patch.
    expect(appended).toHaveLength(1)
    expect(appended[0]?.text).toBe(assistantText)
    expect((appended[0]?.text.match(/```diff/g) ?? []).length).toBe(1)
  })

  it('strips the CLI "┊ review diff" header from inline diff segments', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    const raw = '  \u001b[33m┊ review diff\u001b[0m\n--- a/foo.ts\n+++ b/foo.ts\n@@\n-old\n+new'

    onEvent({ payload: { inline_diff: raw, summary: 'patched', tool_id: 'tool-1' }, type: 'tool.complete' } as any)
    onEvent({ payload: { text: 'done' }, type: 'message.complete' } as any)

    // diff segment first (kind='diff'), final narration second
    expect(appended).toHaveLength(2)
    expect(appended[0]?.kind).toBe('diff')
    expect(appended[0]?.text).not.toContain('┊ review diff')
    expect(appended[0]?.text).toContain('--- a/foo.ts')
    expect(appended[1]?.text).toBe('done')
  })

  it('drops the diff segment when assistant writes its own ```diff fence', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    const inlineDiff = '--- a/foo.ts\n+++ b/foo.ts\n@@\n-old\n+new'
    const assistantText = 'Done. Clean swap:\n\n```diff\n-old\n+new\n```'

    onEvent({ payload: { inline_diff: inlineDiff, summary: 'patched', tool_id: 'tool-1' }, type: 'tool.complete' } as any)
    onEvent({ payload: { text: assistantText }, type: 'message.complete' } as any)

    expect(appended).toHaveLength(1)
    expect(appended[0]?.text).toBe(assistantText)
    expect((appended[0]?.text.match(/```diff/g) ?? []).length).toBe(1)
  })

  it('keeps tool trail terse when inline_diff is present', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    const diff = '--- a/foo.ts\n+++ b/foo.ts\n@@\n-old\n+new'

    onEvent({
      payload: { inline_diff: diff, name: 'review_diff', summary: diff, tool_id: 'tool-1' },
      type: 'tool.complete'
    } as any)
    onEvent({ payload: { text: 'done' }, type: 'message.complete' } as any)

    // Two segments: the diff block (kind='diff', no tool row) and the final
    // narration (tool row belongs here since pendingSegmentTools carries
    // across the flushStreamingSegment call).
    expect(appended).toHaveLength(2)
    expect(appended[0]?.kind).toBe('diff')
    expect(appended[0]?.text).toContain('```diff')
    expect(appended[0]?.tools ?? []).toEqual([])
    expect(appended[1]?.text).toBe('done')
    expect(appended[1]?.tools?.[0]).toContain('Review Diff')
    expect(appended[1]?.tools?.[0]).not.toContain('--- a/foo.ts')
  })

  it('shows setup panel for missing provider startup error', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({
      payload: {
        message:
          'agent init failed: No LLM provider configured. Run `hermes model` to select a provider, or run `hermes setup` for first-time configuration.'
      },
      type: 'error'
    } as any)

    expect(appended).toHaveLength(1)
    expect(appended[0]).toMatchObject({
      kind: 'panel',
      panelData: { title: 'Setup Required' },
      role: 'system'
    })
  })

  it('keeps gateway noise informational and approval out of Activity', async () => {
    const appended: Msg[] = []
    const ctx = buildCtx(appended)
    ctx.gateway.rpc = vi.fn(async () => {
      throw new Error('cold start')
    })

    const onEvent = createGatewayEventHandler(ctx)

    onEvent({ payload: { line: 'Traceback: noisy but non-fatal' }, type: 'gateway.stderr' } as any)
    onEvent({ payload: { preview: 'bad framing' }, type: 'gateway.protocol_error' } as any)
    onEvent({
      payload: { command: 'rm -rf /tmp/nope', description: 'dangerous command' },
      type: 'approval.request'
    } as any)
    onEvent({ payload: {}, type: 'gateway.ready' } as any)

    await Promise.resolve()
    await Promise.resolve()

    expect(getOverlayState().approval).toMatchObject({ description: 'dangerous command' })
    expect(getTurnState().activity).toMatchObject([
      { text: 'Traceback: noisy but non-fatal', tone: 'info' },
      { text: 'protocol noise detected · /logs to inspect', tone: 'info' },
      { text: 'protocol noise: bad framing', tone: 'info' },
      { text: 'command catalog unavailable: cold start', tone: 'info' }
    ])
  })

  it('still surfaces terminal turn failures as errors', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { message: 'boom' }, type: 'error' } as any)

    expect(getTurnState().activity).toMatchObject([{ text: 'boom', tone: 'error' }])
  })
})
