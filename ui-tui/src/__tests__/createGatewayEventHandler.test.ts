import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { estimateTokensRough } from '../lib/text.js'
import type { Msg } from '../types.js'

// Mock the external-URL opener so the billing.step_up.verification test can
// assert it's invoked without spawning a real browser process.
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

  it('archives incomplete todos into transcript flow at end of turn so they scroll up', () => {
    const appended: Msg[] = []

    const todos = [
      { content: 'Gather ingredients', id: 'prep', status: 'completed' },
      { content: 'Boil water', id: 'boil', status: 'in_progress' },
      { content: 'Make sauce', id: 'sauce', status: 'pending' }
    ]

    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { name: 'todo', todos, tool_id: 'todo-1' }, type: 'tool.start' } as any)
    expect(getTurnState().todos).toEqual(todos)

    onEvent({ payload: { text: 'Started a todo list.' }, type: 'message.complete' } as any)

    const trail = appended.find(msg => msg.kind === 'trail' && msg.todos?.length)
    const finalText = appended.find(msg => msg.role === 'assistant' && msg.text === 'Started a todo list.')

    expect(finalText).toBeDefined()
    expect(trail).toMatchObject({ kind: 'trail', role: 'system', todos, todoIncomplete: true })
    // Todo archive must sit ABOVE the final assistant text so the panel
    // doesn't visibly jump across the final answer at end-of-turn.
    expect(appended.indexOf(trail!)).toBeLessThan(appended.indexOf(finalText!))
    expect(getTurnState().todos).toEqual([])
  })

  it('archives completed todos into transcript flow at end of turn', () => {
    const appended: Msg[] = []
    const todos = [{ content: 'Serve tiny latte', id: 'serve', status: 'completed' }]
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { name: 'todo', todos, tool_id: 'todo-1' }, type: 'tool.start' } as any)
    onEvent({ payload: { text: 'done' }, type: 'message.complete' } as any)

    expect(getTurnState().todos).toEqual([])
    expect(appended).toContainEqual({
      kind: 'trail',
      role: 'system',
      text: '',
      todoCollapsedByDefault: true,
      todos
    })
  })

  it('keeps the current todo list visible when the next message starts', () => {
    const appended: Msg[] = []
    const todos = [{ content: 'Boil water', id: 'boil', status: 'in_progress' }]

    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { name: 'todo', todos, tool_id: 'todo-1' }, type: 'tool.start' } as any)
    expect(getTurnState().todos).toEqual(todos)

    onEvent({ payload: {}, type: 'message.start' } as any)

    expect(getTurnState().todos).toEqual(todos)
  })

  it('prints compaction progress status into the transcript', () => {
    const appended: Msg[] = []
    const ctx = buildCtx(appended)
    const onEvent = createGatewayEventHandler(ctx)

    onEvent({
      payload: { kind: 'compressing', text: 'compressing 968 messages (~123,400 tok)…' },
      type: 'status.update'
    } as any)

    expect(ctx.system.sys).toHaveBeenCalledWith('compressing 968 messages (~123,400 tok)…')
  })

  it('keeps goal verdict text in transcript but shows a brief idle status (#goal statusbar)', () => {
    const appended: Msg[] = []
    const ctx = buildCtx(appended)
    const onEvent = createGatewayEventHandler(ctx)
    const verdict = '✓ Goal achieved: long judge reason goes only in transcript, not merged with cwd label.'

    vi.useFakeTimers()

    try {
      onEvent({
        payload: { kind: 'goal', text: verdict },
        type: 'status.update'
      } as any)

      expect(ctx.system.sys).toHaveBeenCalledWith(verdict)
      expect(getUiState().status).toBe('✓ goal complete')

      vi.advanceTimersByTime(6001)
      expect(getUiState().status).toBe('ready')
    } finally {
      vi.useRealTimers()
    }
  })

  it('maps goal status.update prefixes to short status strings', () => {
    const ctx = buildCtx([])
    const onEvent = createGatewayEventHandler(ctx)

    onEvent({
      payload: { kind: 'goal', text: '↻ Continuing toward goal (1/10): reason' },
      type: 'status.update'
    } as any)
    expect(getUiState().status).toBe('↻ goal continuing')

    onEvent({
      payload: { kind: 'goal', text: '⏸ Goal paused — budget exhausted.' },
      type: 'status.update'
    } as any)
    expect(getUiState().status).toBe('⏸ goal paused')
  })

  it('folds runtime mechanism events into the status capsule without upgrading unknown OpenCode/Deep state', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    patchUiState({ mechanismStatusBar: true, sid: 'sess-1' } as any)

    onEvent({ payload: {}, type: 'message.start' } as any)
    expect((getUiState() as any).statusCapsule).toMatchObject({ route: 'Hermes', state: 'running' })

    onEvent({ payload: { name: 'terminal', tool_id: 'tool-1' }, type: 'tool.start' } as any)
    expect((getUiState() as any).statusCapsule).toMatchObject({
      observed: { deep: 'unknown', opencode: 'unknown' },
      state: 'tooling',
      tools: ['terminal']
    })

    onEvent({ payload: { goal: 'check implementation', subagent_id: 'sa-1', task_index: 0 }, type: 'subagent.start' } as any)
    expect((getUiState() as any).statusCapsule.observed.delegate).toBe('observed')

    onEvent({ payload: { count: 2, index: 1, label: 'openrouter:openai/gpt-5.5', text: 'draft' }, type: 'moa.reference' } as any)
    expect((getUiState() as any).statusCapsule).toMatchObject({
      observed: { moa: 'observed', opencode: 'unknown' },
      route: 'MoA'
    })

    onEvent({ payload: { text: 'done', usage: { calls: 1, input: 20, output: 5, total: 25 } }, type: 'message.complete' } as any)
    expect((getUiState() as any).statusCapsule).toMatchObject({
      precision: { elapsed: 'observed', tokens: 'observed' },
      state: 'idle'
    })
  })

  it('keeps resetting hidden mechanism state while the status capsule is disabled', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    patchUiState({ mechanismStatusBar: true, sid: 'sess-1' } as any)
    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { label: 'reference', text: 'draft' }, type: 'moa.reference' } as any)
    expect((getUiState() as any).statusCapsule.route).toBe('MoA')

    patchUiState({ mechanismStatusBar: false } as any)
    onEvent({ payload: {}, type: 'message.start' } as any)

    expect((getUiState() as any).statusCapsule).toMatchObject({
      observed: { delegate: 'unknown', moa: 'unknown', opencode: 'unknown' },
      route: 'Hermes',
      state: 'running'
    })
  })

  it('surfaces self-improvement review summaries as a persistent system line', () => {
    const appended: Msg[] = []
    const ctx = buildCtx(appended)
    const onEvent = createGatewayEventHandler(ctx)

    onEvent({
      payload: { text: "💾 Self-improvement review: Skill 'hermes-release' patched" },
      type: 'review.summary'
    } as any)

    expect(ctx.system.sys).toHaveBeenCalledWith("💾 Self-improvement review: Skill 'hermes-release' patched")
  })

  it('ignores review.summary events with empty or missing text', () => {
    const appended: Msg[] = []
    const ctx = buildCtx(appended)
    const onEvent = createGatewayEventHandler(ctx)

    onEvent({ payload: { text: '' }, type: 'review.summary' } as any)
    onEvent({ payload: { text: '   ' }, type: 'review.summary' } as any)
    onEvent({ payload: undefined, type: 'review.summary' } as any)

    expect(ctx.system.sys).not.toHaveBeenCalled()
  })

  it('clears the visible todo list when the todo tool returns an empty list', () => {
    const appended: Msg[] = []
    const todos = [{ content: 'Boil water', id: 'boil', status: 'in_progress' }]
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { name: 'todo', todos, tool_id: 'todo-1' }, type: 'tool.start' } as any)
    expect(getTurnState().todos).toEqual(todos)

    onEvent({ payload: { name: 'todo', todos: [], tool_id: 'todo-1' }, type: 'tool.complete' } as any)

    expect(getTurnState().todos).toEqual([])
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

    expect(appended).toHaveLength(2)
    expect(appended[0]).toMatchObject({ kind: 'trail', role: 'system', text: '', thinking: 'mapped the page' })
    expect(appended[0]?.tools).toHaveLength(1)
    expect(appended[0]?.tools?.[0]).toContain('hero cards')
    expect(appended[0]?.toolTokens).toBeGreaterThan(0)
    expect(appended[1]).toMatchObject({ role: 'assistant', text: 'final answer' })
  })

  it('groups sequential completed tools into one trail when the turn completes', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { context: 'alpha', name: 'search_files', tool_id: 'tool-1' }, type: 'tool.start' } as any)
    onEvent({
      payload: { name: 'search_files', summary: 'first done', tool_id: 'tool-1' },
      type: 'tool.complete'
    } as any)
    onEvent({ payload: { context: 'beta', name: 'read_file', tool_id: 'tool-2' }, type: 'tool.start' } as any)
    onEvent({ payload: { name: 'read_file', summary: 'second done', tool_id: 'tool-2' }, type: 'tool.complete' } as any)

    expect(getTurnState().streamSegments.filter(msg => msg.kind === 'trail' && msg.tools?.length)).toHaveLength(1)
    expect(getTurnState().streamSegments[0]?.tools).toHaveLength(2)
    expect(getTurnState().streamPendingTools).toEqual([])

    onEvent({ payload: { text: '' }, type: 'message.complete' } as any)

    const toolTrails = appended.filter(msg => msg.kind === 'trail' && msg.tools?.length)
    expect(toolTrails).toHaveLength(1)
    expect(toolTrails[0]?.tools).toHaveLength(2)
    expect(toolTrails[0]?.tools?.[0]).toContain('Search Files')
    expect(toolTrails[0]?.tools?.[1]).toContain('Read File')
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

    expect(appended).toHaveLength(2)
    expect(appended[0]?.tools).toHaveLength(1)
    expect(appended[0]?.toolTokens).toBeGreaterThan(0)
    expect(appended[1]).toMatchObject({ role: 'assistant', text: 'final answer' })
  })

  it('streams legacy thinking.delta into visible reasoning state', () => {
    vi.useFakeTimers()
    const appended: Msg[] = []
    const streamed = 'short streamed reasoning'
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    try {
      onEvent({ payload: {}, type: 'message.start' } as any)
      onEvent({ payload: { text: streamed }, type: 'thinking.delta' } as any)
      vi.runOnlyPendingTimers()

      expect(getTurnState().reasoning).toBe(streamed)
      expect(getTurnState().reasoningActive).toBe(true)
      expect(getTurnState().reasoningTokens).toBe(estimateTokensRough(streamed))
    } finally {
      vi.useRealTimers()
    }
  })

  it('ignores late thinking.delta after the turn has already completed', () => {
    vi.useFakeTimers()
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    try {
      onEvent({ payload: {}, type: 'message.start' } as any)
      onEvent({ payload: { text: 'final answer' }, type: 'message.complete' } as any)
      expect(getUiState().busy).toBe(false)
      expect(getUiState().status).toBe('ready')

      onEvent({ payload: { text: 'thinking...' }, type: 'thinking.delta' } as any)
      vi.runOnlyPendingTimers()

      expect(getUiState().status).toBe('ready')
      expect(getTurnState().reasoning).toBe('')
    } finally {
      vi.useRealTimers()
    }
  })

  it('preserves streamed reasoning as one completed thinking panel after segment flushes', () => {
    const appended: Msg[] = []
    const streamed = 'first reasoning chunk\nsecond reasoning chunk'

    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { text: streamed }, type: 'reasoning.delta' } as any)
    onEvent({ payload: { text: 'Before edit.' }, type: 'message.delta' } as any)
    turnController.flushStreamingSegment()
    onEvent({ payload: { text: 'final answer' }, type: 'message.complete' } as any)

    expect(appended.map(msg => msg.thinking).filter(Boolean)).toEqual([streamed])
    expect(appended[appended.length - 1]).toMatchObject({ role: 'assistant', text: 'final answer' })
  })

  it('filters spinner/status-only reasoning noise from completed thinking', () => {
    const appended: Msg[] = []
    const streamed = '(¬_¬) synthesizing...\nactual plan\n( ͡° ͜ʖ ͡°) pondering...\nnext step'

    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { text: streamed }, type: 'reasoning.delta' } as any)
    onEvent({ payload: { text: 'final answer' }, type: 'message.complete' } as any)

    expect(appended[0]?.thinking).toBe(streamed)
    expect(appended[0]?.text).toBe('')
    expect(appended[appended.length - 1]).toMatchObject({ role: 'assistant', text: 'final answer' })
  })

  it('shows verbose reasoning even when normal reasoning display is off', () => {
    vi.useFakeTimers()
    patchUiState({ showReasoning: false })
    const appended: Msg[] = []
    const streamed = 'verbose-only reasoning'

    try {
      const onEvent = createGatewayEventHandler(buildCtx(appended))

      onEvent({ payload: { text: streamed, verbose: true }, type: 'reasoning.delta' } as any)
      vi.runOnlyPendingTimers()

      expect(turnController.reasoningText).toBe(streamed)
      expect(getTurnState().reasoning).toBe(streamed)
    } finally {
      vi.useRealTimers()
    }
  })

  it('ignores fallback reasoning.available when streamed reasoning already exists', () => {
    const appended: Msg[] = []
    const streamed = 'short streamed reasoning'
    const fallback = 'x'.repeat(400)

    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { text: streamed }, type: 'reasoning.delta' } as any)
    onEvent({ payload: { text: fallback }, type: 'reasoning.available' } as any)
    onEvent({ payload: { text: 'final answer' }, type: 'message.complete' } as any)

    expect(appended).toHaveLength(2)
    expect(appended[0]?.thinking).toBe(streamed)
    expect(appended[0]?.thinkingTokens).toBe(estimateTokensRough(streamed))
    expect(appended[1]).toMatchObject({ role: 'assistant', text: 'final answer' })
  })

  it('renders moa.reference as a labelled thinking-style segment', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({
      payload: { count: 2, index: 1, label: 'openrouter:openai/gpt-5.5', text: 'Paris.' },
      type: 'moa.reference'
    } as any)
    onEvent({
      payload: { count: 2, index: 2, label: 'openrouter:anthropic/claude-opus-4.8', text: 'Paris.' },
      type: 'moa.reference'
    } as any)

    const segments = getTurnState().streamSegments
    const refBlocks = segments.filter(m => typeof m.thinking === 'string' && m.thinking.includes('Reference'))
    expect(refBlocks).toHaveLength(2)
    expect(refBlocks[0]?.thinking).toContain('Reference 1/2 — openrouter:openai/gpt-5.5')
    expect(refBlocks[0]?.thinking).toContain('Paris.')
    expect(refBlocks[1]?.thinking).toContain('Reference 2/2 — openrouter:anthropic/claude-opus-4.8')
  })

  it('renders moa.reference even when showReasoning is off (it is the MoA process, not reasoning)', () => {
    patchUiState({ showReasoning: false })
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({
      payload: { label: 'openrouter:openai/gpt-5.5', text: 'Four.' },
      type: 'moa.reference'
    } as any)

    const segments = getTurnState().streamSegments
    const refBlocks = segments.filter(m => typeof m.thinking === 'string' && m.thinking.includes('Reference'))
    expect(refBlocks).toHaveLength(1)
    expect(refBlocks[0]?.thinking).toContain('openrouter:openai/gpt-5.5')
  })

  it('moa.aggregating does not append a transcript segment', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: {}, type: 'message.start' } as any)
    const before = getTurnState().streamSegments.length
    onEvent({ payload: { aggregator: 'openrouter:anthropic/claude-opus-4.8' }, type: 'moa.aggregating' } as any)
    expect(getTurnState().streamSegments.length).toBe(before)
  })

  it('uses message.complete reasoning when no streamed reasoning ref', () => {
    const appended: Msg[] = []
    const fromServer = 'recovered from last_reasoning'

    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { reasoning: fromServer, text: 'final answer' }, type: 'message.complete' } as any)

    expect(appended).toHaveLength(2)
    expect(appended[0]?.thinking).toBe(fromServer)
    expect(appended[0]?.thinkingTokens).toBe(estimateTokensRough(fromServer))
    expect(appended[1]).toMatchObject({ role: 'assistant', text: 'final answer' })
  })

  it('renders browser.progress events as system transcript lines as they stream in', () => {
    const appended: Msg[] = []
    const ctx = buildCtx(appended)
    const handler = createGatewayEventHandler(ctx)

    handler({
      payload: { message: 'Chromium-family browser launched and listening on port 9222' },
      type: 'browser.progress'
    } as any)

    expect(ctx.system.sys).toHaveBeenCalledWith('Chromium-family browser launched and listening on port 9222')
  })

  it('annotates gateway.start_timeout with stderr tail lines so users can diagnose without /logs', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({
      payload: {
        cwd: '/repo',
        python: '/opt/venv/bin/python',
        stderr_tail:
          '[startup] timed out\nModuleNotFoundError: No module named openai\nFileNotFoundError: ~/.hermes/config.yaml'
      },
      type: 'gateway.start_timeout'
    } as any)

    const messages = getTurnState().activity.map(a => a.text)

    expect(messages.some(m => m.includes('gateway startup timed out'))).toBe(true)
    expect(messages.some(m => m.includes('ModuleNotFoundError'))).toBe(true)
    expect(messages.some(m => m.includes('FileNotFoundError'))).toBe(true)
  })

  it('prefers raw text over Rich-rendered ANSI on message.complete (#16391)', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    const raw = 'Hermes here.\n\nLine two.'
    // Rich-rendered ANSI (`final_response_markdown: render`) used to win,
    // which left visible escape codes in Ink output. Raw text must win.
    const rendered = '\u001b[33mHermes here.\u001b[0m\n\n\u001b[2mLine two.\u001b[0m'

    onEvent({ payload: { rendered, text: raw }, type: 'message.complete' } as any)

    const assistant = appended.find(msg => msg.role === 'assistant')
    expect(assistant?.text).toBe(raw)
    expect(assistant?.text).not.toContain('\u001b[')
  })

  it('falls back to payload.rendered when text is missing on message.complete', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    const rendered = 'fallback when gateway omitted text'

    onEvent({ payload: { rendered }, type: 'message.complete' } as any)

    const assistant = appended.find(msg => msg.role === 'assistant')
    expect(assistant?.text).toBe(rendered)
  })

  it('always accumulates raw text in message.delta and ignores `rendered` (#16391)', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    // Stream of partial text deltas; each delta carries an incremental
    // Rich-ANSI fragment.  Pre-fix code would replace the whole bufRef
    // with the latest fragment, dropping prior text.
    onEvent({ payload: { rendered: '\u001b[33mFi\u001b[0m', text: 'Fi' }, type: 'message.delta' } as any)
    onEvent({ payload: { rendered: '\u001b[33mrst.\u001b[0m', text: 'rst.' }, type: 'message.delta' } as any)
    onEvent({ payload: { text: ' second.' }, type: 'message.delta' } as any)
    onEvent({ payload: {}, type: 'message.complete' } as any)

    const assistant = appended.find(msg => msg.role === 'assistant')
    expect(assistant?.text).toBe('First. second.')
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
      {
        kind: 'diff',
        role: 'assistant',
        text: block,
        tools: [expect.stringMatching(/^Patch\("foo\.ts"\)(?: \([^)]+\))? ✓$/)]
      }
    ])

    onEvent({ payload: { text: 'patch applied' }, type: 'message.complete' } as any)

    expect(appended).toHaveLength(4)
    expect(appended[0]?.text).toBe('Editing the file')
    expect(appended[1]).toMatchObject({ kind: 'diff', text: block })
    expect(appended[1]?.tools?.[0]).toContain('Patch')
    expect(appended[3]?.text).toBe('patch applied')
    expect(appended[3]?.text).not.toContain('```diff')
  })

  it('keeps verbose result text on inline_diff tool completions', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    const diff = '--- a/foo.ts\n+++ b/foo.ts\n@@\n-old\n+new'

    onEvent({
      payload: { args_text: '{ "path": "foo.ts" }', context: 'foo.ts', name: 'patch', tool_id: 'tool-1' },
      type: 'tool.start'
    } as any)
    onEvent({
      payload: { inline_diff: diff, result_text: 'patched result', tool_id: 'tool-1' },
      type: 'tool.complete'
    } as any)

    expect(turnController.segmentMessages[0]).toMatchObject({ kind: 'diff' })
    expect(turnController.segmentMessages[0]?.tools?.[0]).toContain('Args:\n{ "path": "foo.ts" }')
    expect(turnController.segmentMessages[0]?.tools?.[0]).toContain('Result:\npatched result')
  })

  it('keeps full final responses from duplicating flushed pre-diff narration', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    const diff = '--- a/foo.ts\n+++ b/foo.ts\n@@\n-old\n+new'
    const block = `\`\`\`diff\n${diff}\n\`\`\``

    onEvent({ payload: { text: 'Before edit. ' }, type: 'message.delta' } as any)
    onEvent({ payload: { context: 'foo.ts', name: 'patch', tool_id: 'tool-1' }, type: 'tool.start' } as any)
    onEvent({ payload: { inline_diff: diff, summary: 'patched', tool_id: 'tool-1' }, type: 'tool.complete' } as any)
    onEvent({ payload: { text: 'After edit.' }, type: 'message.delta' } as any)
    onEvent({ payload: { text: 'Before edit. After edit.' }, type: 'message.complete' } as any)

    expect(appended.map(msg => msg.text.trim()).filter(Boolean)).toEqual(['Before edit.', block, 'After edit.'])
    expect(appended[1]?.tools?.[0]).toContain('Patch')
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

    // Tool trail first, then diff segment (kind='diff'), then final narration.
    expect(appended).toHaveLength(2)
    expect(appended[0]?.kind).toBe('diff')
    expect(appended[0]?.text).not.toContain('┊ review diff')
    expect(appended[0]?.text).toContain('--- a/foo.ts')
    expect(appended[0]?.tools?.[0]).toContain('Tool')
    expect(appended[1]?.text).toBe('done')
  })

  it('drops the diff segment when assistant writes its own ```diff fence', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    const inlineDiff = '--- a/foo.ts\n+++ b/foo.ts\n@@\n-old\n+new'
    const assistantText = 'Done. Clean swap:\n\n```diff\n-old\n+new\n```'

    onEvent({
      payload: { inline_diff: inlineDiff, summary: 'patched', tool_id: 'tool-1' },
      type: 'tool.complete'
    } as any)
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

    // Tool row is now placed before the diff, so telemetry does not render
    // below the patch that came from that tool.
    expect(appended).toHaveLength(2)
    expect(appended[0]?.kind).toBe('diff')
    expect(appended[0]?.text).toContain('```diff')
    expect(appended[0]?.tools?.[0]).toContain('Review Diff')
    expect(appended[0]?.tools?.[0]).not.toContain('--- a/foo.ts')
    expect(appended[1]?.text).toBe('done')
    expect(appended[1]?.tools ?? []).toEqual([])
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

  it('does not fetch config while constructing the gateway event handler', () => {
    const appended: Msg[] = []
    const ctx = buildCtx(appended)

    ctx.gateway.rpc = vi.fn(async () => null)

    createGatewayEventHandler(ctx)

    expect(ctx.gateway.rpc).not.toHaveBeenCalled()
  })

  it('on gateway.ready with no STARTUP_RESUME_ID and auto_resume off, forges a new session', async () => {
    const appended: Msg[] = []
    const newSession = vi.fn()
    const resumeById = vi.fn()
    const ctx = buildCtx(appended)

    ctx.session.newSession = newSession
    ctx.session.resumeById = resumeById
    ctx.session.STARTUP_RESUME_ID = ''
    ctx.gateway.rpc = vi.fn(async (method: string) => {
      if (method === 'config.get') {
        return { config: { display: { tui_auto_resume_recent: false } } }
      }

      return null
    })

    createGatewayEventHandler(ctx)({ payload: {}, type: 'gateway.ready' } as any)

    await vi.waitFor(() => expect(newSession).toHaveBeenCalled())
    expect(resumeById).not.toHaveBeenCalled()
  })

  it('on gateway.ready after a crash, resumes the recovered session once and skips forge', async () => {
    const appended: Msg[] = []
    const newSession = vi.fn()
    const resumeById = vi.fn()
    const ctx = buildCtx(appended)

    ctx.session.newSession = newSession
    // Mimic resumeById's synchronous status write so the test proves the
    // "recovering session…" label is applied *after* (and survives) it.
    ctx.session.resumeById = resumeById.mockImplementation(() => patchUiState({ status: 'resuming…' }))
    ctx.session.STARTUP_RESUME_ID = ''
    ctx.session.recoverSidRef = ref<null | string>('sess-crashed')

    const onEvent = createGatewayEventHandler(ctx)

    onEvent({ payload: {}, type: 'gateway.ready' } as any)

    await vi.waitFor(() => expect(resumeById).toHaveBeenCalledWith('sess-crashed'))
    expect(newSession).not.toHaveBeenCalled()
    // One-shot: the ref is consumed so a later ordinary restart forges/resumes
    // per config instead of re-resuming the recovered session.
    expect(ctx.session.recoverSidRef.current).toBeNull()
    expect(getUiState().status).toBe('recovering session…')
  })

  it('on gateway.ready with auto_resume on and a recent session, resumes it', async () => {
    const appended: Msg[] = []
    const newSession = vi.fn()
    const resumeById = vi.fn()
    const ctx = buildCtx(appended)

    ctx.session.newSession = newSession
    ctx.session.resumeById = resumeById
    ctx.session.STARTUP_RESUME_ID = ''
    ctx.gateway.rpc = vi.fn(async (method: string) => {
      if (method === 'config.get') {
        return { config: { display: { tui_auto_resume_recent: true } } }
      }

      if (method === 'session.most_recent') {
        return { session_id: 'sess-most-recent' }
      }

      return null
    })

    createGatewayEventHandler(ctx)({ payload: {}, type: 'gateway.ready' } as any)

    await vi.waitFor(() => expect(resumeById).toHaveBeenCalledWith('sess-most-recent'))
    expect(newSession).not.toHaveBeenCalled()
  })

  it('on gateway.ready with auto_resume on but no eligible session, falls back to new', async () => {
    const appended: Msg[] = []
    const newSession = vi.fn()
    const resumeById = vi.fn()
    const ctx = buildCtx(appended)

    ctx.session.newSession = newSession
    ctx.session.resumeById = resumeById
    ctx.session.STARTUP_RESUME_ID = ''
    ctx.gateway.rpc = vi.fn(async (method: string) => {
      if (method === 'config.get') {
        return { config: { display: { tui_auto_resume_recent: true } } }
      }

      if (method === 'session.most_recent') {
        return { session_id: null }
      }

      return null
    })

    createGatewayEventHandler(ctx)({ payload: {}, type: 'gateway.ready' } as any)

    await vi.waitFor(() => expect(newSession).toHaveBeenCalled())
    expect(resumeById).not.toHaveBeenCalled()
  })

  it('on gateway.ready when config.get rejects, falls back to new session', async () => {
    const appended: Msg[] = []
    const newSession = vi.fn()
    const resumeById = vi.fn()
    const ctx = buildCtx(appended)

    ctx.session.newSession = newSession
    ctx.session.resumeById = resumeById
    ctx.session.STARTUP_RESUME_ID = ''
    ctx.gateway.rpc = vi.fn(async (method: string) => {
      if (method === 'config.get') {
        throw new Error('gateway timeout')
      }

      return null
    })

    createGatewayEventHandler(ctx)({ payload: {}, type: 'gateway.ready' } as any)

    await vi.waitFor(() => expect(newSession).toHaveBeenCalled())
    expect(resumeById).not.toHaveBeenCalled()
  })

  it('on gateway.ready when session.most_recent rejects, falls back to new session', async () => {
    const appended: Msg[] = []
    const newSession = vi.fn()
    const resumeById = vi.fn()
    const ctx = buildCtx(appended)

    ctx.session.newSession = newSession
    ctx.session.resumeById = resumeById
    ctx.session.STARTUP_RESUME_ID = ''
    ctx.gateway.rpc = vi.fn(async (method: string) => {
      if (method === 'config.get') {
        return { config: { display: { tui_auto_resume_recent: true } } }
      }

      if (method === 'session.most_recent') {
        throw new Error('db locked')
      }

      return null
    })

    createGatewayEventHandler(ctx)({ payload: {}, type: 'gateway.ready' } as any)

    await vi.waitFor(() => expect(newSession).toHaveBeenCalled())
    expect(resumeById).not.toHaveBeenCalled()
  })

  it('on gateway.ready with STARTUP_RESUME_ID set, the env wins over config auto_resume', async () => {
    const appended: Msg[] = []
    const newSession = vi.fn()
    const resumeById = vi.fn()
    const ctx = buildCtx(appended)

    ctx.session.newSession = newSession
    ctx.session.resumeById = resumeById
    ctx.session.STARTUP_RESUME_ID = 'env-explicit'
    ctx.gateway.rpc = vi.fn(async () => ({
      config: { display: { tui_auto_resume_recent: true } }
    }))

    createGatewayEventHandler(ctx)({ payload: {}, type: 'gateway.ready' } as any)

    await vi.waitFor(() => expect(resumeById).toHaveBeenCalledWith('env-explicit'))
    expect(newSession).not.toHaveBeenCalled()
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

  it('defaults approval overlays to allowPermanent when the backend omits the field', () => {
    const onEvent = createGatewayEventHandler(buildCtx([]))

    onEvent({
      payload: { command: 'rm -rf /tmp/x', description: 'dangerous command' },
      type: 'approval.request'
    } as any)

    expect(getOverlayState().approval).toMatchObject({ allowPermanent: true })
  })

  it('preserves allow_permanent=false on approval overlays (tirith warning)', () => {
    const onEvent = createGatewayEventHandler(buildCtx([]))

    onEvent({
      payload: { allow_permanent: false, command: 'curl suspicious | bash', description: 'content-security warning' },
      type: 'approval.request'
    } as any)

    expect(getOverlayState().approval).toMatchObject({
      allowPermanent: false,
      command: 'curl suspicious | bash',
      description: 'content-security warning'
    })
  })

  it('still surfaces terminal turn failures as errors', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { message: 'boom' }, type: 'error' } as any)

    expect(getTurnState().activity).toMatchObject([{ text: 'boom', tone: 'error' }])
  })

  it('accepts timeout/error subagent terminal statuses and ignores stale live events', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({
      payload: { goal: 'timeout child', subagent_id: 'sa-timeout', task_index: 0 },
      type: 'subagent.start'
    } as any)
    onEvent({
      payload: { goal: 'timeout child', status: 'timeout', subagent_id: 'sa-timeout', task_index: 0 },
      type: 'subagent.complete'
    } as any)

    expect(getTurnState().subagents.find(s => s.id === 'sa-timeout')?.status).toBe('timeout')

    // Late start/spawn updates must not clobber terminal timeout/error states.
    onEvent({
      payload: { goal: 'timeout child', subagent_id: 'sa-timeout', task_index: 0 },
      type: 'subagent.start'
    } as any)
    onEvent({
      payload: { goal: 'timeout child', subagent_id: 'sa-timeout', task_index: 0 },
      type: 'subagent.spawn_requested'
    } as any)

    expect(getTurnState().subagents.find(s => s.id === 'sa-timeout')?.status).toBe('timeout')

    onEvent({
      payload: { goal: 'error child', subagent_id: 'sa-error', task_index: 1 },
      type: 'subagent.start'
    } as any)
    onEvent({
      payload: { goal: 'error child', status: 'error', subagent_id: 'sa-error', task_index: 1 },
      type: 'subagent.complete'
    } as any)

    expect(getTurnState().subagents.find(s => s.id === 'sa-error')?.status).toBe('error')
  })

  it('normalizes unknown subagent.complete statuses to completed', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({
      payload: { goal: 'weird child', subagent_id: 'sa-weird', task_index: 2 },
      type: 'subagent.start'
    } as any)
    onEvent({
      payload: { goal: 'weird child', status: 'mystery_status', subagent_id: 'sa-weird', task_index: 2 },
      type: 'subagent.complete'
    } as any)

    expect(getTurnState().subagents.find(s => s.id === 'sa-weird')?.status).toBe('completed')
  })

  it('nudges toward /agents on the first spawn_requested of a turn', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({
      payload: { goal: 'child a', subagent_id: 'sa-a', task_index: 0 },
      type: 'subagent.spawn_requested'
    } as any)

    const hints = getTurnState().activity.filter(a => a.text.includes('/agents'))
    expect(hints).toHaveLength(1)
    expect(hints[0]).toMatchObject({ tone: 'info' })
  })

  it('nudges toward /agents on subagent.start (spawn_requested dropped in CLI path)', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    // In the real CLI→gateway path the delegate callback drops
    // spawn_requested, so `start` is the first event the TUI sees.
    onEvent({
      payload: { goal: 'child a', subagent_id: 'sa-a', task_index: 0 },
      type: 'subagent.start'
    } as any)

    expect(getTurnState().activity.filter(a => a.text.includes('/agents'))).toHaveLength(1)
  })

  it('nudges at most once per turn and resets on the next message.start', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    // Multiple spawns in one turn → a single hint.
    onEvent({
      payload: { goal: 'child a', subagent_id: 'sa-a', task_index: 0 },
      type: 'subagent.start'
    } as any)
    onEvent({
      payload: { goal: 'child b', subagent_id: 'sa-b', task_index: 1 },
      type: 'subagent.start'
    } as any)
    expect(getTurnState().activity.filter(a => a.text.includes('/agents'))).toHaveLength(1)

    // New turn clears activity AND the once-per-turn guard → nudges again.
    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({
      payload: { goal: 'child c', subagent_id: 'sa-c', task_index: 0 },
      type: 'subagent.start'
    } as any)
    expect(getTurnState().activity.filter(a => a.text.includes('/agents'))).toHaveLength(1)
  })

  it('does not nudge when the /agents overlay is already open', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    // User already has the dashboard open → nothing to advertise.
    patchOverlayState({ agents: true })

    onEvent({
      payload: { goal: 'child a', subagent_id: 'sa-a', task_index: 0 },
      type: 'subagent.start'
    } as any)

    expect(getTurnState().activity.filter(a => a.text.includes('/agents'))).toHaveLength(0)
  })

  it('nudges if the /agents overlay is closed mid-turn while delegation continues', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    // Overlay open on the first delegation event → suppressed, but the
    // turn's nudge credit must NOT be burned (the user is watching).
    patchOverlayState({ agents: true })
    onEvent({
      payload: { goal: 'child a', subagent_id: 'sa-a', task_index: 0 },
      type: 'subagent.start'
    } as any)
    expect(getTurnState().activity.filter(a => a.text.includes('/agents'))).toHaveLength(0)

    // User closes the dashboard mid-turn → the next delegation event nudges.
    patchOverlayState({ agents: false })
    onEvent({
      payload: { goal: 'child b', subagent_id: 'sa-b', task_index: 1 },
      type: 'subagent.start'
    } as any)
    expect(getTurnState().activity.filter(a => a.text.includes('/agents'))).toHaveLength(1)
  })

  it('does not nudge when display.tui_agents_nudge is false', async () => {
    const appended: Msg[] = []
    const ctx = buildCtx(appended)
    // config.get → full returns the disable flag.
    ctx.gateway.rpc = vi.fn(async (method: string) =>
      method === 'config.get' ? { config: { display: { tui_agents_nudge: false } } } : null
    )
    const onEvent = createGatewayEventHandler(ctx)

    // Config fetch starts once the gateway is ready; let it resolve before any
    // spawn (mirrors real usage — config lands well before first delegation).
    onEvent({ payload: {}, type: 'gateway.ready' } as any)
    await Promise.resolve()
    await Promise.resolve()

    onEvent({
      payload: { goal: 'child a', subagent_id: 'sa-a', task_index: 0 },
      type: 'subagent.start'
    } as any)

    expect(getTurnState().activity.filter(a => a.text.includes('/agents'))).toHaveLength(0)
  })

  it('drops stale reasoning/tool/todos events after ctrl-c until the next message starts', () => {
    // Repro for the discord report: ctrl-c interrupts, but late reasoning/tool
    // events from the still-winding-down agent loop kept populating the UI for
    // ~1s, making it look like the interrupt had been ignored.
    //
    // Fake timers because `interruptTurn` schedules a real setTimeout for
    // its cooldown — without flushing it inside this test, the timeout
    // can fire later and mutate uiStore/turnState during unrelated tests
    // (cross-file flake).
    vi.useFakeTimers()

    try {
      const appended: Msg[] = []
      const ctx = buildCtx(appended)
      ctx.gateway.gw.request = vi.fn(async () => ({ status: 'interrupted' }))
      const onEvent = createGatewayEventHandler(ctx)

      patchUiState({ sid: 'sess-1' })
      onEvent({ payload: {}, type: 'message.start' } as any)
      onEvent({
        payload: {
          context: 'pre',
          name: 'search',
          todos: [{ content: 'pre-interrupt', id: 'todo-1', status: 'pending' }],
          tool_id: 't-1'
        },
        type: 'tool.start'
      } as any)

      // Pre-interrupt todos should land in turn state.
      expect(getTurnState().todos).toEqual([{ content: 'pre-interrupt', id: 'todo-1', status: 'pending' }])

      turnController.interruptTurn({
        appendMessage: (msg: Msg) => appended.push(msg),
        gw: ctx.gateway.gw,
        sid: 'sess-1',
        sys: ctx.system.sys
      })

      onEvent({ payload: { text: 'still thinking…' }, type: 'reasoning.delta' } as any)
      // Post-interrupt tool.start with a todos payload — must NOT mutate todos.
      onEvent({
        payload: {
          context: 'post',
          name: 'browser',
          todos: [{ content: 'late ghost', id: 'todo-ghost', status: 'pending' }],
          tool_id: 't-2'
        },
        type: 'tool.start'
      } as any)
      // Late tool.generating must NOT push a 'drafting …' line into the trail.
      const trailBefore = getTurnState().turnTrail.length
      onEvent({ payload: { name: 'browser' }, type: 'tool.generating' } as any)
      expect(getTurnState().turnTrail.length).toBe(trailBefore)
      onEvent({ payload: { name: 'browser', preview: 'loading' }, type: 'tool.progress' } as any)
      onEvent({ payload: { summary: 'done', tool_id: 't-2' }, type: 'tool.complete' } as any)
      onEvent({ payload: { text: 'late chunk' }, type: 'message.delta' } as any)

      expect(getTurnState().tools).toEqual([])
      expect(turnController.reasoningText).toBe('')
      expect(turnController.bufRef).toBe('')
      expect(getTurnState().streamPendingTools).toEqual([])
      expect(getTurnState().streamSegments).toEqual([])
      // Stale post-interrupt todos must not have leaked through.
      // (This test does not assert that pre-interrupt todos are cleared —
      // current interrupt path leaves them visible until the next message.)
      expect(getTurnState().todos.find(t => t.content === 'late ghost')).toBeUndefined()

      onEvent({ payload: {}, type: 'message.start' } as any)
      onEvent({ payload: { text: 'fresh' }, type: 'reasoning.delta' } as any)

      expect(turnController.reasoningText).toBe('fresh')
    } finally {
      // Drain pending fake timers BEFORE restoring real timers so a mid-
      // test assertion failure can't leak the interrupt-cooldown setTimeout
      // across test files (the original Copilot concern).
      vi.runAllTimers()
      vi.useRealTimers()
    }
  })

  it('keepBusy interrupt holds busy until the gateway settles and suppresses the cancelled turn’s final_response', () => {
    // Force-send: interrupt holds busy so the drain waits for the real settle
    // instead of racing it (the race duplicated the bubble, leaked a "queued: …"
    // note, and surfaced the cancelled turn's "Operation interrupted…" reply).
    const appended: Msg[] = []
    const ctx = buildCtx(appended)
    ctx.gateway.gw.request = vi.fn(async () => ({ status: 'interrupted' }))
    const onEvent = createGatewayEventHandler(ctx)

    patchUiState({ sid: 'sess-1' })
    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { text: 'thinking…' }, type: 'reasoning.delta' } as any)
    expect(getUiState().busy).toBe(true)

    turnController.interruptTurn(
      { appendMessage: (msg: Msg) => appended.push(msg), gw: ctx.gateway.gw, sid: 'sess-1', sys: ctx.system.sys },
      { keepBusy: true }
    )

    // Held busy: the drain effect keys off busy→false, so it must not fire yet.
    expect(getUiState().busy).toBe(true)

    // The cancelled turn settles with a backend interrupted final_response.
    const before = appended.length
    onEvent({
      payload: { text: 'Operation interrupted: waiting for model response (4.1s elapsed).' },
      type: 'message.complete'
    } as any)

    // Settle flips busy false (the single drain edge) and the backend
    // "Operation interrupted…" line is suppressed (not appended).
    expect(getUiState().busy).toBe(false)
    expect(
      appended.slice(before).some(m => typeof m.text === 'string' && m.text.includes('Operation interrupted'))
    ).toBe(false)
  })

  it('persists an abandoned (timed-out) clarify into the transcript when the clarify tool completes', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    // Backend clarify timed out: the overlay is still live (Python returned an
    // empty answer), and the clarify tool's own tool.complete then fires.
    patchOverlayState({
      clarify: { choices: ['Scope A', 'Scope B'], question: 'How do you want to scope?', requestId: 'req-1' }
    })

    onEvent({ payload: { duration_s: 300, name: 'clarify', tool_id: 'clar-1' }, type: 'tool.complete' } as any)

    const record = appended.find(msg => msg.role === 'system' && msg.text.startsWith('ask How do you want to scope?'))
    expect(record).toBeDefined()
    expect(record?.text).toContain('1. Scope A')
    expect(record?.text).toContain('2. Scope B')
    expect(record?.text).toContain('timed out — no selection')
    // The live overlay is cleared so it doesn't double-render with the record.
    expect(getOverlayState().clarify).toBeNull()
  })

  it('only persists an abandoned clarify once even if tool.complete fires twice', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    patchOverlayState({
      clarify: { choices: ['A'], question: 'Pick?', requestId: 'req-3' }
    })

    onEvent({ payload: { name: 'clarify', tool_id: 'clar-1' }, type: 'tool.complete' } as any)
    // A duplicate clarify tool.complete must not re-persist the same prompt.
    onEvent({ payload: { name: 'clarify', tool_id: 'clar-1' }, type: 'tool.complete' } as any)

    const records = appended.filter(msg => msg.role === 'system' && msg.text.startsWith('ask Pick?'))
    expect(records).toHaveLength(1)
  })

  it('does not flush the clarify overlay when a non-clarify tool completes', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    // A clarify is live, but it's a *different* tool that just completed — the
    // clarify itself is still pending, so we must not persist or clear it.
    patchOverlayState({
      clarify: { choices: ['A', 'B'], question: 'Pick?', requestId: 'req-4' }
    })

    onEvent({ payload: { name: 'search', tool_id: 'tool-1' }, type: 'tool.complete' } as any)

    expect(appended.some(msg => msg.role === 'system' && msg.text.startsWith('ask '))).toBe(false)
    expect(getOverlayState().clarify).not.toBeNull()
  })

  it('does not persist when an answered clarify already cleared the overlay before tool.complete', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    // Answered path (answerClarify) clears the overlay before the agent's
    // tool.complete arrives, so there's nothing live to persist.
    onEvent({ payload: { duration_s: 4.2, name: 'clarify', tool_id: 'clar-1' }, type: 'tool.complete' } as any)

    expect(appended.some(msg => msg.role === 'system' && msg.text.startsWith('ask '))).toBe(false)
  })

  // ── Credits notice (Strategy B) ──────────────────────────────────────
  describe('credits notice', () => {
    it('shows a notice immediately when idle (no turn in flight)', () => {
      const onEvent = createGatewayEventHandler(buildCtx([]))

      onEvent({
        payload: { key: 'credits.depleted', kind: 'sticky', level: 'error', text: '✕ credits exhausted' },
        type: 'notification.show'
      } as any)

      expect(getUiState().notice).toMatchObject({
        key: 'credits.depleted',
        kind: 'sticky',
        level: 'error',
        text: '✕ credits exhausted'
      })
    })

    it('holds a notice arriving mid-turn (busy) and flushes it at message.complete', () => {
      const onEvent = createGatewayEventHandler(buildCtx([]))

      onEvent({ payload: {}, type: 'message.start' } as any)
      expect(getUiState().busy).toBe(true)

      onEvent({
        payload: { key: 'credits.90', kind: 'sticky', level: 'warn', text: '⚠ 90% used' },
        type: 'notification.show'
      } as any)

      // Mid-turn: busy wins, notice is held, not visible yet.
      expect(getUiState().notice).toBeNull()

      onEvent({ payload: { text: 'done' }, type: 'message.complete' } as any)

      // Turn end flushes the held notice.
      expect(getUiState().notice).toMatchObject({ key: 'credits.90', text: '⚠ 90% used' })
    })

    it('flushes a held notice at interruptTurn (turn-end via ctrl-c)', () => {
      vi.useFakeTimers()

      try {
        const ctx = buildCtx([])
        ctx.gateway.gw.request = vi.fn(async () => ({ status: 'interrupted' }))
        const onEvent = createGatewayEventHandler(ctx)

        patchUiState({ sid: 'sess-1' })
        onEvent({ payload: {}, type: 'message.start' } as any)
        onEvent({
          payload: { key: 'credits.depleted', kind: 'sticky', level: 'error', text: '✕ out' },
          type: 'notification.show'
        } as any)
        expect(getUiState().notice).toBeNull()

        turnController.interruptTurn({
          appendMessage: vi.fn(),
          gw: ctx.gateway.gw,
          sid: 'sess-1',
          sys: ctx.system.sys
        })

        expect(getUiState().notice).toMatchObject({ key: 'credits.depleted', text: '✕ out' })
      } finally {
        vi.runAllTimers()
        vi.useRealTimers()
      }
    })

    it('flushes a held notice at recordError (turn-end via error)', () => {
      const onEvent = createGatewayEventHandler(buildCtx([]))

      onEvent({ payload: {}, type: 'message.start' } as any)
      onEvent({
        payload: { key: 'credits.90', kind: 'sticky', level: 'warn', text: '⚠ 90% used' },
        type: 'notification.show'
      } as any)
      expect(getUiState().notice).toBeNull()

      onEvent({ payload: { message: 'boom' }, type: 'error' } as any)

      expect(getUiState().notice).toMatchObject({ key: 'credits.90', text: '⚠ 90% used' })
    })

    it('latest-wins: a second mid-turn notice replaces the first held one', () => {
      const onEvent = createGatewayEventHandler(buildCtx([]))

      onEvent({ payload: {}, type: 'message.start' } as any)
      onEvent({
        payload: { key: 'credits.90', kind: 'sticky', level: 'warn', text: '⚠ 90% used' },
        type: 'notification.show'
      } as any)
      onEvent({
        payload: { key: 'credits.depleted', kind: 'sticky', level: 'error', text: '✕ exhausted' },
        type: 'notification.show'
      } as any)

      onEvent({ payload: { text: 'done' }, type: 'message.complete' } as any)

      // Only the latest held notice surfaces.
      expect(getUiState().notice).toMatchObject({ key: 'credits.depleted', text: '✕ exhausted' })
    })

    it('clears a visible notice only when the clear key matches (no-op otherwise)', () => {
      const onEvent = createGatewayEventHandler(buildCtx([]))

      onEvent({
        payload: { key: 'credits.grant_spent', kind: 'sticky', level: 'warn', text: '⚠ grant spent' },
        type: 'notification.show'
      } as any)
      expect(getUiState().notice).not.toBeNull()

      // Stale/late clear for a DIFFERENT key must not wipe the newer notice.
      onEvent({ payload: { key: 'credits.something_else' }, type: 'notification.clear' } as any)
      expect(getUiState().notice).toMatchObject({ key: 'credits.grant_spent' })

      // Matching key clears.
      onEvent({ payload: { key: 'credits.grant_spent' }, type: 'notification.clear' } as any)
      expect(getUiState().notice).toBeNull()
    })

    it('drops a held pending notice on a matching clear before it can surface', () => {
      const onEvent = createGatewayEventHandler(buildCtx([]))

      onEvent({ payload: {}, type: 'message.start' } as any)
      onEvent({
        payload: { key: 'credits.grant_spent', kind: 'sticky', level: 'warn', text: '⚠ grant spent' },
        type: 'notification.show'
      } as any)
      // Clear arrives mid-turn before the held notice flushes.
      onEvent({ payload: { key: 'credits.grant_spent' }, type: 'notification.clear' } as any)

      onEvent({ payload: { text: 'done' }, type: 'message.complete' } as any)

      // Nothing surfaces — the pending notice was dropped by the matching clear.
      expect(getUiState().notice).toBeNull()
    })

    it('a ttl notice self-expires after ttl_ms when applied while idle', () => {
      vi.useFakeTimers()

      try {
        const onEvent = createGatewayEventHandler(buildCtx([]))

        onEvent({
          payload: { key: 'credits.restored', kind: 'ttl', level: 'success', text: '✓ access restored', ttl_ms: 8000 },
          type: 'notification.show'
        } as any)
        expect(getUiState().notice).toMatchObject({ key: 'credits.restored' })

        vi.advanceTimersByTime(7999)
        expect(getUiState().notice).not.toBeNull()

        vi.advanceTimersByTime(2)
        expect(getUiState().notice).toBeNull()
      } finally {
        vi.useRealTimers()
      }
    })

    it('R3-C2: a ttl notice self-expires even when statusTimer is also armed (timer isolation)', () => {
      // Regression guard for the whole reason `noticeTimer` is a separate
      // timer from `statusTimer`. A concurrent `status.update` (goal path)
      // arms `statusTimer` via restoreStatusAfter; if the two timers shared
      // a slot, clearing statusTimer would cancel the TTL and the notice
      // would never self-expire.
      vi.useFakeTimers()

      try {
        const ctx = buildCtx([])
        const onEvent = createGatewayEventHandler(ctx)

        // 1. While idle, show a ttl notice → applies immediately, arms noticeTimer.
        onEvent({
          payload: { key: 'credits.restored', kind: 'ttl', level: 'success', text: '✓ restored', ttl_ms: 8000 },
          type: 'notification.show'
        } as any)
        expect(getUiState().notice).toMatchObject({ key: 'credits.restored' })

        // 2. A goal status.update arms turnController.statusTimer (via restoreStatusAfter).
        onEvent({
          payload: { kind: 'goal', text: '✓ Goal achieved: some reason' },
          type: 'status.update'
        } as any)
        // statusTimer is now live; notice must still be visible.
        expect(getUiState().notice).toMatchObject({ key: 'credits.restored' })

        // 3. Advance past the TTL — the notice's own dedicated timer fires.
        vi.advanceTimersByTime(8001)

        // 4. Notice self-expired: statusTimer did NOT cancel noticeTimer.
        expect(getUiState().notice).toBeNull()
      } finally {
        vi.runAllTimers()
        vi.useRealTimers()
      }
    })

    it('starts the ttl clock when the notice becomes VISIBLE (at turn end), not on arrival', () => {
      vi.useFakeTimers()

      try {
        const onEvent = createGatewayEventHandler(buildCtx([]))

        onEvent({ payload: {}, type: 'message.start' } as any)
        onEvent({
          payload: { key: 'credits.restored', kind: 'ttl', level: 'success', text: '✓ restored', ttl_ms: 8000 },
          type: 'notification.show'
        } as any)

        // Long busy turn: the TTL must NOT have started while held.
        vi.advanceTimersByTime(10_000)
        expect(getUiState().notice).toBeNull()

        onEvent({ payload: { text: 'done' }, type: 'message.complete' } as any)
        expect(getUiState().notice).toMatchObject({ key: 'credits.restored' })

        // Full 8s starts now (on apply), so it survives nearly that long.
        vi.advanceTimersByTime(7999)
        expect(getUiState().notice).not.toBeNull()
        vi.advanceTimersByTime(2)
        expect(getUiState().notice).toBeNull()
      } finally {
        vi.useRealTimers()
      }
    })

    it('latest-wins cancels a prior ttl timer so it cannot wipe the newer notice', () => {
      vi.useFakeTimers()

      try {
        const onEvent = createGatewayEventHandler(buildCtx([]))

        onEvent({
          payload: { id: 'a', key: 'credits.restored', kind: 'ttl', level: 'success', text: '✓ a', ttl_ms: 5000 },
          type: 'notification.show'
        } as any)

        vi.advanceTimersByTime(4000)

        // A newer sticky arrives before the first's TTL fires.
        onEvent({
          payload: { id: 'b', key: 'credits.depleted', kind: 'sticky', level: 'error', text: '✕ b' },
          type: 'notification.show'
        } as any)
        expect(getUiState().notice).toMatchObject({ id: 'b' })

        // The first notice's stale TTL must NOT clear the newer one.
        vi.advanceTimersByTime(2000)
        expect(getUiState().notice).toMatchObject({ id: 'b', text: '✕ b' })
      } finally {
        vi.useRealTimers()
      }
    })

    it('sticky survives a turn: applied with no pending notice does not clear it', () => {
      const onEvent = createGatewayEventHandler(buildCtx([]))

      // A standing sticky notice from a prior turn.
      onEvent({
        payload: { key: 'credits.depleted', kind: 'sticky', level: 'error', text: '✕ exhausted' },
        type: 'notification.show'
      } as any)
      expect(getUiState().notice).toMatchObject({ key: 'credits.depleted' })

      // A new turn runs with NO new notice arriving.
      onEvent({ payload: {}, type: 'message.start' } as any)
      onEvent({ payload: { text: 'reply' }, type: 'message.complete' } as any)

      // The standing sticky must REappear untouched at turn end.
      expect(getUiState().notice).toMatchObject({ key: 'credits.depleted', text: '✕ exhausted' })
    })

    it('reset()/fullReset() clears pending + timer + visible notice (no cross-session leak)', () => {
      vi.useFakeTimers()

      try {
        const onEvent = createGatewayEventHandler(buildCtx([]))

        // Session A: a visible sticky + a held pending notice mid-turn.
        onEvent({
          payload: { key: 'credits.depleted', kind: 'sticky', level: 'error', text: '✕ A cut' },
          type: 'notification.show'
        } as any)
        onEvent({ payload: {}, type: 'message.start' } as any)
        onEvent({
          payload: { key: 'credits.90', kind: 'sticky', level: 'warn', text: '⚠ A 90%' },
          type: 'notification.show'
        } as any)
        expect(getUiState().notice).toMatchObject({ key: 'credits.depleted' })

        // Session boundary.
        turnController.fullReset()
        expect(getUiState().notice).toBeNull()

        // Session B: a turn ends with nothing held — A's notice must not bleed in.
        onEvent({ payload: {}, type: 'message.start' } as any)
        onEvent({ payload: { text: 'B reply' }, type: 'message.complete' } as any)
        expect(getUiState().notice).toBeNull()
      } finally {
        vi.runAllTimers()
        vi.useRealTimers()
      }
    })

    it('ignores a notification.show with no text', () => {
      const onEvent = createGatewayEventHandler(buildCtx([]))

      onEvent({ payload: { key: 'credits.90', level: 'warn' }, type: 'notification.show' } as any)
      expect(getUiState().notice).toBeNull()
    })
  })

  describe('billing.step_up.verification', () => {
    beforeEach(() => {
      openExternalUrlMock.mockClear()
    })

    it('renders the verification link + code and opens the browser', () => {
      const ctx = buildCtx([])
      const onEvent = createGatewayEventHandler(ctx)

      onEvent({
        payload: { user_code: 'WXYZ-9999', verification_url: 'https://portal.example/device?code=WXYZ' },
        type: 'billing.step_up.verification'
      } as any)

      const printed = (ctx.system.sys as ReturnType<typeof vi.fn>).mock.calls.map(c => c[0]).join('\n')
      expect(printed).toContain('https://portal.example/device?code=WXYZ')
      expect(printed).toContain('WXYZ-9999')
      expect(openExternalUrlMock).toHaveBeenCalledWith('https://portal.example/device?code=WXYZ')
    })

    it('no-ops on a missing verification_url (never opens a browser)', () => {
      const ctx = buildCtx([])
      const onEvent = createGatewayEventHandler(ctx)

      onEvent({ payload: { verification_url: '' }, type: 'billing.step_up.verification' } as any)

      expect(openExternalUrlMock).not.toHaveBeenCalled()
    })
  })
})
