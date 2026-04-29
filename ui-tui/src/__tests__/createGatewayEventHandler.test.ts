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

    createGatewayEventHandler(buildCtx(appended))({ payload: { text: streamed }, type: 'thinking.delta' } as any)
    vi.runOnlyPendingTimers()

    expect(getTurnState().reasoning).toBe(streamed)
    expect(getTurnState().reasoningActive).toBe(true)
    expect(getTurnState().reasoningTokens).toBe(estimateTokensRough(streamed))
    vi.useRealTimers()
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

  it('still surfaces terminal turn failures as errors', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { message: 'boom' }, type: 'error' } as any)

    expect(getTurnState().activity).toMatchObject([{ text: 'boom', tone: 'error' }])
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
})
