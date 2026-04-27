import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { TUI_SESSION_MODEL_FLAG } from '../domain/slash.js'

describe('createSlashHandler', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
  })

  it('opens the resume picker locally', () => {
    const ctx = buildCtx()

    expect(createSlashHandler(ctx)('/resume')).toBe(true)
    expect(getOverlayState().picker).toBe(true)
  })

  it('treats /provider as a local /model alias', () => {
    const ctx = buildCtx()

    expect(createSlashHandler(ctx)('/provider')).toBe(true)
    expect(getOverlayState().modelPicker).toBe(true)
    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
  })

  it('keeps typed /model switches session-scoped by default', async () => {
    patchUiState({ sid: 'sid-abc' })

    const ctx = buildCtx({
      gateway: {
        ...buildGateway(),
        rpc: vi.fn(() => Promise.resolve({ value: 'x-model' }))
      }
    })

    expect(createSlashHandler(ctx)('/model x-model')).toBe(true)
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('config.set', {
      key: 'model',
      session_id: 'sid-abc',
      value: 'x-model'
    })
  })

  it('honors TUI picker session scope without adding --global', async () => {
    patchUiState({ sid: 'sid-abc' })

    const ctx = buildCtx({
      gateway: {
        ...buildGateway(),
        rpc: vi.fn(() => Promise.resolve({ value: 'anthropic/claude-sonnet-4.6' }))
      }
    })

    expect(
      createSlashHandler(ctx)(`/model anthropic/claude-sonnet-4.6 --provider openrouter ${TUI_SESSION_MODEL_FLAG}`)
    ).toBe(true)
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('config.set', {
      key: 'model',
      session_id: 'sid-abc',
      value: 'anthropic/claude-sonnet-4.6 --provider openrouter'
    })
  })

  it('does not duplicate --global for explicit persistent model switches', () => {
    patchUiState({ sid: 'sid-abc' })
    const ctx = buildCtx()

    createSlashHandler(ctx)('/model x-model --global')
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('config.set', {
      key: 'model',
      session_id: 'sid-abc',
      value: 'x-model --global'
    })
  })

  it('opens the skills hub locally for bare /skills', () => {
    const ctx = buildCtx()

    expect(createSlashHandler(ctx)('/skills')).toBe(true)
    expect(getOverlayState().skillsHub).toBe(true)
    expect(ctx.gateway.rpc).not.toHaveBeenCalled()
    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
  })

  it('routes /skills install <name> to skills.manage without opening overlay', () => {
    const ctx = buildCtx()

    expect(createSlashHandler(ctx)('/skills install foo')).toBe(true)
    expect(getOverlayState().skillsHub).toBe(false)
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('skills.manage', {
      action: 'install',
      query: 'foo'
    })
  })

  it('routes /skills inspect <name> to skills.manage', () => {
    const ctx = buildCtx()

    createSlashHandler(ctx)('/skills inspect my-skill')
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('skills.manage', {
      action: 'inspect',
      query: 'my-skill'
    })
  })

  it('routes /skills search <query> to skills.manage', () => {
    const ctx = buildCtx()

    createSlashHandler(ctx)('/skills search vibe')
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('skills.manage', {
      action: 'search',
      query: 'vibe'
    })
  })

  it('routes /skills browse [page] to skills.manage with a numeric page', () => {
    const ctx = buildCtx()

    createSlashHandler(ctx)('/skills browse 3')
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('skills.manage', {
      action: 'browse',
      page: 3
    })
  })

  it('shows usage for an unknown /skills subcommand', () => {
    const ctx = buildCtx()

    createSlashHandler(ctx)('/skills zzz')
    expect(ctx.gateway.rpc).not.toHaveBeenCalled()
    expect(ctx.transcript.sys).toHaveBeenCalledWith(expect.stringContaining('usage: /skills'))
  })

  it('cycles details mode and persists it', async () => {
    const ctx = buildCtx()

    expect(getUiState().detailsMode).toBe('collapsed')
    expect(createSlashHandler(ctx)('/details toggle')).toBe(true)
    expect(getUiState().detailsMode).toBe('expanded')
    expect(getUiState().detailsModeCommandOverride).toBe(true)
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('config.set', {
      key: 'details_mode',
      value: 'expanded'
    })
    expect(ctx.transcript.sys).toHaveBeenCalledWith('details: expanded')
  })

  it('sets a per-section override and persists it under details_mode.<section>', () => {
    const ctx = buildCtx()

    expect(createSlashHandler(ctx)('/details activity hidden')).toBe(true)
    expect(getUiState().sections.activity).toBe('hidden')
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('config.set', {
      key: 'details_mode.activity',
      value: 'hidden'
    })
    expect(ctx.transcript.sys).toHaveBeenCalledWith('details activity: hidden')
  })

  it('clears a per-section override on /details <section> reset', () => {
    const ctx = buildCtx()
    createSlashHandler(ctx)('/details tools expanded')
    expect(getUiState().sections.tools).toBe('expanded')

    createSlashHandler(ctx)('/details tools reset')
    expect(getUiState().sections.tools).toBeUndefined()
    expect(ctx.gateway.rpc).toHaveBeenLastCalledWith('config.set', {
      key: 'details_mode.tools',
      value: ''
    })
    expect(ctx.transcript.sys).toHaveBeenCalledWith('details tools: reset')
  })

  it('rejects unknown section modes with a usage hint', () => {
    const ctx = buildCtx()
    createSlashHandler(ctx)('/details tools blink')
    expect(getUiState().sections.tools).toBeUndefined()
    expect(ctx.transcript.sys).toHaveBeenCalledWith('usage: /details <section> [hidden|collapsed|expanded|reset]')
  })

  it('shows tool enable usage when names are missing', () => {
    const ctx = buildCtx()

    expect(createSlashHandler(ctx)('/tools enable')).toBe(true)
    expect(ctx.transcript.sys).toHaveBeenNthCalledWith(1, 'usage: /tools enable <name> [name ...]')
    expect(ctx.transcript.sys).toHaveBeenNthCalledWith(2, 'built-in toolset: /tools enable web')
    expect(ctx.transcript.sys).toHaveBeenNthCalledWith(3, 'MCP tool: /tools enable github:create_issue')
  })

  it('drops stale slash.exec output after a newer slash', async () => {
    let resolveLate: (v: { output?: string }) => void
    let slashExecCalls = 0

    const ctx = buildCtx({
      gateway: {
        gw: {
          getLogTail: vi.fn(() => ''),
          request: vi.fn((method: string) => {
            if (method === 'slash.exec') {
              slashExecCalls += 1

              if (slashExecCalls === 1) {
                return new Promise<{ output?: string }>(res => {
                  resolveLate = res
                })
              }

              return Promise.resolve({ output: 'fresh' })
            }

            return Promise.resolve({})
          })
        },
        rpc: vi.fn(() => Promise.resolve({}))
      }
    })

    const h = createSlashHandler(ctx)
    expect(h('/slow')).toBe(true)
    expect(h('/fast')).toBe(true)
    resolveLate!({ output: 'too late' })
    await vi.waitFor(() => {
      expect(ctx.transcript.sys).toHaveBeenCalled()
    })

    expect(ctx.transcript.sys).not.toHaveBeenCalledWith('too late')
  })

  it('dispatches command.dispatch with typed alias', async () => {
    const ctx = buildCtx({
      gateway: {
        gw: {
          getLogTail: vi.fn(() => ''),
          request: vi.fn((method: string) => {
            if (method === 'slash.exec') {
              return Promise.reject(new Error('no'))
            }

            if (method === 'command.dispatch') {
              return Promise.resolve({ type: 'alias', target: 'help' })
            }

            return Promise.resolve({})
          })
        },
        rpc: vi.fn(() => Promise.resolve({}))
      }
    })

    const h = createSlashHandler(ctx)
    expect(h('/zzz')).toBe(true)
    await vi.waitFor(() => {
      expect(ctx.transcript.panel).toHaveBeenCalledWith(expect.any(String), expect.any(Array))
    })
  })

  it('resolves unique local aliases through the catalog', () => {
    const ctx = buildCtx({
      local: {
        catalog: {
          canon: {
            '/h': '/help',
            '/help': '/help'
          }
        }
      }
    })

    expect(createSlashHandler(ctx)('/h')).toBe(true)
    expect(ctx.transcript.panel).toHaveBeenCalledWith(expect.any(String), expect.any(Array))
  })

  it('falls through to command.dispatch for skill commands and sends the message', async () => {
    const skillMessage = 'Use this skill to do X.\n\n## Steps\n1. First step'

    const ctx = buildCtx({
      gateway: {
        gw: {
          getLogTail: vi.fn(() => ''),
          request: vi.fn((method: string) => {
            if (method === 'slash.exec') {
              return Promise.reject(new Error('skill command: use command.dispatch'))
            }

            if (method === 'command.dispatch') {
              return Promise.resolve({ type: 'skill', message: skillMessage, name: 'hermes-agent-dev' })
            }

            return Promise.resolve({})
          })
        },
        rpc: vi.fn(() => Promise.resolve({}))
      }
    })

    const h = createSlashHandler(ctx)
    expect(h('/hermes-agent-dev')).toBe(true)
    await vi.waitFor(() => {
      expect(ctx.transcript.sys).toHaveBeenCalledWith('⚡ loading skill: hermes-agent-dev')
    })
    expect(ctx.transcript.send).toHaveBeenCalledWith(skillMessage)
  })

  it('/history pages the current TUI transcript (user + assistant)', () => {
    const ctx = buildCtx({
      local: {
        ...buildLocal(),
        getHistoryItems: vi.fn(() => [
          { role: 'user', text: 'hello' },
          { role: 'system', text: 'ignore me' },
          { role: 'assistant', text: 'hi there' },
          { role: 'user', text: 'test' }
        ])
      }
    })

    createSlashHandler(ctx)('/history')
    expect(ctx.transcript.page).toHaveBeenCalledTimes(1)

    const [body, title] = ctx.transcript.page.mock.calls[0]!

    expect(title).toBe('History')
    expect(body).toContain('[You #1]')
    expect(body).toContain('hello')
    expect(body).toContain('[Hermes #2]')
    expect(body).toContain('hi there')
    expect(body).toContain('[You #3]')
    expect(body).not.toContain('ignore me')
    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
  })

  it('/history reports empty state without paging', () => {
    const ctx = buildCtx()

    createSlashHandler(ctx)('/history')
    expect(ctx.transcript.page).not.toHaveBeenCalled()
    expect(ctx.transcript.sys).toHaveBeenCalledWith('no conversation yet')
  })

  it('/save forwards to session.save RPC and reports the returned file', async () => {
    patchUiState({ sid: 'sid-abc' })

    const rpc = vi.fn(() => Promise.resolve({ file: '/tmp/hermes_conversation_test.json' }))

    const ctx = buildCtx({
      gateway: { ...buildGateway(), rpc },
      local: {
        ...buildLocal(),
        getHistoryItems: vi.fn(() => [
          { role: 'system', text: 'intro' },
          { role: 'user', text: 'hello' },
          { role: 'assistant', text: 'hi there' }
        ])
      }
    })

    createSlashHandler(ctx)('/save')

    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
    expect(rpc).toHaveBeenCalledWith('session.save', { session_id: 'sid-abc' })

    await vi.waitFor(() => {
      expect(ctx.transcript.sys).toHaveBeenCalledWith('conversation saved to: /tmp/hermes_conversation_test.json')
    })
  })

  it('/save reports empty state without calling the RPC or slash worker', () => {
    const rpc = vi.fn(() => Promise.resolve({}))
    const ctx = buildCtx({ gateway: { ...buildGateway(), rpc } })

    createSlashHandler(ctx)('/save')

    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
    expect(rpc).not.toHaveBeenCalled()
    expect(ctx.transcript.sys).toHaveBeenCalledWith('no conversation yet')
  })

  it('/save without an active session tells the user instead of hitting the RPC', () => {
    // sid stays null (default) but there IS visible conversation
    const rpc = vi.fn(() => Promise.resolve({}))

    const ctx = buildCtx({
      gateway: { ...buildGateway(), rpc },
      local: {
        ...buildLocal(),
        getHistoryItems: vi.fn(() => [{ role: 'user', text: 'hello' }])
      }
    })

    createSlashHandler(ctx)('/save')

    expect(rpc).not.toHaveBeenCalled()
    expect(ctx.transcript.sys).toHaveBeenCalledWith('no active session — nothing to save')
  })

  it('/title <name> uses session.title RPC and bypasses slash.exec', async () => {
    patchUiState({ sid: 'sid-abc' })
    const rpc = vi.fn(() => Promise.resolve({ pending: false, title: 'my title' }))
    const ctx = buildCtx({ gateway: { ...buildGateway(), rpc } })

    createSlashHandler(ctx)('/title my title')

    expect(rpc).toHaveBeenCalledWith('session.title', { session_id: 'sid-abc', title: 'my title' })
    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
    await vi.waitFor(() => {
      expect(ctx.transcript.sys).toHaveBeenCalledWith('session title set: my title')
    })
  })

  it('/title with no args fetches and displays the current title', async () => {
    patchUiState({ sid: 'sid-abc' })
    const rpc = vi.fn(() => Promise.resolve({ title: 'demo title' }))
    const ctx = buildCtx({ gateway: { ...buildGateway(), rpc } })

    createSlashHandler(ctx)('/title')

    expect(rpc).toHaveBeenCalledWith('session.title', { session_id: 'sid-abc' })
    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
    await vi.waitFor(() => {
      expect(ctx.transcript.sys).toHaveBeenCalledWith('title: demo title')
    })
  })
})

const buildCtx = (overrides: Partial<Ctx> = {}): Ctx => ({
  ...overrides,
  slashFlightRef: overrides.slashFlightRef ?? { current: 0 },
  composer: { ...buildComposer(), ...overrides.composer },
  gateway: { ...buildGateway(), ...overrides.gateway },
  local: { ...buildLocal(), ...overrides.local },
  session: { ...buildSession(), ...overrides.session },
  transcript: { ...buildTranscript(), ...overrides.transcript },
  voice: { ...buildVoice(), ...overrides.voice }
})

const buildComposer = () => ({
  enqueue: vi.fn(),
  hasSelection: false,
  paste: vi.fn(),
  queueRef: { current: [] as string[] },
  selection: { copySelection: vi.fn(async () => '') },
  setInput: vi.fn()
})

const buildGateway = () => ({
  gw: {
    getLogTail: vi.fn(() => ''),
    request: vi.fn(() => Promise.resolve({}))
  },
  rpc: vi.fn(() => Promise.resolve({}))
})

const buildLocal = () => ({
  catalog: null,
  getHistoryItems: vi.fn(() => []),
  getLastUserMsg: vi.fn(() => ''),
  maybeWarn: vi.fn()
})

const buildSession = () => ({
  closeSession: vi.fn(() => Promise.resolve(null)),
  die: vi.fn(),
  guardBusySessionSwitch: vi.fn(() => false),
  newSession: vi.fn(),
  resetVisibleHistory: vi.fn(),
  resumeById: vi.fn(),
  setSessionStartedAt: vi.fn()
})

const buildTranscript = () => ({
  page: vi.fn(),
  panel: vi.fn(),
  send: vi.fn(),
  setHistoryItems: vi.fn(),
  sys: vi.fn(),
  trimLastExchange: vi.fn(items => items)
})

const buildVoice = () => ({
  setVoiceEnabled: vi.fn()
})

interface Ctx {
  slashFlightRef: { current: number }
  composer: ReturnType<typeof buildComposer>
  gateway: ReturnType<typeof buildGateway>
  local: ReturnType<typeof buildLocal>
  session: ReturnType<typeof buildSession>
  transcript: ReturnType<typeof buildTranscript>
  voice: ReturnType<typeof buildVoice>
}
