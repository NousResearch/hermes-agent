import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  createSlashHandler,
  LEGACY_SLASH_EXEC_COMMANDS,
  LIVE_SLASH_EXEC_COMMANDS,
  NATIVE_PRODUCT_ROUTE_COMMANDS
} from '../app/createSlashHandler.js'
import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'

describe('createSlashHandler', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
  })

  it('locks the authoritative route boundary lists', () => {
    expect(NATIVE_PRODUCT_ROUTE_COMMANDS).toEqual(['setup', 'skills', 'swarm', 'tools'])
    expect(LIVE_SLASH_EXEC_COMMANDS).toEqual(['handoff', 'init-deep', 'model', 'provider', 'ralph-loop', 'start-work', 'ulw-loop'])
    expect(LEGACY_SLASH_EXEC_COMMANDS).toEqual([
      'agents',
      'browser',
      'config',
      'cron',
      'debug',
      'fast',
      'gquota',
      'history',
      'insights',
      'platforms',
      'plugins',
      'profile',
      'reload',
      'reload-mcp',
      'rollback',
      'save',
      'snapshot',
      'status',
      'stop',
      'title',
      'toolsets'
    ])
  })

  it('opens the native setup wizard locally', () => {
    const ctx = buildCtx()

    expect(createSlashHandler(ctx)('/setup')).toBe(true)
    expect(getOverlayState().setupWizard).toBe(true)
  })

  it('toggles the native swarm surface locally', () => {
    const ctx = buildCtx()

    expect(createSlashHandler(ctx)('/swarm')).toBe(true)
    expect(getOverlayState().swarm).toBe(true)

    expect(createSlashHandler(ctx)('/swarm close')).toBe(true)
    expect(getOverlayState().swarm).toBe(false)
  })

  it('opens the resume picker locally', () => {
    const ctx = buildCtx()

    expect(createSlashHandler(ctx)('/resume')).toBe(true)
    expect(getOverlayState().picker).toBe(true)
  })

  it('cycles details mode and persists it', async () => {
    const ctx = buildCtx()

    expect(getUiState().detailsMode).toBe('collapsed')
    expect(createSlashHandler(ctx)('/details toggle')).toBe(true)
    expect(getUiState().detailsMode).toBe('expanded')
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('config.set', {
      key: 'details_mode',
      value: 'expanded'
    })
    expect(ctx.transcript.sys).toHaveBeenCalledWith('details: expanded')
  })

  it('shows tool enable usage when names are missing', () => {
    const ctx = buildCtx()

    expect(createSlashHandler(ctx)('/tools enable')).toBe(true)
    expect(ctx.transcript.sys).toHaveBeenNthCalledWith(1, 'usage: /tools enable <name> [name ...]')
    expect(ctx.transcript.sys).toHaveBeenNthCalledWith(2, 'built-in toolset: /tools enable web')
    expect(ctx.transcript.sys).toHaveBeenNthCalledWith(3, 'MCP tool: /tools enable github:create_issue')
  })

  it('opens the native tools catalog pager', async () => {
    const ctx = buildCtx({
      gateway: {
        ...buildGateway(),
        rpc: vi.fn((method: string) => {
          if (method === 'tools.catalog') {
            return Promise.resolve({
              mcp_servers: [{ enabled: true, name: 'github' }],
              toolsets: [{ description: 'web tools', enabled: true, kind: 'builtin', name: 'web', title: 'Web' }]
            })
          }

          return Promise.resolve({})
        })
      }
    })

    expect(createSlashHandler(ctx)('/tools')).toBe(true)

    await vi.waitFor(() => {
      expect(ctx.transcript.page).toHaveBeenCalledWith(expect.stringContaining('Tool configuration'), 'Tools')
    })
  })

  it('configures a native tool provider and resets visible history', async () => {
    const ctx = buildCtx({
      gateway: {
        ...buildGateway(),
        rpc: vi.fn((method: string, params?: Record<string, unknown>) => {
          if (method === 'tools.provider.configure') {
            expect(params).toEqual({
              env: { FIRECRAWL_API_KEY: 'sekret' },
              provider: 'firecrawl',
              session_id: null,
              toolset: 'web'
            })

            return Promise.resolve({
              info: { model: 'claude', skills: {}, tools: {} },
              reset: true
            })
          }

          return Promise.resolve({})
        })
      }
    })

    expect(createSlashHandler(ctx)('/tools provider web firecrawl FIRECRAWL_API_KEY=sekret')).toBe(true)

    await vi.waitFor(() => {
      expect(ctx.session.resetVisibleHistory).toHaveBeenCalledWith({ model: 'claude', skills: {}, tools: {} })
    })
    expect(ctx.transcript.sys).toHaveBeenCalledWith('provider configured: web → firecrawl')
  })

  it('routes native skills search through gateway RPC', async () => {
    const ctx = buildCtx({
      gateway: {
        ...buildGateway(),
        rpc: vi.fn((method: string, params?: Record<string, unknown>) => {
          if (method === 'skills.manage') {
            expect(params).toEqual({ action: 'search', query: 'memory tools' })

            return Promise.resolve({
              results: [{ description: 'Persistent memory helpers', identifier: 'official/memory/tools', name: 'memory-tools' }]
            })
          }

          return Promise.resolve({})
        })
      }
    })

    expect(createSlashHandler(ctx)('/skills search memory tools')).toBe(true)

    await vi.waitFor(() => {
      expect(ctx.transcript.sys).toHaveBeenCalledWith(expect.stringContaining('memory-tools'))
    })
  })

  it('drops stale slash.exec output after a newer slash', async () => {
    let resolveLate: (v: { output?: string }) => void
    let slashExecCalls = 0

    const ctx = buildCtx({
      local: {
        ...buildLocal(),
        catalog: {
          canon: {
            '/config': '/config',
            '/profile': '/profile'
          }
        }
      },
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
    expect(h('/config')).toBe(true)
    expect(h('/profile')).toBe(true)
    resolveLate!({ output: 'too late' })
    await vi.waitFor(() => {
      expect(ctx.transcript.sys).toHaveBeenCalledWith('fresh')
    })

    expect(ctx.transcript.sys).not.toHaveBeenCalledWith('too late')
  })

  it('routes cataloged legacy commands through slash.exec only', async () => {
    const ctx = buildCtx({
      local: {
        ...buildLocal(),
        catalog: {
          canon: {
            '/config': '/config'
          }
        }
      },
      gateway: {
        gw: {
          getLogTail: vi.fn(() => ''),
          request: vi.fn((method: string, params?: Record<string, unknown>) => {
            if (method === 'slash.exec') {
              expect(params).toEqual({ command: 'config', session_id: null })

              return Promise.resolve({ output: 'config output' })
            }

            return Promise.resolve({})
          })
        },
        rpc: vi.fn(() => Promise.resolve({}))
      }
    })

    const h = createSlashHandler(ctx)
    expect(h('/config')).toBe(true)

    await vi.waitFor(() => {
      expect(ctx.transcript.sys).toHaveBeenCalledWith('config output')
    })

    expect(ctx.gateway.gw.request).not.toHaveBeenCalledWith('command.dispatch', expect.anything())
  })

  it.each([
    ['/handoff request', 'handoff request', '/handoff started'],
    ['/init-deep investigate', 'init-deep investigate', '/init-deep started'],
    ['/start-work ship it', 'start-work ship it', '/start-work started'],
    ['/ralph-loop close it', 'ralph-loop close it', '/ralph-loop started'],
    ['/ulw-loop finish it', 'ulw-loop finish it', '/ulw-loop started']
  ])('routes live slash commands through slash.exec even without catalog canon: %s', async (input, expectedCommand, expectedOutput) => {
    const ctx = buildCtx({
      gateway: {
        gw: {
          getLogTail: vi.fn(() => ''),
          request: vi.fn((method: string, params?: Record<string, unknown>) => {
            if (method === 'slash.exec') {
              expect(params).toEqual({ command: expectedCommand, session_id: null })

              return Promise.resolve({ output: expectedOutput })
            }

            return Promise.resolve({})
          })
        },
        rpc: vi.fn(() => Promise.resolve({}))
      }
    })

    const h = createSlashHandler(ctx)
    expect(h(input)).toBe(true)

    await vi.waitFor(() => {
      expect(ctx.transcript.sys).toHaveBeenCalledWith(expectedOutput)
    })

    expect(ctx.gateway.gw.request).not.toHaveBeenCalledWith('command.dispatch', expect.anything())
  })

  it('dispatches command.dispatch without probing slash.exec for uncataloged commands', async () => {
    const ctx = buildCtx({
      gateway: {
        gw: {
          getLogTail: vi.fn(() => ''),
          request: vi.fn((method: string) => {
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

    expect(ctx.gateway.gw.request).not.toHaveBeenCalledWith('slash.exec', expect.anything())
  })

  it('promotes /swarm into the authoritative native slash route list', () => {
    const ctx = buildCtx()

    const h = createSlashHandler(ctx)
    expect(h('/swarm')).toBe(true)

    expect(getOverlayState().swarm).toBe(true)
    expect(ctx.gateway.gw.request).not.toHaveBeenCalledWith('command.dispatch', expect.anything())
    expect(ctx.gateway.gw.request).not.toHaveBeenCalledWith('slash.exec', expect.anything())
    expect(ctx.transcript.sys).toHaveBeenCalledWith('swarm surface open')
  })

  it.each([
    ['/handoff request', '/handoff'],
    ['/init-deep investigate', '/init-deep'],
    ['/start-work ship it', '/start-work'],
    ['/ralph-loop close it', '/ralph-loop'],
    ['/ulw-loop finish it', '/ulw-loop']
  ])('does not fall back to command.dispatch when slash.exec reports a busy session: %s', async (input, canon) => {
    const ctx = buildCtx({
      local: {
        ...buildLocal(),
        catalog: {
          canon: {
            [canon]: canon
          }
        }
      },
      gateway: {
        gw: {
          getLogTail: vi.fn(() => ''),
          request: vi.fn((method: string) => {
            if (method === 'slash.exec') {
              return Promise.reject(new Error('session busy'))
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
    expect(h(input)).toBe(true)

    await vi.waitFor(() => {
      expect(ctx.transcript.sys).toHaveBeenCalledWith('error: session busy')
    })

    expect(ctx.gateway.gw.request).not.toHaveBeenCalledWith('command.dispatch', expect.anything())
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
    const sections = ctx.transcript.panel.mock.calls[0]?.[1] ?? []
    expect(JSON.stringify(sections)).toContain('/swarm [open|close|toggle]')
  })

  it('opens native setup from /model when no provider is configured', async () => {
    const ctx = buildCtx({
      gateway: {
        ...buildGateway(),
        rpc: vi.fn((method: string) => {
          if (method === 'setup.status') {
            return Promise.resolve({ provider_configured: false })
          }

          return Promise.resolve({})
        })
      }
    })

    expect(createSlashHandler(ctx)('/model')).toBe(true)

    await vi.waitFor(() => {
      expect(getOverlayState().setupWizard).toBe(true)
    })
  })

  it('routes /provider to the live model command', async () => {
    patchUiState({ sid: 'sid' })

    const ctx = buildCtx({
      gateway: {
        ...buildGateway(),
        rpc: vi.fn((method: string) => {
          if (method === 'setup.status') {
            return Promise.resolve({ provider_configured: true })
          }

          return Promise.resolve({ value: 'claude' })
        })
      }
    })

    expect(createSlashHandler(ctx)('/provider claude')).toBe(true)

    await vi.waitFor(() => {
      expect(ctx.gateway.rpc).toHaveBeenCalledWith('config.set', { key: 'model', session_id: 'sid', value: 'claude' })
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
  selection: { copySelection: vi.fn(() => '') },
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
