import { JsonRpcGatewayError } from '@hermes/shared/json-rpc-channel'
import { beforeEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { noteCanonicalCompletion } from '../app/slash/canonicalSessionCommands.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { resetLocale } from '../i18n/runtime.js'

import { activateZh } from './localeFixture.js'

// pastels §2.5: on the shared (canonical) gateway Ink's /title /undo /retry /usage /tools must use
// the verbs that gateway serves — never the legacy session.title / session.undo / session.usage /
// slash.exec RPCs it answers with unknown_method / unsupported_command.

const flush = () => new Promise(resolve => setImmediate(resolve))
const LEGACY = ['session.title', 'session.undo', 'session.usage', 'slash.exec', 'tools.configure']

function harness(rewindRefusal?: Error) {
  let voiceMode = 'off'

  // T9/T10 owner verbs, one answer per method (kept out of the request ladder below).
  const ownerVerb = (method: string, params: any): unknown => {
    const answers: Record<string, () => unknown> = {
      'config.get:full': () => ({ config: { display: { skin: 'ares', streaming: true }, voice: { record_key: 'ctrl+b' } } }),
      'config.get:voice': () => ({ value: voiceMode }),
      'process.stop': () => ({ killed: 0 }),
      'reload.mcp': () =>
        params.confirm
          ? { message: 'MCP servers reconciled (no server changes)', status: 'reloaded' }
          : { message: 'Reloading MCP refreshes every chat of this profile.', status: 'confirm_required' },
      'slash.exec:voice': () => {
        voiceMode = params.command === 'voice on' ? 'voice_only' : params.command === 'voice off' ? 'off' : voiceMode

        return { output: `voice: ${voiceMode}`, type: 'exec' }
      }
    }

    const sub = method === 'config.get' ? params.key : method === 'slash.exec' ? String(params.command).split(' ')[0] : ''

    return (answers[sub ? `${method}:${sub}` : method] ?? (() => undefined))()
  }

  const messages = [
    { role: 'user', content: 'first', row_id: 11 },
    { role: 'assistant', content: 'a1', row_id: 12 },
    { role: 'user', content: 'second question', row_id: 13 },
    { role: 'assistant', content: 'a2', row_id: 14 }
  ]

  const reply = async (method: string, params: any) => {
    if (method === 'session.resume') {
      return { session_id: 'owner', revision: 7, execution_generation: 3, messages }
    }

    if (method === 'session.mutate' && params.operation === 'rename') {
      return { session_id: 'owner', operation: 'rename', revision: 8, title: params.payload.title }
    }

    if (method === 'session.mutate' && params.operation === 'rewind') {
      return {
        session_id: 'owner',
        operation: 'rewind',
        revision: 8,
        execution_generation: 4,
        rewound_count: 2,
        target_message: { role: 'user', content: 'second question' }
      }
    }

    if (method === 'session.mutate' && params.operation === 'reasoning') {
      return { session_id: 'owner', operation: 'reasoning', revision: 8, execution_generation: 4, effort: params.payload.effort }
    }

    if (method === 'session.stats') {
      return {
        api_calls: 2, compression_count: 0, context_length: 128000, context_source: 'provider_usage',
        input_tokens: 200, last_prompt_tokens: 100, model: 'fake-model', output_tokens: 40, prompt_tokens: 200,
        total_tokens: 240
      }
    }

    if (method === 'prompt.receipt') {
      return { status: 'terminal', result: { model: 'm', input_tokens: 5, output_tokens: 7, total_tokens: 12, api_calls: 1 } }
    }

    if (method === 'tools.show') {
      return {
        sections: [
          { name: 'file', tools: [{ name: 'read_file', description: 'Read a text file.' }] },
          { name: 'terminal', tools: [{ name: 'terminal', description: 'Run a shell command.' }] }
        ],
        total: 2
      }
    }

    const owned = ownerVerb(method, params)

    if (owned !== undefined) {
      return owned
    }

    if (method === 'config.get') {
      return { value: params.key === 'skin' ? 'default' : 'auto' }
    }

    if (method === 'config.set') {
      return { key: params.key, value: params.value }
    }

    if (method === 'slash.exec' && params.command === 'status') {
      return { output: 'Session: owner' }
    }

    if (method === 'slash.exec' && params.command === 'personality') {
      return { type: 'exec', output: '🎭 **Available Personalities**' }
    }

    throw new Error(`unknown_method: ${method}`)
  }

  // A refused rewind (the owner's typed busy refusal) replaces only the rewind reply.
  const request = vi.fn(async (method: string, params: any) =>
    rewindRefusal && method === 'session.mutate' && params.operation === 'rewind'
      ? Promise.reject(rewindRefusal)
      : reply(method, params)
  )

  const sys = vi.fn()
  const send = vi.fn()
  const panel = vi.fn()
  const page = vi.fn()

  const voice = { setVoiceEnabled: vi.fn(), setVoiceRecordKey: vi.fn(), setVoiceTts: vi.fn() }

  const ctx = {
    voice,
    slashFlightRef: { current: 0 },
    gateway: { gw: { isCanonical: true, request }, rpc: request },
    transcript: { page, panel, send, sys, setHistoryItems: vi.fn(), trimLastExchange: (items: unknown[]) => items },
    local: { getHistoryItems: () => messages, getLastUserMsg: () => 'typed elsewhere', maybeWarn: vi.fn() },
    session: { resumeById: vi.fn() },
    composer: { enqueue: vi.fn() }
  }

  return { page, panel, request, send, slash: createSlashHandler(ctx as any), sys, voice }
}

it('/voice on|off|status run on the owner and mirror its per-session voice mode, never voice.toggle (T9)', async () => {
  const { request, slash, sys, voice } = harness()

  slash('/voice on')
  await flush()
  await flush()
  expect(request).toHaveBeenCalledWith('slash.exec', { command: 'voice on', session_id: 'owner' })
  expect(request).toHaveBeenCalledWith('config.get', { key: 'voice', session_id: 'owner' })
  expect(voice.setVoiceEnabled).toHaveBeenLastCalledWith(true)
  expect(voice.setVoiceTts).toHaveBeenLastCalledWith(false)

  slash('/voice status')
  await flush()
  await flush()
  expect(voice.setVoiceEnabled).toHaveBeenLastCalledWith(true)

  slash('/voice off')
  await flush()
  await flush()
  expect(voice.setVoiceEnabled).toHaveBeenLastCalledWith(false)

  expect(request.mock.calls.some(([method]) => method === 'voice.toggle')).toBe(false)
  expect(sys.mock.calls.map(([text]) => String(text))).toContain('voice: voice_only')
})

beforeEach(() => {
  resetLocale()
  resetUiState()
  patchUiState({ sid: 'owner', info: { model: 'm', skills: {}, tools: {}, execution_epoch: '1', execution_generation: 3 } as any })
})

it('/title renames and /undo, /retry rewind through session.mutate with the snapshot CAS and row id', async () => {
  const { request, send, slash, sys } = harness()

  slash('/title Probe Renamed')
  await flush()
  expect(request).toHaveBeenCalledWith('session.mutate', {
    session_id: 'owner',
    request_id: expect.any(String),
    expected_revision: 7,
    operation: 'rename',
    payload: { title: 'Probe Renamed' }
  })
  expect(getUiState().sessionTitle).toBe('Probe Renamed')

  for (const command of ['/undo', '/retry']) {
    slash(command)
    await flush()
  }

  const rewinds = request.mock.calls.filter(([method, params]) => method === 'session.mutate' && params.operation === 'rewind')

  expect(rewinds).toHaveLength(2)

  for (const [, params] of rewinds) {
    expect(params).toMatchObject({
      expected_revision: 7,
      expected_generation: 3,
      payload: { target_message_id: 13 }
    })
  }

  expect(sys).toHaveBeenCalledWith('undid 2 messages')
  // Main's /retry resubmits the rewound turn's own text.
  expect(send).toHaveBeenCalledWith('second question')
  expect(request.mock.calls.some(([method]) => LEGACY.includes(method))).toBe(false)
})

it('/undo and /retry while a turn runs print main\'s stop-first text, not the bare reason code', async () => {
  const busy = new JsonRpcGatewayError('session_busy', { code: 4001, data: { reason: 'session_busy' } })
  const { send, slash, sys } = harness(busy)

  for (const command of ['undo', 'retry']) {
    slash(`/${command}`)
    await flush()
    expect(sys).toHaveBeenLastCalledWith(
      `session busy — Hermes is still replying. Stop the current reply first (Stop button, or Ctrl+C in a terminal), then run /${command}.`
    )
  }

  expect(sys.mock.calls.some(([line]) => String(line).includes('error: session_busy'))).toBe(false)
  expect(send).not.toHaveBeenCalled()
})

it('/undo and /retry while a compression holds the session say to wait for it', async () => {
  const compressing = new JsonRpcGatewayError('session_busy', {
    code: 4001,
    data: { field: 'compression', reason: 'session_busy' }
  })

  const { send, slash, sys } = harness(compressing)

  for (const command of ['undo', 'retry']) {
    slash(`/${command}`)
    await flush()
    expect(sys).toHaveBeenLastCalledWith(`session busy — a /compress is running; run /${command} when it finishes.`)
  }

  expect(send).not.toHaveBeenCalled()
})

it("/usage renders main's session totals from session.stats and /tools lists the owner's tools.show", async () => {
  const { page, panel, request, slash, sys } = harness()

  noteCanonicalCompletion('owner', 'adm-1')
  slash('/usage')
  await flush()
  expect(request).toHaveBeenCalledWith('session.stats', { session_id: 'owner' })
  // Main's one Usage panel with session totals and the context line; no "last turn" panel.
  expect(panel).toHaveBeenCalledTimes(1)
  expect(panel).toHaveBeenCalledWith('Usage', [
    {
      rows: [
        ['Model', 'fake-model'],
        ['Input tokens', '200'],
        ['Output tokens', '40'],
        ['Total tokens', '240'],
        ['API calls', '2']
      ]
    },
    { text: expect.stringContaining('100 / 128,000 (0%)') }
  ])
  expect(sys.mock.calls.some(([line]) => /not available on the shared gateway/.test(line))).toBe(false)
  expect(getUiState().usage).toMatchObject({ calls: 2, total: 240, context_max: 128000 })

  slash('/tools')
  await flush()
  // Main's /tools listing (cli_info_mixin.show_tools layout) from the owner's tools.show, in the pager.
  expect(request).toHaveBeenCalledWith('tools.show', { session_id: 'owner' })
  expect(page).toHaveBeenCalledWith(
    [
      '+------------------------------------------------------------------------------+',
      '|                            (^_^)/ Available Tools                            |',
      '+------------------------------------------------------------------------------+',
      '',
      '  [file]',
      '    * read_file            - Read a text file.',
      '',
      '  [terminal]',
      '    * terminal             - Run a shell command.',
      '',
      '  Total: 2 tools  ヽ(^o^)ノ'
    ].join('\n'),
    'Tools'
  )
  expect(sys.mock.calls.some(([line]) => /not available on the shared gateway/.test(line))).toBe(false)
  expect(sys.mock.calls.some(([line]) => /request_failed|unsupported_command|unknown_method/.test(line))).toBe(false)
  expect(request.mock.calls.some(([method]) => LEGACY.includes(method))).toBe(false)
})

it('/tools enable|disable refuses on the shared gateway instead of listing the unchanged tools', async () => {
  const { page, request, slash, sys } = harness()

  slash('/tools disable web')
  slash('/tools enable browser')
  await flush()

  expect(sys.mock.calls.map(([text]) => String(text))).toEqual([
    '/tools disable is not available on the shared gateway yet',
    '/tools enable is not available on the shared gateway yet'
  ])
  expect(page).not.toHaveBeenCalled()
  expect(request.mock.calls.map(([method]) => method)).not.toContain('tools.show')
  expect(request.mock.calls.some(([method]) => LEGACY.includes(method))).toBe(false)
})

it('/status reads through the gateway', async () => {
  const { page, request, slash } = harness()

  slash('/status')
  await flush()
  expect(request).toHaveBeenCalledWith('slash.exec', { command: 'status', session_id: 'owner' })
  expect(page).toHaveBeenCalledWith('Session: owner', expect.any(String))
  expect(request.mock.calls.map(([method]) => method)).not.toContain('session.status')
})

it('/stop is session-scoped; /reload-mcp runs the owner\'s session-scoped reload.mcp (T10)', async () => {
  const { request, slash, sys } = harness()

  for (const cmd of ['/stop', '/reload-mcp', '/reload-mcp now', '/reload-mcp always']) {
    slash(cmd)
    await flush()
  }

  // dokterdok N2: a sessionless process.stop was the owner-wide kill_all.
  expect(request).toHaveBeenCalledWith('process.stop', { session_id: 'owner' })
  // The owner's reload reconciles only this profile's servers; `always` has no owner-side
  // opt-out (the owner always asks), so it confirms once like `now`.
  expect(request).toHaveBeenCalledWith('reload.mcp', { session_id: 'owner' })
  expect(request.mock.calls.filter(([m, p]) => m === 'reload.mcp' && p.confirm === true)).toHaveLength(2)
  expect(request.mock.calls.every(([m, p]) => m !== 'reload.mcp' || !('always' in p))).toBe(true)

  const notices = sys.mock.calls.map(([text]) => String(text))

  expect(notices.some(line => /not available/.test(line))).toBe(false)
  expect(notices).toContain('Reloading MCP refreshes every chat of this profile.')
  expect(notices).toContain('MCP servers reconciled (no server changes)')
})

it('/reasoning --global is the owner\'s /reasoning handler behind config.set; /skin reads still work', async () => {
  const { request, slash, sys } = harness()

  for (const cmd of ['/reasoning high --global', '/skin']) {
    slash(cmd)
  }

  await flush()

  expect(request).toHaveBeenCalledWith('config.set', { key: 'reasoning', scope: 'global', session_id: 'owner', value: 'high' })
  expect(request).toHaveBeenCalledWith('config.get', { key: 'skin' })

  const notices = sys.mock.calls.map(([text]) => String(text))

  expect(notices.some(line => /not available/.test(line))).toBe(false)
  expect(notices.some(line => /skin: default/.test(line))).toBe(true)
})

it('/voice tts refuses by name: the owner never speaks a TUI reply', async () => {
  const { request, slash, sys } = harness()

  slash('/voice tts')
  await flush()

  expect(request.mock.calls.some(([method]) => method === 'slash.exec' || method === 'voice.toggle')).toBe(false)
  expect(sys.mock.calls.map(([text]) => String(text))).toEqual([
    '/voice tts is not available on the shared gateway yet'
  ])
})

it('bare /personality lists through the owner\'s /personality handler', async () => {
  const { request, slash, sys } = harness()

  slash('/personality')
  await flush()

  expect(request).toHaveBeenCalledWith('slash.exec', { command: 'personality', session_id: 'owner' })
  expect(sys.mock.calls.map(([text]) => String(text))).toEqual(['🎭 **Available Personalities**'])
})

it('/personality <name> is the owner\'s config.set like main', async () => {
  const { request, slash, sys } = harness()

  slash('/personality pirate')
  await flush()

  expect(request).toHaveBeenCalledWith('config.set', { key: 'personality', session_id: 'owner', value: 'pirate' })
  expect(sys.mock.calls.map(([text]) => String(text))).toEqual(['personality: pirate'])
})

it('/skin writes the profile through the owner\'s config.set like main (the owner repaints via skin.changed)', async () => {
  const { request, slash, sys } = harness()

  slash('/skin mono')
  await flush()

  expect(request).toHaveBeenCalledWith('config.set', { key: 'skin', value: 'mono' })
  expect(sys.mock.calls.map(([text]) => String(text))).toEqual(['skin → mono'])
})

it('/fast <mode> is the owner\'s session-scoped config.set like main', async () => {
  const { request, slash, sys } = harness()

  slash('/fast fast')
  await flush()

  expect(request).toHaveBeenCalledWith('config.set', { key: 'fast', session_id: 'owner', value: 'fast' })
  expect(sys.mock.calls.map(([text]) => String(text))).toEqual(['fast mode: fast'])
  expect(getUiState().info).toMatchObject({ fast: true, service_tier: 'priority' })
})

it('/theme and /indicator write the profile through the owner\'s config.set like main', async () => {
  const { request, slash, sys } = harness()

  slash('/theme dark')
  await flush()
  slash('/indicator ascii')
  await flush()

  expect(request).toHaveBeenCalledWith('config.set', { key: 'theme', value: 'dark' })
  expect(request).toHaveBeenCalledWith('config.set', { key: 'indicator', value: 'ascii' })

  const notices = sys.mock.calls.map(([text]) => String(text))

  expect(notices).toEqual(expect.arrayContaining(['theme → dark', 'indicator → ascii']))
  expect(notices.some(line => /not available/.test(line))).toBe(false)
  expect(getUiState().indicatorStyle).toBe('ascii')
})

it('session /reasoning <level> is the owner\'s fenced reasoning mutation (B14); display words are its /reasoning handler', async () => {
  const { request, slash, sys } = harness()

  slash('/reasoning high')
  await flush()
  slash('/reasoning show')
  await flush()

  expect(request).toHaveBeenCalledWith('session.mutate', expect.objectContaining({
    session_id: 'owner', operation: 'reasoning', payload: { effort: 'high' }, expected_revision: 7, expected_generation: 3
  }))
  expect(request).toHaveBeenCalledWith('config.set', { key: 'reasoning', session_id: 'owner', value: 'show' })

  const notices = sys.mock.calls.map(([text]) => String(text))

  expect(notices.some(line => /high/.test(line))).toBe(true)
  expect(notices.some(line => /not available/.test(line))).toBe(false)
  expect(getUiState().showReasoning).toBe(true)
})

it('/config pages the owner\'s presentation config (config.get full), never slash.exec (T10)', async () => {
  const { page, request, slash, sys } = harness()

  slash('/config')
  await flush()

  expect(request).toHaveBeenCalledWith('config.get', { key: 'full', session_id: 'owner' })
  expect(request.mock.calls.some(([method]) => method === 'slash.exec')).toBe(false)
  expect(sys.mock.calls.map(([text]) => String(text)).some(line => /not available|unsupported/.test(line))).toBe(false)
  const [text, title] = page.mock.calls[0]!
  expect(title).toBe('Config')
  expect(text).toContain('display:')
  expect(text).toContain('  skin: ares')
  expect(text).toContain('  record_key: ctrl+b')
})

it('/timestamps toggles display.timestamps on the owner (config.set) and the transcript at once (T10)', async () => {
  const { request, slash, sys } = harness()

  slash('/timestamps on')
  await flush()
  expect(getUiState().timestamps).toBe(true)
  expect(request).toHaveBeenCalledWith('config.set', { key: 'timestamps', value: 'on' })

  slash('/ts')
  await flush()
  expect(getUiState().timestamps).toBe(false)
  expect(request).toHaveBeenCalledWith('config.set', { key: 'timestamps', value: 'off' })

  slash('/timestamps status')
  await flush()
  const notices = sys.mock.calls.map(([text]) => String(text))
  expect(notices).toContain('timestamps off')
  expect(notices.some(line => /not available/.test(line))).toBe(false)
  expect(request.mock.calls.some(([method]) => method === 'slash.exec')).toBe(false)

  // A backend that refuses the write reports it instead of a silent `timestamps on`.
  request.mockImplementation(async (method: string) => {
    if (method === 'config.set') {
      throw new Error('unknown config key: timestamps')
    }

    return {}
  })
  slash('/timestamps on')
  await flush()
  expect(sys.mock.calls.map(([text]) => String(text))).toContain('error: unknown config key: timestamps')
})

it('explains profile-wide MCP confirmation in the active language without confirming it', async () => {
  activateZh()
  const { request, slash, sys } = harness()
  slash('/reload-mcp')
  await flush()
  expect(sys).toHaveBeenCalledWith('重载 MCP 会刷新此配置档的所有会话，并可能使提示缓存失效。运行 /reload-mcp now 确认。')
  expect(sys).toHaveBeenCalledWith('Reloading MCP refreshes every chat of this profile.')
  expect(request.mock.calls.filter(([method]) => method === 'reload.mcp')).toEqual([
    ['reload.mcp', { session_id: 'owner' }]
  ])
})
