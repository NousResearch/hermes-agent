import { beforeEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { noteCanonicalCompletion } from '../app/slash/canonicalSessionCommands.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'

// pastels §2.5: on the shared (canonical) gateway Ink's /title /undo /retry /usage /tools must use
// the verbs that gateway serves — never the legacy session.title / session.undo / session.usage /
// slash.exec RPCs it answers with unknown_method / unsupported_command.

const flush = () => new Promise(resolve => setImmediate(resolve))
const LEGACY = ['session.title', 'session.undo', 'session.usage', 'slash.exec', 'tools.configure']

function harness() {
  const messages = [
    { role: 'user', content: 'first', row_id: 11 },
    { role: 'assistant', content: 'a1', row_id: 12 },
    { role: 'user', content: 'second question', row_id: 13 },
    { role: 'assistant', content: 'a2', row_id: 14 }
  ]

  const request = vi.fn(async (method: string, params: any) => {
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

    if (method === 'prompt.receipt') {
      return { status: 'terminal', result: { model: 'm', input_tokens: 5, output_tokens: 7, total_tokens: 12, api_calls: 1 } }
    }

    if (method === 'session.info') {
      return { launch_request: { toolsets: ['terminal', 'web'] } }
    }

    if (method === 'config.get') {
      return { value: params.key === 'skin' ? 'default' : 'auto' }
    }

    if (method === 'slash.exec' && params.command === 'status') {
      return { output: 'Session: owner' }
    }

    throw new Error(`unknown_method: ${method}`)
  })

  const sys = vi.fn()
  const send = vi.fn()
  const panel = vi.fn()
  const page = vi.fn()

  const ctx = {
    slashFlightRef: { current: 0 },
    gateway: { gw: { isCanonical: true, request }, rpc: request },
    transcript: { page, panel, send, sys, setHistoryItems: vi.fn(), trimLastExchange: (items: unknown[]) => items },
    local: { getHistoryItems: () => messages, getLastUserMsg: () => 'typed elsewhere', maybeWarn: vi.fn() },
    session: { resumeById: vi.fn() },
    composer: { enqueue: vi.fn() }
  }

  return { page, panel, request, send, slash: createSlashHandler(ctx as any), sys }
}

beforeEach(() => {
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

it('/usage renders the committed turn result and /tools says what the shared gateway lacks', async () => {
  const { panel, request, slash, sys } = harness()

  slash('/usage')
  await flush()
  expect(sys).toHaveBeenCalledWith('no completed turn in this view yet')

  noteCanonicalCompletion('owner', 'adm-1')
  slash('/usage')
  await flush()
  expect(request).toHaveBeenCalledWith('prompt.receipt', { admission_id: 'adm-1', include_result: true, session_id: 'owner' })
  expect(panel).toHaveBeenCalledWith('Usage · last turn', [
    { rows: expect.arrayContaining([['Total tokens', '12'], ['API calls', '1']]) }
  ])
  expect(sys).toHaveBeenCalledWith('session totals are not available on the shared gateway yet')

  slash('/tools')
  await flush()
  expect(sys).toHaveBeenCalledWith('toolsets fixed at session launch: terminal, web')
  expect(sys).toHaveBeenCalledWith('/tools is not available on the shared gateway yet')
  expect(sys.mock.calls.some(([line]) => /request_failed|unsupported_command|unknown_method/.test(line))).toBe(false)
  // Session totals are the owner's reviewed /usage read; no other legacy route.
  expect(
    request.mock.calls.some(
      ([method, params]) => LEGACY.includes(method) && !(method === 'slash.exec' && params.command === 'usage')
    )
  ).toBe(false)
})

it('/status reads through the gateway; /save, /bg and /btw refuse instead of calling sidecar-only RPCs', async () => {
  const { page, request, slash, sys } = harness()

  slash('/status')
  await flush()
  expect(request).toHaveBeenCalledWith('slash.exec', { command: 'status', session_id: 'owner' })
  expect(page).toHaveBeenCalledWith('Session: owner', expect.any(String))

  for (const cmd of ['/save', '/bg check the logs', '/btw what was that']) {
    slash(cmd)
  }

  await flush()

  const methods = request.mock.calls.map(([method]) => method)

  for (const method of ['session.status', 'session.save', 'prompt.background', 'prompt.btw']) {
    expect(methods).not.toContain(method)
  }

  const notices = sys.mock.calls.map(([text]) => String(text))

  for (const name of ['save', 'bg', 'btw']) {
    expect(notices).toContain(`/${name} is not available on the shared gateway yet`)
  }
})

it('/stop is session-scoped; process-global /agents pause, /reload-mcp and /reload refuse on the shared gateway', async () => {
  const { request, slash, sys } = harness()

  for (const cmd of ['/stop', '/agents pause', '/reload-mcp now', '/reload']) {
    slash(cmd)
  }

  await flush()

  // dokterdok N2: a sessionless process.stop was the owner-wide kill_all.
  expect(request).toHaveBeenCalledWith('process.stop', { session_id: 'owner' })

  const methods = request.mock.calls.map(([method]) => method)

  for (const method of ['delegation.pause', 'reload.mcp', 'reload.env']) {
    expect(methods).not.toContain(method)
  }

  const notices = sys.mock.calls.map(([text]) => String(text))

  for (const name of ['agents pause', 'reload-mcp', 'reload']) {
    expect(notices).toContain(`/${name} is not available on the shared gateway yet`)
  }
})

it('setting writes the shared gateway has no verb for say so instead of a bare invalid_params; reads still work', async () => {
  const { request, slash, sys } = harness()

  // dokterdok P3: the canonical config.set accepts only busy / verbose / yolo / model.
  for (const cmd of ['/theme dark', '/skin mono', '/indicator ascii', '/reasoning high', '/fast fast', '/personality pirate', '/skin']) {
    slash(cmd)
  }

  await flush()

  expect(request.mock.calls.map(([method]) => method)).not.toContain('config.set')
  expect(request).toHaveBeenCalledWith('config.get', { key: 'skin' })

  const notices = sys.mock.calls.map(([text]) => String(text))

  for (const name of ['theme', 'skin', 'indicator', 'reasoning', 'fast', 'personality']) {
    expect(notices).toContain(`/${name} is not available on the shared gateway yet`)
  }

  expect(notices.some(line => /skin: default/.test(line))).toBe(true)
})
