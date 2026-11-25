import { JsonRpcGatewayError } from '@hermes/shared/json-rpc-channel'
import { beforeEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

// PLG-7: the shared gateway fires `pre_command` for the commands it runs (slash.exec,
// command.dispatch). A registry command Ink runs itself is reported through `command.observe` once
// it ran, as the classic CLI reports its own; a refused one is not, and a forwarded one never twice.
const flush = () => new Promise(resolve => setTimeout(resolve, 5))

beforeEach(() => {
  resetUiState()
  patchUiState({ sid: 'owner', info: { model: 'm', skills: {}, tools: {} } as any })
})

function harness({
  busy = false,
  canonical = true,
  mutate,
  observe
}: {
  busy?: boolean
  canonical?: boolean
  mutate?: () => Promise<unknown>
  observe?: () => Promise<unknown>
} = {}) {
  const request = vi.fn(async (method: string) => {
    if (method === 'command.observe') {
      return observe ? observe() : { queued: true }
    }

    if (method === 'slash.exec') {
      return { output: 'ok', type: 'exec' }
    }

    if (method === 'session.resume') {
      return { execution_generation: 1, revision: 1 }
    }

    if (method === 'session.mutate') {
      return mutate ? mutate() : { effort: 'high', execution_generation: 1 }
    }

    return {}
  })

  const sys = vi.fn()
  const gw = { isCanonical: canonical, request }

  const slash = createSlashHandler({
    composer: { enqueue: vi.fn() },
    gateway: { gw, rpc: request },
    local: { getHistoryItems: () => [], getLastUserMsg: () => '', maybeWarn: vi.fn() },
    session: { guardBusySessionSwitch: vi.fn(() => busy), newSession: vi.fn(), resumeById: vi.fn() },
    slashFlightRef: { current: 0 },
    transcript: {
      page: vi.fn(),
      panel: vi.fn(),
      send: vi.fn(),
      setHistoryItems: vi.fn(),
      sys,
      trimLastExchange: (x: unknown[]) => x
    }
  } as any)

  const observed = () =>
    request.mock.calls.filter(([method]) => method === 'command.observe').map(([, params]) => params)

  return { gw, observed, request, slash, sys }
}

it('reports a registry command Ink answers itself, as typed, once', async () => {
  const { gw, observed, request, slash } = harness()

  slash('/history')
  slash('/usage')
  await flush()

  expect(gw.request).toBe(request) // the same client, its own request restored

  expect(observed()).toEqual([
    { arg: '', name: 'history', session_id: 'owner' },
    { arg: '', name: 'usage', session_id: 'owner' }
  ])
})

it('reports /help, a bare /reasoning and /tools list, each once', async () => {
  const { observed, slash } = harness()

  slash('/help')
  slash('/reasoning')
  slash('/tools list')
  await flush()

  expect(observed()).toEqual([
    { arg: '', name: 'help', session_id: 'owner' },
    { arg: '', name: 'reasoning', session_id: 'owner' },
    { arg: 'list', name: 'tools', session_id: 'owner' }
  ])
})

it('reports nothing for /tools enable or disable, which the shared gateway refuses', async () => {
  const tools = harness()

  tools.slash('/tools disable web')
  tools.slash('/tools enable web')
  await flush()
  expect(tools.sys).toHaveBeenCalled() // the "not available" line
  expect(tools.observed()).toEqual([])
})

it('reports nothing for a /new the busy guard rejects', async () => {
  const busy = harness({ busy: true })

  busy.slash('/new')
  await flush()
  expect(busy.observed()).toEqual([])

  patchUiState({ destructiveSlashConfirm: false })
  const idle = harness()

  idle.slash('/new')
  await flush()
  expect(idle.observed()).toEqual([{ arg: '', name: 'new', session_id: 'owner' }])
})

it('reports a gateway-backed /reasoning level only once the gateway accepts it', async () => {
  const refused = harness({
    mutate: () => Promise.reject(new JsonRpcGatewayError('busy', { code: 4009, data: { reason: 'busy' } }))
  })

  refused.slash('/reasoning high')
  await flush()
  expect(refused.request).toHaveBeenCalledWith('session.mutate', expect.objectContaining({ operation: 'reasoning' }))
  expect(refused.observed()).toEqual([])

  const accepted = harness()

  accepted.slash('/reasoning high')
  await flush()
  expect(accepted.observed()).toEqual([{ arg: 'high', name: 'reasoning', session_id: 'owner' }])
})

it('never reports a command Ink sends to the gateway, which fires pre_command itself', async () => {
  const { observed, request, slash } = harness()

  slash('/status')
  slash('/plugins list')
  slash('/no-such-local-command')
  await flush()

  expect(request).toHaveBeenCalledWith('slash.exec', expect.objectContaining({ command: 'status' }))
  expect(request).toHaveBeenCalledWith('slash.exec', expect.objectContaining({ command: 'plugins list' }))
  expect(observed()).toEqual([])
})

it('reports nothing off the shared gateway, for programmatic calls, or with no session', async () => {
  const legacy = harness({ canonical: false })

  legacy.slash('/history')
  await flush()
  expect(legacy.observed()).toEqual([])

  const programmatic = harness()

  programmatic.slash('/history', undefined, false)
  await flush()
  expect(programmatic.observed()).toEqual([])

  patchUiState({ sid: null })
  const fresh = harness()

  fresh.slash('/history')
  await flush()
  expect(fresh.observed()).toEqual([])
})

it('an older gateway without command.observe shows nothing and is not asked again', async () => {
  const older = harness({
    observe: () => Promise.reject(new JsonRpcGatewayError('unknown_method: command.observe', { code: -32601 }))
  })

  older.slash('/history')
  await flush()
  older.slash('/usage')
  await flush()

  expect(older.observed()).toHaveLength(1)
  expect(older.sys).not.toHaveBeenCalledWith(expect.stringMatching(/observe|unknown_method/))
})
