import { JsonRpcGatewayError } from '@hermes/shared/json-rpc-channel'
import { beforeEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

// PLG-7: the shared gateway fires `pre_command` for the commands it runs (slash.exec,
// command.dispatch). A registry command Ink answers itself is reported through `command.observe`,
// as the classic CLI reports its own; one Ink forwards is never reported twice.
const flush = () => new Promise(resolve => setTimeout(resolve, 5))

beforeEach(() => {
  resetUiState()
  patchUiState({ sid: 'owner', info: { model: 'm', skills: {}, tools: {} } as any })
})

function harness({ canonical = true, observe }: { canonical?: boolean; observe?: () => Promise<unknown> } = {}) {
  const request = vi.fn(async (method: string) => {
    if (method === 'command.observe') {
      return observe ? observe() : { queued: true }
    }

    if (method === 'slash.exec') {
      return { output: 'ok', type: 'exec' }
    }

    return {}
  })

  const sys = vi.fn()
  const gw = { isCanonical: canonical, request }

  const slash = createSlashHandler({
    composer: { enqueue: vi.fn() },
    gateway: { gw, rpc: request },
    local: { getHistoryItems: () => [], getLastUserMsg: () => '', maybeWarn: vi.fn() },
    session: { resumeById: vi.fn() },
    slashFlightRef: { current: 0 },
    transcript: { page: vi.fn(), panel: vi.fn(), send: vi.fn(), setHistoryItems: vi.fn(), sys, trimLastExchange: (x: unknown[]) => x }
  } as any)

  const observed = () => request.mock.calls.filter(([method]) => method === 'command.observe').map(([, params]) => params)

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
