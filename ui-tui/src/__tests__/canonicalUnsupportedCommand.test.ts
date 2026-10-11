import { JsonRpcGatewayError } from '@hermes/shared/json-rpc-channel'
import { beforeEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

// The shared owner's slash.exec runs only its reviewed reads and refuses every other command with
// `unsupported_command` (gateway/session_commands.py). Every Ink path that forwards a command there
// (the generic fallback and the explicit slash-worker commands) must name the refusal like the
// classic CLI, never print the bare wire reason.
const flush = () => new Promise(resolve => setImmediate(resolve))

beforeEach(() => {
  resetUiState()
  patchUiState({ sid: 'owner' })
})

it.each([
  ['/config', 'config'],
  ['/plugins enable foo', 'plugins'],
  ['/pet toggle', 'pet']
])('%s refused by the shared gateway names the command', async (cmd, name) => {
  const request = vi.fn(async (method: string) => {
    if (method === 'slash.exec' || method === 'command.dispatch') {
      throw new JsonRpcGatewayError('unsupported_command', { code: 4001, data: { reason: 'unsupported_command' } })
    }

    return {}
  })

  const sys = vi.fn()

  createSlashHandler({
    composer: { enqueue: vi.fn() },
    gateway: { gw: { isCanonical: true, request }, rpc: request },
    local: { catalog: { canon: { [`/${name}`]: `/${name}` } }, getHistoryItems: () => [], getLastUserMsg: () => '' },
    session: {},
    slashFlightRef: { current: 0 },
    transcript: { page: vi.fn(), panel: vi.fn(), send: vi.fn(), setHistoryItems: vi.fn(), sys }
  } as never)(cmd)
  await flush()

  expect(sys.mock.calls.map(([line]) => String(line))).toEqual([`/${name} is not available on the shared gateway yet`])
})
