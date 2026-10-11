import { beforeEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

const flush = () => new Promise(resolve => setImmediate(resolve))

beforeEach(() => {
  resetUiState()
  patchUiState({ sid: 'owner' })
})

// The rollback.* handlers are sidecar live-session RPCs: on the shared gateway they look the
// owner's session id up in the sidecar's own map and always answer "session not found".
it('/rollback says it is not available on the shared gateway instead of "session not found"', async () => {
  const request = vi.fn(async (method: string) => {
    if (method.startsWith('rollback.')) {
      throw new Error('session not found')
    }

    return {}
  })

  const sys = vi.fn()

  const slash = createSlashHandler({
    composer: { enqueue: vi.fn() },
    gateway: { gw: { isCanonical: true, request }, rpc: request },
    local: { getHistoryItems: () => [], getLastUserMsg: () => '', maybeWarn: vi.fn() },
    session: { resumeById: vi.fn() },
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

  for (const cmd of ['/rollback', '/rollback diff abc123', '/rollback abc123 src/a.py']) {
    slash(cmd)
  }

  await flush()

  expect(request.mock.calls.map(([method]) => method).filter(method => method.startsWith('rollback.'))).toEqual([])
  expect(sys.mock.calls.map(([text]) => String(text))).toEqual(
    Array(3).fill('/rollback is not available on the shared gateway yet')
  )
})
