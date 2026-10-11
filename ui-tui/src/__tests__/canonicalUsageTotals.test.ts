import { beforeEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { noteCanonicalCompletion } from '../app/slash/canonicalSessionCommands.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

const flush = () => new Promise(resolve => setTimeout(resolve, 5))
const REPORT = 'Session token usage\nModel: reads-model\nAPI calls: 3\nTotal tokens: 1,234'

beforeEach(() => {
  resetUiState()
  patchUiState({ sid: 'owner', info: { model: 'm', skills: {}, tools: {} } as any })
})

// The owner serves `/usage` as a reviewed session read (slash.exec, as the classic CLI uses it):
// Ink on the shared gateway shows those session totals instead of "not available yet".
it('/usage shows the session totals the shared gateway reports, then the last turn', async () => {
  const request = vi.fn(async (method: string, params: any) => {
    if (method === 'slash.exec' && params.command === 'usage') {
      return { output: REPORT }
    }

    if (method === 'prompt.receipt') {
      return { result: { api_calls: 1, input_tokens: 5, model: 'm', output_tokens: 7, total_tokens: 12 } }
    }

    throw new Error(`unknown_method: ${method}`)
  })

  const sys = vi.fn()
  const page = vi.fn()
  const panel = vi.fn()

  const slash = createSlashHandler({
    composer: { enqueue: vi.fn() },
    gateway: { gw: { isCanonical: true, request }, rpc: request },
    local: { getHistoryItems: () => [], getLastUserMsg: () => '', maybeWarn: vi.fn() },
    session: { resumeById: vi.fn() },
    slashFlightRef: { current: 0 },
    transcript: { page, panel, send: vi.fn(), setHistoryItems: vi.fn(), sys, trimLastExchange: (x: unknown[]) => x }
  } as any)

  noteCanonicalCompletion('owner', 'adm-1')
  slash('/usage')
  await flush()

  expect(request).toHaveBeenCalledWith('slash.exec', { command: 'usage', session_id: 'owner' })
  expect(page).toHaveBeenCalledWith(REPORT, 'Usage')
  expect(panel).toHaveBeenCalledWith('Usage · last turn', [{ rows: expect.arrayContaining([['Total tokens', '12']]) }])
  expect(sys).not.toHaveBeenCalledWith('session totals are not available on the shared gateway yet')
})
