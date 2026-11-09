import { beforeEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

const flush = () => new Promise(resolve => setTimeout(resolve, 5))

// The shared gateway's /usage shows the route's provider plan limits (a plugin's
// fetch_account_usage: Claude plan limits on DirectSDK) the way the classic CLI does.
const STATS = {
  api_calls: 2,
  compression_count: 0,
  context_length: 128000,
  context_source: 'provider_usage',
  input_tokens: 200,
  last_prompt_tokens: 100,
  model: 'm',
  output_tokens: 40,
  prompt_tokens: 200,
  total_tokens: 240
}

const LIMITS = ['📈 Claude plan limits', 'Current session: 99% remaining (1% used)']

beforeEach(() => {
  resetUiState()
  patchUiState({ sid: 'owner', info: { model: 'm', skills: {}, tools: {} } as any })
})

function run(answers: Record<string, () => unknown>) {
  const request = vi.fn(async (method: string) => {
    const answer = answers[method]

    if (!answer) {
      throw new Error(`unknown_method: ${method}`)
    }

    return answer()
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

  return { request, slash, sys }
}

it('/usage prints the plan limits from session.account_usage before the CTA', async () => {
  const { request, slash, sys } = run({
    'session.account_usage': () => ({ lines: LIMITS }),
    'session.stats': () => STATS
  })

  slash('/usage')
  await flush()

  expect(request).toHaveBeenCalledWith('session.account_usage', { session_id: 'owner' })
  expect(sys.mock.calls.map(([line]) => String(line))).toEqual([
    ...LIMITS,
    'Run /subscription to change plan · /topup to add to your balance'
  ])
})

it('/usage on an owner without session.account_usage keeps the totals and the CTA', async () => {
  const { slash, sys } = run({ 'session.stats': () => STATS })

  slash('/usage')
  await flush()

  expect(sys.mock.calls.map(([line]) => String(line))).toEqual([
    'Run /subscription to change plan · /topup to add to your balance'
  ])
})
