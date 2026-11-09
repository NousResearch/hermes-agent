import { JsonRpcGatewayError } from '@hermes/shared/json-rpc-channel'
import { beforeEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

const flush = () => new Promise(resolve => setTimeout(resolve, 5))

// The shared gateway's /usage shows the route's provider plan limits (a plugin's
// fetch_account_usage: Claude plan limits on DirectSDK) the way the classic CLI does, then the Nous
// section, and the Nous CTA only on a Nous route. session.account_usage is the one source of the
// account lines; billing.state's bars are only for an owner without that read.
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

const CTA = 'Run /subscription to change plan · /topup to add to your balance'

const WINDOW_PLAN = {
  gauges: [{ id: 'day', label: 'Today', left_fraction: 0.88, resets_at_ms: null, scope: 'all', title: 'Today' }],
  kind: 'usage_window',
  plan: 'Nous Window'
}

async function usageLines(answers: Record<string, () => unknown>) {
  const { request, slash, sys } = run({ 'billing.state': () => ({ ok: true, plan_usage: WINDOW_PLAN }), ...answers })

  slash('/usage')
  await flush()

  return { billed: request.mock.calls.some(([method]) => method === 'billing.state'), lines: sys.mock.calls.map(([line]) => String(line)) }
}

it('/usage prints the account lines once, with the Nous CTA and plan bars only where they apply', async () => {
  const OFF_ROUTE = ['Nous Portal balance (a separate account: this session runs on another provider)', 'Top-up credits: $7.75']
  const WINDOW = ['Nous Window', 'Today  [|         ]  12% used', 'Top-up credits: $7.75']
  const stats = { 'session.stats': () => STATS }

  // Another provider with Nous signed in: its limits, then the Nous section, no CTA or Nous bars.
  const claude = await usageLines({ ...stats, 'session.account_usage': () => ({ lines: LIMITS, nous_lines: OFF_ROUTE, nous_route: false }) })
  expect(claude).toEqual({ billed: false, lines: [...LIMITS, ...OFF_ROUTE] })

  // A Nous window plan: the plan name and bars come once, from the account read, then the CTA.
  const nous = await usageLines({ ...stats, 'session.account_usage': () => ({ lines: [], nous_lines: WINDOW, nous_route: true }) })
  expect(nous).toEqual({ billed: false, lines: [...WINDOW, CTA] })

  // A refused read says nothing about the route: no CTA or bars.
  const refused = await usageLines({
    ...stats,
    'session.account_usage': () => {
      throw new JsonRpcGatewayError('launch_credentials_unavailable', { code: 4001, data: { reason: 'launch_credentials_unavailable' } })
    }
  })

  expect(refused).toEqual({ billed: false, lines: [] })

  // An owner without session.account_usage keeps main's CTA and its billing.state bars.
  const older = await usageLines(stats)
  expect(older.lines[0]).toBe(CTA)
  expect(older.lines[1]).toBe('Nous Window')
  expect(older.lines[2]).toMatch(/^Today {2}\[/)
})
