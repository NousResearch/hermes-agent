import { beforeEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { noteCanonicalCompletion } from '../app/slash/canonicalSessionCommands.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

const flush = () => new Promise(resolve => setTimeout(resolve, 5))

// The owner's session-cumulative counters (`gateway/session_usage.py::usage_stats`).
const STATS = {
  api_calls: 11, compression_count: 1, context_length: 128000, context_source: 'provider_usage', input_tokens: 1100,
  last_prompt_tokens: 900, model: 'reads-model', output_tokens: 134, prompt_tokens: 1100, total_tokens: 1234
}

beforeEach(() => {
  resetUiState()
  patchUiState({ sid: 'owner', info: { model: 'm', skills: {}, tools: {} } as any })
})

function harness(stats: Record<string, unknown>) {
  const request = vi.fn(async (method: string) => {
    if (method === 'session.stats') {
      return stats
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

  return { page, panel, request, slash, sys }
}

// FINDINGS rows 5/18: main's /usage is ONE "Usage" panel of session totals (all API calls of the
// session, the context line, compressions) — not the messaging gateway's markdown page and not a
// "last turn" panel with a fabricated $0.0000 cost.
it("/usage shows main's session totals panel from the owner's session.stats", async () => {
  const { page, panel, request, slash, sys } = harness(STATS)

  noteCanonicalCompletion('owner', 'adm-1')
  slash('/usage')
  await flush()

  expect(request).toHaveBeenCalledWith('session.stats', { session_id: 'owner' })
  expect(request).not.toHaveBeenCalledWith('slash.exec', expect.anything())
  expect(request).not.toHaveBeenCalledWith('prompt.receipt', expect.anything())
  expect(page).not.toHaveBeenCalled()
  expect(panel).toHaveBeenCalledTimes(1)
  expect(panel).toHaveBeenCalledWith('Usage', [
    {
      rows: [
        ['Model', 'reads-model'],
        ['Input tokens', '1,100'],
        ['Output tokens', '134'],
        ['Total tokens', '1,234'],
        ['API calls', '11']
      ]
    },
    { text: expect.stringContaining('900 / 128,000 (1%)') },
    { text: expect.stringContaining('1') }
  ])
  expect(sys).not.toHaveBeenCalledWith('session totals are not available on the shared gateway yet')
})

it('/usage before any API call says so, as main did', async () => {
  const { panel, slash, sys } = harness({ ...STATS, api_calls: 0, total_tokens: 0 })

  slash('/usage')
  await flush()

  expect(panel).not.toHaveBeenCalled()
  expect(sys.mock.calls[0]?.[0]).toMatch(/no API calls/i)
})

it('/usage on a usage-window plan adds the allowance bars after the totals', async () => {
  const request = vi.fn(async (method: string) => {
    if (method === 'session.stats') {
      return STATS
    }

    if (method === 'billing.state') {
      return {
        ok: true,
        logged_in: true,
        plan_usage: {
          kind: 'usage_window',
          plan: 'Window Plan',
          gauges: [
            {
              label: 'day',
              title: 'Today',
              left_fraction: 0.69,
              style: 'green',
              resets_at_ms: Date.now() + 5 * 3_600_000 + 60_000
            },
            { label: 'top_tier', title: 'Large models (monthly)', left_fraction: 0, style: 'red', scope: 'top_tier' }
          ]
        }
      }
    }

    throw new Error(`unknown_method: ${method}`)
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

  slash('/usage')
  await flush()

  const lines = sys.mock.calls.map(([text]) => String(text))
  expect(lines).toContain('Window Plan')
  expect(lines).toContain('Today                   [|||       ]  31% used  ·  resets in 5h')
  expect(lines).toContain('Large models (monthly)  [||||||||||]  100% used')
})

it('/usage on any other account adds nothing (the billing state carries no plan_usage)', async () => {
  const { sys, slash } = harness(STATS)

  slash('/usage')
  await flush()

  expect(sys.mock.calls.map(([text]) => String(text))).toEqual([
    'Run /subscription to change plan · /topup to add to your balance'
  ])
})
