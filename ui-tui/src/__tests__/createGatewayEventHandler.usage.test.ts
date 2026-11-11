import { describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { getUiState, patchUiState } from '../app/uiStore.js'
import { usageFromTick, ZERO } from '../domain/usage.js'

const ref = <T>(current: T) => ({ current })

const ctx = () =>
  ({
    composer: {
      dequeue: () => undefined,
      queueEditRef: ref<null | number>(null),
      sendQueued: vi.fn(),
      setInput: vi.fn()
    },
    gateway: { gw: { request: vi.fn(async () => null) }, rpc: vi.fn(async () => null) },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: ref(80),
      newSession: vi.fn(),
      resetSession: vi.fn(),
      resumeById: vi.fn(),
      setCatalog: vi.fn()
    },
    submission: { submitRef: { current: vi.fn() } },
    system: { bellOnComplete: false, sys: vi.fn() },
    transcript: { appendMessage: vi.fn(), panel: vi.fn(), setHistoryItems: vi.fn() },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
  }) as any

// The owner's session-cumulative counters (`gateway/session_usage.py::usage_stats`).
const STATS = {
  api_calls: 2,
  avg_latency_s: 0.14,
  cache_read_tokens: 0,
  compression_count: 1,
  context_length: 128000,
  context_source: 'provider_usage',
  input_tokens: 100,
  last_prompt_tokens: 900,
  model: 'fake-model',
  output_tokens: 40,
  prompt_tokens: 100,
  tokens_per_s: 229.4,
  total_tokens: 140
}

describe('canonical session.usage tick', () => {
  it('maps the top-level session.stats fields onto the status-bar usage (FINDINGS row 5)', () => {
    patchUiState({ sid: 'sess-1', usage: ZERO })
    const onEvent = createGatewayEventHandler(ctx())

    onEvent({ payload: STATS, session_id: 'sess-1', type: 'session.usage' } as any)

    expect(getUiState().usage).toMatchObject({
      avg_latency_s: 0.1,
      avg_tps: 229.4,
      calls: 2,
      compressions: 1,
      context_max: 128000,
      context_percent: 1,
      context_used: 900,
      input: 100,
      output: 40,
      total: 140
    })
  })

  it("passes main's sidecar `{usage}` form through and ignores an empty frame", () => {
    expect(usageFromTick({ usage: { calls: 3 } })).toEqual({ calls: 3 })
    expect(usageFromTick({})).toBeUndefined()
  })

  it('reports cache hit and omits absent figures', () => {
    const usage = usageFromTick({ api_calls: 1, cache_read_tokens: 50, prompt_tokens: 100, total_tokens: 100 })!

    expect(usage.cache_hit_pct).toBe(50)
    expect(usage).not.toHaveProperty('context_max')
    expect(usage).not.toHaveProperty('avg_tps')
  })

  it('maps a partial tick to only the figures it carries', () => {
    expect(usageFromTick({ input_tokens: 1200, output_tokens: 80 })).toEqual({ input: 1200, output: 80 })
    expect(usageFromTick({ api_calls: 3 })).toEqual({ calls: 3 })
    expect(usageFromTick({ context_length: 10000, last_prompt_tokens: 4200 })).toMatchObject({
      context_max: 10000,
      context_percent: 42,
      context_used: 4200
    })
    expect(usageFromTick({ input_tokens: 0, prompt_tokens: 1000 })).toEqual({ input: 1000, prompt: 1000 })
    expect(usageFromTick({ model: 'only-a-model' })).toBeUndefined()
  })
})
