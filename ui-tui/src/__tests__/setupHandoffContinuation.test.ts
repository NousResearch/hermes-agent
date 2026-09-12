import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $freeTierBlocks } from '../app/freeTierGate.js'

import type { RunExternalSetupOptions } from '../app/setupHandoff.js'
import { runExternalSetup } from '../app/setupHandoff.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { FREE_TIER_LIMIT_KEY } from '../content/setup.js'

function options(runtime: Record<string, unknown>) {
  const rpc = vi.fn(async (method: string) => {
    if (method === 'setup.runtime_check') return runtime
    if (method === 'config.set') return { value: runtime.model }
    return {}
  })
  const newSession = vi.fn()
  const sys = vi.fn()
  return {
    args: ['setup', 'model'], done: 'done', launcher: vi.fn(async () => ({ code: 0 })),
    suspend: async (run: () => Promise<void>) => { await run() },
    ctx: { gateway: { rpc }, session: { newSession }, transcript: { sys } }
  }
}

describe('continuation through the existing setup wizard', () => {
  beforeEach(() => {
    resetUiState()
    $freeTierBlocks.set({})
    patchUiState({ sid: 'existing-chat', notice: { key: FREE_TIER_LIMIT_KEY, text: 'Choose a provider.' } })
  })

  it('applies a working configured local route to the existing conversation', async () => {
    const opts = options({ ok: true, free_tier: false, model: 'local-model', provider: 'llamacpp' })
    await runExternalSetup(opts as unknown as RunExternalSetupOptions)
    expect(opts.ctx.gateway.rpc).toHaveBeenCalledWith('config.set', {
      key: 'model', session_id: 'existing-chat', value: 'local-model --provider llamacpp --session'
    })
    expect(opts.ctx.session.newSession).not.toHaveBeenCalled()
    expect(getUiState().sid).toBe('existing-chat')
    expect(getUiState().notice).toBeNull()
  })

  it('keeps the gate when setup exits without changing the free route', async () => {
    const opts = options({ ok: true, free_tier: true, model: 'nous/welcome', provider: 'nous' })
    await runExternalSetup(opts as unknown as RunExternalSetupOptions)
    expect(opts.ctx.gateway.rpc.mock.calls.some(([method]) => method === 'config.set')).toBe(false)
    expect(opts.ctx.session.newSession).not.toHaveBeenCalled()
    expect(getUiState().notice?.key).toBe(FREE_TIER_LIMIT_KEY)
  })
})
