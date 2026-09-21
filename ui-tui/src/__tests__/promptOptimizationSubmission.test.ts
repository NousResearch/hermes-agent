import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SubmissionOptions } from '../app/interfaces.js'
import { submitPrompt, type SubmitPromptDeps } from '../app/submissionCore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { resolvePromptOptimizationChoice } from '../app/useMainApp.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { PromptOptimizationPreview } from '../types.js'

const PREVIEW: PromptOptimizationPreview = {
  session_key: 'k',
  original: 'original text',
  rewritten: 'rewritten text',
  quality_before: 0.4,
  quality_after: 0.9,
  token_delta_pct: 10,
  model_profile: 'p'
}

// ── Contract A: the overlay choice → submission-options decision ──────────────
describe('resolvePromptOptimizationChoice — flag contract', () => {
  it('accept: submits the rewritten text once with skipOptimization=true', () => {
    const submit = vi.fn()
    const setInput = vi.fn()

    resolvePromptOptimizationChoice('accept', PREVIEW, { setInput, submit, sys: vi.fn() })

    expect(setInput).not.toHaveBeenCalled()
    expect(submit).toHaveBeenCalledTimes(1)
    expect(submit).toHaveBeenCalledWith('rewritten text', { showUserMessage: true, skipOptimization: true })
  })

  it('reject: submits the ORIGINAL text with skipOptimization=true', () => {
    const submit = vi.fn()

    resolvePromptOptimizationChoice('reject', PREVIEW, { setInput: vi.fn(), submit, sys: vi.fn() })

    expect(submit).toHaveBeenCalledTimes(1)
    expect(submit).toHaveBeenCalledWith('original text', { showUserMessage: true, skipOptimization: true })
  })

  it('edit: loads rewritten text into the composer and does NOT submit', () => {
    const submit = vi.fn()
    const setInput = vi.fn()
    const sys = vi.fn()

    resolvePromptOptimizationChoice('edit', PREVIEW, { setInput, submit, sys })

    expect(submit).not.toHaveBeenCalled()
    expect(setInput).toHaveBeenCalledWith('rewritten text')
    expect(sys).toHaveBeenCalledWith('optimised prompt loaded for editing')
  })
})

// ── Contract B: options reach the REAL optimisation gate (submitPrompt) ───────
function makeGateway() {
  const calls: string[] = []

  const gw = {
    request: vi.fn((method: string) => {
      calls.push(method)

      if (method === 'input.detect_drop') {
        return Promise.resolve({ matched: false })
      }

      return Promise.resolve({ status: 'streaming' })
    })
  } as unknown as GatewayClient

  return { calls, gw }
}

function makeDeps(gw: GatewayClient): SubmitPromptDeps {
  return {
    appendMessage: vi.fn(),
    enqueue: vi.fn(),
    expand: (t: string) => t,
    gw,
    maybeGoodVibes: vi.fn(),
    setLastUserMsg: vi.fn(),
    sys: vi.fn()
  }
}

describe('accept flow reaches submitPrompt with skipOptimization=true (no second preview)', () => {
  beforeEach(() => {
    resetUiState()
    patchUiState({ sid: 'sess-1' })
  })

  it('accept path: exactly one prompt.submit, NO prompt.optimize.preview', async () => {
    const { calls, gw } = makeGateway()
    const deps = makeDeps(gw)

    // Wire the resolver's submit to the REAL submitPrompt (the actual gate),
    // exactly as the hook does: options → (showUserMessage, skipOptimization).
    const submit = (value: string, options?: SubmissionOptions) =>
      submitPrompt(
        value,
        deps,
        options?.showUserMessage ?? true,
        undefined,
        options?.skipOptimization ?? false
      )

    resolvePromptOptimizationChoice('accept', PREVIEW, { setInput: vi.fn(), submit, sys: vi.fn() })

    await Promise.resolve()
    await Promise.resolve()

    expect(calls).toContain('prompt.submit')
    expect(calls).not.toContain('prompt.optimize.preview')
    expect(calls).not.toContain('input.detect_drop')
    expect(calls.filter(c => c === 'prompt.submit')).toHaveLength(1)
  })

  it('control: dropping skipOptimization DOES trigger a second prompt.optimize.preview', async () => {
    const { calls, gw } = makeGateway()
    const deps = makeDeps(gw)

    // Simulate the regression: the flag never reaches the gate.
    submitPrompt('rewritten text', deps, true, false)

    await Promise.resolve()
    await Promise.resolve()

    expect(calls).toContain('input.detect_drop')
    expect(calls).toContain('prompt.optimize.preview')
  })
})
