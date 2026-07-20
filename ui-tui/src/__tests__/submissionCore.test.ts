import { beforeEach, describe, expect, it, vi } from 'vitest'

import { isSessionBusyError, markSubmitting, submitPrompt, type SubmitPromptDeps } from '../app/submissionCore.js'
import { turnController } from '../app/turnController.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import type { GatewayClient } from '../gatewayClient.js'

function deferred<T = unknown>() {
  let reject!: (reason?: unknown) => void
  let resolve!: (value: T) => void

  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, reject, resolve }
}

const flushPromises = async () => {
  await Promise.resolve()
  await Promise.resolve()
}

// A gateway double whose `input.detect_drop` resolution we control, so we can
// observe UI state DURING the async gap — the exact window the queue-mode race
// lived in.
function makeDeferredGateway() {
  let resolveDrop: (v: unknown) => void = () => {}

  const dropPromise = new Promise(res => {
    resolveDrop = res
  })

  const calls: string[] = []

  const gw = {
    request: vi.fn((method: string) => {
      calls.push(method)

      if (method === 'input.detect_drop') {
        return dropPromise
      }

      // prompt.submit et al: resolve immediately with a success shape.
      return Promise.resolve({ status: 'streaming' })
    })
  } as unknown as GatewayClient

  return { calls, gw, resolveDrop: (v: unknown = { matched: false }) => resolveDrop(v) }
}

function makeDeps(gw: GatewayClient, over: Partial<SubmitPromptDeps> = {}): SubmitPromptDeps {
  return {
    appendMessage: vi.fn(),
    enqueue: vi.fn(),
    expand: (t: string) => t,
    gw,
    setLastUserMsg: vi.fn(),
    sys: vi.fn(),
    ...over
  }
}

function makeLifecycleGateway() {
  const calls: Array<{ method: string; params?: Record<string, unknown> }> = []
  const drops: ReturnType<typeof deferred>[] = []
  const submits: ReturnType<typeof deferred>[] = []

  const gw = {
    request: vi.fn((method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })
      const pending = deferred()

      if (method === 'input.detect_drop') {
        drops.push(pending)
      } else if (method === 'prompt.submit') {
        submits.push(pending)
      }

      return pending.promise
    })
  } as unknown as GatewayClient

  return { calls, drops, gw, submits }
}

describe('submissionCore.submitPrompt — synchronous busy (queue-race fix)', () => {
  beforeEach(() => {
    turnController.reset()
    resetUiState()
    patchUiState({ sid: 'sess-1' })
  })

  it('flips busy=true SYNCHRONOUSLY, before input.detect_drop resolves', async () => {
    const { gw, resolveDrop } = makeDeferredGateway()

    expect(getUiState().busy).toBe(false)

    submitPrompt('hello', makeDeps(gw))

    // The critical invariant: busy is already true even though the
    // detect_drop RPC has NOT resolved yet. This is what makes a second,
    // rapid submit take the local-enqueue branch instead of racing a second
    // prompt.submit onto the backend.
    expect(getUiState().busy).toBe(true)
    expect(getUiState().status).toBe('running…')

    resolveDrop()
    await flushPromises()
  })

  it('regression: two back-to-back sends — the SECOND sees busy=true in the gap', async () => {
    const { gw, resolveDrop } = makeDeferredGateway()

    // Emulate dispatchSubmission's routing decision: it sends only when
    // busy===false, otherwise it would enqueue. We assert the state the
    // router reads, which is the real regression.
    submitPrompt('first message', makeDeps(gw))

    // Before the fix, busy was still false here (set only inside detect_drop's
    // .then), so a second Enter would wrongly route into send() again.
    const busyWhenSecondArrives = getUiState().busy
    expect(busyWhenSecondArrives).toBe(true)

    resolveDrop()
    await Promise.resolve()
  })

  it('does not submit when there is no session, and does not mark busy', () => {
    resetUiState() // sid: null
    const { gw, calls } = makeDeferredGateway()
    const sys = vi.fn()

    submitPrompt('hello', makeDeps(gw, { sys }))

    expect(getUiState().busy).toBe(false)
    expect(sys).toHaveBeenCalledWith('session not ready yet')
    expect(calls).not.toContain('input.detect_drop')
  })

  it('after detect_drop resolves (no file), it issues prompt.submit', async () => {
    const { calls, gw, resolveDrop } = makeDeferredGateway()

    submitPrompt('hi there', makeDeps(gw))
    expect(calls).toEqual(['input.detect_drop'])

    resolveDrop({ matched: false })
    await flushPromises()

    expect(calls).toContain('prompt.submit')
  })
})

describe('submissionCore.submitPrompt — asynchronous ownership', () => {
  beforeEach(() => {
    turnController.reset()
    resetUiState()
    patchUiState({ sid: 'sess-1' })
  })

  it('drops an old detect-drop completion after interrupt', async () => {
    const { drops, gw, submits } = makeLifecycleGateway()
    const appendMessage = vi.fn()

    submitPrompt('old prompt', makeDeps(gw, { appendMessage }))

    turnController.interruptTurn({
      appendMessage: vi.fn(),
      gw: { request: vi.fn(() => Promise.resolve({})) },
      sid: 'sess-1',
      sys: vi.fn()
    })
    drops[0]!.reject(new Error('old detect failure'))
    await flushPromises()

    expect(submits).toHaveLength(0)
    expect(appendMessage).not.toHaveBeenCalled()
  })

  it('does not resurrect an old preflight in a switched session after a newer submit', async () => {
    const { calls, drops, gw, submits } = makeLifecycleGateway()
    const appendMessage = vi.fn()
    const deps = makeDeps(gw, { appendMessage })

    submitPrompt('old prompt', deps)
    patchUiState({ sid: 'sess-2' })
    submitPrompt('new prompt', deps)

    drops[0]!.resolve({ matched: false })
    await flushPromises()
    expect(submits).toHaveLength(0)

    drops[1]!.resolve({ matched: false })
    await flushPromises()

    expect(submits).toHaveLength(1)
    expect(calls.filter(call => call.method === 'prompt.submit')).toEqual([
      { method: 'prompt.submit', params: { session_id: 'sess-2', text: 'new prompt' } }
    ])
    expect(appendMessage).toHaveBeenCalledTimes(1)
    expect(appendMessage).toHaveBeenCalledWith({ role: 'user', text: 'new prompt' })

    submits[0]!.resolve({ status: 'streaming' })
    await flushPromises()
  })

  it('ignores a stale prompt rejection while a newer prompt owns busy state', async () => {
    const { drops, gw, submits } = makeLifecycleGateway()
    const enqueue = vi.fn()
    const sys = vi.fn()
    const deps = makeDeps(gw, { enqueue, sys })

    submitPrompt('old prompt', deps)
    drops[0]!.resolve({ matched: false })
    await flushPromises()

    submitPrompt('new prompt', deps)
    drops[1]!.resolve({ matched: false })
    await flushPromises()

    submits[0]!.reject(new Error('old failure'))
    await flushPromises()

    expect(getUiState()).toMatchObject({ busy: true, sid: 'sess-1', status: 'running…' })
    expect(enqueue).not.toHaveBeenCalled()
    expect(sys).not.toHaveBeenCalled()

    submits[1]!.resolve({ status: 'streaming' })
    await flushPromises()
  })

  it.each([
    {
      busy: false,
      error: new Error('provider failed'),
      expectedStatus: 'ready',
      expectedSys: 'error: provider failed',
      queued: false
    },
    {
      busy: true,
      error: new Error('session busy'),
      expectedStatus: 'queued for next turn',
      expectedSys: 'queued: "current prompt"',
      queued: true
    }
  ])('lets the current rejection perform its expected error/queue behavior', async scenario => {
    const { drops, gw, submits } = makeLifecycleGateway()
    const enqueue = vi.fn()
    const sys = vi.fn()

    submitPrompt('current prompt', makeDeps(gw, { enqueue, sys }))
    drops[0]!.resolve({ matched: false })
    await flushPromises()
    submits[0]!.reject(scenario.error)
    await flushPromises()

    expect(getUiState()).toMatchObject({ busy: scenario.busy, status: scenario.expectedStatus })
    expect(sys).toHaveBeenCalledWith(scenario.expectedSys)
    expect(enqueue).toHaveBeenCalledTimes(scenario.queued ? 1 : 0)

    if (scenario.queued) {
      expect(enqueue).toHaveBeenCalledWith('current prompt')
    }
  })
})

describe('submissionCore.markSubmitting', () => {
  beforeEach(() => resetUiState())

  it('sets busy + running status', () => {
    markSubmitting()
    expect(getUiState().busy).toBe(true)
    expect(getUiState().status).toBe('running…')
  })
})

describe('submissionCore.isSessionBusyError', () => {
  it('matches the legacy busy rejections but not arbitrary errors', () => {
    expect(isSessionBusyError(new Error('session busy'))).toBe(true)
    expect(isSessionBusyError(new Error('waiting for model response'))).toBe(true)
    expect(isSessionBusyError(new Error('some other failure'))).toBe(false)
    expect(isSessionBusyError('not an error')).toBe(false)
  })
})
