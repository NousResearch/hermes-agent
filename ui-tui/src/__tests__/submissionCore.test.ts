import { beforeEach, describe, expect, it, vi } from 'vitest'

import { isSessionBusyError, markSubmitting, submitPrompt, type SubmitPromptDeps } from '../app/submissionCore.js'
import { captureDestination } from '../app/submissionDestination.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import type { GatewayClient } from '../gatewayClient.js'

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

describe('submissionCore.submitPrompt — synchronous busy (queue-race fix)', () => {
  beforeEach(() => {
    resetUiState()
    patchUiState({ sid: 'sess-1' })
  })

  it('flips busy=true SYNCHRONOUSLY, before input.detect_drop resolves', () => {
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
    await Promise.resolve()
    await Promise.resolve()

    expect(calls).toContain('prompt.submit')
  })
})

describe('submissionCore.submitPrompt — literal submissions (startup -q queries)', () => {
  beforeEach(() => {
    resetUiState()
    patchUiState({ sid: 'sess-1' })
  })

  it('skipDetectDrop submits directly without the detect_drop round-trip', async () => {
    const { calls, gw } = makeDeferredGateway()

    submitPrompt('!echo not-a-shell-escape', makeDeps(gw), true, undefined, { skipDetectDrop: true })

    await Promise.resolve()
    await Promise.resolve()

    expect(calls).not.toContain('input.detect_drop')
    expect(calls).toContain('prompt.submit')
  })

  it('literal text reaches prompt.submit verbatim', async () => {
    const submitted: string[] = []

    const gw = {
      request: vi.fn((method: string, params?: { text?: string }) => {
        if (method === 'prompt.submit' && params?.text) {
          submitted.push(params.text)
        }

        return Promise.resolve({ status: 'streaming' })
      })
    } as unknown as GatewayClient

    submitPrompt('/model $(rm -rf ~)', makeDeps(gw), true, undefined, { skipDetectDrop: true })

    await Promise.resolve()
    await Promise.resolve()

    expect(submitted).toEqual(['/model $(rm -rf ~)'])
  })
})

it('keeps the submit destination across preprocessing and never mutates the newly focused session', async () => {
  resetUiState()
  patchUiState({ sid: 'original' })
  const { gw, resolveDrop } = makeDeferredGateway()
  const deps = makeDeps(gw)
  submitPrompt('private', deps)
  patchUiState({ sid: 'other', busy: false, status: 'other ready' })
  resolveDrop()
  await Promise.resolve()
  await Promise.resolve()
  expect(gw.request).toHaveBeenCalledWith(
    'prompt.submit',
    expect.objectContaining({ session_id: 'original', text: 'private' })
  )
  expect(deps.appendMessage).not.toHaveBeenCalled()
  expect(getUiState()).toMatchObject({ sid: 'other', busy: false, status: 'other ready' })
})

it('retains a queued submission on ambiguous response and retries the exact identity until durable acknowledgement', async () => {
  resetUiState()
  patchUiState({ sid: 'owner', info: { model: 'test', tools: {}, skills: {}, stored_session_id: 'stored-owner' } })
  const settle = vi.fn()
  const item = { text: 'private', display: 'private', submissionId: 'stable-id', settle }

  const request = vi.fn().mockResolvedValueOnce({ status: 'streaming' }).mockResolvedValueOnce({
    admission_id: 'stable-id',
    target_session_id: 'stored-owner',
    target_profile_home: captureDestination().profileHome,
    status: 'queued'
  })

  const deps = makeDeps({ request } as unknown as GatewayClient)
  submitPrompt(item.text, deps, true, undefined, { skipDetectDrop: true, queueItem: item })
  await Promise.resolve()
  expect(settle).toHaveBeenLastCalledWith(false)
  submitPrompt(item.text, deps, true, undefined, { skipDetectDrop: true, queueItem: item })
  await Promise.resolve()
  expect(request.mock.calls.map(call => call[1])).toEqual([
    { session_id: 'owner', text: 'private', submission_id: 'stable-id', queued: true },
    { session_id: 'owner', text: 'private', submission_id: 'stable-id', queued: true }
  ])
  expect(settle).toHaveBeenLastCalledWith(true)

  for (const stored_session_id of [undefined, 'wrong-target']) {
    patchUiState({ info: { model: 'test', tools: {}, skills: {}, stored_session_id } })
    request.mockResolvedValueOnce({ admission_id: 'stable-id', target_session_id: 'owner',
      target_profile_home: captureDestination().profileHome, status: 'queued' })
    submitPrompt(item.text, deps, true, undefined, { skipDetectDrop: true, queueItem: item })
    await Promise.resolve()
    expect(settle).toHaveBeenLastCalledWith(false)
  }
})

it('preserves legacy isolated submission after explicit unsupported admission without repeating preprocessing', async () => {
  resetUiState()
  patchUiState({ sid: 'owner' })
  const settle = vi.fn()
  const item = { text: 'private', display: 'private', submissionId: 'unsupported-id', queued: true, settle }
  let reject!: (error: unknown) => void

  const request = vi.fn().mockReturnValueOnce(new Promise((_, fail) => { reject = fail }))
    .mockResolvedValueOnce({ status: 'queued' })

  const deps = makeDeps({ request } as unknown as GatewayClient)
  submitPrompt(item.text, deps, true, undefined, { skipDetectDrop: true, queueItem: item })
  patchUiState({ sid: 'other', busy: false, status: 'other ready' })
  reject(Object.assign(new Error('unsupported'), { code: 4094 }))
  await vi.waitFor(() => expect(settle).toHaveBeenLastCalledWith(true))
  expect(request.mock.calls.map(call => call[1])).toEqual([
    { session_id: 'owner', text: 'private', submission_id: 'unsupported-id', queued: true },
    { session_id: 'owner', text: 'private', queued: true }
  ])
  expect(deps.appendMessage).toHaveBeenCalledTimes(1)
  expect(getUiState()).toMatchObject({ sid: 'other', busy: false, status: 'other ready' })
})

it('never downgrades ambiguous or conflicting durable submissions to legacy delivery', async () => {
  for (const code of [undefined, 4093, 5071]) {
    resetUiState()
    patchUiState({ sid: 'owner' })
    const settle = vi.fn()
    const item = { text: 'private', display: 'private', submissionId: 'retained-id', settle }
    const request = vi.fn().mockRejectedValue(Object.assign(new Error('not admitted'), { code }))
    submitPrompt(item.text, makeDeps({ request } as unknown as GatewayClient), true, undefined,
      { skipDetectDrop: true, queueItem: item })
    await vi.waitFor(() => expect(settle).toHaveBeenLastCalledWith(false))
    expect(request).toHaveBeenCalledTimes(1)
  }
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
