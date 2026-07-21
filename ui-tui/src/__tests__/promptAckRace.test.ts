import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { OverlayState } from '../app/interfaces.js'
import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { resolveAnsweredPrompt } from '../app/useMainApp.js'

// Regression: the backend FIFO can emit prompt B after removing prompt A but
// before A's `*.respond` RPC resolves. If a newer B of the SAME kind has taken
// the overlay slot by the time A's late ACK lands, the old code cleared it and
// reset the status to running — leaving B's backend waiter invisible. These
// tests drive resolveAnsweredPrompt at the exact moment the answer callbacks
// invoke it (in the settled `.then` of the respond RPC), with B installed while
// the RPC is still pending, and assert B survives and the status is preserved.

type PromptKind = 'approval' | 'clarify' | 'secret' | 'sudo'

const KINDS: PromptKind[] = ['approval', 'clarify', 'secret', 'sudo']

// The waiting status createGatewayEventHandler installs for each prompt kind.
const BUSY_STATUS: Record<PromptKind, string> = {
  approval: 'approval needed',
  clarify: 'waiting for input…',
  secret: 'secret input needed',
  sudo: 'sudo password needed'
}

const makeReq = (kind: PromptKind, requestId: string) => {
  switch (kind) {
    case 'approval':
      return { command: 'rm -rf /tmp/x', description: 'delete tmp', requestId }

    case 'clarify':
      return { choices: null, question: 'which path?', requestId }

    case 'secret':
      return { envVar: 'API_TOKEN', prompt: 'enter token', requestId }

    case 'sudo':
      return { requestId }
  }
}

// Install/replace a prompt overlay the way createGatewayEventHandler does.
const installPrompt = (kind: PromptKind, requestId: string) =>
  patchOverlayState({ [kind]: makeReq(kind, requestId) } as Partial<OverlayState>)

const deferred = <T>() => {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(r => {
    resolve = r
  })

  return { promise, resolve }
}

describe('resolveAnsweredPrompt — late prompt ACK must not erase a newer same-kind prompt', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
  })

  afterEach(() => {
    resetOverlayState()
    resetUiState()
  })

  it.each(KINDS)(
    'preserves a newer %s prompt B installed while A’s response RPC was in flight',
    async kind => {
      // Prompt A is on screen; the UI shows A's waiting status.
      installPrompt(kind, 'req-A')
      patchUiState({ status: BUSY_STATUS[kind] })

      // Answer A: capture its requestId and dispatch the (still pending) RPC.
      const requestId = 'req-A'
      const rpc = deferred<{ ok: true }>()
      const flow = rpc.promise.then(() => resolveAnsweredPrompt(kind, requestId))

      // Backend FIFO removes A and installs a NEWER B (same kind) before A's
      // ACK arrives — exactly what createGatewayEventHandler does on the event.
      installPrompt(kind, 'req-B')
      patchUiState({ status: BUSY_STATUS[kind] })

      // A's late ACK finally resolves.
      rpc.resolve({ ok: true })
      const result = await flow

      // B is untouched and still usable, and the status was not reset to running.
      expect(getOverlayState()[kind]).toEqual(makeReq(kind, 'req-B'))
      expect(getUiState().status).toBe(BUSY_STATUS[kind])
      expect(result.supersededByNewer).toBe(true)
    }
  )

  it.each(KINDS)('clears the %s overlay and resumes running when no newer prompt is pending', async kind => {
    installPrompt(kind, 'req-A')
    patchUiState({ status: BUSY_STATUS[kind] })

    const requestId = 'req-A'
    const rpc = deferred<{ ok: true }>()
    const flow = rpc.promise.then(() => resolveAnsweredPrompt(kind, requestId))

    rpc.resolve({ ok: true })
    const result = await flow

    expect(getOverlayState()[kind]).toBeNull()
    expect(getUiState().status).toBe('running…')
    expect(result.supersededByNewer).toBe(false)
  })

  it('cancel path (resetStatus:false) drops A without resuming running, and leaves a newer B intact', () => {
    // clarify (cancel) and sudo/secret (empty input) share this branch: A is
    // dismissed but the status must not be forced back to running, and a queued
    // newer B must survive.
    installPrompt('sudo', 'req-A')
    patchUiState({ status: BUSY_STATUS.sudo })

    // No newer prompt: A is dropped, status left as-is (not running…).
    const dropped = resolveAnsweredPrompt('sudo', 'req-A', { resetStatus: false })
    expect(getOverlayState().sudo).toBeNull()
    expect(getUiState().status).toBe(BUSY_STATUS.sudo)
    expect(dropped.supersededByNewer).toBe(false)

    // Newer B present: it survives, status untouched.
    installPrompt('sudo', 'req-B')
    const superseded = resolveAnsweredPrompt('sudo', 'req-A', { resetStatus: false })
    expect(getOverlayState().sudo).toEqual(makeReq('sudo', 'req-B'))
    expect(getUiState().status).toBe(BUSY_STATUS.sudo)
    expect(superseded.supersededByNewer).toBe(true)
  })
})
