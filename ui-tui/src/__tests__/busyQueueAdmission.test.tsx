import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { captureDestination } from '../app/submissionDestination.js'
import { turnController } from '../app/turnController.js'
import { $uiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useSubmission } from '../app/useSubmission.js'
import { canonicalEvent, canonicalResult } from '../canonicalGateway.js'
import { useQueue } from '../hooks/useQueue.js'
import { loadPendingInputs } from '../lib/pendingInputs.js'

// Canonical authority rows: `pending` carries the server-issued admission_id,
// the client's input_id and the public text under `ref` destination fields.
const row = (admission_id: string, input_id: string, text: string, status = 'queued') => ({
  admission_id, input_id, text, status, sequence: 1, outcome: null, authority_epoch: 1, execution_generation: null,
  ref: { profile_id: '/tmp/profile', session_id: 'stored-owner' }
})

function mount(busyInputMode: 'queue' | 'interrupt' | 'steer' = 'queue', cancel: () => Promise<unknown> = () => Promise.resolve({ status: 'terminal', outcome: 'cancelled' })) {
  const home = mkdtempSync(join(tmpdir(), 'ink-busy-admit-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()

  const info = { model: 'test', skills: {}, tools: {}, profile_name: 'default',
    stored_session_id: 'stored-owner', execution_epoch: '1', execution_generation: 1, running: true }

  patchUiState({ sid: 'owner', info, busy: true, status: 'running…', busyInputMode })
  const calls: Array<{ method: string; params: any }> = []

  const request = vi.fn((method: string, params: any) => {
    calls.push({ method, params })

    if (method === 'input.detect_drop') { return Promise.resolve({ matched: false }) }

    if (method === 'prompt.submit') {
      return Promise.resolve({ admission_id: `adm-${params.submission_id}`, input_id: params.submission_id,
        target_session_id: 'stored-owner', target_profile_home: home, status: 'queued' })
    }

    if (method === 'prompt.cancel') { return cancel() }
    throw new Error(`unexpected RPC: ${method}`)
  })

  const gw = { request, isCanonical: true } as any
  let queue!: ReturnType<typeof useQueue>
  let submission!: ReturnType<typeof useSubmission>

  const noop = () => {}

  const onEvent = createGatewayEventHandler({
    composer: { dequeue: () => undefined, queueEditRef: { current: null }, sendQueued: noop, setInput: noop },
    gateway: { gw, rpc: async () => null },
    session: { STARTUP_RESUME_ID: '', colsRef: { current: 80 }, newSession: noop, resetSession: noop, resumeById: noop, setCatalog: noop },
    submission: { submitRef: { current: noop } }, system: { bellOnComplete: false, sys: noop },
    transcript: { appendMessage: noop, panel: noop, setHistoryItems: noop },
    voice: { setProcessing: noop, setRecording: noop, setVoiceEnabled: noop }
  } as any)

  // The wire shape the gateway fans out; the client translates it exactly once.
  const fanout = (pending: any[]) => onEvent(canonicalEvent({ type: 'session.info', session_id: 'owner',
    payload: { stored_session_id: 'stored-owner', pending, running: true, execution_generation: 1, revision: 3,
      execution_epoch: '1' } } as any) as any)

  function Harness() {
    const ui = useStore($uiState)
    queue = useQueue(gw)
    submission = useSubmission({
      appendMessage: noop,
      composerActions: { ...queue, prependQueue: queue.prependQ, takeQueue: queue.takeQ, removeQueue: queue.removeQ,
        pushHistory: noop, clearIn: noop } as any,
      composerRefs: { queueRef: queue.queueRef, queueEditRef: queue.queueEditRef, tokensRef: { current: [] } } as any,
      composerState: { input: '', inputBuf: [], completions: [] } as any,
      gw, setLastUserMsg: noop, slashRef: { current: () => true },
      submitRef: { current: noop }, sys: noop
    })

    return <Text>{ui.status} {queue.queuedDisplay.join('|')}</Text>
  }

  const instance = renderSync(<Harness />, {
    stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any, patchConsole: false
  })

  return {
    calls, request, fanout,
    get queue() { return queue }, get submission() { return submission },
    cleanup() { instance.unmount(); turnController.fullReset(); resetUiState(); vi.unstubAllEnvs(); rmSync(home, { recursive: true, force: true }) }
  }
}

it('admits busy queue-mode input to the authority immediately with a stable input_id instead of holding it locally', async () => {
  const h = mount('queue')

  try {
    h.submission.dispatchSubmission('follower while busy')
    await expect.poll(() => h.calls.filter(c => c.method === 'prompt.submit').length).toBe(1)
    const submit = h.calls.find(c => c.method === 'prompt.submit')!.params
    expect(submit).toMatchObject({ session_id: 'owner', text: 'follower while busy', queued: true })
    expect(typeof submit.submission_id).toBe('string')
    expect(submit.submission_id.length).toBeGreaterThan(8)
    // The receipt settles the local staging row; the durable FIFO now owns it.
    await expect.poll(() => h.queue.queueRef.current.length).toBe(0)
    // The in-flight turn's streaming buffer and busy latch are untouched.
    expect($uiState.get()).toMatchObject({ busy: true, status: 'running…' })
  } finally { h.cleanup() }
})

it('renders the queue from the session.info pending fanout, excluding the started row, and clears it when the fanout empties', async () => {
  const h = mount('queue')

  try {
    h.fanout([row('adm-1', 'in-1', 'currently running', 'started'), row('adm-2', 'in-2', 'second in line')])
    await expect.poll(() => h.queue.queuedDisplay).toEqual(['[queued] second in line'])
    h.fanout([row('adm-1', 'in-1', 'currently running', 'started')])
    await expect.poll(() => h.queue.queuedDisplay).toEqual([])
  } finally { h.cleanup() }
})

it('deletes a server-queued row through prompt.cancel rather than local removal', async () => {
  const h = mount('queue')

  try {
    h.fanout([row('adm-9', 'in-9', 'cancel me')])
    await expect.poll(() => h.queue.queuedDisplay).toEqual(['[queued] cancel me'])
    h.queue.removeQ(0)
    await expect.poll(() => h.calls.filter(c => c.method === 'prompt.cancel').length).toBe(1)
    expect(h.calls.find(c => c.method === 'prompt.cancel')!.params).toEqual({ session_id: 'owner', admission_id: 'adm-9' })
    // Only the authority's next fanout retires the row.
    expect(h.queue.queuedDisplay).toEqual(['[queued] cancel me'])
    h.fanout([])
    await expect.poll(() => h.queue.queuedDisplay).toEqual([])
  } finally { h.cleanup() }
})

it('retains disconnected input durably on its old destination without dispatching or draining', async () => {
  const h = mount()

  try {
    patchUiState({ gatewayConnected: false, busy: false } as any)
    const destination = captureDestination()
    h.submission.dispatchSubmission('DISCONNECTED_LOSS_SENTINEL')
    await new Promise(resolve => setImmediate(resolve))
    expect(h.calls).toEqual([])
    expect(loadPendingInputs(destination)).toEqual([expect.objectContaining({ text: 'DISCONNECTED_LOSS_SENTINEL', destination })])
    expect(h.queue.dequeue()).toBeUndefined()
  } finally { h.cleanup() }
})

it('refuses unbound Enter without clearing the draft or claiming a queued input', () => {
  const h = mount()
  try {
    patchUiState({ sid: null })
    h.submission.dispatchSubmission('unbound draft')
    expect(h.queue.queueRef.current).toEqual([])
    expect(h.calls).toEqual([])
  } finally { h.cleanup() }
})

it('discards unknown execution with its generation, retaining the row on refusal', async () => {
  const h = mount()

  try {
    h.request.mockImplementation(async (method, params) => { h.calls.push({method, params}); throw new Error('stale_generation') })
    h.fanout([{ ...row('unknown-admission', 'unknown-input', 'interrupted input', 'unknown'), execution_generation: 7 }])
    await expect.poll(() => h.queue.queuedDisplay).toEqual(['[unknown] interrupted input'])
    h.queue.removeQ(0)
    await expect.poll(() => h.calls.length).toBe(1)
    expect(h.calls[0]).toEqual({ method: 'prompt.resolve_unknown', params: { session_id: 'owner', admission_id: 'unknown-admission', execution_generation: 7 } })
    await expect.poll(() => $uiState.get().status).toContain('stale_generation')
    expect(h.queue.queuedDisplay).toEqual(['[unknown] interrupted input'])
  } finally { h.cleanup() }
})

it('never replays an ambiguous non-idempotent busy correction or falls back to queue', async () => {
  const h = mount('steer')

  try {
    h.request.mockImplementation(async (method, params) => { h.calls.push({ method, params }); throw new Error('invalid_params') })
    h.submission.dispatchSubmission('correction')
    await expect.poll(() => h.calls.length).toBe(1)
    expect(h.calls[0]).toMatchObject({ method: 'session.steer', params: { session_id: 'owner', text: 'correction', execution_generation: 1 } })
    const first = h.calls[0]!.params
    await expect.poll(() => h.queue.queueRef.current[0]?.failed).toBe(true)
    patchUiState({ busyInputMode: 'interrupt', info: { ...$uiState.get().info!, execution_generation: 2 } })
    h.submission.sendQueued(h.queue.dequeue(true)!)
    await new Promise(resolve => setImmediate(resolve))
    expect(h.calls).toEqual([{ method: 'session.steer', params: first }])
    expect(first).not.toHaveProperty('submission_id')
    expect(loadPendingInputs(captureDestination())[0]).toMatchObject({ controlMethod: 'session.steer', executionGeneration: 1 })
  } finally { h.cleanup() }
})

it('retires a generation-matched redirect acknowledgement without resetting the live turn', async () => {
  const h = mount('interrupt')

  try {
    h.request.mockImplementation(async (method, params) => {
      h.calls.push({method, params});

 return { status: 'redirected', execution_generation: params.execution_generation } as any
    })
    h.submission.dispatchSubmission('redirect correction')
    await expect.poll(() => h.calls.length).toBe(1)
    expect(h.calls[0]).toEqual({ method: 'session.redirect', params: { session_id: 'owner', text: 'redirect correction', execution_generation: 1 } })
    await expect.poll(() => h.queue.queueRef.current.length).toBe(0)
    expect(loadPendingInputs(captureDestination())).toEqual([])
    expect($uiState.get()).toMatchObject({ busy: true, status: 'running…' })
  } finally { h.cleanup() }
})

it('projects canonical pending rows onto the legacy pending_submissions shape for resume snapshots and fanout', () => {
  const pending = [row('adm-3', 'in-3', 'hello')]

  const snapshot = canonicalResult('session.resume', { session_id: 'sid', stored_session_id: 'sid', authority_epoch: 1,
    execution_generation: 2, running: true, messages: [], pending }, {})

  expect(snapshot.info.pending_submissions).toEqual([expect.objectContaining({ admission_id: 'adm-3', input_id: 'in-3', user: 'hello',
    status: 'queued', target_session_id: 'stored-owner', target_profile_home: '/tmp/profile' })])
  const ev = canonicalEvent({ type: 'session.info', session_id: 'sid', payload: { pending } } as any)
  expect((ev.payload as any).pending_submissions[0]).toMatchObject({ admission_id: 'adm-3', user: 'hello' })
})

it('does not admit an edited replacement while the original row is still being retired, and keeps it as a retryable draft when retirement is refused', async () => {
  let release!: (value: unknown) => void
  const h = mount('queue', () => new Promise(resolve => { release = resolve }))

  try {
    h.fanout([row('adm-orig', 'in-orig', 'ORIGINAL_EFFECT')])
    await expect.poll(() => h.queue.queuedDisplay).toEqual(['[queued] ORIGINAL_EFFECT'])
    h.queue.setQueueEdit(0)
    h.submission.dispatchSubmission('EDITED_EFFECT')
    await expect.poll(() => h.calls.filter(c => c.method === 'prompt.cancel').length).toBe(1)
    await new Promise(resolve => setTimeout(resolve, 20))
    // Retirement is still pending: nothing has been re-admitted yet.
    expect(h.calls.filter(c => c.method === 'prompt.submit')).toEqual([])
    release({ admission_id: 'adm-orig', status: 'terminal', outcome: 'cancelled' })
    await expect.poll(() => h.calls.filter(c => c.method === 'prompt.submit').length).toBe(1)
    expect(h.calls.find(c => c.method === 'prompt.submit')!.params).toMatchObject({ text: 'EDITED_EFFECT', queued: true })
  } finally { h.cleanup() }
})

it('keeps the edited text as an unconfirmed durable draft when the store refuses to cancel the original', async () => {
  const h = mount('queue', () => Promise.reject(new Error('stale_generation')))

  try {
    h.fanout([row('adm-orig', 'in-orig', 'ORIGINAL_EFFECT')])
    await expect.poll(() => h.queue.queuedDisplay).toEqual(['[queued] ORIGINAL_EFFECT'])
    h.queue.setQueueEdit(0)
    h.submission.dispatchSubmission('EDITED_EFFECT')
    await expect.poll(() => h.calls.filter(c => c.method === 'prompt.cancel').length).toBe(1)
    await expect.poll(() => $uiState.get().status).toContain('discard failed')
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(h.calls.filter(c => c.method === 'prompt.submit')).toEqual([])
    expect(h.queue.queuedDisplay).toEqual(['[unconfirmed · Alt+K retry] EDITED_EFFECT', '[queued] ORIGINAL_EFFECT'])
    expect(loadPendingInputs(captureDestination()).map(item => [item.text, item.failed])).toEqual([['EDITED_EFFECT', true]])
  } finally { h.cleanup() }
})
