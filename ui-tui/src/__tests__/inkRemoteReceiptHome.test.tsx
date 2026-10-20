import { mkdtempSync, realpathSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { turnController } from '../app/turnController.js'
import { $uiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useSubmission } from '../app/useSubmission.js'
import { canonicalEvent } from '../canonicalGateway.js'
import { useQueue } from '../hooks/useQueue.js'
import { t } from '../i18n/runtime.js'

// C22: an explicit HERMES_TUI_GATEWAY_URL owner on another home/machine answers receipts naming ITS
// profile_id; Ink validates them against the owner identity from the snapshot, not its local
// HERMES_HOME, and still rejects a wrong-session receipt.
const row = (admission_id: string, input_id: string, text: string, status = 'queued') => ({
  admission_id, input_id, text, status, sequence: 1, outcome: null, authority_epoch: 1, execution_generation: null,
  ref: { profile_id: '/tmp/profile', session_id: 'stored-owner' }
})

function mount(receiptHome: string, receiptSession = 'stored-owner', busyInputMode: 'queue' | 'interrupt' | 'steer' = 'queue', cancel: () => Promise<unknown> = () => Promise.resolve({ status: 'terminal', outcome: 'cancelled' })) {
  // macOS: tmpdir() is /var/..., a symlink to /private/var; captureDestination() realpaths
  // HERMES_HOME, so receipts must carry the canonical home or target_profile_home never matches.
  const home = realpathSync(mkdtempSync(join(tmpdir(), 'ink-busy-admit-')))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()

  vi.stubEnv('HERMES_TUI_GATEWAY_URL', 'ws://remote.test/api/ws')

  const info = { model: 'test', skills: {}, tools: {}, profile_name: 'default', profile_id: '/srv/remote-owner/.hermes',
    stored_session_id: 'stored-owner', execution_epoch: '1', execution_generation: 1, running: true }

  patchUiState({ sid: 'owner', info, busy: true, status: 'running…', busyInputMode })
  const calls: Array<{ method: string; params: any }> = []

  const request = vi.fn((method: string, params: any) => {
    calls.push({ method, params })

    if (method === 'input.detect_drop') { return Promise.resolve({ matched: false }) }

    if (method === 'prompt.submit') {
      return Promise.resolve({ admission_id: `adm-${params.submission_id}`, input_id: params.submission_id,
        target_session_id: receiptSession, target_profile_home: receiptHome === 'local' ? home : receiptHome, status: 'queued' })
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
      composerRefs: { queueRef: queue.queueRef, queueEditRef: queue.queueEditRef, steerEditRef: { current: null }, tokensRef: { current: [] } } as any,
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

it.each([
  ['the remote owner home', '/srv/remote-owner/.hermes', 'stored-owner', false],
  ['a wrong session', '/srv/remote-owner/.hermes', 'someone-else', true]
] as const)('a remote gateway receipt naming %s is confirmed only for the owner destination', async (_label, receiptHome, receiptSession, refused) => {
  const h = mount(receiptHome, receiptSession)

  try {
    h.submission.dispatchSubmission('follower while busy')
    await expect.poll(() => h.calls.filter(c => c.method === 'prompt.submit').length).toBe(1)
    // A confirmed receipt hands the row to the durable FIFO; a refused one keeps it local.
    await expect.poll(() => h.queue.queueRef.current.length).toBe(refused ? 1 : 0)
    expect($uiState.get().status === t('canonical.submit.statusAdmissionUnconfirmed')).toBe(refused)
  } finally { h.cleanup() }
})
