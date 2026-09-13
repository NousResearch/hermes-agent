import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import React, { useEffect } from 'react'
import { expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { turnController } from '../app/turnController.js'
import { $uiState, getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useSubmission } from '../app/useSubmission.js'
import { useQueue } from '../hooks/useQueue.js'

function deferred() {
  let resolve!: (value: any) => void
  let reject!: (error: Error) => void
  const promise = new Promise<any>((yes, no) => { resolve = yes; reject = no })

  return { promise, resolve, reject }
}

function mount() {
  const home = mkdtempSync(join(tmpdir(), 'ink-receipt-busy-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()

  const info = { model: 'test', skills: {}, tools: {}, profile_name: 'default',
    stored_session_id: 'stored-owner', execution_epoch: 'owner-epoch', execution_generation: 1, running: false }

  patchUiState({ sid: 'owner', info, busy: false, status: 'ready' })
  const submits: ReturnType<typeof deferred>[] = []
  const snapshots: ReturnType<typeof deferred>[] = []

  const request = vi.fn((method: string) => {
    if (method === 'input.detect_drop') { return Promise.resolve({ matched: false }) }
    const pending = deferred()

    if (method === 'prompt.submit') { submits.push(pending) }
    else if (method === 'session.activate') { snapshots.push(pending) }
    else { throw new Error(`unexpected RPC: ${method}`) }

    return pending.promise
  })

  let queue!: ReturnType<typeof useQueue>
  let submission!: ReturnType<typeof useSubmission>

  const noop = () => {}

  const onEvent = createGatewayEventHandler({
    composer: { dequeue: () => undefined, queueEditRef: { current: null }, sendQueued: noop, setInput: noop },
    gateway: { gw: { request }, rpc: async () => null },
    session: { STARTUP_RESUME_ID: '', colsRef: { current: 80 }, newSession: noop, resetSession: noop, resumeById: noop, setCatalog: noop },
    submission: { submitRef: { current: noop } }, system: { bellOnComplete: false, sys: noop },
    transcript: { appendMessage: noop, panel: noop, setHistoryItems: noop },
    voice: { setProcessing: noop, setRecording: noop, setVoiceEnabled: noop }
  } as any)

  const emit = (type: string, generation: number) => onEvent({ type, session_id: 'owner',
    payload: { execution_epoch: info.execution_epoch, execution_generation: generation, text: 'completed' } } as any)

  function Harness() {
    const ui = useStore($uiState)
    queue = useQueue()
    submission = useSubmission({
      appendMessage: noop,
      composerActions: { ...queue, prependQueue: queue.prependQ, takeQueue: queue.takeQ } as any,
      composerRefs: { queueRef: queue.queueRef, queueEditRef: queue.queueEditRef, tokensRef: { current: [] } } as any,
      composerState: { input: '', inputBuf: [], completions: [] } as any,
      gw: { request } as any, setLastUserMsg: noop, slashRef: { current: () => true },
      submitRef: { current: noop }, sys: noop
    })
    useEffect(() => {
      if (!ui.sid || ui.busy || queue.queueEditRef.current !== null || !queue.queueRef.current.length) { return }
      const next = queue.dequeue()

      if (next) { submission.sendQueued(next) }
    }, [ui.sid, ui.busy, queue, submission])

    return <Text>{ui.status} {queue.queuedDisplay.join('|')}</Text>
  }

  const instance = renderSync(<Harness />, {
    stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any, patchConsole: false
  })

  return {
    submits, snapshots, request, emit,
    get queue() { return queue }, get submission() { return submission },
    receipt: (id: string, status: string) => ({ admission_id: id, target_session_id: info.stored_session_id,
      target_profile_home: home, status }),
    snapshot: (running: boolean, generation: number) => ({ session_id: 'owner', session_key: info.stored_session_id,
      running, info: { ...info, running, execution_generation: generation } }),
    cleanup() { instance.unmount(); turnController.fullReset(); resetUiState(); vi.unstubAllEnvs(); rmSync(home, { recursive: true, force: true }) }
  }
}

it('reconciles terminal and unknown retries to idle so queued follow-ups can drain without another terminal event', async () => {
  for (const status of ['terminal', 'unknown']) {
    const h = mount()

    try {
      const item = h.queue.stage('original')
      h.submission.sendQueued(item)
      await expect.poll(() => h.submits.length).toBe(1)
      h.emit('message.start', 1)
      h.emit('message.complete', 1)
      h.submits[0]!.reject(new Error('lost acknowledgement'))
      await expect.poll(() => item.failed).toBe(true)
      h.submission.sendQueued(h.queue.dequeue(true)!)
      await expect.poll(() => h.submits.length).toBe(2)
      h.submits[1]!.resolve(h.receipt(item.submissionId!, status))
      await expect.poll(() => h.queue.queueRef.current.length).toBe(0)
      await expect.poll(() => h.snapshots.length).toBe(1)
      h.snapshots[0]!.resolve(h.snapshot(false, 1))
      await expect.poll(() => getUiState().busy).toBe(false)
      expect(getUiState().status).toBe('ready')
      h.queue.enqueue('follow-up')
      await expect.poll(() => h.submits.length).toBe(3)
      expect(h.request).toHaveBeenLastCalledWith('prompt.submit', expect.objectContaining({ text: 'follow-up' }))
    } finally { h.cleanup() }
  }
})

it('preserves a newer running generation across old receipts and delayed idle snapshots', async () => {
  for (const delayed of [false, true]) {
    const h = mount()

    try {
      const item = h.queue.stage('retry old identity')
      h.submission.sendQueued(item)
      await expect.poll(() => h.submits.length).toBe(1)

      if (!delayed) { h.emit('message.start', 2) }
      h.submits[0]!.resolve(h.receipt(item.submissionId!, 'terminal'))
      await expect.poll(() => h.queue.queueRef.current.length).toBe(0)
      await expect.poll(() => h.snapshots.length).toBe(1)

      if (delayed) { h.emit('message.start', 2) }
      h.snapshots[0]!.resolve(h.snapshot(!delayed, delayed ? 1 : 2))
      await new Promise(resolve => setTimeout(resolve, 20))
      expect(getUiState()).toMatchObject({ busy: true, status: 'running…', info: { execution_generation: 2 } })
      h.queue.enqueue('must wait')
      await new Promise(resolve => setTimeout(resolve, 20))
      expect(h.submits).toHaveLength(1)
      expect(h.queue.queueRef.current).toHaveLength(1)
      h.emit('message.complete', 2)
      await expect.poll(() => h.submits.length).toBe(2)
    } finally { h.cleanup() }
  }
})
