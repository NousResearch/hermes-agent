import { beforeEach, describe, expect, it, vi } from 'vitest'

import { patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { steerOrQueue } from '../app/steerOrQueue.js'
import { captureDestination } from '../app/submissionDestination.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { approvalAnswers } from '../app/useApprovalAnswers.js'
import { approvalAction, approvalRows } from '../components/prompts.js'

const req = { command: 'rm -rf build', description: 'recursive delete', requestId: 'r1' }

const owner = () => ({ ...captureDestination(), sid: 'owner' })

beforeEach(() => {
  resetUiState()
  resetOverlayState()
  patchUiState({ sid: 'owner' })
})

describe('"No, and tell Hermes what to do instead" (#25693)', () => {
  it('is offered after Deny on every prompt that has Deny, and picked by its number', () => {
    const rows = approvalRows(req)

    expect(rows).toEqual(['once', 'session', 'always', 'deny', 'tell'])
    expect(approvalAction('5', {}, 0, rows)).toEqual({ kind: 'choose', choice: 'tell' })
    expect(approvalRows({ ...req, smartDenied: true })).toEqual(['once', 'deny', 'tell'])
    // Esc still denies outright: the tell row never becomes the default answer.
    expect(approvalAction('', { escape: true }, 4, rows)).toEqual({ kind: 'choose', choice: 'deny' })
  })

  it('is not offered when the gateway sends no Deny choice', () => {
    expect(approvalRows({ ...req, choices: ['once', 'session'] })).toEqual(['once', 'session'])
  })

  it('steers the instruction into the running turn and queues nothing', async () => {
    const rpc = vi.fn(async () => ({ status: 'queued' }))
    const enqueue = vi.fn()
    const sys = vi.fn()

    await steerOrQueue({ destination: owner(), enqueue, generation: 3, rpc: rpc as any, sys, text: 'use INSERT OR REPLACE' })

    expect(rpc).toHaveBeenCalledWith('session.steer', {
      execution_generation: 3,
      session_id: 'owner',
      text: 'use INSERT OR REPLACE'
    })
    expect(enqueue).not.toHaveBeenCalled()
  })

  it('keeps the words as the next prompt when the steer is refused or fails', async () => {
    for (const rpc of [vi.fn(async () => ({ status: 'rejected' })), vi.fn(async () => Promise.reject(new Error('gone')))]) {
      const enqueue = vi.fn()

      await steerOrQueue({ destination: owner(), enqueue, rpc: rpc as any, sys: vi.fn(), text: 'do X instead' })

      expect(enqueue).toHaveBeenCalledWith('do X instead', 'do X instead', expect.objectContaining({ sid: 'owner' }))
    }
  })

  it('a late refusal queues into the session it was steered at, even after the user switched away', async () => {
    let refuse!: (value: unknown) => void
    const rpc = vi.fn(() => new Promise(resolve => (refuse = resolve)))
    const enqueue = vi.fn()
    const sys = vi.fn()
    const steered = steerOrQueue({ destination: owner(), enqueue, rpc: rpc as any, sys, text: 'do X instead' })

    patchUiState({ sid: 'other' })
    refuse({ status: 'rejected' })
    await steered

    expect(enqueue).toHaveBeenCalledWith('do X instead', 'do X instead', expect.objectContaining({ sid: 'owner' }))
    expect(sys).not.toHaveBeenCalled()
  })

  it('a submitted instruction locks the prompt: repeat Enter or an allow during the steer sends nothing extra', async () => {
    const approval = { ...req }
    let reply!: (value: unknown) => void
    const rpc = vi.fn(() => new Promise(resolve => (reply = resolve)))
    const respondWith = vi.fn()

    patchOverlayState({ approval })

    const { answerApproval, answerApprovalTell } = approvalAnswers({
      approval,
      canonical: false,
      enqueue: vi.fn(),
      respondWith,
      rpc: rpc as any,
      sys: vi.fn()
    })

    answerApprovalTell('use INSERT OR REPLACE')
    answerApprovalTell('use INSERT OR REPLACE')
    answerApproval('once')

    expect(rpc).toHaveBeenCalledTimes(1)
    expect(respondWith).not.toHaveBeenCalled()

    reply({ status: 'queued' })
    await new Promise(resolve => setImmediate(resolve))
    answerApproval('once')

    expect(respondWith.mock.calls.map(([, result]) => result)).toEqual([{ choice: 'deny' }])
  })
})
