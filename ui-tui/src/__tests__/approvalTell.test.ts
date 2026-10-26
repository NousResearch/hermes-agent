import { describe, expect, it, vi } from 'vitest'

import { steerOrQueue } from '../app/steerOrQueue.js'
import { approvalAction, approvalRows } from '../components/prompts.js'

const req = { command: 'rm -rf build', description: 'recursive delete', requestId: 'r1' }

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

    await steerOrQueue({ enqueue, generation: 3, rpc: rpc as any, sid: 'owner', sys, text: 'use INSERT OR REPLACE' })

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

      await steerOrQueue({ enqueue, rpc: rpc as any, sid: 'owner', sys: vi.fn(), text: 'do X instead' })

      expect(enqueue).toHaveBeenCalledWith('do X instead')
    }
  })
})
