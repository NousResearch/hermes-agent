import { describe, expect, it, vi } from 'vitest'

import { findSlashCommand } from '../app/slash/registry.js'

const guarded =
  <T>(fn: (r: T) => void) =>
  (r: null | T) => {
    if (r) {
      fn(r)
    }
  }

const runSteer = async (arg: string, steerResult: unknown, busy = true) => {
  const sys = vi.fn()
  const enqueue = vi.fn()
  const rpc = vi.fn((_method: string, _params: unknown) => Promise.resolve(steerResult))

  const ctx = {
    composer: { enqueue },
    gateway: { gw: { isCanonical: false }, rpc },
    guarded,
    guardedErr: vi.fn(),
    sid: 'sid-1',
    transcript: { sys },
    ui: { busy }
  }

  findSlashCommand('steer')!.run(arg, ctx as never, `/steer ${arg}`)
  await rpc.mock.results[0]?.value
  await Promise.resolve()

  return { enqueue, printed: sys.mock.calls.map(c => String(c[0])).join('\n'), rpc }
}

describe('/steer', () => {
  it('steers the live turn when the gateway accepts', async () => {
    const { enqueue, printed, rpc } = await runSteer('check the logs', { status: 'queued', text: 'check the logs' })

    expect(rpc).toHaveBeenCalledWith('session.steer', { session_id: 'sid-1', text: 'check the logs' })
    expect(enqueue).not.toHaveBeenCalled()
    expect(printed).toContain('steer queued')
  })

  // #64578: the turn can end between the client's busy check and the RPC; the gateway then
  // answers 'rejected'. The text must fall back to the next-turn queue, not vanish.
  it('queues the text for the next turn when the gateway rejects the steer', async () => {
    const { enqueue, printed } = await runSteer('check the logs', { status: 'rejected', text: 'check the logs' })

    expect(enqueue).toHaveBeenCalledWith('check the logs')
    expect(printed).toContain('queued for next turn')
  })
})
