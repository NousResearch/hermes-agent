import { describe, expect, it, vi } from 'vitest'

import { coreCommands } from '../app/slash/commands/core.js'

const steerCommand = coreCommands.find(command => command.name === 'steer')!

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
    gateway: { rpc },
    guarded,
    guardedErr: vi.fn(),
    sid: 'sid-1',
    transcript: { sys },
    ui: { busy }
  }

  steerCommand.run(arg, ctx as never, `/steer ${arg}`)
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

  it('reports a capability refusal without queueing a different action', async () => {
    const error = Object.assign(new Error('agent does not support steer'), { code: 4010 })
    const enqueue = vi.fn()
    const guardedErr = vi.fn()
    const ctx = {
      composer: { enqueue },
      gateway: { rpc: vi.fn().mockRejectedValue(error) },
      guarded,
      guardedErr,
      sid: 'sid-1',
      transcript: { sys: vi.fn() },
      ui: { busy: true }
    }
    steerCommand.run('check the logs', ctx as never, '/steer check the logs')
    await vi.waitFor(() => expect(guardedErr).toHaveBeenCalledWith(error))
    expect(enqueue).not.toHaveBeenCalled()
  })
})
