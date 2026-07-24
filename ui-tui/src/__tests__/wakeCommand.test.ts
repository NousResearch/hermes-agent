import { beforeEach, describe, expect, it, vi } from 'vitest'

import { wakeCommands } from '../app/slash/commands/wake.js'
import { isWakeUserDisabled, setWakeUserDisabled } from '../app/wakeState.js'

const wakeCommand = wakeCommands.find(cmd => cmd.name === 'wake')!

const guarded =
  <T>(fn: (r: T) => void) =>
  (r: null | T) => {
    if (r) {
      fn(r)
    }
  }

/** Build a ctx whose rpc routes by method name to a supplied map of results. */
const buildCtx = (results: Record<string, unknown>) => {
  const sys = vi.fn()

  const rpc = vi.fn((method: string, _params: unknown) => Promise.resolve(results[method]))

  const ctx = {
    gateway: { rpc },
    guarded,
    guardedErr: vi.fn(),
    sid: 'sid-1',
    stale: () => false,
    transcript: { page: vi.fn(), sys }
  }

  const run = async (arg: string) => {
    wakeCommand.run(arg, ctx as any, `/wake${arg ? ` ${arg}` : ''}`)
    await rpc.mock.results[0]?.value
    await Promise.resolve()
    await Promise.resolve()
  }

  return { ctx, rpc, run, sys }
}

const printed = (sys: ReturnType<typeof vi.fn>) => sys.mock.calls.map(c => c[0]).join('\n')

describe('/wake slash command', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setWakeUserDisabled(false)
  })

  it('registers with usage metadata', () => {
    expect(wakeCommand).toBeDefined()
    expect(wakeCommand.usage).toBe('/wake [on|off|status]')
  })

  it('/wake on calls wake.start with surface tui and reports listening', async () => {
    const { rpc, run, sys } = buildCtx({
      'wake.start': { phrase: 'hey hermes', provider: 'openwakeword', started: true }
    })

    await run('on')

    expect(rpc).toHaveBeenCalledWith('wake.start', { surface: 'tui' })
    expect(printed(sys)).toContain('listening')
    expect(printed(sys)).toContain('hey hermes')
    expect(printed(sys)).toContain('openwakeword')
  })

  it('/wake on clears the session opt-out flag', async () => {
    setWakeUserDisabled(true)

    const { run } = buildCtx({ 'wake.start': { started: true } })

    await run('on')

    expect(isWakeUserDisabled()).toBe(false)
  })

  it('/wake on prints the reason when the gateway refuses', async () => {
    const { run, sys } = buildCtx({
      'wake.start': { owner_surface: 'gui', reason: 'owned', started: false }
    })

    await run('on')

    const out = printed(sys)
    expect(out).toContain('not started')
    expect(out).toContain('another surface owns the listener')
    expect(out).toContain('gui')
  })

  it('/wake on surfaces the hint when unavailable', async () => {
    const { run, sys } = buildCtx({
      'wake.start': { hint: 'pip install openwakeword', reason: 'unavailable', started: false }
    })

    await run('on')

    const out = printed(sys)
    expect(out).toContain('unavailable')
    expect(out).toContain('pip install openwakeword')
  })

  it('/wake off calls wake.stop, remembers the opt-out, and reports', async () => {
    const { rpc, run, sys } = buildCtx({ 'wake.stop': { stopped: true } })

    await run('off')

    expect(rpc).toHaveBeenCalledWith('wake.stop', {})
    expect(isWakeUserDisabled()).toBe(true)
    expect(printed(sys)).toContain('listener off')
  })

  it('/wake off explains a not_owner refusal but still records the opt-out', async () => {
    const { run, sys } = buildCtx({ 'wake.stop': { reason: 'not_owner', stopped: false } })

    await run('off')

    expect(isWakeUserDisabled()).toBe(true)
    expect(printed(sys)).toContain('nothing to stop')
    expect(printed(sys)).toContain('doesn’t own the listener')
  })

  it('/wake status prints a listening one-liner', async () => {
    const { rpc, run, sys } = buildCtx({
      'wake.status': {
        available: true,
        listening: true,
        owned_by_caller: true,
        owner_surface: 'tui',
        phrase: 'hey hermes',
        provider: 'openwakeword'
      }
    })

    await run('status')

    expect(rpc).toHaveBeenCalledWith('wake.status', {})

    const out = printed(sys)
    expect(out).toContain('listening')
    expect(out).toContain('hey hermes')
    expect(out).toContain('openwakeword')
  })

  it('bare /wake behaves like /wake status', async () => {
    const { rpc, run } = buildCtx({ 'wake.status': { available: true, listening: false } })

    await run('')

    expect(rpc).toHaveBeenCalledWith('wake.status', {})
  })

  it('status reports another surface owning the listener', async () => {
    const { run, sys } = buildCtx({
      'wake.status': {
        available: true,
        listening: false,
        owned_by_caller: false,
        owner_surface: 'gui',
        phrase: 'hey hermes'
      }
    })

    await run('status')

    const out = printed(sys)
    expect(out).toContain('off here')
    expect(out).toContain('gui')
  })

  it('status surfaces the hint when the wake word is unavailable', async () => {
    const { run, sys } = buildCtx({
      'wake.status': { available: false, hint: 'no microphone detected', listening: false }
    })

    await run('status')

    const out = printed(sys)
    expect(out).toContain('unavailable')
    expect(out).toContain('no microphone detected')
  })

  it('rejects unknown subcommands with usage text', async () => {
    const { rpc, run, sys } = buildCtx({})

    await run('banana')

    expect(rpc).not.toHaveBeenCalled()
    expect(printed(sys)).toContain('usage: /wake [on|off|status]')
  })
})
