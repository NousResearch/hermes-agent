import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { coreCommands } from '../app/slash/commands/core.js'
import { t } from '../i18n/runtime.js'

const quit = coreCommands.find(command => command.name === 'quit')!

interface RpcCall {
  method: string
  params?: Record<string, unknown>
}

const buildCtx = (options: { canonical?: boolean; mutate?: () => Promise<unknown>; resume?: () => Promise<unknown> } = {}) => {
  const calls: RpcCall[] = []
  const die = vi.fn()
  const sys = vi.fn()

  const gw = {
    isCanonical: options.canonical ?? true,
    request: vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'session.resume') {
        return options.resume ? options.resume() : { execution_generation: 4, revision: 12 }
      }

      if (method === 'session.mutate') {
        return options.mutate ? options.mutate() : { operation: 'delete', session_id: 'sid-1' }
      }

      if (method === 'session.delete') {
        return { deleted: params?.session_id }
      }

      return {}
    })
  }

  const ctx = {
    gateway: { gw, rpc: vi.fn() },
    guarded: <T>(fn: (result: T) => void) => fn,
    guardedErr: vi.fn(),
    session: { die },
    sid: 'sid-1',
    stale: () => false,
    transcript: { page: vi.fn(), panel: vi.fn(), sys },
    ui: { destructiveSlashConfirm: false, storedSid: 'stored-1' }
  }

  const run = (arg: string) => quit.run(arg, ctx as never, `/quit ${arg}`.trim())

  return { calls, ctx, die, run, sys }
}

const flush = async () => {
  await vi.advanceTimersByTimeAsync(200)
}

describe('/quit --delete', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('deletes through the canonical session.mutate before exiting', async () => {
    const { calls, die, run, sys } = buildCtx()

    run('--delete')
    await flush()

    expect(calls.map(call => call.method)).toEqual(['session.resume', 'session.mutate'])
    expect(calls[1]?.params).toMatchObject({ operation: 'delete', payload: {}, session_id: 'sid-1' })
    expect(sys).toHaveBeenCalledWith(t('slashCmd.core.quit.deleted'))
    expect(die).toHaveBeenCalledTimes(1)
  })

  it('accepts -d as the short form', async () => {
    const { calls, die, run } = buildCtx()

    run('-d')
    await flush()

    expect(calls.at(-1)?.method).toBe('session.mutate')
    expect(die).toHaveBeenCalledTimes(1)
  })

  it('stays in the session and reports the reason when the gateway refuses the delete', async () => {
    const { die, run, sys } = buildCtx({
      mutate: () => Promise.reject(Object.assign(new Error('busy'), { data: { reason: 'busy' } }))
    })

    run('--delete')
    await flush()

    expect(die).not.toHaveBeenCalled()
    expect(sys).toHaveBeenCalledWith(t('slashCmd.core.quit.deleteFailed', 'busy'))
  })

  it('quits without an RPC when no delete was asked for', async () => {
    const { calls, die, run } = buildCtx()

    run('')
    await flush()

    expect(calls).toEqual([])
    expect(die).toHaveBeenCalledTimes(1)
  })

  it('refuses an unknown argument instead of quitting silently', async () => {
    const { calls, die, run, sys } = buildCtx()

    run('--nope')
    await flush()

    expect(calls).toEqual([])
    expect(die).not.toHaveBeenCalled()
    expect(sys).toHaveBeenCalledWith(t('slashCmd.core.quit.deleteUsage'))
  })

  it('closes the live session, then deletes the stored id on a legacy transport', async () => {
    const { calls, die, run } = buildCtx({ canonical: false })

    run('--delete')
    await flush()

    // `session.delete` takes the stored id and refuses a session still live here (4023), so the
    // runtime view is closed first — the order the switcher's ^d uses.
    expect(calls).toEqual([
      { method: 'session.close', params: { session_id: 'sid-1' } },
      { method: 'session.delete', params: { session_id: 'stored-1' } }
    ])
    expect(die).toHaveBeenCalledTimes(1)
  })

  it('stays in the session when a legacy delete is refused', async () => {
    const { ctx, die, run, sys } = buildCtx({ canonical: false })

    ctx.gateway.gw.request = vi.fn(async (method: string) =>
      method === 'session.delete' ? Promise.reject(Object.assign(new Error('live'), { data: { reason: 'live' } })) : {}
    )
    run('--delete')
    await flush()

    expect(die).not.toHaveBeenCalled()
    expect(sys).toHaveBeenCalledWith(t('slashCmd.core.quit.deleteFailed', 'live'))
  })
})
