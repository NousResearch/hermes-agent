import { beforeEach, describe, expect, it, vi } from 'vitest'

import { coreCommands } from '../app/slash/commands/core.js'

const focusCommand = coreCommands.find(cmd => cmd.name === 'focus')!

/**
 * Regression: /focus must carry the live session id.
 *
 * The backend (tui_gateway/methods_config_set.py::_set_focus) gates
 * tool.start/tool.complete emission on the SESSION's own copy of
 * `tool_progress_mode`, snapshotted at session start. It only rewrites that
 * copy when `config.set` arrives with a session_id that resolves to a live
 * session; otherwise it writes config.yaml and returns, leaving the running
 * session streaming tool rows while the status-bar badge claims focus is on.
 */
const buildCtx = (focusView: boolean) => {
  const sys = vi.fn()
  const rpc = vi.fn(() => Promise.resolve({ key: 'focus', tool_progress: 'off', value: 'on' }))

  const ctx = {
    gateway: { rpc },
    guarded: <T>(fn: (r: T) => void) => fn,
    guardedErr: vi.fn(),
    sid: 'sid-1',
    stale: () => false,
    transcript: { page: vi.fn(), sys },
    ui: { focusView }
  }

  return { ctx, rpc, sys }
}

describe('/focus slash command', () => {
  beforeEach(() => vi.clearAllMocks())

  it('sends session_id so the live session stops emitting tool rows', () => {
    const { ctx, rpc } = buildCtx(false)

    focusCommand.run('on', ctx as any, '/focus on')

    expect(rpc).toHaveBeenCalledWith('config.set', { key: 'focus', session_id: 'sid-1', value: 'on' })
  })

  it('restores the stashed tool-progress mode on the same session', () => {
    const { ctx, rpc } = buildCtx(true)

    focusCommand.run('off', ctx as any, '/focus off')

    expect(rpc).toHaveBeenCalledWith('config.set', { key: 'focus', session_id: 'sid-1', value: 'off' })
  })

  it('/focus status reports without touching the gateway', () => {
    const { ctx, rpc, sys } = buildCtx(true)

    focusCommand.run('status', ctx as any, '/focus status')

    expect(rpc).not.toHaveBeenCalled()
    expect(sys.mock.calls.map(c => c[0]).join('\n')).toContain('focus view on')
  })
})
