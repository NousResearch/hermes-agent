import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  getAttentionHook,
  getNotifyOnInteract,
  resetAttentionConfigForTests,
  setAttentionHook
} from '../app/attentionConfigStore.js'
import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { createServerRequestHandler } from '../app/createServerRequestHandler.js'
import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { resetServerRequestsForTests } from '../app/serverRequestStore.js'
import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { ATTENTION_EVENT_TITLES, notifyAttention, ringBell } from '../lib/notify.js'
import type { Msg } from '../types.js'

// Attention plumbing is a subprocess surface; intercept the spawn so tests
// assert the contract without executing real commands. notify.ts uses only
// `spawn`/`execFile`/`platform` from these modules, so plain factories suffice.
const spawnMock = vi.fn(() => ({
  on: () => undefined,
  unref: () => undefined
}) as never)

vi.mock('node:child_process', () => ({
  execFile: vi.fn(),
  spawn: (...args: unknown[]) => spawnMock(...(args as []))
}))

vi.mock('node:os', () => ({ platform: () => 'linux' }))

const ref = <T>(current: T) => ({ current })

const buildCtx = (appended: Msg[], bellOnComplete = false) =>
  ({
    composer: {
      dequeue: () => undefined,
      queueEditRef: ref<null | number>(null),
      sendQueued: vi.fn(),
      setInput: vi.fn()
    },
    gateway: {
      gw: { request: vi.fn() },
      rpc: vi.fn(async () => null)
    },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: ref(80),
      newSession: vi.fn(),
      resetSession: vi.fn(),
      resumeById: vi.fn(),
      setCatalog: vi.fn()
    },
    submission: {
      submitRef: { current: vi.fn() }
    },
    system: {
      bellOnComplete,
      stdout: { isTTY: false, write: vi.fn() },
      sys: vi.fn()
    },
    transcript: {
      appendMessage: (msg: Msg) => appended.push(msg),
      panel: vi.fn(),
      setHistoryItems: vi.fn()
    },
    voice: {
      setProcessing: vi.fn(),
      setRecording: vi.fn(),
      setVoiceEnabled: vi.fn()
    }
  }) as any

/** One attention hook spawn, decoded: [file, ...args] + env. */
const hookCall = () => {
  const call = spawnMock.mock.calls.at(-1)

  if (!call) {
    return null
  }

  const [file, args, options] = call as unknown as [string, string[], { env?: Record<string, string> }]

  return { args, env: options?.env ?? {}, file }
}

describe('notify — attention hook (#46357)', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetOverlayState()
    resetUiState()
    resetTurnState()
    resetServerRequestsForTests()
    resetAttentionConfigForTests()
    turnController.fullReset()
    spawnMock.mockImplementation(
      () =>
        ({
          on: () => undefined,
          unref: () => undefined
        }) as never
    )
  })

  it('fires the hook on turn.completed with sanitized payload', () => {
    patchUiState({ sid: 'sess-42' })
    setAttentionHook({ command: 'notify-send -a Hermes', enabled: true })
    const onEvent = createGatewayEventHandler(buildCtx([]))

    onEvent({ payload: { text: 'done — all tests pass' }, type: 'message.complete' } as any)

    const call = hookCall()

    expect(call).not.toBeNull()
    expect(call!.file).toBe('notify-send')
    expect(call!.args).toEqual(['-a', 'Hermes', 'turn.completed', 'Turn complete', 'done — all tests pass'])
    expect(call!.env.HERMES_ATTENTION_EVENT).toBe('turn.completed')
    expect(call!.env.HERMES_ATTENTION_SESSION_ID).toBe('sess-42')
  })

  it('fires turn.blocked on a failed turn and collapses whitespace in the message', () => {
    setAttentionHook({ command: 'hook', enabled: true })
    const onEvent = createGatewayEventHandler(buildCtx([]))

    onEvent({
      payload: { error: 'boom\n\nsecond   line', status: 'error' },
      type: 'message.complete'
    } as any)

    const call = hookCall()

    expect(call).not.toBeNull()
    expect(call!.env.HERMES_ATTENTION_EVENT).toBe('turn.blocked')
    expect(call!.args[0]).toBe('turn.blocked')
    // The failed-turn message is describeTurnFailure()'s copy with the raw
    // error folded in; whitespace collapses to single spaces (sanitize contract).
    expect(call!.env.HERMES_ATTENTION_MESSAGE).toContain('boom second line')
    expect(call!.env.HERMES_ATTENTION_MESSAGE).not.toMatch(/\n/)
  })

  it('caps long messages and skips when disabled or commandless', () => {
    setAttentionHook({ command: 'hook', enabled: true })
    const onEvent = createGatewayEventHandler(buildCtx([]))

    onEvent({
      payload: { text: 'x'.repeat(500) },
      type: 'message.complete'
    } as any)

    expect(hookCall()!.env.HERMES_ATTENTION_MESSAGE.length).toBeLessThanOrEqual(202)

    // Disabled → no spawn at all.
    setAttentionHook({ command: 'hook', enabled: false })
    spawnMock.mockClear()
    onEvent({ payload: { text: 'again' }, type: 'message.complete' } as any)
    expect(spawnMock).not.toHaveBeenCalled()

    // Enabled but empty command → no spawn.
    setAttentionHook({ command: '', enabled: true })
    onEvent({ payload: { text: 'once more' }, type: 'message.complete' } as any)
    expect(spawnMock).not.toHaveBeenCalled()
  })

  it('does not fire on interrupted turns', () => {
    setAttentionHook({ command: 'hook', enabled: true })
    const onEvent = createGatewayEventHandler(buildCtx([]))

    // A Ctrl+C-sealed turn: turnController.interrupted is true when the
    // completion lands — the attention path skips it like the bell does.
    turnController.interrupted = true
    onEvent({ payload: { text: 'partial' }, type: 'message.complete' } as any)

    expect(spawnMock).not.toHaveBeenCalled()
  })

  it('fires input.needed / approval.needed / sudo.needed when blocking prompts open, never on replay', () => {
    setAttentionHook({ command: 'hook', enabled: true })
    patchUiState({ sid: 'sess-7' })

    const handler = createServerRequestHandler({
      notifyPromptAttention: payload => notifyAttention(payload),
      setStatus: () => undefined
    })

    handler({ fail: vi.fn(), id: 'a', method: 'clarify', params: { question: 'Which DB?', session_id: 'sess-7' }, respond: vi.fn() } as any)
    expect(hookCall()!.env.HERMES_ATTENTION_EVENT).toBe('input.needed')
    expect(hookCall()!.env.HERMES_ATTENTION_MESSAGE).toBe('Which DB?')

    handler({ fail: vi.fn(), id: 'b', method: 'approval', params: { command: 'rm -rf /', session_id: 'sess-7' }, respond: vi.fn() } as any)
    expect(hookCall()!.env.HERMES_ATTENTION_EVENT).toBe('approval.needed')

    handler({ fail: vi.fn(), id: 'c', method: 'sudo', params: { session_id: 'sess-7' }, respond: vi.fn() } as any)
    expect(hookCall()!.env.HERMES_ATTENTION_EVENT).toBe('sudo.needed')

    // A replayed prompt (reconnect) must not re-fire attention.
    spawnMock.mockClear()
    handler({
      fail: vi.fn(),
      id: 'd',
      method: 'clarify',
      params: { question: 'replayed?', session_id: 'sess-7' },
      replayed: true,
      respond: vi.fn()
    } as any)
    expect(getOverlayState().clarify).not.toBeNull()
    expect(spawnMock).not.toHaveBeenCalled()
  })

  it('exposes a stable title per event category', () => {
    expect(ATTENTION_EVENT_TITLES['turn.completed']).toBe('Turn complete')
    expect(ATTENTION_EVENT_TITLES['input.needed']).toBe('Input needed')
  })
})

describe('notify — ringBell (#25022)', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('writes BEL on a TTY and plays the paplay sound on Linux', () => {
    const write = vi.fn()
    const execMock = vi.fn()

    ringBell({ isTTY: true, write } as unknown as NodeJS.WriteStream, {
      exec: execMock as never,
      platform: () => 'linux'
    })
    expect(write).toHaveBeenCalledWith('\x07')
    expect(execMock).toHaveBeenCalledWith(
      'paplay',
      ['/usr/share/sounds/freedesktop/stereo/message-new-instant.oga'],
      { timeout: 10_000 },
      expect.any(Function)
    )
  })

  it('plays the paplay sound even when BEL cannot (no TTY / non-Linux)', () => {
    const execMock = vi.fn()
    const write = vi.fn()

    // No TTY: BEL is impossible, but the paplay fallback still fires on Linux —
    // that is exactly the Wayland/SSH case the sound exists for.
    ringBell({ isTTY: false, write } as unknown as NodeJS.WriteStream, {
      exec: execMock as never,
      platform: () => 'linux'
    })
    expect(write).not.toHaveBeenCalled()
    expect(execMock).toHaveBeenCalled()

    // Non-Linux: no paplay exists, BEL alone.
    execMock.mockClear()
    ringBell({ isTTY: true, write } as unknown as NodeJS.WriteStream, {
      exec: execMock as never,
      platform: () => 'darwin'
    })
    expect(write).toHaveBeenCalledWith('\x07')
    expect(execMock).not.toHaveBeenCalled()

    // No stdout at all: sound only.
    execMock.mockClear()
    ringBell(undefined, { exec: execMock as never, platform: () => 'linux' })
    expect(execMock).toHaveBeenCalled()
  })

  it('applyDisplay syncs notify_on_interact and tui_attention_hook into the store', async () => {
    const { hydrateFullConfig } = await import('../app/useConfigSync.js')

    const gw = {
      request: async () => ({
        config: {
          display: {
            notify_on_interact: true,
            tui_attention_hook: { command: '  notify-send -a Hermes  ', enabled: true }
          }
        }
      })
    } as never

    await hydrateFullConfig(gw, () => undefined)

    expect(getNotifyOnInteract()).toBe(true)
    expect(getAttentionHook()).toEqual({ command: 'notify-send -a Hermes', enabled: true })

    // A failed config fetch keeps the last-known flags (quietRpc → null).
    const failing = { request: async () => Promise.reject(new Error('rpc down')) } as never

    await hydrateFullConfig(failing, () => undefined)
    expect(getNotifyOnInteract()).toBe(true)
    expect(getAttentionHook().enabled).toBe(true)

    // A later successful fetch with the keys absent reverts to defaults.
    const bare = { request: async () => ({ config: { display: {} } }) } as never

    await hydrateFullConfig(bare, () => undefined)
    expect(getNotifyOnInteract()).toBe(false)
    expect(getAttentionHook()).toEqual({ command: '', enabled: false })
  })
})
