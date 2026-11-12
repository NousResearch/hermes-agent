import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const ink = vi.hoisted(() => ({ supported: true }))

vi.mock('@hermes/ink', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  isProgressReportingAvailable: () => ink.supported
}))

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { turnController } from '../app/turnController.js'
import { patchUiState } from '../app/uiStore.js'

const INDETERMINATE = '\x1b]9;4;3;\x07'
const CLEAR = '\x1b]9;4;0;\x07'
const ERROR = '\x1b]9;4;2;100\x07'

const ctx = () =>
  ({
    composer: { dequeue: () => undefined, queueEditRef: { current: null }, sendQueued: vi.fn(), setInput: vi.fn() },
    gateway: { gw: { request: vi.fn(async () => null) }, rpc: vi.fn(async () => null) },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: { current: 80 },
      newSession: vi.fn(),
      resetSession: vi.fn(),
      resumeById: vi.fn(),
      setCatalog: vi.fn()
    },
    submission: { submitRef: { current: vi.fn() } },
    system: { bellOnComplete: false, sys: vi.fn() },
    transcript: { appendMessage: vi.fn(), panel: vi.fn(), setHistoryItems: vi.fn() },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
  }) as any

let written: string[]

const progressWrites = () => written.filter(s => s.startsWith('\x1b]9;4;'))

describe('terminal progress (OSC 9;4) across a turn', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.stubEnv('TMUX', '')
    vi.stubEnv('STY', '')
    ink.supported = true
    written = []
    vi.spyOn(process.stdout, 'write').mockImplementation(((chunk: string) => {
      written.push(String(chunk))

      return true
    }) as any)
    patchUiState({ busy: false, sid: 'sess-1', terminalProgress: true })
  })

  afterEach(() => {
    turnController.reset()
    vi.restoreAllMocks()
    vi.unstubAllEnvs()
    vi.useRealTimers()
  })

  it('pulses on message.start, keeps it alive, and clears on a completed turn', () => {
    const onEvent = createGatewayEventHandler(ctx())

    onEvent({ payload: {}, type: 'message.start' } as any)
    expect(progressWrites()).toEqual([INDETERMINATE])

    vi.advanceTimersByTime(5_000)
    expect(progressWrites()).toEqual([INDETERMINATE, INDETERMINATE])

    onEvent({ payload: { status: 'complete', text: 'done' }, type: 'message.complete' } as any)
    vi.advanceTimersByTime(20_000)
    expect(progressWrites()).toEqual([INDETERMINATE, INDETERMINATE, CLEAR])
  })

  it('settles red on a failed completion and on an error event', () => {
    const onEvent = createGatewayEventHandler(ctx())

    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { error: 'boom', status: 'error', text: 'Error: boom' }, type: 'message.complete' } as any)
    expect(progressWrites()).toEqual([INDETERMINATE, ERROR])

    written = []
    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { message: 'provider down' }, type: 'error' } as any)
    vi.advanceTimersByTime(20_000)
    expect(progressWrites()).toEqual([INDETERMINATE, ERROR])
  })

  it('clears when the user interrupts the turn', async () => {
    const onEvent = createGatewayEventHandler(ctx())

    onEvent({ payload: {}, type: 'message.start' } as any)
    await turnController.interruptTurn({
      appendMessage: vi.fn(),
      gw: { isCanonical: false, request: vi.fn(async () => null) } as any,
      sid: 'sess-1',
      sys: vi.fn()
    })
    vi.advanceTimersByTime(20_000)
    expect(progressWrites()).toEqual([INDETERMINATE, CLEAR])
  })

  it('writes nothing when display.terminal_progress is off or the terminal cannot show it', () => {
    const onEvent = createGatewayEventHandler(ctx())

    patchUiState({ terminalProgress: false })
    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { status: 'error', text: 'x' }, type: 'message.complete' } as any)

    patchUiState({ terminalProgress: true })
    ink.supported = false
    onEvent({ payload: {}, type: 'message.start' } as any)
    vi.advanceTimersByTime(20_000)
    onEvent({ payload: { status: 'complete', text: 'done' }, type: 'message.complete' } as any)

    expect(progressWrites()).toEqual([])
  })
})
