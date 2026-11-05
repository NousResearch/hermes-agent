import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React, { useEffect } from 'react'
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import {
  buildOsc7Sequence,
  isLocalTerminalBackend,
  noteTerminalToolComplete,
  reportCwd,
  reportSessionCwd,
  resetOsc7State,
  terminalCwdFromResult
} from '../lib/osc7.js'
import { wrapForMultiplexer } from '../lib/osc52.js'

const ESC = '\x1b'
const ST = '\x1b\\'
const BEL = '\x07'
const SCREEN_STY = '12345.pts-0.host'
const TMUX_SOCK = '/tmp/tmux-123/default,1,0'

const envBackup = { ...process.env }
const isTTYBackup = process.stdout.isTTY

// Real, distinct, existing local directories (reportCwd only reports a directory that exists here).
let scratch = ''
let DIR_A = ''
let DIR_B = ''
let DIR_C = ''

const ref = <T>(current: T) => ({ current })

const fakeStream = () => {
  const writes: string[] = []

  return {
    stream: {
      isTTY: true,
      write: (s: string) => {
        writes.push(String(s))

        return true
      }
    } as any,
    writes
  }
}

const spyStdout = () => vi.spyOn(process.stdout, 'write').mockImplementation(() => true as any)

const osc7Writes = (spy: ReturnType<typeof spyStdout>) =>
  spy.mock.calls.map(c => String(c[0] ?? '')).filter(s => s.includes(']7;file://'))

const buildCtx = () =>
  ({
    composer: { dequeue: () => undefined, queueEditRef: ref<null | number>(null), sendQueued: vi.fn(), setInput: vi.fn() },
    gateway: { gw: { request: vi.fn(async () => null) }, rpc: vi.fn(async () => null) },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: ref(80),
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

/** A real handler bound to a live session 'live' whose durable id is 'stored'. */
const handlerFor = (info: Record<string, unknown>) => {
  patchUiState({ info: { model: 'test', skills: {}, tools: {}, ...info } as any, sid: 'live', storedSid: 'stored' })

  const onEvent = createGatewayEventHandler(buildCtx())

  return (result: unknown, args: Record<string, unknown> = {}) =>
    onEvent({ payload: { args, name: 'terminal', result, tool_id: 'osc7-review' }, session_id: 'live', type: 'tool.complete' } as any)
}

beforeAll(() => {
  scratch = mkdtempSync(join(tmpdir(), 'osc7-review-'))
  DIR_A = mkdtempSync(join(scratch, 'dir-a-'))
  DIR_B = mkdtempSync(join(scratch, 'dir-b-'))
  DIR_C = mkdtempSync(join(scratch, 'dir-c-'))
})

afterAll(() => {
  rmSync(scratch, { force: true, recursive: true })
})

beforeEach(() => {
  resetUiState()
  resetTurnState()
  turnController.fullReset()
  resetOsc7State()
  delete process.env.TMUX
  delete process.env.STY
  ;(process.stdout as any).isTTY = true
})

afterEach(() => {
  vi.restoreAllMocks()
  process.env = { ...envBackup }
  ;(process.stdout as any).isTTY = isTTYBackup
  resetOsc7State()
})

describe('finding 1: GNU screen framing keeps the inner OSC terminated', () => {
  it('reportCwd under STY: the only ST is the DCS terminator, and the inner OSC 7 ends in BEL', () => {
    process.env.STY = SCREEN_STY
    const { stream, writes } = fakeStream()
    const seq = buildOsc7Sequence(DIR_A)

    expect(reportCwd(DIR_A, stream)).toBe(true)
    expect(writes).toHaveLength(1)

    const out = writes[0]!

    expect(out.startsWith(`${ESC}P`)).toBe(true)
    expect(out.endsWith(ST)).toBe(true)
    // screen ends the DCS at the FIRST ST: it must be the last two bytes.
    expect(out.indexOf(ST)).toBe(out.length - ST.length)

    const inner = out.slice(2, -ST.length)

    expect(inner.endsWith(BEL)).toBe(true)
    expect(inner).toBe(`${seq.slice(0, -ST.length)}${BEL}`)
  })

  it('wrapForMultiplexer under STY swaps an ST terminator for BEL and keeps the payload byte-for-byte', () => {
    process.env.STY = SCREEN_STY
    const seq = buildOsc7Sequence('/Users/josh/my dir', 'host.example')

    expect(seq.endsWith(ST)).toBe(true)
    expect(wrapForMultiplexer(seq)).toBe(`${ESC}P${seq.slice(0, -ST.length)}${BEL}${ST}`)
  })

  it('wrapForMultiplexer under STY leaves a BEL-terminated OSC 52 unchanged inside the DCS', () => {
    process.env.STY = SCREEN_STY
    const osc52 = `${ESC}]52;c;${Buffer.from('hi').toString('base64')}${BEL}`

    expect(wrapForMultiplexer(osc52)).toBe(`${ESC}P${osc52}${ST}`)
  })

  it('TMUX still wins over STY and keeps the ST-terminated OSC with every ESC doubled', () => {
    process.env.TMUX = TMUX_SOCK
    process.env.STY = SCREEN_STY
    const seq = buildOsc7Sequence('/tmp/x', 'host.example')

    expect(wrapForMultiplexer(seq)).toBe(`${ESC}Ptmux;${seq.split(ESC).join(ESC + ESC)}${ST}`)
  })

  it('without a multiplexer the sequence goes out untouched', () => {
    const seq = buildOsc7Sequence('/tmp/x', 'host.example')

    expect(wrapForMultiplexer(seq)).toBe(seq)
  })
})

describe('finding 2: only a local terminal backend reports or remembers a cwd', () => {
  it('isLocalTerminalBackend: unreported or "local" is local, every named remote backend is not', () => {
    for (const local of [undefined, null, '', 'local']) {
      expect(isLocalTerminalBackend(local), String(local)).toBe(true)
    }

    for (const remote of ['ssh', 'docker', 'modal', 'singularity', 'daytona']) {
      expect(isLocalTerminalBackend(remote), remote).toBe(false)
    }
  })

  it('noteTerminalToolComplete for a non-local backend writes nothing and remembers nothing', () => {
    const write = spyStdout()

    noteTerminalToolComplete('s1', 'terminal', { cwd: DIR_A }, { backend: 'ssh' })
    expect(osc7Writes(write)).toEqual([])

    reportSessionCwd('s1', DIR_B)
    expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_B)])
  })

  it('noteTerminalToolComplete for a local or unreported backend writes and remembers (control)', () => {
    const write = spyStdout()

    noteTerminalToolComplete('s1', 'terminal', { cwd: DIR_A }, { backend: 'local' })
    noteTerminalToolComplete('s2', 'terminal', { cwd: DIR_B })
    expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_A), buildOsc7Sequence(DIR_B)])

    write.mockClear()
    reportSessionCwd('s1', DIR_C)
    reportSessionCwd('s2', DIR_C)
    expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_A), buildOsc7Sequence(DIR_B)])
  })

  it('reportSessionCwd for a non-local backend writes neither the remembered cwd nor the fallback', () => {
    const write = spyStdout()

    noteTerminalToolComplete('s1', 'terminal', { cwd: DIR_A })
    write.mockClear()
    reportSessionCwd('s1', DIR_B, 'docker')
    reportSessionCwd('other', DIR_B, 'ssh')
    expect(osc7Writes(write)).toEqual([])

    // The gate drops the report, not the memory.
    reportSessionCwd('s1', DIR_B, 'local')
    expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_A)])
  })

  it('the real handler with session.info terminal_backend "ssh" drops a "/" cwd that exists locally', () => {
    const terminal = handlerFor({ cwd: '/', terminal_backend: 'ssh' })
    const write = spyStdout()

    terminal(JSON.stringify({ cwd: '/', exit_code: 0, output: '/' }), { command: 'cd /' })
    expect(write.mock.calls.map(c => String(c[0]))).toEqual([])

    // ...and did not remember it for the stored session either.
    reportSessionCwd('stored', DIR_B)
    expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_B)])
  })

  it('the real handler with terminal_backend "local" reports the same result (control)', () => {
    const terminal = handlerFor({ cwd: '/', terminal_backend: 'local' })
    const write = spyStdout()

    terminal(JSON.stringify({ cwd: '/', exit_code: 0, output: '/' }), { command: 'cd /' })
    expect(osc7Writes(write)).toEqual([buildOsc7Sequence('/')])
  })
})

describe('finding 3: a one-off workdir command neither moves the terminal nor the memory', () => {
  it('through the real handler: remembered cwd survives a workdir call and a following plain pwd', () => {
    const terminal = handlerFor({ cwd: DIR_C, terminal_backend: 'local' })
    const write = spyStdout()

    terminal({ cwd: DIR_A, exit_code: 0, output: '' }, { command: `cd ${DIR_A}` })
    expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_A)])

    write.mockClear()
    terminal(JSON.stringify({ cwd: DIR_B, exit_code: 0, output: '' }), { command: `cd ${DIR_B}`, workdir: DIR_C })
    expect(osc7Writes(write)).toEqual([])

    terminal({ exit_code: 0, output: DIR_A }, { command: 'pwd' })
    expect(osc7Writes(write)).toEqual([])

    reportSessionCwd('stored', DIR_C)
    expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_A)])
  })

  it('through the real handler: a workdir call with nothing remembered leaves the session on its fallback', () => {
    const terminal = handlerFor({ cwd: DIR_C, terminal_backend: 'local' })
    const write = spyStdout()

    terminal({ cwd: '/', exit_code: 0, output: '' }, { command: 'cd /', workdir: DIR_B })
    expect(osc7Writes(write)).toEqual([])

    reportSessionCwd('stored', DIR_C)
    expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_C)])
  })

  it('a falsy workdir is not a one-off: the cwd is reported and remembered (control)', () => {
    const terminal = handlerFor({ cwd: DIR_C, terminal_backend: 'local' })
    const write = spyStdout()

    terminal({ cwd: DIR_A }, { command: `cd ${DIR_A}`, workdir: '' })
    expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_A)])

    write.mockClear()
    reportSessionCwd('stored', DIR_C)
    expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_A)])
  })

  it('helper level: args.workdir skips both the write and the memory', () => {
    const write = spyStdout()

    noteTerminalToolComplete('s1', 'terminal', { cwd: DIR_B }, { args: { command: 'ls', workdir: DIR_B } })
    expect(osc7Writes(write)).toEqual([])

    reportSessionCwd('s1', DIR_A)
    expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_A)])
  })
})

describe('finding 4 (TUI side): the bounded envelope still carries the cwd to the report', () => {
  it('terminalCwdFromResult reads cwd out of a truncated envelope', () => {
    expect(terminalCwdFromResult('terminal', JSON.stringify({ cwd: '/x', output: '...', truncated: true }))).toBe('/x')
  })

  it('the real handler reports the cwd of a truncated envelope', () => {
    const terminal = handlerFor({ cwd: DIR_C, terminal_backend: 'local' })
    const write = spyStdout()

    terminal(JSON.stringify({ cwd: DIR_A, output: 'x'.repeat(32768), truncated: true }), { command: `cd ${DIR_A} && noisy` })
    expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_A)])
  })
})

describe('finding 5: the real useSessionLifecycle reports the cwd on create / activate / resume', () => {
  let instance: { unmount: () => void } | undefined

  beforeEach(() => {
    delete process.env.HERMES_TUI_ACTIVE_SESSION_FILE
  })

  afterEach(() => {
    instance?.unmount()
    instance = undefined
  })

  const mount = (terminalBackend: string) => {
    let api: ReturnType<typeof useSessionLifecycle> | undefined

    const info = {
      cwd: DIR_C,
      model: 'test',
      skills: {},
      stored_session_id: 'stored',
      terminal_backend: terminalBackend,
      tools: {}
    }

    const request = vi.fn(async () => ({
      info,
      messages: [],
      running: false,
      session_id: 'live',
      status: 'idle',
      stored_session_id: 'stored'
    }))

    function Probe() {
      const lifecycle = useSessionLifecycle({
        colsRef: { current: 80 },
        composerActions: { setComposerTokens: vi.fn() } as any,
        gw: { isCanonical: true, request } as any,
        panel: vi.fn(),
        rpc: request as any,
        scrollRef: { current: null },
        setHistoryItems: vi.fn(),
        setLastUserMsg: vi.fn(),
        setSessionStartedAt: vi.fn(),
        setStickyPrompt: vi.fn(),
        setVoiceProcessing: vi.fn(),
        setVoiceRecording: vi.fn(),
        sys: vi.fn()
      })

      useEffect(() => {
        api = lifecycle
      })

      return null
    }

    const stream = () => Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })

    instance = renderSync(React.createElement(Probe), {
      patchConsole: false,
      stderr: stream() as any,
      stdin: stream() as any,
      stdout: stream() as any
    })

    return () => api!
  }

  const run = async (api: () => ReturnType<typeof useSessionLifecycle>, method: string) => {
    if (method === 'newSession') {
      await api().newSession()
    } else {
      await (api() as any)[method]('stored')
    }

    await vi.waitFor(() => expect(getUiState().sid).toBe('live'))
  }

  it('newSession reports session.info cwd exactly once', async () => {
    const api = mount('local')

    await vi.waitFor(() => expect(api()).toBeTruthy())

    const write = spyStdout()

    await run(api, 'newSession')
    expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_C)])
  })

  it.each(['activateLiveSession', 'resumeById'])(
    '%s reports the remembered terminal cwd (not session.info cwd) exactly once',
    async method => {
      const api = mount('local')

      await vi.waitFor(() => expect(api()).toBeTruthy())
      const write = spyStdout()

      noteTerminalToolComplete('stored', 'terminal', { cwd: DIR_A })
      write.mockClear()

      await run(api, method)
      expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_A)])
    }
  )

  it.each(['newSession', 'activateLiveSession', 'resumeById'])(
    '%s with terminal_backend "ssh" reports nothing',
    async method => {
      const api = mount('ssh')

      await vi.waitFor(() => expect(api()).toBeTruthy())
      const write = spyStdout()

      noteTerminalToolComplete('stored', 'terminal', { cwd: DIR_A })
      write.mockClear()

      await run(api, method)
      expect(osc7Writes(write)).toEqual([])
    }
  )
})
