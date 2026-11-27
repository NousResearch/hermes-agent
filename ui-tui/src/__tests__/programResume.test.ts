import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useProgramResume } from '../app/useProgramResume.js'
import { ZERO } from '../domain/usage.js'
import {
  $persistedSessions,
  $programResumeEnabled,
  buildTuiResumeArgv,
  clearProgramResume,
  clearProgramResumeOnExit,
  consumeProgramResumeEnv,
  encodeProgramResume,
  encodeProgramResumeClear,
  isValidOsc7577Token,
  markProgramResumeReported,
  markProgramResumeUserQuit,
  markSessionPersisted,
  resetProgramResumeStateForTests,
  suppressProgramResumeClear
} from '../lib/programResume.js'

const SPEC_TOKEN = 'dummy-token-replace-me'
const SPEC_TIME = 1795543200
const SPEC_PID = 74877

const decodeArgv = (seq: string): string[] => {
  const body = seq.slice('\x1b]7577;'.length, -'\x1b\\'.length)
  const argvPart = body.split(':').find(part => part.startsWith('argv='))

  return argvPart ? argvPart.slice('argv='.length).split(',').map(b64 => Buffer.from(b64, 'base64').toString('utf8')) : []
}

const PREFIX = ['/usr/local/bin/hermes', '-p', 'default', '--tui']

// useProgramResume rendered against a TTY stand-in: `writes` collects what reaches the terminal and
// `settle()` re-renders past the hook's settle delay.
function mountProbe(env: NodeJS.ProcessEnv = {}) {
  vi.useFakeTimers()
  vi.setSystemTime(SPEC_TIME * 1000)
  consumeProgramResumeEnv({
    HERMES_TUI_OWNER_PID: String(SPEC_PID),
    HERMES_TUI_RESUME_ARGV: JSON.stringify(PREFIX),
    OSC7577_TOKEN: SPEC_TOKEN,
    ...env
  })

  const writes: string[] = []
  const ttyOut = Object.assign(new PassThrough(), { columns: 80, isTTY: true, rows: 24 })
  const origWrite = ttyOut.write.bind(ttyOut)
  ttyOut.write = ((chunk: unknown, ...rest: unknown[]) => {
    writes.push(String(chunk))

    return (origWrite as (...args: unknown[]) => boolean)(chunk, ...rest)
  }) as typeof ttyOut.write

  function Probe() {
    useProgramResume(ttyOut as unknown as NodeJS.WriteStream)

    return null
  }

  const app = renderSync(React.createElement(Probe), {
    patchConsole: false,
    stderr: Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 }) as unknown as NodeJS.WriteStream,
    stdin: Object.assign(new PassThrough(), { isTTY: false }) as unknown as NodeJS.ReadStream,
    stdout: ttyOut as unknown as NodeJS.WriteStream
  })

  const settle = () => {
    app.rerender(React.createElement(Probe))
    vi.advanceTimersByTime(60)
  }

  return { settle, unmount: () => app.unmount(), writes }
}

describe('OSC 7577 program resume encoder', () => {
  beforeEach(() => {
    resetProgramResumeStateForTests()
    resetUiState()
    $programResumeEnabled.set(null)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllEnvs()
    resetProgramResumeStateForTests()
  })

  it('matches all three official OSC 7577 rev 0.5 spec test vectors byte-for-byte', () => {
    expect(
      encodeProgramResume(['nvim', '-S', '/home/user/.local/state/nvim/sessions/8f2c1d.vim'], {
        now: SPEC_TIME,
        pid: SPEC_PID,
        token: SPEC_TOKEN
      })
    ).toBe(
      '\x1b]7577;pid=74877:time=1795543200:' +
        'argv=bnZpbQ==,LVM=,L2hvbWUvdXNlci8ubG9jYWwvc3RhdGUvbnZpbS9zZXNzaW9ucy84ZjJjMWQudmlt:' +
        'mac=9dbb99bfb856749a87563839d7ce9062818d78cf25909db93f309e0cd3b1909b\x1b\\'
    )

    expect(encodeProgramResumeClear({ now: SPEC_TIME, pid: SPEC_PID, token: SPEC_TOKEN })).toBe(
      '\x1b]7577;pid=74877:time=1795543200:clear=1:' +
        'mac=c30603369f98fef581a4098be889175b0c74f0641e49f8b25f4326cb38f97c93\x1b\\'
    )

    expect(
      encodeProgramResume(['lazygit'], {
        cwd: '/home/user/src/project',
        now: SPEC_TIME,
        pid: SPEC_PID,
        token: SPEC_TOKEN
      })
    ).toBe(
      '\x1b]7577;pid=74877:time=1795543200:argv=bGF6eWdpdA==:' +
        'cwd=L2hvbWUvdXNlci9zcmMvcHJvamVjdA==:' +
        'mac=35cb909805c380336efd3689a125acc774a9dec8422a1d21492ebec06ca758e1\x1b\\'
    )
  })

  it('validates tokens, argv elements, cwd, and the 2048-byte ceiling', () => {
    expect(isValidOsc7577Token(SPEC_TOKEN)).toBe(true)

    for (const badToken of ['', 'short', 'a'.repeat(65), `${SPEC_TOKEN}\n`, ` ${SPEC_TOKEN}`, 'bad$token_123456789012']) {
      expect(isValidOsc7577Token(badToken)).toBe(false)
      expect(encodeProgramResume(['hermes'], { now: 1, pid: 1, token: badToken })).toBeNull()
    }

    for (const badArgv of [[], ['./hermes'], ['bin/hermes'], ['-hermes'], [''], ['hermes', 'x\n'], ['hermes', 'x\u0085'], Array(65).fill('a')]) {
      expect(encodeProgramResume(badArgv, { now: 1, pid: 1, token: SPEC_TOKEN })).toBeNull()
    }

    expect(encodeProgramResume(['hermes'], { cwd: 'relative', now: 1, pid: 1, token: SPEC_TOKEN })).toBeNull()
    expect(encodeProgramResume(['hermes', '--resume', 's'.repeat(2000)], { now: 1, pid: 1, token: SPEC_TOKEN })).toBeNull()
  })

  it('caches env metadata and strips OSC7577_TOKEN from process.env before spawning children', () => {
    const fakeEnv: NodeJS.ProcessEnv = {
      HERMES_TUI_OWNER_PID: String(SPEC_PID),
      HERMES_TUI_RESUME_ARGV: JSON.stringify(['/usr/local/bin/hermes', '-p', 'work', '--tui']),
      OSC7577_TOKEN: SPEC_TOKEN,
      OTHER_VAR: 'keep'
    }

    consumeProgramResumeEnv(fakeEnv)

    expect(fakeEnv.OSC7577_TOKEN).toBeUndefined()
    expect(fakeEnv.HERMES_TUI_OWNER_PID).toBeUndefined()
    expect(fakeEnv.HERMES_TUI_RESUME_ARGV).toBeUndefined()
    expect(fakeEnv.OTHER_VAR).toBe('keep')

    expect(buildTuiResumeArgv('sess-42')).toEqual(['/usr/local/bin/hermes', '-p', 'work', '--tui', '--resume', 'sess-42'])

    const seq = encodeProgramResume(buildTuiResumeArgv(null)!, { now: SPEC_TIME })
    expect(seq).toContain(`pid=${SPEC_PID}:time=${SPEC_TIME}:`)
  })

  it('a bypass launch reports only a saved session (no bare form)', () => {
    const env: NodeJS.ProcessEnv = {
      HERMES_TUI_RESUME_ARGV: JSON.stringify(['/usr/local/bin/hermes', '-p', 'default', '--tui', '--safe-mode']),
      HERMES_TUI_RESUME_SESSION_ONLY: '1',
      OSC7577_TOKEN: SPEC_TOKEN
    }

    consumeProgramResumeEnv(env)
    expect(env.HERMES_TUI_RESUME_SESSION_ONLY).toBeUndefined()
    expect(buildTuiResumeArgv(null)).toBeNull()
    expect(buildTuiResumeArgv('s1')).toEqual(['/usr/local/bin/hermes', '-p', 'default', '--tui', '--safe-mode', '--resume', 's1'])
  })

  it('clears on user quit, stays silent on SIGHUP/SIGTERM/EIO, and never throws on dead tty', () => {
    const out: string[] = []
    const write = (seq: string) => void out.push(seq)

    expect(clearProgramResume(write, { now: SPEC_TIME, pid: SPEC_PID, token: SPEC_TOKEN })).toBe(false)
    expect(out).toEqual([])

    markProgramResumeReported()
    expect(clearProgramResume(write, { now: SPEC_TIME, pid: SPEC_PID, token: SPEC_TOKEN })).toBe(true)
    expect(clearProgramResume(write, { now: SPEC_TIME, pid: SPEC_PID, token: SPEC_TOKEN })).toBe(false)
    expect(out).toHaveLength(1)
    expect(out[0]).toContain(':clear=1:mac=')

    markProgramResumeReported()
    suppressProgramResumeClear()
    expect(clearProgramResume(write, { now: SPEC_TIME, pid: SPEC_PID, token: SPEC_TOKEN })).toBe(false)
    expect(out).toHaveLength(1)

    resetProgramResumeStateForTests()
    markProgramResumeReported()
    expect(
      clearProgramResume(
        () => {
          throw new Error('EIO')
        },
        { now: SPEC_TIME, pid: SPEC_PID, token: SPEC_TOKEN }
      )
    ).toBe(false)
  })

  it('argv[0] rules match the Python reporter', () => {
    // Same table as tests/hermes_cli/test_program_resume.py ARGV0_CASES.
    const cases: [string, boolean][] = [
      ['/usr/local/bin/hermes', true],
      ['hermes', true],
      ['C:\\Program Files\\hermes.exe', true],
      ['c:/hermes/hermes.exe', true],
      ['\\\\server\\share\\hermes.exe', true],
      ['\\hermes.exe', false],
      ['C:hermes.exe', true], // no separator: a bare command name
      ['./hermes', false],
      ['bin/hermes', false],
      ['-hermes', false],
      ['', false],
      ['/usr/bin/her\x1bmes', false]
    ]

    for (const [arg, valid] of cases) {
      const seq = encodeProgramResume([arg], { now: SPEC_TIME, pid: SPEC_PID, token: SPEC_TOKEN })
      expect([arg, seq !== null]).toEqual([arg, valid])
    }
  })

  it('reports nothing without the launcher prefix (a home no command can reach)', () => {
    const probe = mountProbe({ HERMES_TUI_RESUME_ARGV: '' })
    expect(buildTuiResumeArgv('sess-1')).toBeNull()
    $programResumeEnabled.set(true)
    patchUiState({ storedSid: 'sess-1' })
    probe.settle()
    expect(probe.writes).toEqual([])
    probe.unmount()
  })

  it('reports --resume only for a session the gateway confirmed saved', () => {
    const probe = mountProbe()

    probe.settle()
    expect(probe.writes).toEqual([]) // $programResumeEnabled is null until the first config read

    $programResumeEnabled.set(true)
    // A built agent, a pending submit and token usage are no proof of a state.db row.
    patchUiState({ busy: true, info: { model: 'test', skills: {}, tools: {} }, storedSid: 'fresh-1', usage: { ...ZERO, input: 10 } })
    probe.settle()
    expect(probe.writes.map(decodeArgv)).toEqual([PREFIX])

    markSessionPersisted('fresh-1') // an inline turn's prompt.submit, or a resume with stored history
    probe.settle()
    expect(decodeArgv(probe.writes.at(-1)!)).toEqual([...PREFIX, '--resume', 'fresh-1'])

    patchUiState({ busy: false, storedSid: 'fresh-2', usage: ZERO }) // /new
    probe.settle()
    expect(decodeArgv(probe.writes.at(-1)!)).toEqual(PREFIX)

    markSessionPersisted('x'.repeat(2000))
    patchUiState({ storedSid: 'x'.repeat(2000) }) // unreportable: clear the stale record
    probe.settle()
    expect(probe.writes.at(-1)).toContain(':clear=1:mac=')
    probe.unmount()
  })

  it('tracks saved sessions only when this pane reports', () => {
    consumeProgramResumeEnv({})
    markSessionPersisted('s1')
    expect($persistedSessions.get().size).toBe(0)
  })

  it('reports again after SIGCONT (back in the foreground)', () => {
    const probe = mountProbe()
    $programResumeEnabled.set(true)
    patchUiState({ storedSid: 'cont-1' })
    probe.settle()
    expect(probe.writes).toHaveLength(1)

    process.emit('SIGCONT', 'SIGCONT')
    probe.settle()
    expect(probe.writes).toHaveLength(2)
    expect(decodeArgv(probe.writes[1]!)).toEqual(decodeArgv(probe.writes[0]!))

    probe.unmount()
    expect(process.listenerCount('SIGCONT')).toBe(0)
  })

  it('turning display.program_resume off clears the reported record, even right after SIGCONT', () => {
    const probe = mountProbe()
    $programResumeEnabled.set(true)
    patchUiState({ storedSid: 'on-1' })
    probe.settle()
    expect(probe.writes).toHaveLength(1)

    process.emit('SIGCONT', 'SIGCONT') // resets the dedupe key before the delayed re-report
    $programResumeEnabled.set(false)
    probe.settle()
    expect(probe.writes).toHaveLength(2)
    expect(probe.writes[1]).toContain(':clear=1:mac=')

    $programResumeEnabled.set(true)
    probe.settle()
    expect(probe.writes.map(decodeArgv).at(-1)).toEqual(PREFIX)
    probe.unmount()
  })

  it('clears on exit only after a user quit: /update (42) and a crash keep the record', () => {
    const out: string[] = []
    consumeProgramResumeEnv({ OSC7577_TOKEN: SPEC_TOKEN })

    markProgramResumeReported()
    expect(clearProgramResumeOnExit(seq => void out.push(seq))).toBe(false)
    expect(out).toEqual([])

    markProgramResumeUserQuit()
    expect(clearProgramResumeOnExit(seq => void out.push(seq))).toBe(true)
    expect(out).toHaveLength(1)
    expect(out[0]).toContain(':clear=1:mac=')
  })

  it('suppresses OSC 7577 completely in DASHBOARD_TUI_MODE and non-TTY streams', async () => {
    vi.useFakeTimers()
    vi.stubEnv('HERMES_TUI_DASHBOARD', '1')
    vi.resetModules()

    const { useProgramResume: useDashboardResume } = await import('../app/useProgramResume.js')

    const { $programResumeEnabled: dashboardEnabled, consumeProgramResumeEnv: consumeDash } = await import(
      '../lib/programResume.js'
    )

    consumeDash({ OSC7577_TOKEN: SPEC_TOKEN })
    dashboardEnabled.set(true)

    const writes: string[] = []
    const ttyOut = Object.assign(new PassThrough(), { columns: 80, isTTY: true, rows: 24 })
    ttyOut.write = ((chunk: unknown) => {
      writes.push(String(chunk))

      return true
    }) as typeof ttyOut.write

    function DashProbe() {
      useDashboardResume(ttyOut as unknown as NodeJS.WriteStream)

      return null
    }

    const app = renderSync(React.createElement(DashProbe), {
      patchConsole: false,
      stderr: Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 }) as unknown as NodeJS.WriteStream,
      stdin: Object.assign(new PassThrough(), { isTTY: false }) as unknown as NodeJS.ReadStream,
      stdout: ttyOut as unknown as NodeJS.WriteStream
    })

    vi.advanceTimersByTime(100)
    expect(writes).toEqual([])
    app.unmount()
  })
})
