import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useProgramResume } from '../app/useProgramResume.js'
import { ZERO } from '../domain/usage.js'
import {
  $programResumeEnabled,
  buildTuiResumeArgv,
  clearProgramResume,
  consumeProgramResumeEnv,
  encodeProgramResume,
  encodeProgramResumeClear,
  isValidOsc7577Token,
  markProgramResumeReported,
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

    const seq = encodeProgramResume(buildTuiResumeArgv(null), { now: SPEC_TIME })
    expect(seq).toContain(`pid=${SPEC_PID}:time=${SPEC_TIME}:`)
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

  it('tracks session durability transitions across /new, busy, and config disable in useProgramResume', () => {
    vi.useFakeTimers()
    vi.setSystemTime(SPEC_TIME * 1000)

    consumeProgramResumeEnv({
      HERMES_TUI_OWNER_PID: String(SPEC_PID),
      HERMES_TUI_RESUME_ARGV: JSON.stringify(['/usr/local/bin/hermes', '--tui']),
      OSC7577_TOKEN: SPEC_TOKEN
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

    // Not enabled yet ($programResumeEnabled is null): nothing emitted.
    vi.advanceTimersByTime(100)
    expect(writes).toEqual([])

    // Enabled on an empty lazy session: emits prefix without --resume.
    $programResumeEnabled.set(true)
    patchUiState({ info: { lazy: true, model: 'test', skills: {}, tools: {} }, storedSid: 'lazy-1' })
    app.rerender(React.createElement(Probe))
    vi.advanceTimersByTime(60)
    expect(writes).toHaveLength(1)
    expect(decodeArgv(writes[0]!)).toEqual(['/usr/local/bin/hermes', '--tui'])

    // Session becomes durable (busy): emits --resume lazy-1.
    patchUiState({ busy: true, usage: { ...ZERO, input: 10 } })
    app.rerender(React.createElement(Probe))
    vi.advanceTimersByTime(60)
    expect(writes).toHaveLength(2)
    expect(decodeArgv(writes[1]!)).toEqual(['/usr/local/bin/hermes', '--tui', '--resume', 'lazy-1'])

    // /new creates a fresh lazy session: drops --resume.
    patchUiState({
      busy: false,
      info: { lazy: true, model: 'test', skills: {}, tools: {} },
      sessionTitle: '',
      storedSid: 'lazy-2',
      usage: ZERO
    })
    app.rerender(React.createElement(Probe))
    vi.advanceTimersByTime(60)
    expect(writes).toHaveLength(3)
    expect(decodeArgv(writes[2]!)).toEqual(['/usr/local/bin/hermes', '--tui'])

    // Oversized storedSid after an active report clears the stale record.
    patchUiState({ busy: true, storedSid: 'x'.repeat(2000) })
    app.rerender(React.createElement(Probe))
    vi.advanceTimersByTime(60)
    expect(writes).toHaveLength(4)
    expect(writes[3]).toContain(':clear=1:mac=')

    app.unmount()
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
