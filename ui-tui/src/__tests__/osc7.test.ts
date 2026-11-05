import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { hostname, tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import {
  buildOsc7Sequence,
  buildOsc7Url,
  noteTerminalToolComplete,
  reportCwd,
  reportSessionCwd,
  resetOsc7State,
  terminalCwdFromResult
} from '../lib/osc7.js'
import { wrapForMultiplexer } from '../lib/osc52.js'

const OSC7_PREFIX = '\x1b]7;file://'
const ST = '\x1b\\'
const MISSING_DIR = '/workspace/definitely-not-here-xyz'

const envBackup = { ...process.env }
const isTTYBackup = process.stdout.isTTY

// Two real, distinct, existing directories plus one whose name needs encoding.
let DIR_A = ''
let DIR_B = ''
let DIR_ENCODED = ''
let PLAIN_FILE = ''
let scratch = ''

/** Percent-encode each path segment, keeping the '/' separators. */
const encodePath = (p: string) => p.split('/').map(encodeURIComponent).join('/')

/** Extract the file:// URL path (decoded) out of a raw OSC 7 sequence. */
const decodedPathOf = (seq: string, host: string) => decodeURIComponent(seq.slice(`${OSC7_PREFIX}${host}`.length, -ST.length))

const fakeStream = (isTTY: boolean | undefined) => {
  const writes: string[] = []

  return {
    stream: {
      isTTY,
      write: (s: string) => {
        writes.push(String(s))

        return true
      }
    } as any,
    writes
  }
}

const rawEnv = () => {
  delete process.env.TMUX
  delete process.env.STY
}

const spyStdout = () => vi.spyOn(process.stdout, 'write').mockImplementation(() => true as any)

const osc7Writes = (spy: ReturnType<typeof spyStdout>) =>
  spy.mock.calls.map(c => String(c[0] ?? '')).filter(s => s.includes(']7;file://'))

beforeAll(() => {
  scratch = mkdtempSync(join(tmpdir(), 'osc7-lock-'))
  DIR_A = mkdtempSync(join(scratch, 'dir-a-'))
  DIR_B = mkdtempSync(join(scratch, 'dir-b-'))
  DIR_ENCODED = mkdtempSync(join(scratch, 'my dir ü-'))
  PLAIN_FILE = join(scratch, 'plain-file.txt')
  writeFileSync(PLAIN_FILE, 'not a directory')
})

afterAll(() => {
  rmSync(scratch, { force: true, recursive: true })
})

beforeEach(() => {
  resetOsc7State()
})

afterEach(() => {
  vi.restoreAllMocks()
  process.env = { ...envBackup }
  ;(process.stdout as any).isTTY = isTTYBackup
  resetOsc7State()
})

describe('buildOsc7Sequence / buildOsc7Url', () => {
  it('frames the URL as ESC ] 7 ; file://HOST PATH terminated by ST (not BEL)', () => {
    const seq = buildOsc7Sequence('/Users/josh/dev', 'host.local')

    expect(seq.startsWith(OSC7_PREFIX)).toBe(true)
    expect(seq.endsWith(ST)).toBe(true)
    expect(seq).not.toContain('\x07')
    expect(seq).toBe(`\x1b]7;${buildOsc7Url('/Users/josh/dev', 'host.local')}${ST}`)
    expect(seq).toBe('\x1b]7;file://host.local/Users/josh/dev\x1b\\')
  })

  it('uses os.hostname() when no host is given and the explicit host when one is', () => {
    expect(buildOsc7Url('/Users/josh/dev')).toBe(`file://${hostname()}/Users/josh/dev`)
    expect(buildOsc7Sequence('/Users/josh/dev').startsWith(`${OSC7_PREFIX}${hostname()}/`)).toBe(true)
    expect(buildOsc7Url('/Users/josh/dev', 'other.host')).toBe('file://other.host/Users/josh/dev')
  })

  it("keeps '/' separators and percent-encodes space and non-ASCII as UTF-8", () => {
    expect(buildOsc7Sequence('/Users/josh/dev/my dir/ü', 'host.local')).toBe(
      '\x1b]7;file://host.local/Users/josh/dev/my%20dir/%C3%BC\x1b\\'
    )
  })

  it("leaves ASCII letters, digits, '-', '_' and '.' untouched", () => {
    const p = '/Ab-Cd_09/x.y/z-1_2.3'

    expect(buildOsc7Url(p, 'h')).toBe(`file://h${p}`)
  })

  it('round-trips: decoding the URL path gives back the original path', () => {
    const p = '/tmp/a b/ü/ç%/#?&'
    const seq = buildOsc7Sequence(p, 'h')

    expect(decodedPathOf(seq, 'h')).toBe(p)
    // No raw space, '#', '?' survive into the URL.
    expect(seq.slice(OSC7_PREFIX.length, -ST.length)).not.toMatch(/[ #?]/)
  })
})

describe('reportCwd', () => {
  it('writes the raw sequence to a tty stream for an existing directory and returns true', () => {
    rawEnv()
    const { stream, writes } = fakeStream(true)

    expect(reportCwd(DIR_A, stream)).toBe(true)
    expect(writes).toEqual([buildOsc7Sequence(DIR_A)])
    expect(writes[0]!.startsWith(`${OSC7_PREFIX}${hostname()}`)).toBe(true)
    expect(writes[0]!.endsWith(`${encodePath(DIR_A)}${ST}`)).toBe(true)
  })

  it('percent-encodes a real directory whose name has a space and a non-ASCII char', () => {
    rawEnv()
    const { stream, writes } = fakeStream(true)

    expect(reportCwd(DIR_ENCODED, stream)).toBe(true)
    expect(writes).toHaveLength(1)
    expect(writes[0]).toContain('my%20dir%20%C3%BC-')
    expect(decodedPathOf(writes[0]!, hostname())).toBe(DIR_ENCODED)
  })

  it.each([false, undefined])('writes nothing and returns false when isTTY is %s', isTTY => {
    rawEnv()
    const { stream, writes } = fakeStream(isTTY)

    expect(reportCwd(DIR_A, stream)).toBe(false)
    expect(writes).toEqual([])
  })

  it('writes nothing and returns false for a path that does not exist', () => {
    rawEnv()
    const { stream, writes } = fakeStream(true)

    expect(reportCwd(MISSING_DIR, stream)).toBe(false)
    expect(writes).toEqual([])
  })

  it('writes nothing and returns false for an existing path that is a file, not a directory', () => {
    rawEnv()
    const { stream, writes } = fakeStream(true)

    expect(reportCwd(PLAIN_FILE, stream)).toBe(false)
    expect(writes).toEqual([])
  })

  it.each([null, undefined, ''])('writes nothing and returns false for cwd %j', cwd => {
    rawEnv()
    const { stream, writes } = fakeStream(true)

    expect(reportCwd(cwd as any, stream)).toBe(false)
    expect(writes).toEqual([])
  })

  it('defaults to process.stdout and respects its isTTY', () => {
    rawEnv()

    const write = spyStdout()

    ;(process.stdout as any).isTTY = false
    expect(reportCwd(DIR_A)).toBe(false)
    expect(write).not.toHaveBeenCalled()

    ;(process.stdout as any).isTTY = true
    expect(reportCwd(DIR_A)).toBe(true)
    expect(write).toHaveBeenCalledTimes(1)
    expect(String(write.mock.calls[0]?.[0])).toBe(buildOsc7Sequence(DIR_A))
  })
})

describe('reportCwd multiplexer wrapping', () => {
  it('wraps for tmux passthrough with every ESC doubled when TMUX is set', () => {
    process.env.TMUX = '/tmp/tmux-123/default,1,0'
    delete process.env.STY
    const { stream, writes } = fakeStream(true)
    const seq = buildOsc7Sequence(DIR_A)

    expect(reportCwd(DIR_A, stream)).toBe(true)
    expect(writes).toEqual([`\x1bPtmux;${seq.replaceAll('\x1b', '\x1b\x1b')}${ST}`])
  })

  it('wraps as a plain DCS when only STY (GNU screen) is set', () => {
    delete process.env.TMUX
    process.env.STY = '12345.pts-0.host'
    const { stream, writes } = fakeStream(true)
    const seq = buildOsc7Sequence(DIR_A)

    expect(reportCwd(DIR_A, stream)).toBe(true)
    expect(writes).toEqual([`\x1bP${seq.slice(0, -ST.length)}\x07${ST}`])
  })

  it('prefers tmux wrapping when both TMUX and STY are set', () => {
    process.env.TMUX = '/tmp/tmux-123/default,1,0'
    process.env.STY = '12345.pts-0.host'
    const { stream, writes } = fakeStream(true)

    reportCwd(DIR_A, stream)
    expect(writes).toHaveLength(1)
    expect(writes[0]!.startsWith('\x1bPtmux;\x1b\x1b]7;file://')).toBe(true)
  })

  it('writes exactly what wrapForMultiplexer produces for the sequence, in every env', () => {
    const envs: Array<Record<string, string | undefined>> = [
      { STY: undefined, TMUX: undefined },
      { STY: undefined, TMUX: '/tmp/tmux-1/default,1,0' },
      { STY: '1.pts-0.host', TMUX: undefined }
    ]

    for (const env of envs) {
      rawEnv()

      for (const [k, v] of Object.entries(env)) {
        if (v !== undefined) {
          process.env[k] = v
        }
      }

      const { stream, writes } = fakeStream(true)
      reportCwd(DIR_B, stream)
      expect(writes).toEqual([wrapForMultiplexer(buildOsc7Sequence(DIR_B))])
    }
  })
})

describe('terminalCwdFromResult', () => {
  it('returns the cwd from a parsed-object result and from the same result as JSON text', () => {
    const obj = { cwd: '/some/dir', exit_code: 0, output: '' }

    expect(terminalCwdFromResult('terminal', obj)).toBe('/some/dir')
    expect(terminalCwdFromResult('terminal', JSON.stringify(obj))).toBe('/some/dir')
    expect(terminalCwdFromResult('terminal', '{"output":"","exit_code":0,"cwd":"/some/dir"}')).toBe('/some/dir')
  })

  it('returns null for any tool other than terminal, even with a cwd in the result', () => {
    expect(terminalCwdFromResult('read_file', { cwd: '/tmp' })).toBeNull()
    expect(terminalCwdFromResult('read_file', '{"cwd":"/tmp"}')).toBeNull()
    expect(terminalCwdFromResult(undefined, { cwd: '/tmp' })).toBeNull()
  })

  it('returns null when the result has no cwd, is non-JSON text, or is missing', () => {
    expect(terminalCwdFromResult('terminal', { exit_code: 0, output: 'no cwd key' })).toBeNull()
    expect(terminalCwdFromResult('terminal', '{"output":"no cwd key","exit_code":0}')).toBeNull()
    expect(terminalCwdFromResult('terminal', 'plain text output, not json')).toBeNull()
    expect(terminalCwdFromResult('terminal', undefined)).toBeNull()
    expect(terminalCwdFromResult('terminal', null)).toBeNull()
  })

  it('returns null when cwd is not a string', () => {
    expect(terminalCwdFromResult('terminal', { cwd: 42 })).toBeNull()
    expect(terminalCwdFromResult('terminal', '{"cwd":null}')).toBeNull()
  })
})

describe('noteTerminalToolComplete', () => {
  beforeEach(() => {
    rawEnv()
    ;(process.stdout as any).isTTY = true
  })

  it.each([
    ['object', () => ({ cwd: DIR_A, exit_code: 0, output: '' })],
    ['JSON string', () => JSON.stringify({ output: '', exit_code: 0, cwd: DIR_A })]
  ])('writes exactly one OSC 7 for the cwd of a terminal result (%s form)', (_label, result) => {
    const write = spyStdout()

    noteTerminalToolComplete('sess-1', 'terminal', result())
    expect(write).toHaveBeenCalledTimes(1)
    expect(String(write.mock.calls[0]?.[0])).toBe(buildOsc7Sequence(DIR_A))
  })

  it('object and string forms of the same result produce identical writes', () => {
    const write = spyStdout()
    const obj = { cwd: DIR_B, exit_code: 0, output: '' }

    noteTerminalToolComplete('sess-1', 'terminal', obj)
    noteTerminalToolComplete('sess-2', 'terminal', JSON.stringify(obj))
    expect(write).toHaveBeenCalledTimes(2)
    expect(String(write.mock.calls[0]?.[0])).toBe(String(write.mock.calls[1]?.[0]))
  })

  it.each([
    ['another tool name', 'read_file', () => ({ cwd: DIR_A })],
    ['no cwd key', 'terminal', () => ({ exit_code: 0, output: 'no cwd key' })],
    ['non-JSON string', 'terminal', () => 'total 0\ndrwxr-xr-x  2 josh staff 64 .'],
    ['missing result', 'terminal', () => undefined],
    ['cwd that is not a directory', 'terminal', () => ({ cwd: MISSING_DIR, exit_code: 0, output: '' })]
  ])('writes nothing for %s', (_label, name, result) => {
    const write = spyStdout()

    noteTerminalToolComplete('sess-1', name, result())
    expect(write).not.toHaveBeenCalled()
  })

  it('writes nothing when stdout is not a tty', () => {
    ;(process.stdout as any).isTTY = false
    const write = spyStdout()

    noteTerminalToolComplete('sess-1', 'terminal', { cwd: DIR_A, exit_code: 0, output: '' })
    expect(write).not.toHaveBeenCalled()
  })

  it('applies multiplexer wrapping to what it writes', () => {
    process.env.TMUX = '/tmp/tmux-123/default,1,0'
    const write = spyStdout()

    noteTerminalToolComplete('sess-1', 'terminal', { cwd: DIR_A, exit_code: 0, output: '' })
    expect(write).toHaveBeenCalledTimes(1)
    expect(String(write.mock.calls[0]?.[0])).toBe(wrapForMultiplexer(buildOsc7Sequence(DIR_A)))
  })
})

describe('reportSessionCwd', () => {
  beforeEach(() => {
    rawEnv()
    ;(process.stdout as any).isTTY = true
  })

  it('writes the remembered terminal cwd for the session rather than the fallback', () => {
    noteTerminalToolComplete('stored-1', 'terminal', { cwd: DIR_A, exit_code: 0, output: '' })
    const write = spyStdout()

    reportSessionCwd('stored-1', DIR_B)
    expect(write).toHaveBeenCalledTimes(1)
    expect(String(write.mock.calls[0]?.[0])).toBe(buildOsc7Sequence(DIR_A))
  })

  it('remembers the cwd from the JSON-string result form too', () => {
    noteTerminalToolComplete('stored-1', 'terminal', JSON.stringify({ output: '', exit_code: 0, cwd: DIR_A }))
    const write = spyStdout()

    reportSessionCwd('stored-1', DIR_B)
    expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_A)])
  })

  it('remembers the latest terminal cwd per session, independently', () => {
    noteTerminalToolComplete('s-1', 'terminal', { cwd: DIR_B })
    noteTerminalToolComplete('s-1', 'terminal', { cwd: DIR_A })
    noteTerminalToolComplete('s-2', 'terminal', { cwd: DIR_B })
    const write = spyStdout()

    reportSessionCwd('s-1', DIR_ENCODED)
    reportSessionCwd('s-2', DIR_ENCODED)
    expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_A), buildOsc7Sequence(DIR_B)])
  })

  it('does not remember a cwd reported by a non-terminal tool', () => {
    noteTerminalToolComplete('stored-1', 'read_file', { cwd: DIR_A })
    const write = spyStdout()

    reportSessionCwd('stored-1', DIR_B)
    expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_B)])
  })

  it('writes the fallback cwd for a session it has never seen', () => {
    noteTerminalToolComplete('stored-1', 'terminal', { cwd: DIR_A })
    const write = spyStdout()

    reportSessionCwd('never-seen', DIR_B)
    expect(write).toHaveBeenCalledTimes(1)
    expect(String(write.mock.calls[0]?.[0])).toBe(buildOsc7Sequence(DIR_B))
  })

  it('writes nothing when nothing is remembered and there is no fallback', () => {
    const write = spyStdout()

    reportSessionCwd('x', undefined)
    reportSessionCwd('x', null)
    expect(write).not.toHaveBeenCalled()
  })

  it('resetOsc7State forgets every remembered cwd', () => {
    noteTerminalToolComplete('stored-1', 'terminal', { cwd: DIR_A })
    noteTerminalToolComplete('stored-2', 'terminal', { cwd: DIR_A })
    resetOsc7State()
    const write = spyStdout()

    reportSessionCwd('stored-1', DIR_B)
    reportSessionCwd('stored-2', undefined)
    expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_B)])
  })

  it('writes nothing when stdout is not a tty', () => {
    ;(process.stdout as any).isTTY = false
    const write = spyStdout()

    reportSessionCwd('never-seen', DIR_B)
    expect(write).not.toHaveBeenCalled()
  })

  it('applies multiplexer wrapping to what it writes', () => {
    process.env.TMUX = '/tmp/tmux-123/default,1,0'
    const write = spyStdout()

    reportSessionCwd('never-seen', DIR_B)
    expect(write).toHaveBeenCalledTimes(1)
    expect(String(write.mock.calls[0]?.[0])).toBe(wrapForMultiplexer(buildOsc7Sequence(DIR_B)))
  })
})

describe('tool.complete through the real createGatewayEventHandler', () => {
  const ref = <T>(current: T) => ({ current })

  const buildCtx = () =>
    ({
      composer: {
        dequeue: () => undefined,
        queueEditRef: ref<null | number>(null),
        sendQueued: vi.fn(),
        setInput: vi.fn()
      },
      gateway: {
        gw: { request: vi.fn(async () => null) },
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
      submission: { submitRef: { current: vi.fn() } },
      system: { bellOnComplete: false, sys: vi.fn() },
      transcript: { appendMessage: vi.fn(), panel: vi.fn(), setHistoryItems: vi.fn() },
      voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
    }) as any

  beforeEach(() => {
    resetUiState()
    resetTurnState()
    turnController.fullReset()
    patchUiState({ sid: 'live-1', storedSid: 'stored-1' })
    rawEnv()
    ;(process.stdout as any).isTTY = true
  })

  const resultText = () => JSON.stringify({ output: DIR_A, exit_code: 0, error: null, cwd: DIR_A })

  it('a terminal tool.complete whose result is JSON text writes exactly one OSC 7 for its cwd', () => {
    const onEvent = createGatewayEventHandler(buildCtx())
    const write = spyStdout()

    onEvent({
      payload: { name: 'terminal', result: resultText(), tool_id: 'c1' },
      session_id: 'live-1',
      type: 'tool.complete'
    } as any)

    const seqs = osc7Writes(write)
    expect(seqs).toHaveLength(1)
    expect(seqs[0]).toContain(']7;file://')
    expect(seqs[0]!.endsWith(`${encodePath(DIR_A)}${ST}`)).toBe(true)
  })

  it('a terminal tool.complete whose result is a parsed object writes the same OSC 7', () => {
    const onEvent = createGatewayEventHandler(buildCtx())
    const write = spyStdout()

    onEvent({
      payload: { name: 'terminal', result: JSON.parse(resultText()), tool_id: 'c1' },
      type: 'tool.complete'
    } as any)

    expect(osc7Writes(write)).toEqual([buildOsc7Sequence(DIR_A)])
  })

  it('a read_file tool.complete with the same result writes no OSC 7', () => {
    const onEvent = createGatewayEventHandler(buildCtx())
    const write = spyStdout()

    onEvent({
      payload: { name: 'read_file', result: resultText(), tool_id: 'c2' },
      session_id: 'live-1',
      type: 'tool.complete'
    } as any)

    expect(osc7Writes(write)).toEqual([])
  })

  it('a terminal tool.complete for another session is dropped and writes no OSC 7', () => {
    const onEvent = createGatewayEventHandler(buildCtx())
    const write = spyStdout()

    onEvent({
      payload: { name: 'terminal', result: resultText(), tool_id: 'c3' },
      session_id: 'someone-else',
      type: 'tool.complete'
    } as any)

    expect(osc7Writes(write)).toEqual([])
  })
})
