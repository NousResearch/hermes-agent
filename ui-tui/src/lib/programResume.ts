// OSC 7577 Program Resume Protocol (rev 0.5):
// the TUI's twin of hermes_cli/terminal_notify.py::program_resume_sequence.
import { createHash } from 'node:crypto'

import { atom } from 'nanostores'

const TOKEN_RE = /^[A-Za-z0-9_-]{22,64}$/
// eslint-disable-next-line no-control-regex
const CONTROLS_RE = /[\u0000-\u001f\u007f-\u009f]/
const MAX_BYTES = 2048
const MAX_ARGV = 64

// null until the first config read, so an opted-out user never sees a startup report.
export const $programResumeEnabled = atom<boolean | null>(null)

let cachedToken: null | string = null
let cachedOwnerPid: null | number = null
let cachedResumePrefix: null | readonly string[] = null
let envConsumed = false
let reported = false
let clearSuppressed = false

export const isValidOsc7577Token = (token: null | string | undefined): token is string =>
  typeof token === 'string' && TOKEN_RE.test(token)

const isValidAbsPath = (path: string): boolean => {
  if (!path || CONTROLS_RE.test(path)) {
    return false
  }

  return path.startsWith('/') || /^[A-Za-z]:[\\/]/.test(path) || path.startsWith('\\\\')
}

const isValidArgv0 = (arg: string): boolean => {
  if (!arg || CONTROLS_RE.test(arg)) {
    return false
  }

  return isValidAbsPath(arg) || (!arg.includes('/') && !arg.includes('\\') && !arg.startsWith('-'))
}

/** Cache OSC 7577 launch metadata from *env* and strip those variables before spawning children. */
export function consumeProgramResumeEnv(env: NodeJS.ProcessEnv = process.env): void {
  const rawToken = env.OSC7577_TOKEN
  const rawPid = env.HERMES_TUI_OWNER_PID
  const rawArgv = env.HERMES_TUI_RESUME_ARGV

  delete env.OSC7577_TOKEN
  delete env.HERMES_TUI_OWNER_PID
  delete env.HERMES_TUI_RESUME_ARGV

  cachedToken = isValidOsc7577Token(rawToken) ? rawToken : null

  const trimmedPid = (rawPid ?? '').trim()
  const parsedPid = /^[1-9][0-9]*$/.test(trimmedPid) ? Number(trimmedPid) : NaN
  cachedOwnerPid = Number.isSafeInteger(parsedPid) && parsedPid >= 1 ? parsedPid : null

  cachedResumePrefix = null

  if (rawArgv) {
    try {
      const parsed: unknown = JSON.parse(rawArgv)

      if (Array.isArray(parsed) && parsed.length > 0 && parsed.every(item => typeof item === 'string' && item.length > 0)) {
        cachedResumePrefix = parsed
      }
    } catch {
      cachedResumePrefix = null
    }
  }

  envConsumed = true
}

const resolveToken = (token?: null | string): null | string => {
  if (token !== undefined) {
    return isValidOsc7577Token(token) ? token : null
  }

  if (cachedToken) {
    return cachedToken
  }

  if (!envConsumed && isValidOsc7577Token(process.env.OSC7577_TOKEN)) {
    return process.env.OSC7577_TOKEN
  }

  return null
}

const resolveMeta = (opts?: { now?: number; pid?: number; token?: null | string }) => {
  const token = resolveToken(opts?.token)

  if (!token) {
    return null
  }

  const pid = opts?.pid ?? cachedOwnerPid ?? process.pid
  const now = opts?.now ?? Math.floor(Date.now() / 1000)

  if (!Number.isSafeInteger(pid) || pid < 1 || !Number.isSafeInteger(now) || now < 1) {
    return null
  }

  return { now, pid, token }
}

const finalizeSequence = (body: string, token: string): null | string => {
  const mac = createHash('sha256').update(`${body}:${token}`, 'ascii').digest('hex')
  const seq = `\x1b]7577;${body}:mac=${mac}\x1b\\`

  return Buffer.byteLength(seq, 'utf8') <= MAX_BYTES ? seq : null
}

export function buildTuiResumeArgv(sessionId?: null | string, prefix?: null | readonly string[]): string[] {
  const base = prefix?.length
    ? [...prefix]
    : cachedResumePrefix?.length
      ? [...cachedResumePrefix]
      : ['hermes', '--tui']

  if (sessionId) {
    base.push('--resume', sessionId)
  }

  return base
}

export function encodeProgramResume(
  argv: readonly string[],
  opts?: { cwd?: string; now?: number; pid?: number; token?: null | string }
): null | string {
  const meta = resolveMeta(opts)

  if (!meta) {
    return null
  }

  if (argv.length < 1 || argv.length > MAX_ARGV) {
    return null
  }

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]

    if (typeof arg !== 'string' || CONTROLS_RE.test(arg)) {
      return null
    }

    if (i === 0 && !isValidArgv0(arg)) {
      return null
    }
  }

  const encodedArgs = argv.map(arg => Buffer.from(arg, 'utf8').toString('base64')).join(',')
  let body = `pid=${meta.pid}:time=${meta.now}:argv=${encodedArgs}`

  if (opts?.cwd !== undefined) {
    if (!isValidAbsPath(opts.cwd)) {
      return null
    }

    body += `:cwd=${Buffer.from(opts.cwd, 'utf8').toString('base64')}`
  }

  return finalizeSequence(body, meta.token)
}

export function encodeProgramResumeClear(opts?: { now?: number; pid?: number; token?: null | string }): null | string {
  const meta = resolveMeta(opts)

  if (!meta) {
    return null
  }

  return finalizeSequence(`pid=${meta.pid}:time=${meta.now}:clear=1`, meta.token)
}

export const markProgramResumeReported = (): void => {
  reported = true
}

export const suppressProgramResumeClear = (): void => {
  clearSuppressed = true
}

// Quit: clear the pane's resume record on explicit user exit (never on SIGHUP/SIGTERM/EIO).
export function clearProgramResume(
  write: (seq: string) => void,
  opts?: { now?: number; pid?: number; token?: null | string }
): boolean {
  if (!reported || clearSuppressed) {
    return false
  }

  const seq = encodeProgramResumeClear(opts)

  if (!seq) {
    return false
  }

  reported = false

  try {
    write(seq)

    return true
  } catch {
    return false
  }
}

export function resetProgramResumeStateForTests(): void {
  cachedToken = null
  cachedOwnerPid = null
  cachedResumePrefix = null
  envConsumed = false
  reported = false
  clearSuppressed = false
}
