import { statSync } from 'node:fs'
import { hostname } from 'node:os'

import { wrapForMultiplexer } from './osc52.js'

const ESC = '\x1b'
const ST = `${ESC}\\`

/** `file://<host><path>`, every path segment percent-encoded the way zsh/bash shell integration writes it. */
export function buildOsc7Url(path: string, host: string = hostname()): string {
  return `file://${host}${path.split('/').map(encodeURIComponent).join('/')}`
}

/** `ESC ] 7 ; file://host/path ESC \` — the cwd report terminals (Tern, iTerm2, WezTerm, Ghostty, kitty) read. */
export function buildOsc7Sequence(path: string, host?: string): string {
  return `${ESC}]7;${buildOsc7Url(path, host)}${ST}`
}

type CwdStream = Pick<NodeJS.WriteStream, 'isTTY' | 'write'>

const isDirectory = (path: string): boolean => {
  try {
    return statSync(path).isDirectory()
  } catch {
    return false
  }
}

/**
 * Tell the terminal where the session's shell is (OSC 7), so its Files pane / cwd chip
 * follows the terminal tool's `cd` the way it follows a shell's. The terminal tool runs in a
 * detached subshell the host terminal never sees, so without this the pane stays wherever
 * Hermes was launched. Silent when stdout is not a tty or the path is not a local directory
 * (a container / ssh backend's cwd means nothing to this host). Terminals that do not
 * understand OSC 7 ignore it.
 */
export function reportCwd(cwd: null | string | undefined, stream: CwdStream = process.stdout): boolean {
  if (!cwd || !stream.isTTY || !isDirectory(cwd)) {
    return false
  }

  try {
    stream.write(wrapForMultiplexer(buildOsc7Sequence(cwd)))

    return true
  } catch {
    return false
  }
}

// The last cwd the terminal tool reported per session (keyed by the durable session id). The
// gateway's session.info `cwd` is the workspace, which a plain `cd` deliberately does not move,
// so this is the only record of where the shell actually is when a session comes back.
const lastTerminalCwd = new Map<string, string>()

/** The terminal tool's result JSON carries `cwd` only when the command changed directory. The
 *  legacy tui_gateway ships `result` parsed; the canonical gateway ships it as the JSON text. */
export function terminalCwdFromResult(name: unknown, result: unknown): null | string {
  if (name !== 'terminal' || !result) {
    return null
  }

  let parsed: unknown = result

  if (typeof result === 'string') {
    try {
      parsed = JSON.parse(result)
    } catch {
      return null
    }
  }

  if (!parsed || typeof parsed !== 'object') {
    return null
  }

  const cwd = (parsed as { cwd?: unknown }).cwd

  return typeof cwd === 'string' && cwd ? cwd : null
}

/** Where the terminal tool runs, from session.info `terminal_backend`. Only a local shell's cwd is this
 *  host's: a container / ssh path that happens to exist here too (`/`, `/home`) is not it. Unreported
 *  (older gateways) counts as local. */
export const isLocalTerminalBackend = (backend: unknown): boolean => !backend || backend === 'local'

/** A one-off `workdir` command does not move the session's shell (the terminal tool keeps its recorded cwd,
 *  tools/terminal_tool_result.py), so its cwd echo must not move the terminal or the memory either. */
const isTransientWorkdirCall = (args: unknown): boolean =>
  !!args && typeof args === 'object' && Boolean((args as { workdir?: unknown }).workdir)

export type TerminalCwdContext = {
  /** The call's args (`tool.complete.args`). */
  args?: unknown
  /** session.info `terminal_backend` of the session the event belongs to. */
  backend?: unknown
}

/** `tool.complete`: remember where this session's shell is and tell the terminal. */
export function noteTerminalToolComplete(
  sessionId: null | string | undefined,
  name: unknown,
  result: unknown,
  context: TerminalCwdContext = {}
): void {
  if (!isLocalTerminalBackend(context.backend) || isTransientWorkdirCall(context.args)) {
    return
  }

  const cwd = terminalCwdFromResult(name, result)

  if (!cwd) {
    return
  }

  if (sessionId) {
    lastTerminalCwd.set(sessionId, cwd)
  }

  reportCwd(cwd)
}

/** A session became the active one (create / activate / resume): report where its shell last was, else its cwd. */
export function reportSessionCwd(
  sessionId: null | string | undefined,
  fallbackCwd: null | string | undefined,
  backend?: unknown
): void {
  if (!isLocalTerminalBackend(backend)) {
    return
  }

  const remembered = sessionId ? lastTerminalCwd.get(sessionId) : undefined

  reportCwd(remembered ?? fallbackCwd)
}

/** Test seam. */
export function resetOsc7State(): void {
  lastTerminalCwd.clear()
}
