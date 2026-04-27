import type { Writable } from 'stream'

import { coerce } from 'semver'

import { env } from '../utils/env.js'
import { gte } from '../utils/semver.js'

import { getClearTerminalSequence } from './clearTerminal.js'
import type { Diff } from './frame.js'
import { cursorMove, cursorTo, eraseLines } from './termio/csi.js'
import { BSU, ESU, HIDE_CURSOR, SHOW_CURSOR } from './termio/dec.js'
import { link } from './termio/osc.js'

export type Progress = {
  state: 'running' | 'completed' | 'error' | 'indeterminate'
  percentage?: number
}

/**
 * Checks if the terminal supports OSC 9;4 progress reporting.
 * Supported terminals:
 * - ConEmu (Windows) - all versions
 * - Ghostty 1.2.0+
 * - iTerm2 3.6.6+
 *
 * Note: Windows Terminal interprets OSC 9;4 as notifications, not progress.
 */
export function isProgressReportingAvailable(): boolean {
  // Only available if we have a TTY (not piped)
  if (!process.stdout.isTTY) {
    return false
  }

  // Explicitly exclude Windows Terminal, which interprets OSC 9;4 as
  // notifications rather than progress indicators
  if (process.env.WT_SESSION) {
    return false
  }

  // ConEmu supports OSC 9;4 for progress (all versions)
  if (process.env.ConEmuANSI || process.env.ConEmuPID || process.env.ConEmuTask) {
    return true
  }

  const version = coerce(process.env.TERM_PROGRAM_VERSION)

  if (!version) {
    return false
  }

  // Ghostty 1.2.0+ supports OSC 9;4 for progress
  // https://ghostty.org/docs/install/release-notes/1-2-0
  if (process.env.TERM_PROGRAM === 'ghostty') {
    return gte(version.version, '1.2.0')
  }

  // iTerm2 3.6.6+ supports OSC 9;4 for progress
  // https://iterm2.com/downloads.html
  if (process.env.TERM_PROGRAM === 'iTerm.app') {
    return gte(version.version, '3.6.6')
  }

  return false
}

/**
 * Checks if the terminal supports DEC mode 2026 (synchronized output).
 * When supported, BSU/ESU sequences prevent visible flicker during redraws.
 */
export function isSynchronizedOutputSupported(): boolean {
  // tmux parses and proxies every byte but doesn't implement DEC 2026.
  // BSU/ESU pass through to the outer terminal but tmux has already
  // broken atomicity by chunking. Skip to save 16 bytes/frame + parser work.
  if (process.env.TMUX) {
    return false
  }

  const termProgram = process.env.TERM_PROGRAM
  const term = process.env.TERM

  // Modern terminals with known DEC 2026 support
  if (
    termProgram === 'iTerm.app' ||
    termProgram === 'WezTerm' ||
    termProgram === 'WarpTerminal' ||
    termProgram === 'ghostty' ||
    termProgram === 'contour' ||
    termProgram === 'vscode' ||
    termProgram === 'alacritty'
  ) {
    return true
  }

  // kitty sets TERM=xterm-kitty or KITTY_WINDOW_ID
  if (term?.includes('kitty') || process.env.KITTY_WINDOW_ID) {
    return true
  }

  // Ghostty may set TERM=xterm-ghostty without TERM_PROGRAM
  if (term === 'xterm-ghostty') {
    return true
  }

  // foot sets TERM=foot or TERM=foot-extra
  if (term?.startsWith('foot')) {
    return true
  }

  // Alacritty may set TERM containing 'alacritty'
  if (term?.includes('alacritty')) {
    return true
  }

  // Zed uses the alacritty_terminal crate which supports DEC 2026
  if (process.env.ZED_TERM) {
    return true
  }

  // Windows Terminal
  if (process.env.WT_SESSION) {
    return true
  }

  // VTE-based terminals (GNOME Terminal, Tilix, etc.) since VTE 0.68
  const vteVersion = process.env.VTE_VERSION

  if (vteVersion) {
    const version = parseInt(vteVersion, 10)

    if (version >= 6800) {
      return true
    }
  }

  return false
}

// -- XTVERSION-detected terminal name (populated async at startup) --
//
// TERM_PROGRAM is not forwarded over SSH by default, so env-based detection
// fails when the process runs remotely inside a VS Code integrated terminal.
// XTVERSION (CSI > 0 q → DCS > | name ST) goes through the pty — the query
// reaches the *client* terminal and the reply comes back through stdin.
// App.tsx fires the query when raw mode enables; setXtversionName() is called
// from the response handler. Readers should treat undefined as "not yet known"
// and fall back to env-var detection.

let xtversionName: string | undefined

/** Record the XTVERSION response. Called once from App.tsx when the reply
 *  arrives on stdin. No-op if already set (defend against re-probe). */
export function setXtversionName(name: string): void {
  if (xtversionName === undefined) {
    xtversionName = name
  }
}

/** True if running in an xterm.js-based terminal (VS Code, Cursor, Windsurf
 *  integrated terminals). Combines TERM_PROGRAM env check (fast, sync, but
 *  not forwarded over SSH) with the XTVERSION probe result (async, survives
 *  SSH — query/reply goes through the pty). Early calls may miss the probe
 *  reply — call lazily (e.g. in an event handler) if SSH detection matters. */
export function isXtermJs(): boolean {
  if (process.env.TERM_PROGRAM === 'vscode') {
    return true
  }

  return xtversionName?.startsWith('xterm.js') ?? false
}

// Terminals known to correctly implement the Kitty keyboard protocol
// (CSI >1u) and/or xterm modifyOtherKeys (CSI >4;2m) for ctrl+shift+<letter>
// disambiguation. We previously enabled unconditionally (#23350), assuming
// terminals silently ignore unknown CSI — but some terminals honor the enable
// and emit codepoints our input parser doesn't handle (notably over SSH and
// in xterm.js-based terminals like VS Code). tmux is allowlisted because it
// accepts modifyOtherKeys and doesn't forward the kitty sequence to the outer
// terminal.
const EXTENDED_KEYS_TERMINALS = ['iTerm.app', 'kitty', 'WezTerm', 'ghostty', 'tmux', 'windows-terminal', 'vscode']

/** True if this terminal correctly handles extended key reporting
 *  (Kitty keyboard protocol + xterm modifyOtherKeys). */
export function supportsExtendedKeys(): boolean {
  return EXTENDED_KEYS_TERMINALS.includes(env.terminal ?? '')
}

/** True if the terminal scrolls the viewport when it receives cursor-up
 *  sequences that reach above the visible area. On Windows, conhost's
 *  SetConsoleCursorPosition follows the cursor into scrollback
 *  (microsoft/terminal#14774), yanking users to the top of their buffer
 *  mid-stream. WT_SESSION catches WSL-in-Windows-Terminal where platform
 *  is linux but output still routes through conhost. */
export function hasCursorUpViewportYankBug(): boolean {
  return process.platform === 'win32' || !!process.env.WT_SESSION
}

// Computed once at module load — terminal capabilities don't change mid-session.
// Exported so callers can pass a sync-skip hint gated to specific modes.
export const SYNC_OUTPUT_SUPPORTED = isSynchronizedOutputSupported()

export type Terminal = {
  stdout: Writable
  stderr: Writable
}

export function writeDiffToTerminal(
  terminal: Terminal,
  diff: Diff,
  skipSyncMarkers = false,
  onDrain?: () => void
): { bytes: number; backpressure: boolean } {
  // No output if there are no patches
  if (diff.length === 0) {
    return { bytes: 0, backpressure: false }
  }

  // BSU/ESU wrapping is opt-out to keep main-screen behavior unchanged.
  // Callers pass skipSyncMarkers=true when the terminal doesn't support
  // DEC 2026 (e.g. tmux) AND the cost matters (high-frequency alt-screen).
  const useSync = !skipSyncMarkers

  // Buffer all writes into a single string to avoid multiple write calls
  let buffer = useSync ? BSU : ''

  for (const patch of diff) {
    switch (patch.type) {
      case 'stdout':
        buffer += patch.content

        break

      case 'clear':
        if (patch.count > 0) {
          buffer += eraseLines(patch.count)
        }

        break

      case 'clearTerminal':
        buffer += getClearTerminalSequence()

        break

      case 'cursorHide':
        buffer += HIDE_CURSOR

        break

      case 'cursorShow':
        buffer += SHOW_CURSOR

        break

      case 'cursorMove':
        buffer += cursorMove(patch.x, patch.y)

        break

      case 'cursorTo':
        buffer += cursorTo(patch.col)

        break

      case 'carriageReturn':
        buffer += '\r'

        break

      case 'hyperlink':
        buffer += link(patch.uri)

        break

      case 'styleStr':
        buffer += patch.str

        break
    }
  }

  // Add synchronized update end and flush buffer
  if (useSync) {
    buffer += ESU
  }

  // Node's Writable.write returns false when the internal buffer is full
  // (backpressure). On a slow terminal parser that's the tell: we're
  // producing bytes faster than the outer terminal can consume them.
  // The 2-arg form attaches a drain callback that fires once the chunk
  // is actually flushed to the OS socket/pipe — giving us end-to-end
  // drain timing, not just "queued in Node".
  const wrote = onDrain ? terminal.stdout.write(buffer, () => onDrain()) : terminal.stdout.write(buffer)

  return { bytes: Buffer.byteLength(buffer, 'utf8'), backpressure: !wrote }
}
