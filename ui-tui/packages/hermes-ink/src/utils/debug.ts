import { appendFileSync, mkdirSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'

// Default destination lives alongside the other `~/.hermes/logs/*.log` files
// (agent.log, errors.log, gateway.log) so `hermes logs` output and TUI-side
// debug breadcrumbs sit in one place. Respects HERMES_HOME for profile-aware
// installs, mirroring the resolution used by ui-tui/src/lib/parentLog.ts.
const DEFAULT_LOG_PATH = join(process.env.HERMES_HOME?.trim() || join(homedir(), '.hermes'), 'logs', 'tui-debug.log')

/**
 * Debug-only logging for the Ink TUI.
 *
 * A no-op unless HERMES_INK_DEBUG_LOG is set — this must never write to
 * stdout/stderr, since patchConsole()/patchStderr() (ink.tsx) route through
 * here while the alt-screen is mounted, and printing would corrupt the
 * terminal buffer. When enabled, appends a timestamped line to a log file
 * instead (default `~/.hermes/logs/tui-debug.log`, overridable with
 * HERMES_INK_DEBUG_LOG_PATH) so in-TUI debugging has somewhere to look.
 *
 * Failures to write are swallowed: there's nowhere safe to report them from
 * inside the alt-screen, and debug logging must never crash the TUI.
 */
export function logForDebugging(
  message: string,
  options: {
    level?: string
  } = {}
): void {
  if (!process.env.HERMES_INK_DEBUG_LOG) {
    return
  }

  try {
    const path = process.env.HERMES_INK_DEBUG_LOG_PATH?.trim() || DEFAULT_LOG_PATH
    const level = options.level ?? 'debug'

    mkdirSync(dirname(path), { recursive: true })
    appendFileSync(path, `${new Date().toISOString()} [${level}] ${message}\n`)
  } catch {
    // Best-effort — see doc comment above.
  }
}
