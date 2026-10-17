import { appendFileSync, closeSync, mkdirSync, openSync, renameSync, statSync, unlinkSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

// Mirror the Python gateway's panic log (tui_gateway/server.py::_CRASH_LOG) from
// the Node parent so lifecycle breadcrumbs interleave, by timestamp, with the
// child's `=== SIGTERM received ===` / `=== gateway exit ===` entries.
//
// A backend SIGTERM is *usually* a parent action — `gw.kill()` (graceful-exit on
// a signal to Node, or an explicit /quit) or `start()` replacing a live child —
// but it can also come straight from an external supervisor (s6, a cgroup OOM
// reaper, a stray `kill`) signalling the child directly. Telling those apart is
// exactly the point: #31051 left these breadcrumbs in an in-memory CircularBuffer
// that dies with the process, so SIGTERM crash reports arrived with no parent
// context. A `[tui-parent]` line immediately before the child's panic means a
// parent kill; its absence *suggests* an external signal — not definitive,
// since this logger is best-effort (disabled under VITEST, and a failed append
// is swallowed). Persisting the death-explaining events here is what makes that
// distinction (and a memory-critical `process.exit(137)`, which closes stdin →
// clean EOF, not SIGTERM) diagnosable after the fact.
const logDir = join(process.env.HERMES_HOME?.trim() || join(homedir(), '.hermes'), 'logs')
const CRASH_LOG = join(logDir, 'tui_gateway_crash.log')

// Skipped under vitest so unit tests exercising start()/kill() can't write into
// a real ~/.hermes (tests must stay hermetic — see AGENTS.md).
const enabled = !process.env.VITEST
// Slice a single breadcrumb's value to MAX_BREADCRUMB chars (a short
// "[truncated …]" marker is appended, so the written line is slightly longer)
// so a pathological value (e.g. a giant error) can't bloat the shared crash log
// or add noticeable blocking on the synchronous append. Mirrors the spirit of
// GatewayClient's in-memory log-line cap.
const MAX_BREADCRUMB = 4096
// Size cap so the shared crash log cannot grow without bound: recurring
// lifecycle breadcrumbs and the child's panics used to append forever
// (15 MB / 778 records on one host). One kept generation (.1) bounds the
// total at 2x the cap. Rotation takes the same O_CREAT|O_EXCL lock file as
// the Python gateway (never waiting; a stale lock is broken) and re-checks
// the size under it: a rotator acting on a stale stat could otherwise rename
// a small fresh log over the retained .1 generation and erase it (#136633
// review). A line lost to a concurrent rename matches the best-effort
// contract this logger already has.
const MAX_CRASH_LOG_BYTES = 5 * 1024 * 1024
// A rotation lock older than this belongs to a killed writer: break it
// instead of skipping rotation forever.
const ROTATE_LOCK_STALE_MS = 15_000
let warned = false

function acquireRotateLock(): number | null {
  const lockPath = `${CRASH_LOG}.rotate.lock`

  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      return openSync(lockPath, 'wx')
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'EEXIST') {
        return null
      }
    }

    try {
      if (Date.now() - statSync(lockPath).mtimeMs <= ROTATE_LOCK_STALE_MS) {
        return null
      }
    } catch {
      continue // Vanished between open and stat: retry the exclusive create.
    }

    try {
      unlinkSync(lockPath) // Stale lock: break it and retry.
    } catch {
      return null
    }
  }

  return null
}

function releaseRotateLock(fd: number): void {
  try {
    closeSync(fd)
  } catch {
    // Already closed.
  }

  try {
    unlinkSync(`${CRASH_LOG}.rotate.lock`)
  } catch {
    // Gone.
  }
}

function rotateCrashLogIfOverCap(): void {
  try {
    if (statSync(CRASH_LOG).size <= MAX_CRASH_LOG_BYTES) {
      return
    }
  } catch {
    return // Missing file: the append below recreates the log.
  }

  const fd = acquireRotateLock()

  if (fd === null) {
    return // A concurrent writer owns this rotation.
  }

  try {
    try {
      if (statSync(CRASH_LOG).size <= MAX_CRASH_LOG_BYTES) {
        return // Already rotated away.
      }
    } catch {
      return
    }

    renameSync(CRASH_LOG, `${CRASH_LOG}.1`)
  } finally {
    releaseRotateLock(fd)
  }
}

export function recordParentLifecycle(line: string): void {
  if (!enabled) {
    return
  }

  try {
    // Collapse embedded newlines so a multi-line value (e.g. an error message)
    // stays one breadcrumb and can't masquerade as a separate log entry or as
    // the child's panic output sharing this file.
    const oneLine = line.replace(/[\r\n]+/g, ' ↵ ')

    const capped =
      oneLine.length > MAX_BREADCRUMB
        ? `${oneLine.slice(0, MAX_BREADCRUMB)}… [truncated ${oneLine.length} chars]`
        : oneLine

    mkdirSync(logDir, { recursive: true })
    rotateCrashLogIfOverCap()
    appendFileSync(CRASH_LOG, `[tui-parent] ${new Date().toISOString()} ${capped}\n`)
  } catch {
    if (!warned) {
      warned = true
      process.stderr.write('hermes-tui: parent lifecycle log unavailable\n')
    }
  }
}
