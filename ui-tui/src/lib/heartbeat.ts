/**
 * Renderer heartbeat (Stage 3 frozen-detection).
 *
 * The orchestrator's reaper distinguishes a renderer that has EXITED (poll()
 * catches it) from one that is ALIVE BUT FROZEN (event loop wedged — poll()
 * still says alive). The only way to detect the latter from outside is a
 * liveness file the renderer touches on a timer: if the loop wedges, the timer
 * stops firing, the file's mtime goes stale, and the reaper reaps it.
 *
 * Deliberately tiny + dependency-free: a setInterval that touches a file. When
 * HERMES_TUI_HEARTBEAT_FILE is unset (not under the orchestrator), it's a no-op.
 */

import { closeSync, openSync, utimesSync } from 'node:fs'

export const HEARTBEAT_INTERVAL_MS = 15_000

/**
 * Start touching the heartbeat file every intervalMs. Returns a stop function.
 * No-op (returns a no-op stopper) when no heartbeat file is configured.
 *
 * `touch` and `now` are injectable for deterministic tests.
 */
export function startHeartbeat(
  file: string | undefined = process.env.HERMES_TUI_HEARTBEAT_FILE,
  intervalMs: number = HEARTBEAT_INTERVAL_MS,
  touch: (path: string) => void = defaultTouch
): () => void {
  if (!file || !file.trim()) {
    return () => {}
  }
  const path = file.trim()

  const beat = () => {
    try {
      touch(path)
    } catch {
      // best-effort; a failed touch just risks a false-frozen reap, which the
      // orchestrator respawns from anyway — never throw out of the timer.
    }
  }

  beat() // immediate first beat so a fresh renderer isn't briefly "stale"
  const timer = setInterval(beat, intervalMs)
  // Do NOT keep the event loop alive just for the heartbeat.
  if (typeof timer.unref === 'function') {
    timer.unref()
  }
  return () => clearInterval(timer)
}

/** Touch a file's mtime, creating it if absent. */
export function defaultTouch(path: string): void {
  let fd: number | undefined
  try {
    fd = openSync(path, 'a') // create if missing, no truncate
  } finally {
    if (fd !== undefined) {
      const now = new Date()
      try {
        utimesSync(path, now, now)
      } catch {
        // file exists but utimes failed — opening in 'a' already bumped ctime;
        // mtime may not move, but the reaper's stale window is generous.
      }
      closeSync(fd)
    }
  }
}
