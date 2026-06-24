/**
 * Recycle bridge (Stage 1).
 *
 * The memory monitor lives in entry.tsx (module scope, before React mounts);
 * the live sid + scroll handle live inside the React app. To let a sustained-
 * pressure signal trigger a SEAMLESS renderer recycle, the app registers a
 * recycle callback here and entry.tsx invokes it.
 *
 * A recycle is ONLY safe when the renderer is a disposable client of a durable
 * gateway — i.e. running in attach mode under the orchestrator
 * (HERMES_TUI_GATEWAY_URL set). In spawned-gateway mode the renderer OWNS the
 * gateway, so exiting would kill the session — there we must NOT recycle, only
 * warn (today's behaviour). `canRecycle()` encodes that guard.
 */

let _recycle: (() => void) | null = null

/**
 * Renderer exit code that tells the supervisor (tui_gateway/orchestrator.py) a
 * renderer exit was a DELIBERATE seamless recycle, not a user /quit.
 *
 * This is the single source of truth for the recycle-vs-quit contract; the
 * Python side mirrors this exact number as RECYCLE_EXIT_CODE in
 * tui_gateway/orchestrator.py. The supervisor maps exits as:
 *   - 0                 -> genuine user /quit -> tear the whole tree down.
 *   - RECYCLE_EXIT_CODE -> deliberate recycle -> keep the gateway + in-flight
 *                          turn alive, respawn a fresh renderer that re-attaches
 *                          and resumes the live session.
 *   - any other non-zero -> crash/OOM -> same session-preserving respawn path.
 *
 * 97 sits clear of the 0-2 shell range, the 126-165 signal range, and the
 * sysexits.h band (64-78), so it can never collide with a real quit/crash code.
 */
export const RECYCLE_EXIT_CODE = 97

/** The app registers its recycle implementation once on mount. */
export function registerRecycleHandler(fn: () => void): () => void {
  _recycle = fn
  return () => {
    if (_recycle === fn) {
      _recycle = null
    }
  }
}

/**
 * True only when a recycle is safe: the renderer is an attached client of a
 * durable gateway (orchestrator/attach mode). In spawned-gateway mode the
 * renderer exiting would take the session with it, so recycle is unsafe.
 */
export function canRecycle(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.HERMES_TUI_GATEWAY_URL?.trim())
}

/**
 * Trigger a recycle if one is registered AND safe. Returns true when a recycle
 * was actually initiated; false when it was skipped (no handler, or unsafe
 * spawned-gateway mode) so the caller can fall back to a warning.
 */
export function triggerRecycle(env: NodeJS.ProcessEnv = process.env): boolean {
  if (!_recycle || !canRecycle(env)) {
    return false
  }
  _recycle()
  return true
}
