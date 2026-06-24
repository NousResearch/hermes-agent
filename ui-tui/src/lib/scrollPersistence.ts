/**
 * Scroll-position persistence across a renderer recycle (Stage 1).
 *
 * THE GAP this fills: `resumeById` always `scrollToBottom()` after a resume
 * (useSessionLifecycle.ts:340). On a CRASH that's fine — but for a deliberate
 * renderer RECYCLE (the inversion's "kill the disposable renderer, respawn a
 * fresh one" path), snapping to the bottom loses the user's scroll position and
 * makes the recycle *visible*. The whole point of the inversion is that a
 * recycle is invisible. So we persist {top, atBottom} keyed by sid when a
 * renderer is about to exit, and the fresh renderer restores it on resume.
 *
 * Everything the gateway holds (transcript, in-flight turn) already survives via
 * the durable anchor + session.resume live-session adoption. Scroll position is
 * the ONE piece of pure-renderer view state that dies with the renderer, so it's
 * the only thing this module needs to carry across.
 *
 * Storage: a tiny JSON file under the runtime dir, keyed by sid. Best-effort —
 * a missing/corrupt file just falls back to the existing scrollToBottom default,
 * so this can never make resume worse than today, only better.
 */

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'

export interface ScrollState {
  /** Scroll-top in rows at recycle time. Ignored when atBottom is true. */
  top: number
  /** Was the view pinned to the bottom (sticky)? The common case. */
  atBottom: boolean
  /** When persisted (ms). Lets a stale entry be ignored after a TTL. */
  savedAt: number
}

/** Entries older than this are treated as stale and ignored on restore. */
export const SCROLL_STATE_TTL_MS = 10 * 60 * 1000 // 10 min

function stateDir(): string {
  // Prefer an explicit runtime dir (the orchestrator sets one) so the persisted
  // file lives beside the session's other runtime state; fall back to tmp.
  const base = process.env.HERMES_TUI_RUNTIME_DIR?.trim() || join(tmpdir(), 'hermes-tui-scroll')
  return base
}

/** Per-sid file path. sid is a hex session id, safe as a filename. */
export function scrollStatePath(sid: string, dir = stateDir()): string {
  return join(dir, `scroll-${sid}.json`)
}

/**
 * Persist scroll state for `sid`. Called when the renderer is about to exit for
 * a recycle. Best-effort: any error is swallowed (a failed persist just means
 * the fresh renderer falls back to scrollToBottom, the current behaviour).
 */
export function persistScrollState(sid: string, state: Omit<ScrollState, 'savedAt'>, dir = stateDir()): boolean {
  if (!sid) {
    return false
  }
  try {
    const path = scrollStatePath(sid, dir)
    mkdirSync(dirname(path), { recursive: true })
    const payload: ScrollState = { ...state, savedAt: Date.now() }
    writeFileSync(path, JSON.stringify(payload), 'utf8')
    return true
  } catch {
    return false
  }
}

/**
 * Restore scroll state for `sid`, or null when absent/stale/corrupt. The caller
 * uses null to mean "keep the existing scrollToBottom default".
 *
 * `now` is injectable for deterministic TTL tests.
 */
export function restoreScrollState(sid: string, dir = stateDir(), now: number = Date.now()): ScrollState | null {
  if (!sid) {
    return null
  }
  try {
    const raw = readFileSync(scrollStatePath(sid, dir), 'utf8')
    const parsed = JSON.parse(raw) as Partial<ScrollState>
    if (
      typeof parsed?.top !== 'number' ||
      typeof parsed?.atBottom !== 'boolean' ||
      typeof parsed?.savedAt !== 'number'
    ) {
      return null
    }
    if (now - parsed.savedAt > SCROLL_STATE_TTL_MS) {
      return null // stale — fall back to default
    }
    return { top: parsed.top, atBottom: parsed.atBottom, savedAt: parsed.savedAt }
  } catch {
    return null
  }
}

/**
 * Apply a restored scroll state to a ScrollBox-like handle. Returns true if a
 * non-default (mid-history) position was applied, false when the caller should
 * use its existing scrollToBottom default (atBottom, or nothing to restore).
 *
 * Kept handle-shape-minimal so it's trivially testable with a fake.
 */
export interface ScrollApplyTarget {
  scrollTo: (y: number) => void
  scrollToBottom: () => void
}

export function applyScrollState(target: ScrollApplyTarget, state: ScrollState | null): boolean {
  if (!state || state.atBottom) {
    target.scrollToBottom()
    return false
  }
  target.scrollTo(Math.max(0, state.top))
  return true
}
