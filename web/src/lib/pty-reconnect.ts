import { reconnectBackoffDelayMs } from '@hermes/shared'

export type PtyConnectionState = 'connecting' | 'open' | 'reconnecting' | 'closed' | 'ended'

export const PTY_RECONNECT_INPUT_MESSAGE = 'Chat is reconnecting. Input will resume when connected.'

// Minimum gap (ms) between page-resume-triggered reconnect attempts, so a
// burst of visibilitychange/pageshow/focus/online events on tab-return
// collapses into a single reconnect.
export const PTY_RESUME_RECONNECT_THROTTLE_MS = 1000

// If a socket sits in WS_CONNECTING past this budget it is treated as wedged
// (e.g. a half-open mobile socket after a radio handoff — the NS-591 case)
// and force-closed so `onclose` → scheduleReconnect can recover it.
export const PTY_CONNECTING_TIMEOUT_MS = 8000

// The same budget for the phase *before* the socket exists: in gated mode a
// connect first awaits a fresh single-use ticket. That request produces no
// WebSocket, so a rejection or a hang is invisible to both `onclose` and the
// CONNECTING timer above — the tab would sit on "connecting" forever with no
// retry. Bound it so the failure routes into the ordinary backoff instead.
export const PTY_TICKET_TIMEOUT_MS = 8000

// Short ladder, tight cap: the PTY is the user's live terminal, so a transient
// drop must come back fast and a dead backend must stop chasing quickly (the
// banner offers a manual Reconnect). Deterministic so the banner can print it.
export const PTY_RECONNECT_BASE_MS = 250
export const PTY_RECONNECT_MAX_MS = 3000
export const PTY_RECONNECT_MAX_ATTEMPTS = 5

// Browsers cannot emit WebSocket ping frames directly. A resize control frame
// is consumed by `/api/pty` without reaching the child process, so it is a
// safe application-level keepalive for quiet terminals behind idle-closing
// proxies.
export const PTY_KEEPALIVE_INTERVAL_MS = 20_000

/** Delay before PTY reconnect `attempt` (1-based: ChatPage bumps its counter before scheduling). */
export function ptyReconnectDelayMs(attempt: number): number {
  return reconnectBackoffDelayMs(attempt - 1, {
    baseDelayMs: PTY_RECONNECT_BASE_MS,
    capMs: PTY_RECONNECT_MAX_MS,
    jitter: false
  })
}

// How long after a resumed socket opens we keep suppressing ANSI erase codes
// (`ESC[K` / `ESC[X`) from the PTY stream. Ink's two-pass virtual scroll emits
// them while replaying a long session; past that replay they are legitimate
// in-place redraws (spinners, progress bars, status lines) and must reach
// xterm or stale glyphs are left on screen. The replay of a 200+ message
// session takes ~10-20s, so this is deliberately generous — over-running the
// window only costs a few stale cells on a buffer that is about to be
// repainted, while under-running it re-opens the blank-viewport bug.
export const PTY_RESUME_SANITIZE_WINDOW_MS = 30000

export interface PtyResumeReconnectInput {
  isActive: boolean
  visibilityState?: DocumentVisibilityState
  online: boolean
  socketReadyState?: number | null
  ptyState: PtyConnectionState
  connectInFlight?: boolean
  /** The automatic ladder used its last attempt; only the explicit Reconnect button restarts it. */
  reconnectGaveUp?: boolean
}

const WS_CONNECTING = 0
const WS_OPEN = 1
const WS_CLOSING = 2
const WS_CLOSED = 3

export function shouldReconnectPtyOnPageResume({
  isActive,
  visibilityState,
  online,
  socketReadyState,
  ptyState,
  connectInFlight,
  reconnectGaveUp
}: PtyResumeReconnectInput): boolean {
  if (!isActive || !online || visibilityState === 'hidden') {
    return false
  }
  // The overlay says retries stopped: a tab focus or network blip must not
  // silently restart the whole ladder behind it.
  if (reconnectGaveUp) {
    return false
  }
  if (ptyState === 'ended') {
    return false
  }
  if (socketReadyState === WS_OPEN) {
    return false
  }
  // A connect is mid-flight (the async socket-open IIFE is awaiting its
  // ticket URL and hasn't assigned wsRef yet, or the socket is still
  // CONNECTING on a non-stuck attempt). Don't fire a redundant reconnect
  // into that window unless the tab already believes it is reconnecting or
  // closed and needs a fresh attempt.
  if ((connectInFlight || socketReadyState === WS_CONNECTING) && ptyState !== 'reconnecting' && ptyState !== 'closed') {
    return false
  }
  return (
    socketReadyState === null ||
    socketReadyState === undefined ||
    socketReadyState === WS_CLOSING ||
    socketReadyState === WS_CLOSED ||
    ptyState === 'reconnecting' ||
    ptyState === 'closed'
  )
}

export function shouldBlockPtyInput(ptyState: PtyConnectionState): boolean {
  return ptyState !== 'open'
}

// Keystrokes typed while the socket is still coming up (the first seconds
// after a page load, or after a transient drop) are held and delivered once
// it opens, instead of being dropped (#145511). Bounded so a long outage
// cannot turn into an unbounded burst on reconnect.
export const PTY_QUEUED_INPUT_LIMIT = 4096

// Gap between `onopen` and flushing held input, so the server has attached
// the PTY before the bytes arrive.
export const PTY_QUEUED_INPUT_FLUSH_DELAY_MS = 150

/** Input blocked in these states is held for the next open socket; in `closed`/`ended` it is dropped. */
export function shouldQueuePtyInput(ptyState: PtyConnectionState): boolean {
  return ptyState === 'connecting' || ptyState === 'reconnecting'
}

/**
 * Append one terminal `onData` chunk to the held-input queue, dropping the
 * oldest whole chunks once the total passes `limit` characters. Chunks are
 * kept intact so an escape sequence (arrow keys, Ctrl combos) is never split
 * in half; a single chunk larger than the limit (a big paste) is dropped
 * rather than truncated.
 */
export function enqueuePtyInput(
  queue: readonly string[],
  data: string,
  limit: number = PTY_QUEUED_INPUT_LIMIT
): string[] {
  if (!data || data.length > limit) {
    return [...queue]
  }

  const next = [...queue, data]
  let total = next.reduce((sum, chunk) => sum + chunk.length, 0)

  while (total > limit) {
    total -= next.shift()?.length ?? 0
  }

  return next
}
