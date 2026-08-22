/**
 * Reconnect policy for the ChatSidebar `/api/events` subscriber socket.
 *
 * Pure helpers, no DOM: the component owns the socket and the timer, this
 * module owns the arithmetic and the "is this close code worth retrying"
 * decision so both can be unit-tested without a fake WebSocket.
 */

export const EVENTS_RECONNECT_BASE_MS = 1_000
export const EVENTS_RECONNECT_MAX_MS = 30_000
export const EVENTS_MAX_RECONNECT_ATTEMPTS = 15
/** Bound ticket minting plus the WebSocket opening handshake. */
export const EVENTS_CONNECT_TIMEOUT_MS = 15_000

/** Normal closure — the server said goodbye, don't chase it. */
const WS_CLOSE_NORMAL = 1000
/** Ticket rejected / forbidden: retrying just burns tickets, user must reload. */
const WS_CLOSE_AUTH_CODES = new Set([4401, 4403])

/**
 * Exponential backoff, 1s → 2s → 4s → … → 30s cap.
 *
 * `attempt` is 0-based: attempt 0 is the first retry after the initial
 * connection dropped.
 */
export function eventsReconnectDelayMs(attempt: number): number {
  const exponent = Math.max(0, Math.trunc(attempt))

  // 2 ** exponent overflows to Infinity long before it matters; Math.min
  // still clamps correctly, but guard anyway so the delay stays a number.
  const raw = EVENTS_RECONNECT_BASE_MS * 2 ** Math.min(exponent, 32)

  return Math.min(raw, EVENTS_RECONNECT_MAX_MS)
}

/**
 * Whether a close code should trigger a retry.
 *
 * Auth rejections are terminal (the banner tells the user to reload) and a
 * normal 1000 close is intentional. Everything else — gateway restart,
 * network drop, 1005/1006, proxy timeout — is worth retrying.
 */
export function shouldRetryEventsClose(code: number | undefined): boolean {
  if (code === undefined) {
    return true
  }

  return code !== WS_CLOSE_NORMAL && !WS_CLOSE_AUTH_CODES.has(code)
}

export function isEventsAuthRejection(code: number | undefined): boolean {
  return code !== undefined && WS_CLOSE_AUTH_CODES.has(code)
}
