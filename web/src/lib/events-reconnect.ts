/**
 * Reconnect policy for the ChatSidebar `/api/events` subscriber socket.
 *
 * Pure helpers, no DOM: the component owns the socket and the timer, this
 * module owns the arithmetic and the "is this close code worth retrying"
 * decision so both can be unit-tested without a fake WebSocket.
 */

import { reconnectBackoffDelayMs } from '@hermes/shared'

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
 * Exponential backoff, 1s → 2s → 4s → … → 30s cap. Deterministic (no jitter)
 * because the banner prints the exact delay.
 *
 * `attempt` is 0-based: attempt 0 is the first retry after the initial
 * connection dropped.
 */
export function eventsReconnectDelayMs(attempt: number): number {
  return reconnectBackoffDelayMs(attempt, {
    baseDelayMs: EVENTS_RECONNECT_BASE_MS,
    capMs: EVENTS_RECONNECT_MAX_MS,
    jitter: false
  })
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

// The sidebar's banner is shared with `info.credential_warning` and with the
// JSON-RPC sidecar's errors, so the events socket may only clear a message it
// wrote itself. Everything this module can put in the banner starts with
// EVENTS_FEED_PREFIX so `isEventsFeedMessage` can recognise it.
//
// "Live tool activity" is what the feed is to the user: the sidebar's tool
// list and the chat title. Never name the transport or print a close code —
// those go to the console (ChatSidebar logs them) for diagnosis.
const EVENTS_FEED_PREFIX = 'Live tool activity '

export const EVENTS_DISCONNECTED_MESSAGE = `${EVENTS_FEED_PREFIX}paused — the chat title may not update`

export function eventsReconnectingMessage(delayMs: number): string {
  return `${EVENTS_FEED_PREFIX}paused — reconnecting in ${Math.round(delayMs / 1000)}s…`
}

/** Auth rejection (4401/4403): the login expired, only a reload mints a new one.
 *  The code is logged by the caller; it never appears in user text. */
export function eventsRejectedMessage(code: number): string {
  void code
  return `${EVENTS_FEED_PREFIX}stopped (your login expired). Reload the page to resume.`
}

export function eventsGaveUpMessage(): string {
  return `${EVENTS_FEED_PREFIX}stopped after ${EVENTS_MAX_RECONNECT_ATTEMPTS} reconnect attempts. Click Reconnect side panel, or reload the page.`
}

/** True for the auth-rejection message, which needs a Reload button rather than Reconnect. */
export function isEventsAuthRejectionMessage(message: string | null): boolean {
  return message === eventsRejectedMessage(4401)
}

/**
 * True when `message` is one this module produced, i.e. safe to clear on a
 * successful reconnect. Guards against stomping a `credential_warning` or a
 * sidecar error that happens to be showing when the feed recovers.
 */
export function isEventsFeedMessage(message: string | null): boolean {
  if (!message) {
    return false
  }

  return message.startsWith(EVENTS_FEED_PREFIX)
}
