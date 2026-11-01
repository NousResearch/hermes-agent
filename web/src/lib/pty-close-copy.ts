import { en } from '@/i18n/en'
/**
 * User-facing copy for the dashboard chat (PTY) connection lifecycle.
 *
 * Pure: ChatPage owns the socket and state, this module owns the words and
 * the "which button goes with this failure" decision, so both can be tested
 * without xterm or a fake WebSocket. WebSocket close codes never appear in
 * user text — ChatPage logs them to the console instead.
 */

export type PtyBannerAction = 'reload' | 'check-server' | null

export interface PtyBanner {
  kind:
    | 'sessionTokenUnavailable'
    | 'authFailed'
    | 'originRefused'
    | 'websocketUnavailable'
    | 'localClientRefused'
    | 'reconnectGaveUp'
  text: string
  action: PtyBannerAction
}

/** Chat tab opened without the injected login token (loopback mode only). */
export const PTY_TOKEN_MISSING_BANNER: PtyBanner = {
  kind: 'sessionTokenUnavailable',
  text: en.chatSidebar.sessionTokenUnavailable,
  action: 'reload'
}

/** Rejection close codes the server sends before any PTY exists. `reason` is logged, not shown. */
const REJECTION_BANNERS: Record<number, PtyBanner> = {
  4401: {
    kind: 'authFailed',
    text: en.chatSidebar.authFailed,
    action: 'reload'
  },
  4403: {
    kind: 'originRefused',
    text: en.chatSidebar.originRefused,
    action: null
  },
  4404: {
    kind: 'websocketUnavailable',
    text: en.chatSidebar.websocketUnavailable,
    action: 'reload'
  },
  4408: {
    kind: 'localClientRefused',
    text: en.chatSidebar.localClientRefused,
    action: null
  }
}

export function ptyRejectionBanner(code: number): PtyBanner | null {
  return REJECTION_BANNERS[code] ?? null
}

/** Shown while the automatic reconnect ladder is running. No close code. */
export const PTY_RECONNECTING_BANNER = 'Chat connection interrupted. Reconnecting...'

/** Shown after the last automatic attempt failed (overlay + banner). */
export const PTY_GAVE_UP_BANNER: PtyBanner = {
  kind: 'reconnectGaveUp',
  text: en.chatSidebar.reconnectGaveUp,
  action: 'check-server'
}

/** Overlay copy: the hermes --tui child exited; a crash looks identical to `/exit`. */
export const PTY_SESSION_ENDED_MESSAGE =
  'Chat session ended. If you did not end it yourself, the agent may have crashed — open Logs to see why, or start a new session.'

/** Overlay copy when the server could not start the chat at all (close 1011; details are in the
 *  terminal). Neutral on purpose: 1011 also covers "no terminal support on this platform"
 *  (native Windows), where retrying cannot help, so the text must not promise a fix. */
export const PTY_START_FAILED_MESSAGE = 'Chat could not start. The reason is printed above.'

/** Terminal footer line replacing `[session ended (code N)]`. */
export const PTY_SESSION_ENDED_TERMINAL_LINE = '[chat session ended]'

/** True once the ladder has used its last attempt and that attempt also failed. */
export function ptyReconnectExhausted(attempt: number, maxAttempts: number): boolean {
  return attempt >= maxAttempts
}
