import type { GatewayClient } from '../gatewayClient.js'
import type { SessionActiveListResponse } from '../gatewayTypes.js'

/** The live-session roster the status bar's title and count read: on the shared owner its own
 * `session.list` (live scope); the sidecar's `session.active_list` never holds those sessions. */
export const requestLiveSessions = (gw: Pick<GatewayClient, 'isCanonical' | 'request'>, currentSid: null | string) =>
  gw.isCanonical
    ? gw.request<SessionActiveListResponse>('session.list', { limit: 200 })
    : gw.request<SessionActiveListResponse>('session.active_list', { current_session_id: currentSid })
