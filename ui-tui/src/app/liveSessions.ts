import type { SessionActiveListResponse } from '@hermes/shared/gateway-types'

import type { GatewayClient } from '../gatewayClient.js'
import type { SessionActiveItem, SessionActiveListResponse } from '@hermes/shared/gateway-types'

import { getUiState, patchUiState } from './uiStore.js'

/** The live-session roster the status bar's title and count read: on the shared owner its own
 * `session.list` (live scope); the sidecar's `session.active_list` never holds those sessions. */
export const requestLiveSessions = (gw: Pick<GatewayClient, 'isCanonical' | 'request'>, currentSid: null | string) =>
  gw.isCanonical
    ? gw.request<SessionActiveListResponse>('session.list', { limit: 200 })
    : gw.request<SessionActiveListResponse>('session.active_list', { current_session_id: currentSid })

/** The status bar's live count and the terminal titlebar's session title from one poll. Patches only
 * on a change: patchUiState always notifies every $uiState subscriber, and patching on each 1.5 s
 * poll re-renders the whole TUI (idle flicker). */
export function applyLiveSessionRoster(sessions: SessionActiveItem[]) {
  const prev = getUiState()
  const liveSessionCount = sessions.length
  const sessionTitle = sessions.find(s => s.current || s.id === prev.sid)?.title?.trim() ?? ''

  if (prev.liveSessionCount !== liveSessionCount || prev.sessionTitle !== sessionTitle) {
    patchUiState({ liveSessionCount, sessionTitle })
  }
}
