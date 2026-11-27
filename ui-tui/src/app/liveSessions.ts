import type {
  SessionActiveItem,
  SessionActiveListResponse
} from '@hermes/shared/gateway-types'

import type { GatewayClient } from '../gatewayClient.js'

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
  const sessionTitle = sessions.find(s => s.current || s.id === prev.sid)?.title?.trim() ?? ''
  // display.tui_quiet_idle: an idle TUI must write nothing, so the
  // count holds between turns (other sessions starting/ending would repaint
  // it) and catches up on the first poll of the next turn. The initial
  // hydrate (count still 0) always lands.
  const holdCount = prev.quietIdle && !prev.busy && prev.liveSessionCount > 0
  const liveSessionCount = holdCount ? prev.liveSessionCount : sessions.length

  if (prev.liveSessionCount !== liveSessionCount || prev.sessionTitle !== sessionTitle) {
    patchUiState({ liveSessionCount, sessionTitle })
  }
}
