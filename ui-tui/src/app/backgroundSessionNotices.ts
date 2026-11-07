import type { SessionActiveListResponse } from '@hermes/shared/gateway-types'

import type { GatewayClient } from '../gatewayClient.js'
import { t } from '../i18n/runtime.js'
import { asRpcResult } from '../lib/rpc.js'

import { getUiState } from './uiStore.js'

// Sessions this view left running (/new mid-turn, the switcher) keep working out of sight; the
// 1.5 s live-session poll is what notices their turn ending or a question waiting, and says so in
// the focused transcript. Only sessions this TUI has shown count: on the shared gateway the live
// feed also lists other clients' chats, which are not ours to announce.

interface LiveRow {
  id?: string
  running?: boolean
  session_id?: string
  status?: string
  title?: string
}

type Phase = 'busy' | 'idle' | 'waiting'

const phaseOf = (row: LiveRow): Phase => {
  const status = row.status ?? (row.running ? 'working' : 'idle')

  return status === 'waiting' ? 'waiting' : status === 'working' || status === 'starting' ? 'busy' : 'idle'
}

const label = (id: string, title?: string) => title?.trim() || id.slice(0, 12)

export function createBackgroundSessionWatch() {
  const shown = new Set<string>()
  const last = new Map<string, Phase>()

  return {
    /** Mark *sid* as a session this view displayed (its later turns are worth announcing). */
    focus(sid: null | string) {
      if (sid) {
        shown.add(sid)
      }
    },

    /** *sid* was just left mid-turn: it is ours and busy, whatever the last poll sampled (a turn
     * started after an idle poll and finished before the next one still gets its notice). */
    handoff(sid: null | string) {
      if (sid) {
        shown.add(sid)
        last.set(sid, 'busy')
      }
    },

    /** Whether a session this view showed is live off screen (only then is a status read worth it). */
    watching(rows: LiveRow[], focused: null | string): boolean {
      return rows.some(row => {
        const id = row.session_id || row.id

        return Boolean(id) && id !== focused && shown.has(id!)
      })
    },

    /** Notices for this poll: a shown, non-focused session whose turn finished or that now waits. */
    update(rows: LiveRow[], focused: null | string): string[] {
      const notices: string[] = []
      const seen = new Set<string>()

      for (const row of rows) {
        const id = row.session_id || row.id

        if (!id) {
          continue
        }

        seen.add(id)
        const phase = phaseOf(row)
        const before = last.get(id)
        last.set(id, phase)

        if (id === focused || !shown.has(id) || before === undefined || before === phase) {
          continue
        }

        if (phase === 'waiting') {
          notices.push(t('session.lifecycle.backgroundWaiting', label(id, row.title)))
        } else if (phase === 'idle' && before === 'busy') {
          notices.push(t('session.lifecycle.backgroundFinished', label(id, row.title)))
        }
      }

      for (const id of [...last.keys()]) {
        if (!seen.has(id)) {
          last.delete(id)
        }
      }

      return notices
    }
  }
}

export type BackgroundSessionWatch = ReturnType<typeof createBackgroundSessionWatch>

/** The poll's notices. The shared gateway's live `session.list` says only running or not; while a
 * shown session is off screen its `session.active_list` is read too, which says `waiting` when a
 * question (approval, clarify) holds the turn. */
export async function backgroundSessionNotices(
  watch: BackgroundSessionWatch,
  gw: Pick<GatewayClient, 'isCanonical' | 'request'>,
  rows: LiveRow[],
  focused: null | string,
  fresh: () => boolean = () => true
): Promise<string[]> {
  if (!gw.isCanonical || !watch.watching(rows, focused)) {
    return watch.update(rows, focused)
  }

  const raw = await gw.request<SessionActiveListResponse>('session.active_list', { current_session_id: focused })

  return fresh() ? watch.update(asRpcResult<SessionActiveListResponse>(raw)?.sessions ?? rows, focused) : []
}

// One per process, like the overlay and server-request stores: the TUI has one transcript.
const watch = createBackgroundSessionWatch()

/** A session this view leaves while its turn runs (`/new` mid-turn, the switcher). */
export const noteBackgroundHandoff = (sid: null | string) => watch.handoff(sid)

/** The live-session poll's entry: marks the focused session as shown and writes the notices for
 * the sessions this view left running into the focused transcript. *isCurrent* is the poll
 * effect's liveness: a reply that lands after the view switched sessions is dropped whole (the
 * watch is not advanced either, so the next poll still sees the transition). */
export function noteLiveSessions(
  gw: Pick<GatewayClient, 'isCanonical' | 'request'>,
  rows: LiveRow[],
  sys: (text: string) => void,
  isCurrent: () => boolean = () => true
) {
  const focused = getUiState().sid
  watch.focus(focused)

  const fresh = () => isCurrent() && getUiState().sid === focused

  void backgroundSessionNotices(watch, gw, rows, focused, fresh)
    .then(notices => notices.forEach(sys))
    .catch(() => {})
}
