import { t } from '../i18n/runtime.js'

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
