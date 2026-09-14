import type { SessionInfo } from '../types.js'

export type PendingRow = NonNullable<SessionInfo['pending_submissions']>[number]

// Inputs whose transcript bubble this client already painted at submit time.
// A busy-time admission defers its bubble until the authority starts it, so
// the bubble lands after the in-flight assistant reply instead of inside it.
const shown = new Set<string>()

export const markBubbleShown = (inputId?: string) => { if (inputId) { shown.add(inputId) } }
export const resetBubbleLedger = () => shown.clear()

// Rows the authority just moved queued → started whose bubble nobody painted.
// Only a transition observed by THIS client counts: a resume snapshot's started
// row already lives in the replayed history.
export function newlyStartedRows(previous: PendingRow[] | undefined, incoming: PendingRow[] | undefined): PendingRow[] {
  if (!previous || !incoming) { return [] }
  const wasQueued = new Set(previous.filter(row => row.status === 'queued').map(row => row.admission_id))

  return incoming.filter(row => row.status === 'started' && wasQueued.has(row.admission_id) && !(row.input_id && shown.has(row.input_id)))
}

export const isServerQueued = (row: PendingRow) => row.status === 'queued' || row.status === 'unknown'
