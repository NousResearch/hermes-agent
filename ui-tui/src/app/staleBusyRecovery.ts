import type { SessionActiveItem } from '../gatewayTypes.js'

interface CompletionSnapshot {
  awaiting: boolean
  epoch: number
}

export const shouldRecoverStaleBusy = ({
  busy,
  currentCompletion,
  currentSessionId,
  requestedCompletion,
  requestedSessionId,
  sessions
}: {
  busy: boolean
  currentCompletion: CompletionSnapshot
  currentSessionId: null | string
  requestedCompletion: CompletionSnapshot
  requestedSessionId: null | string
  sessions: SessionActiveItem[]
}): boolean => {
  if (
    !busy ||
    !currentSessionId ||
    requestedSessionId !== currentSessionId ||
    !requestedCompletion.awaiting ||
    !currentCompletion.awaiting ||
    requestedCompletion.epoch !== currentCompletion.epoch
  ) {
    return false
  }

  const current = sessions.find(session => session.id === currentSessionId)

  return current?.status === 'idle'
}
