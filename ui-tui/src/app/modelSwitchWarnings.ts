import type { Msg } from '../types.js'

const MAX_WARNING_SESSIONS = 64
const MAX_WARNINGS_PER_SESSION = 64
const warningsBySession = new Map<string, Array<{ occurrence?: string; message: Msg }>>()

function boundWarningCache(session: string, warnings: Array<{ occurrence?: string; message: Msg }>) {
  if (!warningsBySession.has(session) && warningsBySession.size >= MAX_WARNING_SESSIONS) {
    const oldest = warningsBySession.keys().next().value

    if (oldest !== undefined) {
      warningsBySession.delete(oldest)
    }
  }

  if (warnings.length > MAX_WARNINGS_PER_SESSION) {
    warnings.splice(0, warnings.length - MAX_WARNINGS_PER_SESSION)
  }

  warningsBySession.set(session, warnings)
}

/** Keep display-only status events with their stored session across transcript replacement. */
export function rememberModelSwitchWarning(
  session: string,
  text: string,
  timestamp?: number,
  occurrence?: string
): boolean {
  const warnings = warningsBySession.get(session) ?? []

  if (occurrence && warnings.some(warning => warning.occurrence === occurrence)) {
    return false
  }

  warnings.push({ occurrence, message: { role: 'system', text: `warning: ${text}`, createdAt: timestamp } })
  boundWarningCache(session, warnings)

  return true
}

export function migrateModelSwitchWarnings(fromSession: string, toSession: string): void {
  if (!fromSession || !toSession || fromSession === toSession) {
    return
  }

  const source = warningsBySession.get(fromSession)

  if (!source?.length) {
    return
  }

  const destination = warningsBySession.get(toSession) ?? []
  const merged: Array<{ occurrence?: string; message: Msg }> = []

  // The predecessor session is chronologically older than the canonical successor. Keep that order so
  // bounding removes the oldest rows rather than the successor's newest warning.
  for (const warning of [...source, ...destination]) {
    if (warning.occurrence && merged.some(existing => existing.occurrence === warning.occurrence)) {
      continue
    }

    merged.push(warning)
  }

  warningsBySession.delete(fromSession)
  boundWarningCache(toSession, merged)
}

export function restoreModelSwitchWarnings(session: string, transcript: Msg[]): Msg[] {
  const warnings = warningsBySession.get(session)

  if (!warnings?.length) {
    return transcript
  }

  const result = [...transcript]
  const claimed = new Set<Msg>()

  for (const { message } of warnings) {
    const stored = result.find(
      row =>
        !claimed.has(row) &&
        row.role === 'system' &&
        row.text === message.text &&
        (row.createdAt === undefined || message.createdAt === undefined || row.createdAt === message.createdAt)
    )

    if (stored) {
      claimed.add(stored)

      continue
    }

    const following = result.findIndex(
      row => row.createdAt !== undefined && message.createdAt !== undefined && row.createdAt > message.createdAt
    )

    result.splice(following < 0 ? result.length : following, 0, message)
    claimed.add(message)
  }

  return result
}
