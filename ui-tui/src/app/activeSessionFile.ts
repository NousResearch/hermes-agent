import { randomUUID } from 'node:crypto'
import { closeSync, openSync, renameSync, unlinkSync, writeFileSync } from 'node:fs'

export const writeActiveSessionFile = (sessionId: null | string, file = process.env.HERMES_TUI_ACTIVE_SESSION_FILE) => {
  if (!file || !sessionId) {
    return
  }

  let pending: string | undefined

  try {
    // Same-directory rename keeps dashboard reconnect readers on complete JSON.
    const staging = `${file}.${randomUUID()}.tmp`
    const fd = openSync(staging, 'wx', 0o600)
    pending = staging

    try {
      writeFileSync(fd, JSON.stringify({ session_id: sessionId }))
    } finally {
      closeSync(fd)
    }

    renameSync(staging, file)
    pending = undefined
  } catch {
    // Reconnect / shell epilogue hint only; never break live session changes.
  } finally {
    if (pending) {
      try {
        unlinkSync(pending)
      } catch {
        // Cleanup is also best-effort if the directory became unavailable.
      }
    }
  }
}
