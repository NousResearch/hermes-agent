export const SESSION_EXIT_TIMEOUT_MS = 5000

export interface SessionExitOptions {
  closeSession: (sessionId: string) => Promise<unknown>
  exit: (code: number) => void
  getSessionId: () => null | string
}

export function createSessionExit({ closeSession, exit, getSessionId }: SessionExitOptions) {
  let pending: null | Promise<void> = null

  return (code = 0) => {
    pending ??= (async () => {
      const sessionId = getSessionId()

      if (sessionId) {
        let timeout: ReturnType<typeof setTimeout> | undefined

        try {
          // session.close acknowledges after provider finalization. Keep the
          // transport alive for that receipt, but allow quitting a dead backend.
          await Promise.race([
            closeSession(sessionId),
            new Promise<void>(resolve => {
              timeout = setTimeout(resolve, SESSION_EXIT_TIMEOUT_MS)
            })
          ])
        } catch {
          // A disconnected backend must not trap the user in the terminal.
        } finally {
          clearTimeout(timeout)
        }
      }

      exit(code)
    })()

    return pending
  }
}
