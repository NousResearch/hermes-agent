/** Deadlines for the Hermes Console websocket connect.
 *
 *  Two phases can wedge without ever producing a `close` event, which is the
 *  only signal the modal's error path listens to:
 *
 *  - the single-use ticket `api.buildWsUrl` mints, which runs *before* any
 *    socket exists, so a hang produces no socket and no `close`;
 *  - the socket itself sitting in `CONNECTING` after a radio handoff or
 *    behind a proxy that accepts and stalls.
 *
 *  Mirrors `pty-reconnect`'s PTY_TICKET_TIMEOUT_MS / PTY_CONNECTING_TIMEOUT_MS
 *  for the console surface.
 */

export const CONSOLE_TICKET_TIMEOUT_MS = 8000

export const CONSOLE_CONNECTING_TIMEOUT_MS = 8000

export type TicketOutcome<T> =
  | { status: 'failed'; error: unknown }
  | { status: 'ok'; value: T }
  | { status: 'timeout' }

/** Await `mint()` under a deadline.
 *
 *  Resolves `timeout` when the budget runs out first — and keeps resolving
 *  `timeout` if the request lands afterwards, so a late ticket can never open
 *  a socket behind the failure the caller already reported.
 */
export function raceTicket<T>(
  mint: () => Promise<T>,
  timeoutMs: number = CONSOLE_TICKET_TIMEOUT_MS,
): Promise<TicketOutcome<T>> {
  return new Promise<TicketOutcome<T>>(resolve => {
    let settled = false
    const finish = (outcome: TicketOutcome<T>) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolve(outcome)
    }
    const timer = setTimeout(() => finish({ status: 'timeout' }), timeoutMs)

    mint().then(
      value => finish({ status: 'ok', value }),
      error => finish({ error, status: 'failed' }),
    )
  })
}
