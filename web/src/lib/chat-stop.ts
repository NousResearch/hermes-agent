export interface SessionInterruptGateway {
  connect(): Promise<void>
  request<T>(method: string, params?: Record<string, unknown>): Promise<T>
}

export function isCurrentStopRequest(
  requestGeneration: number,
  currentGeneration: number,
  requestedChannel: string,
  currentChannel: string | null,
  requestedSessionId: string,
  currentSessionId: string | null,
): boolean {
  return (
    requestGeneration === currentGeneration &&
    requestedChannel === currentChannel &&
    requestedSessionId === currentSessionId
  )
}

export interface SessionInterruptResponse {
  status?: string
  interrupted?: boolean
  turn_isolation?: boolean
}

/**
 * Interrupt the live dashboard TUI turn through the authenticated gateway.
 *
 * The caller supplies the runtime session id observed from the current PTY's
 * structured event stream. The id is passed unchanged so a stale or malformed
 * value cannot be silently redirected to another session.
 */
export async function interruptCurrentSession(
  gateway: SessionInterruptGateway,
  sessionId: string
): Promise<SessionInterruptResponse> {
  if (!sessionId || !sessionId.trim()) {
    throw new Error('Current chat session is not available')
  }

  await gateway.connect()
  const response = await gateway.request<SessionInterruptResponse>('session.interrupt', {
    session_id: sessionId
  })

  if (response?.status !== 'interrupted') {
    throw new Error('Stop was not acknowledged')
  }

  return response
}
