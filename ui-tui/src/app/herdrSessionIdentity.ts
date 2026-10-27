import { spawn } from 'node:child_process'

import { $uiState } from './uiStore.js'

let lastSequence = 0n

/** Observe committed view identity, not gateway process environment or optional shell breadcrumbs. */
export function watchHerdrSessionIdentity(env: NodeJS.ProcessEnv = process.env) {
  // Explicit endpoints can belong to another machine (even when forwarded through localhost).
  // Only the locally bootstrapped owner has a database the local Herdr can safely resume.
  if (env.HERDR_ENV !== '1' || !env.HERDR_PANE_ID?.trim() ||
      env.HERMES_TUI_GATEWAY_URL?.trim() || env.HERMES_TUI_SIDECAR_URL?.trim()) {
    return () => {}
  }

  const pane = env.HERDR_PANE_ID
  const binary = env.HERDR_BIN_PATH || 'herdr'
  let stopped = false
  let running = false
  let pending: string | null = null
  let previousSid: string | null = null
  let previousStoredSid: string | null = null

  const drain = () => {
    if (stopped || running || !pending) {
      return
    }

    const sessionId = pending
    pending = null
    running = true
    const wallNanos = BigInt(Date.now()) * 1_000_000n
    lastSequence = wallNanos > lastSequence ? wallNanos : lastSequence + 1n
    let finished = false
    const finish = () => {
      if (finished) {
        return
      }

      finished = true
      running = false
      drain()
    }

    try {
      const child = spawn(binary, [
        'pane', 'report-agent-session', pane,
        '--source', 'herdr:hermes', '--agent', 'hermes',
        '--seq', String(lastSequence), '--agent-session-id', sessionId,
        '--session-start-source', 'resume'
      ], { stdio: 'ignore', timeout: 1000, killSignal: 'SIGKILL' })
      // Serialize writes so an older slow process cannot overwrite the current conversation.
      child.once('close', finish)
      child.once('error', finish)
    } catch {
      finish()
    }
  }

  const unlisten = $uiState.subscribe(state => {
    if (state.sid === previousSid && state.storedSid === previousStoredSid) {
      return
    }

    previousSid = state.sid
    previousStoredSid = state.storedSid
    pending = state.sid && state.storedSid ? state.storedSid : null
    drain()
  })

  return () => {
    stopped = true
    pending = null
    unlisten()
  }
}
