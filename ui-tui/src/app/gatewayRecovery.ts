// Crash-recovery budget for the gateway exit handler. A gateway that
// crash-loops on startup must not let the TUI spawn-storm, so respawn+resume
// attempts are capped to GATEWAY_RECOVERY_LIMIT within a sliding
// GATEWAY_RECOVERY_WINDOW_MS; past the budget the app falls back to the inert
// "gateway exited" state. Kept pure (no refs/UI) so the bound — including the
// crash-loop case — is unit-testable.
export const GATEWAY_RECOVERY_LIMIT = 3
export const GATEWAY_RECOVERY_WINDOW_MS = 60_000

export interface RecoveryPlan {
  // Attempt timestamps to persist (the pruned window, plus `now` iff recovering).
  attempts: number[]
  recover: boolean
  // Session to resume — the live sid, or the not-yet-consumed recovery target
  // when the live sid was already cleared by a prior exit.
  sid: null | string
}

// Decide whether to respawn+resume after a gateway death. `liveSid` is the
// current session (nulled on the first exit); `recoverSid` is a pending
// recovery target carried across a respawn that died before gateway.ready —
// so a startup crash-loop keeps retrying the same session up to the budget
// instead of stranding it after one attempt.
export function planGatewayRecovery(
  liveSid: null | string,
  recoverSid: null | string,
  attempts: number[],
  now: number
): RecoveryPlan {
  const sid = liveSid ?? recoverSid
  const recent = attempts.filter(t => now - t < GATEWAY_RECOVERY_WINDOW_MS)
  const recover = Boolean(sid) && recent.length < GATEWAY_RECOVERY_LIMIT

  return { attempts: recover ? [...recent, now] : recent, recover, sid }
}

// Pick the id to carry into the next gateway.ready recovery (#94935). The
// durable persisted id (state.db row key) wins whenever we have one: the live
// 8-hex sid is minted per gateway process and dies with its in-memory record,
// so a ws_orphan_reap during the very outage being recovered from makes a
// resume-by-live-sid fail with 4007 ("session not found") and strands whatever
// the gateway persisted for that conversation. Falls back to the live sid for
// sessions with nothing persisted yet (fresh/lazy chats).
export function pickRecoverySessionId(durableId: string, liveSid: null | string): null | string {
  return durableId || liveSid || null
}
