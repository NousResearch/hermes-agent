// Crash-recovery budget for the gateway exit handler. A gateway that
// crash-loops on startup must not let the TUI spawn-storm, so respawn+resume
// attempts are capped to GATEWAY_RECOVERY_LIMIT within a sliding
// GATEWAY_RECOVERY_WINDOW_MS; past the budget the app falls back to the inert
// "gateway exited" state. Kept pure (no refs/UI) so the bound is unit-testable.
export const GATEWAY_RECOVERY_LIMIT = 3
export const GATEWAY_RECOVERY_WINDOW_MS = 60_000

// Given prior attempt timestamps and the current time, return the timestamps
// still inside the window plus whether another recovery is allowed. The caller
// appends `now` to `recent` only when it actually starts a recovery.
export function evalRecovery(attempts: number[], now: number): { allowed: boolean; recent: number[] } {
  const recent = attempts.filter(t => now - t < GATEWAY_RECOVERY_WINDOW_MS)

  return { allowed: recent.length < GATEWAY_RECOVERY_LIMIT, recent }
}
