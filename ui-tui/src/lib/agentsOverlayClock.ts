interface StatusLike {
  status: string
}

/**
 * Agents overlay clock policy:
 * - live running/queued agents need the 500ms gantt/elapsed cadence;
 * - running background processes need 1s elapsed updates;
 * - settled process rows still need 1s wall-clock so PROCESS_RETAIN_SECONDS
 *   ("Ns ago" / retain window) can elapse and rows can leave the dock;
 * - static replay with no processes (only settled agents) has no wall-clock-
 *   dependent presentation and returns null.
 */
export const agentsOverlayClockIntervalMs = (
  replayMode: boolean,
  agents: readonly StatusLike[],
  processes: readonly StatusLike[]
): number | null => {
  if (!replayMode && agents.some(item => item.status === 'running' || item.status === 'queued')) {
    return 500
  }

  if (processes.some(item => item.status === 'running')) {
    return 1000
  }

  // Exited/done/failed/killed/lost rows stay visible for PROCESS_RETAIN_SECONDS;
  // freeze the clock and `now` never advances enough for them to leave.
  if (processes.length > 0) {
    return 1000
  }

  return null
}
