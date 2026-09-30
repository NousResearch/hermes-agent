interface StatusLike {
  status: string
}

/**
 * Agents overlay clock policy:
 * - live running/queued agents need the 500ms gantt/elapsed cadence;
 * - running background processes only need 1s elapsed updates;
 * - settled/replay content has no wall-clock-dependent presentation.
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

  return null
}
