// slashCmd.handoff — replies and status lines of app/slash/commands/handoff.ts.
// Command NAMES/aliases/arg syntax inside usage strings stay literal.

export const slashCmdHandoffEn = {
  handoff: {
    usage: 'usage: /handoff <platform>',
    noSession: 'no active session — nothing to hand off',
    waitForWork: 'wait for the current turn and queued/background work before handoff',
    inProgress: 'handoff in progress — wait for the result before changing this session',
    statusPending: 'handoff pending…',
    pending: (platform: string, home: string) => `handoff pending: ${platform} · ${home}`,
    invalidAck: 'invalid handoff acknowledgement',
    completed: 'handoff completed — continue on the destination platform; /new starts a new session here',
    statusCompleted: 'handoff completed',
    unknownError: 'unknown error',
    failed: (error: string) =>
      `handoff failed: ${error} — check the destination before resuming; /new starts a fresh session`,
    statusFailed: (error: string) => `handoff failed: ${error}`,
    unknown: 'handoff outcome unknown — check the destination before resuming; /new starts a fresh session',
    unknownWithError: (error: string) =>
      `handoff outcome unknown: ${error} — check the destination before resuming; /new starts a fresh session`,
    statusUnknown: 'handoff outcome unknown',
    pollTimedOut: 'handoff status polling timed out',
    rejected: (error: string) => `handoff: ${error}`
  }
}
