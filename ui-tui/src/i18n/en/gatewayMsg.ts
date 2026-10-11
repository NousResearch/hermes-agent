// Lane t2 — app/createGatewayEventHandler.ts: notices/activity lines rendered from
// gateway events. Owned namespace: `gatewayMsg`.
// Function leaves take positional args; packs use `{0}`, `{1}` (argument order documented per leaf).
//
// Not keyed on purpose: `[bg <id>]` / `[btw "…"]` transcript tags (locale-neutral
// chips), the `${n} subagents` spawn-tree label (persisted to the backend), the
// startup image question (sent to the model), `error: ` machine prefixes, and
// state values other code compares against ('ready').

export const gatewayMsgEn = {
  gatewayMsg: {
    // Free-text status-bar phrases nothing compares against.
    status: {
      recoveringSession: 'recovering session…',
      forgingSession: 'forging session…',
      resumingMostRecent: 'resuming most recent…',
      protocolWarning: 'protocol warning'
    },
    startup: {
      querySkipped: 'startup query skipped: no active session',
      imageAttachFailed: (message: string) => `startup image attach failed: ${message}`,
      commandCatalogUnavailable: (message: string) => `command catalog unavailable: ${message}`
    },
    clarify: {
      // Reason shown as "(…)" after an abandoned clarify prompt.
      timedOut: 'timed out'
    },
    agents: {
      workingNudge: 'subagents working · /agents to watch live'
    },
    // Brief status-bar echo of a goal status.update; leading glyph mirrors the backend line.
    goal: {
      complete: '✓ goal complete',
      continuing: '↻ goal continuing',
      paused: '⏸ goal paused'
    },
    billing: {
      openLinkRemoteSpending: '💳 Open this link to allow Remote Spending:',
      enterCode: (code: string) => `If prompted, enter code: ${code}`
    },
    voice: {
      stopPhrase: 'voice: stop phrase — voice chat ended',
      noSpeechLimit: 'voice: no speech detected 3 times, continuous mode stopped'
    },
    wake: {
      // {0}=profile name (used twice: in the notice and in the suggested command)
      otherProfile: (profile: string) => `wake phrase for profile '${profile}' — run: hermes -p ${profile} --tui`,
      failed: (message: string) => `wake: ${message}`
    },
    protocol: {
      noiseDetected: 'protocol noise detected · /logs to inspect',
      noise: (preview: string) => `protocol noise: ${preview}`
    },
    moa: {
      // {0}=references done {1}=references total (pre-formatted strings)
      refs: (done: string, total: string) => `MoA: refs ${done}/${total}`,
      aggregating: 'MoA: aggregating…'
    },
    error: {
      unknown: 'unknown error'
    }
  }
}
