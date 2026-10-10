// Lane t2 — app/ session flow: useMainApp, useSessionLifecycle, useInputHandlers,
// turnController, createServerRequestHandler, setupHandoff. Owned namespace: `session`.
// Function leaves take positional args; packs use `{0}`, `{1}` (argument order documented per leaf).
//
// Not keyed on purpose: the status-bar state values the app layer compares
// against ('ready', 'running…', 'interrupted', 'summoning hermes…') and the
// transient trail marker 'analyzing tool output…' — those are mapped to the
// `status` namespace at render time (appChrome.displayStatus / thinking.tsx).

export const sessionEn = {
  session: {
    // Free-text status-bar phrases (patchUiState({ status })) that nothing compares against.
    status: {
      setupRequired: 'setup required',
      setupRunning: 'setup running…',
      startingAgent: 'starting agent…',
      waitingForInput: 'waiting for input…',
      switchingSession: 'switching session…',
      resuming: 'resuming…',
      interrupting: 'interrupting…',
      reconnecting: 'reconnecting…',
      restarting: 'restarting…',
      stopped: 'stopped',
      closingSession: 'closing session…',
      approvalNeeded: 'approval needed',
      sudoPasswordNeeded: 'sudo password needed',
      secretInputNeeded: 'secret input needed',
      unlockVault: (displayName: string) => `unlock ${displayName}`,
      saveLogin: (site: string) => `save login for ${site}`,
      verificationCode: (site: string) => `verification code for ${site}`
    },
    // Shared by the rpc wrapper (useMainApp) and the lifecycle paths; `error: ` prefix stays literal.
    common: {
      invalidResponse: (method: string) => `invalid response: ${method}`,
      // useSubmission.interpolate: a `{!cmd}` that could not run refuses the whole submit.
      interpolationFailed: (message: string) => `not sent: {!…} could not run (${message}) — input retained`
    },
    lifecycle: {
      newLiveSessionStarted: 'new live session started',
      sessionTitleSet: (title: string) => `session title set: ${title}`,
      titleQueuedSuffix: ' (queued while session initializes)',
      failedToSetTitle: (message: string) => `failed to set session title: ${message}`,
      switchSessions: 'switch sessions',
      interruptBeforeSwitch: (what: string) => `interrupt the current turn before trying to ${what}`
    },
    main: {
      failedToStartLiveSession: 'failed to start new live session',
      invalidModelSwitchResponse: 'invalid response: model switch',
      modelSwitched: (model: string) => `model → ${model}`,
      widgetLive: (id: string) => `widget /${id} is live — type /${id} to open`,
      widgetRemoved: (id: string) => `widget /${id} removed (file deleted)`,
      // {0}=file {1}=message
      widgetFailedToLoad: (file: string, message: string) => `widget ${file} failed to load: ${message}`,
      clarifyQuestionsOne: (count: string) => `${count} question`,
      clarifyQuestionsOther: (count: string) => `${count} questions`,
      skipped: '(skipped)',
      voiceRec: '● REC',
      voiceStt: '◉ STT',
      voiceOn: 'voice on',
      voiceOff: 'voice off',
      voiceTtsSuffix: ' [tts]'
    },
    approval: {
      denied: 'denied',
      approved: (choice: string) => `approved (${choice})`
    },
    input: {
      dashboardNewSession: 'starting a fresh dashboard chat...',
      voiceStillTranscribing: 'voice: still transcribing; try again shortly',
      voiceModeOff: 'voice: mode is off — enable with /voice on',
      voiceError: (message: string) => `voice error: ${message}`,
      sudoCancelled: 'sudo cancelled',
      secretCancelled: 'secret entry cancelled',
      vaultStaysLocked: (displayName: string) => `${displayName} stays locked`,
      loginNotSaved: (site: string) => `login for ${site} not saved`,
      codeSkipped: (site: string) => `verification code for ${site} skipped`,
      failedToOpenEditor: 'failed to open editor',
      failedToOpenEditorWith: (message: string) => `failed to open editor: ${message}`,
      yoloNeedsSession: 'yolo needs an active session',
      yoloOn: 'yolo on',
      yoloOff: 'yolo off',
      yoloToggleFailed: 'failed to toggle yolo'
    },
    turn: {
      interrupted: 'interrupted'
    },
    request: {
      dangerousCommand: 'dangerous command'
    },
    handoff: {
      launching: (command: string) => `launching \`hermes ${command}\`…`,
      launchError: (error: string) => `error launching hermes: ${error}`,
      // {0}=subcommand {1}=exit code
      exitedWithCode: (subcommand: string, code: string) => `hermes ${subcommand} exited with code ${code}`,
      stillNoProvider: 'still no provider configured'
    }
  }
}
