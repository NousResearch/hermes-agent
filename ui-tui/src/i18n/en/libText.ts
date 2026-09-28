// Lane t2 — gatewayClient displayed reasons, lib/*, domain/*, hooks/* user-facing text.
// Owned namespace: `libText`.
// Function leaves take positional args; packs use `{0}`, `{1}` (argument order documented per leaf).

export const libTextEn = {
  libText: {
    // gatewayClient.ts — transport-drop reasons that reject in-flight RPCs and
    // therefore surface in the transcript. `gateway not connected` /
    // `gateway not running` are deliberately NOT here: app/userMessages.ts
    // parses them (NOT_CONNECTED_RE) and rewrites them into plain words.
    gateway: {
      restarting: 'gateway restarting',
      exited: 'gateway exited',
      /** {0} = exit code */
      exitedWithCode: (code: string) => `gateway exited (${code})`,
      /** {0} = child process error message */
      error: (message: string) => `gateway error: ${message}`,
      websocketUnavailable: 'gateway websocket unavailable',
      websocketConnectionFailed: 'gateway websocket connection failed',
      /** {0} = websocket close code */
      websocketClosedDuringConnect: (code: string) => `gateway websocket closed (${code}) during connect`,
      websocketClosed: 'gateway websocket closed',
      /** {0} = websocket close code */
      websocketClosedWithCode: (code: string) => `gateway websocket closed (${code})`,
      websocketStartupFailed: 'gateway websocket startup failed',
      attachUrlChanged: 'gateway attach url changed',
      closed: 'gateway closed'
    },

    // lib/terminalSetup.ts — /terminal-setup result messages. {0} = IDE label
    // (VS Code / Cursor / Windsurf) unless noted.
    terminalSetup: {
      mustRunLocally: (label: string) =>
        `${label} terminal setup must be run on the local machine, not inside an SSH session.`,
      settingsPathUnknown: (label: string) => `Could not determine ${label} settings path on this platform.`,
      /** {0} = IDE label, {1} = keybindings.json path */
      keybindingsNotArray: (label: string, file: string) => `${label} keybindings.json is not a JSON array: ${file}`,
      /** {0} = IDE label, {1} = error text */
      readFailed: (label: string, error: string) => `Failed to read ${label} keybindings: ${error}`,
      /** {0} = keybindings.json path, {1} = comma-joined conflicting key chords */
      conflicts: (file: string, keys: string) => `Existing terminal keybindings would conflict in ${file}: ${keys}`,
      alreadyConfigured: (label: string) => `${label} terminal keybindings already configured.`,
      /** {0} = count, {1} = IDE label */
      addedOne: (count: string, label: string) => `Added ${count} ${label} terminal keybinding`,
      /** {0} = count, {1} = IDE label */
      addedOther: (count: string, label: string) => `Added ${count} ${label} terminal keybindings`,
      /** {0} = count */
      migratedOne: (count: string) => `migrated ${count} legacy binding to CSI u encoding`,
      /** {0} = count */
      migratedOther: (count: string) => `migrated ${count} legacy bindings to CSI u encoding`,
      /** {0} = comma-joined summary parts, {1} = keybindings.json path */
      summaryIn: (parts: string, file: string) => `${parts} in ${file}`,
      /** {0} = IDE label, {1} = error text */
      configureFailed: (label: string, error: string) => `Failed to configure ${label} terminal shortcuts: ${error}`,
      noSupportedIde: 'No supported IDE terminal detected. Supported: VS Code, Cursor, Windsurf.'
    },

    // lib/terminalParity.ts — startup hints about the host terminal.
    terminalParity: {
      /** {0} = detected IDE id (vscode / cursor / windsurf) */
      ideSetup: (terminal: string) =>
        `Detected ${terminal} terminal · run /terminal-setup for best Cmd+Enter / undo parity`,
      appleTerminal:
        'Apple Terminal detected · use /paste for image-only clipboard fallback, and try Ctrl+A / Ctrl+E / Ctrl+U if Cmd+←/→/⌫ gets rewritten',
      tmux: 'tmux detected · clipboard copy/paste uses passthrough when available; allow-passthrough improves OSC52 reliability',
      remote:
        'SSH session detected · text clipboard can bridge via OSC52, but image clipboard and local screenshot paths still depend on the machine running Hermes'
    },

    // lib/billingDialog.ts — the out-of-credits confirm dialog.
    billingDialog: {
      dismiss: 'Dismiss',
      topUp: 'Top up',
      nousDetail: 'Your Nous credit balance is exhausted — top up to keep going.',
      nousTitle: 'Out of Nous credits',
      yourProvider: 'your provider',
      openBillingPage: 'Open billing page',
      switchProvider: 'Switch provider',
      /** {0} = provider label */
      providerDetail: (label: string) => `${label} reports your credits or billing are exhausted.`,
      /** {0} = provider label */
      providerTitle: (label: string) => `Out of credits · ${label}`
    },

    // lib/subagentTree.ts — the `d2 · 7 agents · 124 tools · 2m 14s` summary chip.
    subagentTree: {
      /** {0} = pre-formatted count */
      agentsOne: (count: string) => `${count} agent`,
      /** {0} = pre-formatted count */
      agentsOther: (count: string) => `${count} agents`,
      /** {0} = pre-formatted count */
      toolsOne: (count: string) => `${count} tool`,
      /** {0} = pre-formatted count */
      toolsOther: (count: string) => `${count} tools`,
      /** {0} = compact token count (`12k`) */
      tokens: (count: string) => `${count} tok`
    },

    // lib/text.ts — transcript trail / clarify prose.
    text: {
      showingLiveTail: 'showing live tail',
      /** {0} = label prefix, {1} = omitted line count, {2} = omitted char count (both pre-formatted) */
      omittedLinesChars: (prefix: string, lines: string, chars: string) =>
        `[${prefix}; omitted ${lines} lines / ${chars} chars]`,
      /** {0} = label prefix, {1} = omitted char count (pre-formatted) */
      omittedChars: (prefix: string, chars: string) => `[${prefix}; omitted ${chars} chars]`,
      argsLabel: 'Args',
      resultLabel: 'Result',
      errorLabel: 'Error',
      /** {0} = pre-formatted line count; sits inside a `[[ … ]]` composer token */
      pasteLinesChip: (count: string) => `[${count} lines]`,
      /** {0} = the question text */
      clarifyHead: (question: string) => `ask ${question}`,
      /** {0} = why the prompt ended ("timed out", "cancelled") */
      clarifyNoSelection: (reason: string) => `(${reason} — no selection)`,
      /** {0} = question count */
      clarifyBatchHead: (count: string) => `ask (${count} questions)`,
      /** {0} = question, {1} = locked answer */
      clarifyAnswered: (question: string, answer: string) => `✓ ${question} → ${answer}`,
      /** {0} = question */
      clarifyUnanswered: (question: string) => `· ${question} (no answer)`,
      /** {0} = why the batch ended */
      clarifyBatchReason: (reason: string) => `(${reason})`
    },

    // lib/agentRows.ts — docked agents panel row fallbacks.
    agentRows: {
      starting: 'Starting…',
      agent: 'agent',
      resultReady: 'result ready'
    },

    // domain/messages.ts — transcript replay projections.
    messages: {
      messageFallback: '(message)',
      /** {0} = the first few words of the message */
      longMessage: (prefix: string) => `${prefix} [long message]`,
      modelChanged: 'model changed',
      resumedInterruptedTurn: 'resumed interrupted turn',
      personalityChanged: 'personality changed',
      backgroundProcessFinished: 'background process finished',
      backgroundAgentWorkFinished: 'background agent work finished',
      /** {0} = count */
      backgroundAgentsFinishedOne: (count: string) => `${count} background agent finished`,
      /** {0} = count */
      backgroundAgentsFinishedOther: (count: string) => `${count} background agents finished`
    },

    // domain/attachments.ts — the composer token an attached image renders as.
    attachments: {
      /** {0} = 1-based image index; must stay inside `[[ … ]]` (PASTE_SNIPPET_RE) */
      imageToken: (index: string) => `[[ Image ${index} ]]`
    },

    // hooks/useCompletion.ts — the placeholder row when the completer RPC fails.
    completion: {
      unavailable: 'completion unavailable',
      unavailableMeta: 'unavailable'
    }
  }
}
