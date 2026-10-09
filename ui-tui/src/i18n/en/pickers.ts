// Picker overlays: the /model provider→model→effort picker, the Ctrl+X live
// session switcher (orchestrator), and the /pet gallery picker. Owned
// namespace: `pickers`.
//
// Hotkey chords (`Enter`, `Esc`, `^g`, `↑/↓`…) are tokens and stay as-is inside
// the keyed hint sentences; model ids, provider slugs and RPC method names are
// passed in as arguments, never translated.

export const pickersEn = {
  pickers: {
    common: {
      // {0} = the human-readable error message (never a machine code).
      error: (message: string) => `error: ${message}`,
      // {0} = the RPC method name (e.g. `model.options`) — not translated.
      invalidResponse: (method: string) => `invalid response: ${method}`,
      moreAbove: (count: number) => ` ↑ ${count} more`,
      moreBelow: (count: number) => ` ↓ ${count} more`,
      filter: (query: string) => `filter: ${query}`,
      typeToFilter: 'type to filter'
    },
    model: {
      loading: 'loading models…',
      noProviders: 'no providers available',
      failedToSaveKey: 'failed to save key',
      // {0} = the provider's key env var name (e.g. OPENAI_API_KEY).
      pasteKeyToActivate: (keyEnv: string) => `paste ${keyEnv} to activate`,
      runHermesModelToConfigure: 'run `hermes model` to configure',
      escQCancelHint: 'Esc/q cancel',
      reasoning: {
        none: 'none (disable reasoning)',
        keepCurrent: 'Keep current effort'
      },
      key: {
        // {0} = provider display name.
        title: (providerName: string) => `Configure ${providerName}`,
        pasteBelow: 'Paste your API key below (saved to ~/.hermes/.env)',
        empty: '(empty)',
        saving: 'saving…',
        hint: 'Enter save · Ctrl+U clear · Esc back'
      },
      disconnect: {
        // {0} = provider display name.
        title: (providerName: string) => `Disconnect ${providerName}?`,
        // {0} = provider display name.
        removesCredentials: (providerName: string) => `This removes saved credentials for ${providerName}.`,
        reauthLater: 'You can re-authenticate later by selecting it again.',
        disconnecting: 'disconnecting…',
        hint: 'y/Enter confirm · n/Esc cancel'
      },
      provider: {
        title: 'Select provider (step 1/3)',
        subtitle: 'Full model IDs on the next step · Enter to continue',
        // {0} = current model id (or the `unknown` leaf below).
        current: (model: string) => `Current: ${model}`,
        unknown: '(unknown)',
        noKey: '(no key)',
        needsSetup: '(needs setup)',
        modelCountOne: (count: number) => `${count} model`,
        modelCountOther: (count: number) => `${count} models`,
        filterHint: 'type to filter · ↑/↓ select',
        // {0} = provider warning text from the backend.
        warning: (warning: string) => `warning: ${warning}`,
        noMatches: 'no providers match',
        hint: '↑/↓ select · Enter choose · ^d disconnect · Esc clear/back · q close'
      },
      persist: {
        // {0} = one of `scopeGlobal` / `scopeSession`.
        label: (scope: string) => `persist: ${scope}`,
        scopeGlobal: 'global',
        scopeSession: 'session',
        toggleHint: ' · ^g toggle',
        onlySuffix: ' only'
      },
      effort: {
        title: 'Reasoning effort (step 3/3)',
        // {0} = the pending model id.
        subtitle: (model: string) => `${model} · applies with the switch (same scope) · Esc back`,
        hint: '↑/↓ select · Enter switch · Esc back · q close'
      },
      modelStage: {
        title: 'Select model (step 2/3)',
        // {0} = provider display name (or the `unknownProvider` leaf below).
        subtitle: (providerName: string) => `${providerName} · Esc back`,
        unknownProvider: '(unknown provider)',
        noMatches: 'no models match filter',
        noneListed: 'no models listed for this provider',
        hint: '↑/↓ select · Enter next · Esc clear/back · q close',
        emptyHint: 'Esc back · q close'
      }
    },
    session: {
      title: 'Sessions',
      loading: 'loading sessions…',
      modelUnknown: 'model?',
      currentOrDefault: 'current/default',
      liveSessionsOne: (count: number) => `${count} live session`,
      liveSessionsOther: (count: number) => `${count} live sessions`,
      // {0} = live session count, {1} = resumable session count.
      counts: (liveCount: number, resumableCount: number) => `${liveCount} live · ${resumableCount} resumable`,
      age: {
        today: 'today',
        yesterday: 'yesterday',
        daysAgo: (days: number) => `${days}d ago`
      },
      status: {
        idle: 'idle',
        starting: 'starting',
        waiting: 'waiting',
        working: 'working'
      },
      errors: {
        couldNotLoadResumable: 'could not load resumable sessions',
        alreadyClosed: 'session was already closed'
      },
      row: {
        new: 'new',
        draft: '✎ draft',
        startNew: 'Start a new live session',
        current: 'current',
        untitled: '(untitled)',
        closing: 'closing…',
        deleting: 'deleting…',
        pressDAgain: 'press d again to delete',
        messageCount: (count: number) => `${count} msgs`
      },
      noOtherSessions: 'no other sessions — Enter on +new to start one',
      promptLabel: 'prompt › ',
      // {0} = the draft model label (short model id or `currentOrDefault`).
      draftModel: (label: string) => `model: ${label}`,
      selectNewPrefix: 'Select ',
      newRowMarker: '+new',
      selectNewSuffix: ' to type a prompt',
      // Hint fragments are concatenated with hotkey tokens in code, so the
      // leading/trailing spaces and ` · ` separators are part of each leaf.
      hint: {
        resumableLabel: 'Resumable:',
        resume: ' resume · ',
        delete: ' delete',
        newRowLabel: 'New row:',
        typePrompt: ' type prompt · ',
        start: ' start · ',
        model: ' model',
        sessionRowLabel: 'Session row:',
        switch: ' switch · ',
        close: ' close',
        move: ' move · ',
        new: ' new · ',
        refresh: ' refresh · ',
        searchHint: ' · / search · /search <text> full-text search'
      }
    },
    pet: {
      title: 'Pets',
      loading: 'loading pets…',
      adopting: 'adopting…',
      petCountOne: (count: number) => `${count} pet`,
      petCountOther: (count: number) => `${count} pets`,
      // {0} = the filter text the user typed.
      noMatches: (query: string) => `no pets match "${query}"`,
      noneAvailable: 'no pets available',
      officialTag: ' · official',
      escCancelHint: 'Esc cancel',
      hint: '↑/↓ select · Enter adopt · type to filter · Esc cancel'
    }
  }
}
