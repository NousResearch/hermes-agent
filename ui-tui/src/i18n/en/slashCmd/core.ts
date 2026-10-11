// slashCmd.core / slashCmd.debug / slashCmd.setup — replies and usage hints of
// app/slash/commands/{core,debug,setup}.ts. Command NAMES/aliases/arg syntax stay literal.
// Grouped one level deeper by command so `slashCmd.core.<command>.<leaf>`.

export const slashCmdCoreEn = {
  core: {
    help: {
      skillCommandsAvailable: (count: string) => `${count} skill commands available — /skills to browse`,
      detailsGlobal: 'set global agent detail visibility mode',
      detailsSection: 'override one section (thinking/tools/subagents/activity)',
      fortune: 'show a random or daily local fortune',
      gridTest: 'open the interactive widget-grid demo',
      dialogTest: 'open a sample dialog overlay with a faked backdrop',
      tuiSection: 'TUI'
    },
    quit: {
      dashboardDisabled: 'exit is disabled in hosted dashboard chat — use /new to start a fresh session'
    },
    update: {
      dashboardDisabled: 'update is disabled in hosted dashboard chat — the hosted environment is managed separately',
      exiting: 'exiting TUI to run update...'
    },
    mouse: {
      usage: 'usage: /mouse [on|off|toggle|wheel|buttons|all]',
      tracking: (mode: string) => `mouse tracking ${mode}`
    },
    clear: {
      switchSessions: 'switch sessions',
      forgingSession: 'forging session…',
      newSessionStarted: 'new session started',
      cancelLabel: 'No, keep going',
      confirmNew: 'Yes, start a new session',
      confirmClear: 'Yes, clear the session',
      detail: 'This ends the current conversation and clears the transcript.',
      titleNew: 'Start a new session?',
      titleClear: 'Clear the current session?'
    },
    redraw: {
      done: 'ui redrawn'
    },
    status: {
      noActiveSession: 'no active session',
      empty: '(no status)',
      pageTitle: 'Status'
    },
    title: {
      noActiveSession: 'no active session',
      current: (title: string) => `title: ${title}`,
      none: 'no title set',
      usage: 'usage: /title <your session title>',
      queuedSuffix: ' (queued while session initializes)',
      // {0}=title, {1}=suffix ('' or queuedSuffix)
      set: (title: string, suffix: string) => `session title set: ${title}${suffix}`
    },
    density: {
      usage: 'usage: /density [on|off|toggle]',
      state: (mode: string) => `density ${mode}`
    },
    details: {
      usage:
        'usage: /details [hidden|collapsed|expanded|cycle]  or  /details <section> [hidden|collapsed|expanded|reset]',
      sectionUsage: 'usage: /details <section> [hidden|collapsed|expanded|reset]',
      // {0}=mode, {1}=overrides summary ('' or '  (a=b c=d)')
      current: (mode: string, overrides: string) => `details: ${mode}${overrides}`,
      // {0}=section, {1}=mode or 'reset'
      section: (section: string, mode: string) => `details ${section}: ${mode}`,
      reset: 'reset'
    },
    fortune: {
      usage: 'usage: /fortune [random|daily]'
    },
    copy: {
      copiedCharsOne: (count: string) => `copied ${count} character`,
      copiedCharsOther: (count: string) => `copied ${count} characters`,
      clipboardFailed: 'clipboard copy failed — try HERMES_TUI_FORCE_OSC52=1 to force the escape sequence',
      usage: 'usage: /copy [number] | /copy code [n] | /copy cmd [n]',
      nothingToCopy: 'nothing to copy — start a conversation first',
      sentOsc52: 'sent OSC52 copy sequence (terminal support required)',
      copied: 'copied to clipboard',
      failed: (error: string) => `copy failed: ${error}`,
      noCode: 'no complete code blocks in assistant responses yet',
      noCmd: 'no shell commands have run yet',
      invalidItem: (max: string) => `invalid item number — use 1-${max}`,
      pickCode: (count: string) => `${count} code blocks in the latest response:`,
      pickCmd: (count: string) => `${count} commands in the latest turn:`,
      pickHint: (scope: string, max: string) => `run /copy ${scope} <1-${max}> to copy one`,
      copiedItem: (index: string, label: string) => `copied #${index} (${label})`
    },
    paste: {
      usage: 'usage: /paste'
    },
    prompt: {
      editorFailed: (error: string) => `editor failed: ${error}`
    },
    terminalSetup: {
      usage: 'usage: /terminal-setup [auto|vscode|cursor|windsurf]',
      restartIde: 'restart the IDE terminal for the new keybindings to take effect',
      failed: (error: string) => `terminal setup failed: ${error}`
    },
    logs: {
      pageTitle: 'Logs',
      none: 'no gateway logs'
    },
    history: {
      noConversation: 'no conversation yet',
      youTag: (index: string) => `You #${index}`,
      hermesTag: (index: string) => `Hermes #${index}`,
      toolCallsOne: (count: string) => `(${count} tool call)`,
      toolCallsOther: (count: string) => `(${count} tool calls)`,
      empty: '(empty)',
      pageTitle: 'History'
    },
    save: {
      noConversation: 'no conversation yet',
      noActiveSession: 'no active session — nothing to save',
      saved: (file: string) => `conversation saved to: ${file}`,
      failed: 'failed to save'
    },
    focus: {
      statusOn: 'focus view on — only your prompt and the final response',
      statusOff: 'focus view off',
      usage: 'usage: /focus [on|off|status]',
      enabled: 'focus view enabled — just your prompt and the final response',
      disabled: 'focus view disabled'
    },
    statusbar: {
      usage: 'usage: /statusbar [on|off|top|bottom|toggle]',
      state: (mode: string) => `status bar ${mode}`
    },
    battery: {
      // {0}=on|off, {1}=plug icon, {2}=percent
      statusLive: (state: string, icon: string, percent: string) =>
        `battery indicator ${state} — currently ${icon} ${percent}%`,
      statusNoBattery: (state: string) => `battery indicator ${state} — no battery detected on this machine`,
      state: (state: string) => `battery indicator ${state}`,
      usage: 'usage: /battery [on|off|status]'
    },
    queue: {
      countOne: (count: string) => `${count} queued message`,
      countOther: (count: string) => `${count} queued messages`,
      queued: (preview: string) => `queued: "${preview}"`
    },
    steer: {
      usage: 'usage: /steer <prompt>',
      noActiveTurnQueued: (preview: string) => `no active turn — queued for next: "${preview}"`,
      queued: (preview: string) => `steer queued — arrives after next tool call: "${preview}"`,
      rejected: 'steer rejected — no active turn, queued for next turn'
    },
    undo: {
      nothing: 'nothing to undo',
      undidOne: (count: string) => `undid ${count} message`,
      undidOther: (count: string) => `undid ${count} messages`
    },
    retry: {
      nothing: 'nothing to retry'
    }
  },
  debug: {
    widgetsReload: {
      loaded: (names: string) => `loaded: ${names}`,
      none: 'no user widgets found',
      summary: (parts: string) => `widgets — ${parts}`
    },
    heapdump: {
      // {0}=heap size, {1}=rss size (pre-formatted)
      writing: (heap: string, rss: string) => `writing heap dump (heap ${heap} · rss ${rss})…`,
      failed: (error: string) => `heapdump failed: ${error}`,
      unknownError: 'unknown error',
      heapPath: (path: string) => `heapdump: ${path}`,
      diagPath: (path: string) => `diagnostics: ${path}`
    },
    themeInfo: {
      panelTitle: 'Theme',
      osc11Background: 'OSC-11 background',
      noReply: '(no reply)',
      unset: '(unset)',
      detectedMode: 'detected mode',
      light: 'light',
      dark: 'dark'
    },
    mem: {
      panelTitle: 'Memory',
      heapUsed: 'heap used',
      heapTotal: 'heap total',
      external: 'external',
      arrayBuffers: 'array buffers',
      rss: 'rss',
      uptime: 'uptime',
      seconds: (count: string) => `${count}s`
    }
  },
  setup: {
    setup: {
      done: 'setup complete — starting session…'
    }
  }
}
