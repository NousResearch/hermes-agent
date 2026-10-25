// slashCmd.ops / slashCmd.wake — replies and usage hints of app/slash/commands/{ops,wake}.ts.
// Command NAMES/aliases/arg syntax inside usage strings stay literal.

export const slashCmdOpsEn = {
  ops: {
    stop: {
      stoppedOne: (count: string) => `stopped ${count} background process`,
      stoppedOther: (count: string) => `stopped ${count} background processes`
    },
    reloadMcp: {
      confirmRequired: '/reload-mcp requires confirmation',
      reloaded: 'MCP servers reloaded',
      reloadedAlways: 'MCP servers reloaded · future /reload-mcp will run without confirmation',
      complete: 'reload complete'
    },
    reloadEnv: {
      reloadedOne: (count: string) => `reloaded .env (${count} var updated)`,
      reloadedOther: (count: string) => `reloaded .env (${count} vars updated)`
    },
    browser: {
      usage: 'usage: /browser [connect|disconnect|status|use] [url] · persistent: set browser.cdp_url in config.yaml',
      checking: (url: string) => `checking Chromium-family browser remote debugging at ${url}...`,
      connected: (url: string) => `browser connected: ${url}`,
      urlUnavailable: '(url unavailable)',
      notConnected: 'browser not connected (try /browser connect <url> or set browser.cdp_url in config.yaml)',
      disconnected: 'browser disconnected',
      connectedLive: 'Browser connected to live Chromium-family browser via CDP',
      endpoint: (url: string) => `Endpoint: ${url}`,
      nextCallUsesEndpoint: 'next browser tool call will use this CDP endpoint',
      useUsage: 'Usage: /browser use [off]',
      useEnabled: 'Browser Use mode enabled — browser_exec via the Browser Use CLI 3.0',
      useDisabled: 'Browser Use mode disabled — built-in browser tools restored',
      newSessionsOnly: 'applies to new sessions — this one keeps its current tools (/new to start one)',
      modeBrowserUse: 'Browser: Browser Use mode (browser_exec via the Browser Use CLI 3.0)'
    },
    rollback: {
      noSession: 'no active session — nothing to rollback',
      notEnabled: 'checkpoints are not enabled',
      noneFound: 'no checkpoints found',
      listTitle: 'Rollback checkpoints',
      noMetadata: '(no metadata)',
      usageDiff: 'usage: /rollback diff <checkpoint>',
      noChanges: 'no changes since this checkpoint',
      diffTitle: 'Rollback diff',
      failed: (error: string) => `rollback failed: ${error}`,
      unknownError: 'unknown error',
      workspaceTarget: 'workspace',
      restoredDetail: 'restored',
      // {0}=target (file path or "workspace"), {1}=detail
      restored: (target: string, detail: string) => `rollback restored ${target}: ${detail}`
    },
    agents: {
      paused: 'delegation · paused',
      resumed: 'delegation · resumed',
      statePaused: 'paused',
      stateActive: 'active',
      // {0}=state (paused/active), {1}=max spawn depth, {2}=max concurrent children
      status: (state: string, depth: string, children: string) => `delegation · ${state} · caps d${depth}/${children}`
    },
    replay: {
      noneOnDisk: 'no archived spawn trees on disk for this session',
      subagentsLabel: (count: string) => `${count} subagents`,
      archivedTitle: 'Archived spawn trees',
      usageLoad: 'usage: /replay load <path>',
      snapshotEmpty: 'snapshot empty or unreadable',
      noneThisSession: 'no completed spawn trees this session · try /replay list',
      indexOutOfRange: (max: string) => `replay: index out of range 1..${max} · use /replay list for disk`
    },
    replayDiff: {
      usage: 'usage: /replay-diff <a> <b>  (e.g. /replay-diff 1 2 for last two)',
      unresolved: (count: string) => `replay-diff: could not resolve indices · history has ${count} entries`
    },
    reloadSkills: {
      reloaded: 'skills reloaded',
      pageTitle: 'Reload Skills'
    },
    slashWorker: {
      warning: (warning: string) => `warning: ${warning}`,
      skillsNoOutput: '/skills: no output',
      skillsTitle: 'Skills',
      pluginsNoOutput: '/plugins: no output',
      pluginsTitle: 'Plugins',
      toolsNoOutput: '/tools: no output',
      toolsTitle: 'Tools'
    },
    skills: {
      listTitle: 'Skills',
      usageInspect: 'usage: /skills inspect <name>',
      unknownSkill: (name: string) => `unknown skill: ${name}`,
      rowName: 'Name',
      rowCategory: 'Category',
      rowPath: 'Path',
      inspectTitle: 'Skill',
      usageSearch: 'usage: /skills search <query>',
      noResults: (query: string) => `no results for: ${query}`,
      searchTitle: (query: string) => `Search: ${query}`,
      usageInstall: 'usage: /skills install <name or url>',
      installing: (name: string) => `installing ${name}…`,
      installed: (name: string) => `installed ${name}`,
      installFailed: 'install failed',
      usageBrowse: 'usage: /skills browse [page]  (page must be a positive number)',
      fetching: 'fetching community skills (scans 6 sources, may take ~15s)…',
      noneOnPage: (page: string) => `no skills on page ${page}`,
      // {0}=page, {1}=total skills
      noneOnPageWithTotal: (page: string, total: string) => `no skills on page ${page} (total ${total})`,
      // {0}=page, {1}=total pages
      pageOf: (page: string, total: string) => `page ${page} of ${total}`,
      skillsTotal: (count: string) => `${count} skills total`,
      browseMore: (nextPage: string) => `/skills browse ${nextPage} for more`,
      browseTitle: 'Browse Skills',
      browseTitlePage: (page: string) => `Browse Skills — p${page}`
    },
    tools: {
      usage: (subcommand: string) => `usage: /tools ${subcommand} <name> [name ...]`,
      builtinExample: (subcommand: string) => `built-in toolset: /tools ${subcommand} web`,
      mcpExample: (subcommand: string) => `MCP tool: /tools ${subcommand} github:create_issue`,
      disabled: (names: string) => `disabled: ${names}`,
      enabled: (names: string) => `enabled: ${names}`,
      unknownToolsets: (names: string) => `unknown toolsets: ${names}`,
      missingServers: (names: string) => `missing MCP servers: ${names}`,
      sessionReset: 'session reset. new tool configuration is active.'
    }
  },
  wake: {
    usage: 'usage: /wake [on|off|status]',
    reason: {
      disabled: 'disabled (config wake_word.enabled)',
      disabledForSurface: 'scoped to another surface (config wake_word.surface)',
      notOwner: 'another surface owns the listener',
      owned: 'another surface owns the listener',
      unavailable: 'unavailable',
      unknown: 'unknown'
    },
    ownedBy: (surface: string) => `owned by ${surface}`,
    // {0}=text, {1}=hint from the gateway
    withHint: (text: string, hint: string) => `${text} — ${hint}`,
    notStarted: (reason: string) => `wake: not started — ${reason}`,
    forPhrase: (phrase: string) => `for “${phrase}”`,
    // {0}=detail suffix (" for “phrase” · provider · enabled in config" or empty)
    listening: (details: string) => `wake: listening${details}`,
    micSilent: '⚠ mic delivers only silence',
    // {0}=owner surface, {1}=detail suffix
    offOwnedBy: (surface: string, details: string) => `wake: off here · listener owned by ${surface}${details}`,
    unavailable: 'wake: unavailable',
    off: (details: string) => `wake: off${details} · /wake on to arm`,
    enabledInConfig: 'enabled in config',
    disabledInConfig: 'disabled in config',
    listenerOff: (details: string) => `wake: listener off${details}`,
    notOwnerStop: 'this surface doesn’t own the listener',
    notRunning: 'not running',
    // {0}=reason, {1}=detail suffix
    nothingToStop: (reason: string, details: string) => `wake: nothing to stop — ${reason}${details}`
  }
}
