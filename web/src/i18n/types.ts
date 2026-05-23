export type Locale =
  | "en"
  | "zh"
  | "zh-hant"
  | "ja"
  | "de"
  | "es"
  | "fr"
  | "tr"
  | "uk"
  | "af"
  | "ko"
  | "it"
  | "ga"
  | "pt"
  | "ru"
  | "hu";

export interface Translations {
  // ── Common ──
  common: {
    save: string;
    saving: string;
    cancel: string;
    close: string;
    confirm: string;
    delete: string;
    refresh: string;
    retry: string;
    search: string;
    loading: string;
    create: string;
    creating: string;
    set: string;
    replace: string;
    clear: string;
    live: string;
    off: string;
    enabled: string;
    disabled: string;
    active: string;
    inactive: string;
    unknown: string;
    untitled: string;
    all?: string;
    none: string;
    form: string;
    noResults: string;
    of: string;
    page: string;
    msgs: string;
    tools: string;
    match: string;
    other: string;
    configured: string;
    removed: string;
    failedToToggle: string;
    failedToRemove: string;
    failedToReveal: string;
    collapse: string;
    expand: string;
    general: string;
    messaging: string;
    pluginLoadFailed: string;
    pluginNotRegistered: string;
    copied: string;
    installSuccess: string;
    installFailed: string;
    rescanFailed: string;
    saveFailed: string;
    removeSuccess: string;
  };

  // ── App shell ──
  app: {
    brand: string;
    brandShort: string;
    closeNavigation: string;
    closeModelTools: string;
    footer: {
      org: string;
    };
    activeSessionsLabel: string;
    gatewayStatusLabel: string;
    gatewayStrip: {
      failed: string;
      off: string;
      running: string;
      starting: string;
      stopped: string;
    };
    nav: {
      analytics: string;
      chat: string;
      config: string;
      cron: string;
      documentation: string;
      keys: string;
      logs: string;
      models: string;
      profiles: string;
      plugins: string;
      sessions: string;
      skills: string;
    };
    modelToolsSheetSubtitle: string;
    modelToolsSheetTitle: string;
    navigation: string;
    openDocumentation: string;
    openNavigation: string;
    pluginNavSection: string;
    sessionsActiveCount: string;
    statusOverview: string;
    system: string;
    webUi: string;
    pluginLabels?: Record<string, string>;
  };

  // ── Status page ──
  status: {
    actionFailed: string;
    actionFinished: string;
    actions: string;
    agent: string;
    connected: string;
    connectedPlatforms: string;
    disconnected: string;
    error: string;
    failed: string;
    gateway: string;
    gatewayFailedToStart: string;
    lastUpdate: string;
    noneRunning: string;
    notRunning: string;
    pid: string;
    platformDisconnected: string;
    platformError: string;
    activeSessions: string;
    recentSessions: string;
    restartGateway: string;
    restartingGateway: string;
    running: string;
    runningRemote: string;
    startFailed: string;
    starting: string;
    startedInBackground: string;
    stopped: string;
    updateHermes: string;
    updatingHermes: string;
    waitingForOutput: string;
  };

  // ── Sessions page ──
  sessions: {
    title: string;
    searchPlaceholder: string;
    noSessions: string;
    noMatch: string;
    startConversation: string;
    noMessages: string;
    untitledSession: string;
    deleteSession: string;
    confirmDeleteTitle: string;
    confirmDeleteMessage: string;
    sessionDeleted: string;
    failedToDelete: string;
    resumeInChat: string;
    previousPage: string;
    nextPage: string;
    sourceLocal: string;
    roles: {
      user: string;
      assistant: string;
      system: string;
      tool: string;
    };
  };

  // ── Analytics page ──
  analytics: {
    period: string;
    totalTokens: string;
    totalSessions: string;
    apiCalls: string;
    dailyTokenUsage: string;
    dailyBreakdown: string;
    perModelBreakdown: string;
    topSkills: string;
    skill: string;
    loads: string;
    edits: string;
    lastUsed: string;
    input: string;
    output: string;
    total: string;
    noUsageData: string;
    startSession: string;
    date: string;
    model: string;
    tokens: string;
    perDayAvg: string;
    acrossModels: string;
    inOut: string;
    tokenAnalyticsHidden: string;
    tokenAnalyticsDesc1: string;
    tokenAnalyticsDesc2: string;
    tokenAnalyticsDesc3: string;
    tokenAnalyticsConfigure: string;
  };

  // ── Models page ──
  models: {
    modelsUsed: string;
    estimatedCost: string;
    tokens: string;
    sessions: string;
    avgPerSession: string;
    apiCalls: string;
    toolCalls: string;
    noModelsData: string;
    startSession: string;
    modelSettings: string;
    appliesToNewSessions: string;
    mainModel: string;
    unset: string;
    change: string;
    auxiliaryTasks: string;
    overrideCount: string;
    allAuto: string;
    configure: string;
    setMainModel: string;
    tokenBarCacheRead: string;
    tokenBarReasoning: string;
    tokenBarInput: string;
    tokenBarOutput: string;
    badgeTools: string;
    badgeVision: string;
    badgeReasoning: string;
    useAs: string;
    current: string;
    auxiliaryTask: string;
    allAuxiliaryTasks: string;
    mainBadge: string;
    auxBadge: string;
    ctxLabel: string;
    outLabel: string;
    auxTasksModalTitle: string;
    resetAllToAuto: string;
    auxTasksDesc: string;
    autoUseMain: string;
    providerDefault: string;
    setAuxiliary: string;
    resetAuxTitle: string;
    resetAuxDesc: string;
    resetAll: string;
    auxTasksSummary: string;
    sessionTokenUnavailable: string;
  };

  // ── Logs page ──
  logs: {
    title: string;
    autoRefresh: string;
    file: string;
    level: string;
    component: string;
    lines: string;
    noLogLines: string;
    fileAgent?: string;
    fileErrors?: string;
    fileGateway?: string;
    levelDebug?: string;
    levelInfo?: string;
    levelWarning?: string;
    levelError?: string;
    compGateway?: string;
    compAgent?: string;
    compTools?: string;
    compCli?: string;
    compCron?: string;
  };

  // ── Cron page ──
  cron: {
    confirmDeleteMessage: string;
    confirmDeleteTitle: string;
    newJob: string;
    nameOptional: string;
    namePlaceholder: string;
    prompt: string;
    promptPlaceholder: string;
    schedule: string;
    schedulePlaceholder: string;
    deliverTo: string;
    scheduledJobs: string;
    noJobs: string;
    last: string;
    next: string;
    pause: string;
    resume: string;
    triggerNow: string;
    delivery: {
      local: string;
      telegram: string;
      discord: string;
      slack: string;
      email: string;
    };
    profile?: string;
    allProfiles?: string;
    jobStateIdle?: string;
    jobStateRunning?: string;
    jobStatePaused?: string;
    jobStateError?: string;
    jobStateScheduled?: string;
    jobStateDisabled?: string;
  };

  // ── Plugins page ──
  pluginsPage: {
    contextEngineLabel: string;
    dashboardSlots: string;
    disableRuntime: string;
    enableAfterInstall: string;
    enableRuntime: string;
    forceReinstall: string;
    headline: string;
    identifierLabel: string;
    inactive: string;
    installBtn: string;
    installHeading: string;
    installHint: string;
    memoryProviderLabel: string;
    missingEnvWarn: string;
    noDashboardTab: string;
    openTab: string;
    orphanHeading: string;
    pluginListHeading: string;
    providerDefaults: string;
    providersHeading: string;
    providersHint: string;
    refreshDashboard: string;
    removeConfirm: string;
    removeHint: string;
    removeConfirmDesc: string;
    rescanHeading: string;
    rescanHint: string;
    runtimeHeading: string;
    saveProviders: string;
    savedProviders: string;
    sourceBadge: string;
    authRequired: string;
    authRequiredHint: string;
    updateGit: string;
    versionBadge: string;
    showInSidebar: string;
    hideFromSidebar: string;
    placeholderOwnerRepo: string;
    orphanSeparator: string;
    compressorLabel: string;
    installSuccess: string;
    installFailed: string;
    rescanFailed: string;
    saveFailed: string;
    failed: string;
    removeSuccess: string;
  };

  // ── Profiles page ──
  profiles: {
    newProfile: string;
    name: string;
    namePlaceholder: string;
    nameRequired: string;
    nameRule: string;
    invalidName: string;
    cloneFromDefault: string;
    allProfiles: string;
    noProfiles: string;
    defaultBadge: string;
    hasEnv: string;
    model: string;
    skills: string;
    rename: string;
    editSoul: string;
    soulSection: string;
    soulPlaceholder: string;
    saveSoul: string;
    soulSaved: string;
    openInTerminal: string;
    commandCopied: string;
    copyFailed: string;
    confirmDeleteTitle: string;
    confirmDeleteMessage: string;
    created: string;
    deleted: string;
    renamed: string;
  };

  // ── Skills page ──
  skills: {
    title: string;
    searchPlaceholder: string;
    enabledOf: string;
    all: string;
    categories: string;
    filters: string;
    noSkills: string;
    noSkillsMatch: string;
    skillCount: string;
    resultCount: string;
    noDescription: string;
    toolsets: string;
    toolsetLabel: string;
    noToolsetsMatch: string;
    setupNeeded: string;
    disabledForCli: string;
    more: string;
    categoryMlops: string;
    categoryMlopsCloud: string;
    categoryMlopsEval: string;
    categoryMlopsInference: string;
    categoryMlopsModels: string;
    categoryMlopsTraining: string;
    categoryMlopsVdb: string;
    categoryMcp: string;
    categoryRedTeaming: string;
    categoryOcr: string;
    categoryP5js: string;
    categoryAi: string;
    categoryUx: string;
    categoryUi: string;
  };

  // ── Config page ──
  config: {
    configPath: string;
    filters: string;
    sections: string;
    exportConfig: string;
    importConfig: string;
    resetDefaults: string;
    resetScopeTooltip: string;
    confirmResetScope: string;
    resetScopeToast: string;
    rawYaml: string;
    rawYamlTab: string;
    searchResults: string;
    fields: string;
    noFieldsMatch: string;
    configSaved: string;
    yamlConfigSaved: string;
    failedToSave: string;
    failedToSaveYaml: string;
    failedToLoadRaw: string;
    configImported: string;
    invalidJson: string;
    confirmResetDescription: string;
    themes?: {
      labels: Record<string, string>;
      descriptions: Record<string, string>;
    };
    categories: {
      general: string;
      agent: string;
      terminal: string;
      display: string;
      delegation: string;
      memory: string;
      compression: string;
      security: string;
      browser: string;
      voice: string;
      tts: string;
      stt: string;
      logging: string;
      discord: string;
      auxiliary: string;
    };
  };

  // ── Env / Keys page ──
  env: {
    changesNote: string;
    confirmClearMessage: string;
    confirmClearTitle: string;
    description: string;
    enterValue: string;
    getKey: string;
    hideAdvanced: string;
    hideValue: string;
    keysCount: string;
    llmProviders: string;
    notConfigured: string;
    notSet: string;
    providersConfigured: string;
    replaceCurrentValue: string;
    showAdvanced: string;
    showValue: string;
    providerGroupNous: string;
    providerGroupAnthropic: string;
    providerGroupDashscope: string;
    providerGroupDeepseek: string;
    providerGroupGemini: string;
    providerGroupZai: string;
    providerGroupHuggingface: string;
    providerGroupKimi: string;
    providerGroupMinimaxCn: string;
    providerGroupMinimax: string;
    providerGroupOpenCodeGo: string;
    providerGroupOpenCodeZen: string;
    providerGroupOpenrouter: string;
    providerGroupXiaomi: string;
    navOauth: string;
    navProviders: string;
    navTools: string;
    navMessaging: string;
    navSettings: string;
    navAria: string;
    ariaReveal: string;
    ariaHide: string;
  };

  // ── OAuth ──
  oauth: {
    title: string;
    providerLogins: string;
    description: string;
    connected: string;
    expired: string;
    notConnected: string;
    runInTerminal: string;
    noProviders: string;
    login: string;
    disconnect: string;
    managedExternally: string;
    copied: string;
    cli: string;
    copyCliCommand: string;
    connect: string;
    sessionExpires: string;
    initiatingLogin: string;
    exchangingCode: string;
    connectedClosing: string;
    loginFailed: string;
    sessionExpired: string;
    reOpenAuth: string;
    reOpenVerification: string;
    submitCode: string;
    pasteCode: string;
    waitingAuth: string;
    enterCodePrompt: string;
    pkceStep1: string;
    pkceStep2: string;
    pkceStep3: string;
    flowLabels: {
      pkce: string;
      device_code: string;
      external: string;
    };
    expiresIn: string;
  };

  // ── Language switcher ──
  language: {
    switchTo: string;
  };

  // ── Theme switcher ──
  theme: {
    title: string;
    switchTheme: string;
  };

  // ── Chat page ──
  chat: {
    authFailed: string;
    localhostOnly: string;
    sessionEnded: string;
    copyTooltip: string;
    copyAria: string;
    copyText: string;
    copiedText: string;
  };

  // ── Chat page ──
  achievements: {
    hero: {
      kicker: string;
      title: string;
      subtitle: string;
      scan_subtitle: string;
    };
    actions: {
      rescan: string;
    };
    stats: {
      unlocked: string;
      unlocked_hint: string;
      discovered: string;
      discovered_hint: string;
      secrets: string;
      secrets_hint: string;
      highest_tier: string;
      highest_tier_hint: string;
      latest: string;
      latest_hint_empty: string;
      none_yet: string;
    };
    state: {
      unlocked: string;
      discovered: string;
      secret: string;
    };
    tier: {
      target: string;
      hidden: string;
      complete: string;
      objective: string;
    };
    progress: {
      hidden: string;
    };
    scan: {
      building_headline: string;
      building_detail: string;
      starting_headline: string;
      progress_detail: string;
      idle_detail: string;
    };
    guide: {
      tiers_header: string;
      secret_header: string;
      secret_body: string;
      scan_status_header: string;
      scan_status_body: string;
      what_scanned_header: string;
      what_scanned_body: string;
    };
    card: {
      share_title: string;
      share_label: string;
      share_text: string;
      how_to_reveal: string;
      what_counts: string;
      evidence_label: string;
      evidence_session_fallback: string;
      no_evidence: string;
    };
    latest: {
      header: string;
    };
    empty: {
      no_secrets_header: string;
      no_secrets_body: string;
      check_guide: string;
    };
    categories?: Record<string, string>;
    filters?: {
      all_categories?: string;
      visibility_all?: string;
      visibility_unlocked?: string;
      visibility_discovered?: string;
      visibility_secret?: string;
    };
    achievementData?: Record<string, { name?: string; description: string }>;
  };

  // ── Kanban (plugin) ──
  kanban?: {
    columnLabels: Record<string, string>;
    columnHelp: Record<string, string>;
    confirmDone: string;
    confirmArchive: string;
    confirmBlocked: string;
    completionBlockedHallucination: string;
    suspectedHallucinatedReferences: string;
    trash: { label: string; title: string; confirm: string; confirmMany: string; dropHint: string; };
    completionSummary: string;
    completionSummaryRequired: string;
    renderingError: string;
    reloadView: string;
    wsAuthFailed: string;
    moveFailed: string;
    taskCreatedWarning: string;
    bulkFailed: string;
    loading: string;
    loadFailed: string;
    loadFailedHint: string;
    taskNeedsAttention: string;
    tasksNeedAttention: string;
    hide: string;
    show: string;
    untitled: string;
    unassigned: string;
    diagnostic: string;
    open: string;
    copyCommand: string;
    copied: string;
    unblockedMessage: string;
    unblockFailed: string;
    reclaimedMessage: string;
    reclaimFailed: string;
    pickProfileFirst: string;
    reassignedMessage: string;
    reassignFailed: string;
    reassignTo: string;
    diagnostics: string;
    boardSwitcherHint: string;
    newBoard: string;
    board: string;
    archiveBoardConfirm: string;
    archiveBoardTitle: string;
    archive: string;
    newBoardTitle: string;
    newBoardDescription: string;
    slug: string;
    slugHint: string;
    displayName: string;
    filterCards: string;
    tenant: string;
    assignee: string;
    allTenants: string;
    allProfiles: string;
    showArchived: string;
    lanesByProfile: string;
    nudgeDispatcher: string;
    refresh: string;
    clearFilters: string;
    selectAllTasks: string;
    tasksInColumn: string;
    noTasks: string;
    needsAssignee: string;
    needsAssigneeHint: string;
    clickToEditAssignee: string;
    emptyAssignee: string;
  };
};