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
    // Optional: non-English locales fall back to the English literal in the
    // component until translated, matching the enriched-profiles keys.
    gateway?: string;
    gatewayHint?: string;
    pluginLoadFailed: string;
    pluginNotRegistered: string;
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
    /** Optional — fall back to English literals until translated. */
    managingProfile?: string;
    currentProfileOption?: string;
    managingProfileBanner?: string;
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
    history: string;
    overview: string;
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
    deleteEmpty: string;
    deleteEmptyConfirmTitle: string;
    deleteEmptyConfirmMessage: string;
    emptySessionsDeleted: string;
    failedToDeleteEmpty: string;
    selectSession: string;
    selectAllOnPage: string;
    clearSelection: string;
    selectedCount: string;
    deleteSelected: string;
    deleteSelectedConfirmTitle: string;
    deleteSelectedConfirmMessage: string;
    selectedSessionsDeleted: string;
    failedToDeleteSelected: string;
    resumeInChat: string;
    newChat: string;
    previousPage: string;
    nextPage: string;
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
    scheduleMode: string;
    scheduleModes: {
      interval: string;
      daily: string;
      weekly: string;
      monthly: string;
      once: string;
      custom: string;
      intervalEvery: string;
      intervalUnit: string;
      unitMinutes: string;
      unitHours: string;
      unitDays: string;
      timeOfDay: string;
      weekdays: string;
      weekdaysShort: [string, string, string, string, string, string, string];
      dayOfMonth: string;
      onceAt: string;
      customLabel: string;
      customPlaceholder: string;
      customHint: string;
      preview: string;
      previewEmpty: string;
    };
    scheduleDescribe: {
      none: string;
      everyMinutes: string;
      everyHours: string;
      everyDays: string;
      dailyAt: string;
      weeklyAt: string;
      monthlyAt: string;
      onceAt: string;
    };
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
      needsHomeChannel?: string;
      noneConfigured?: string;
    };
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
  };

  // ── Profiles page ──
  profiles: {
    newProfile: string;
    name: string;
    namePlaceholder: string;
    nameRequired: string;
    nameRule: string;
    invalidName: string;
    cloneFrom: string;
    cloneFromNone: string;
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
    // Optional keys added for the enriched profiles experience. Non-English
    // locales fall back to the English literal in the component until
    // translated, so these are optional to avoid churning every locale file.
    activeProfile?: string;
    activeBadge?: string;
    setActive?: string;
    activeSet?: string;
    gatewayRunning?: string;
    gatewayStopped?: string;
    gatewayRunningWarning?: string;
    aliasBadge?: string;
    description?: string;
    descriptionPlaceholder?: string;
    noDescription?: string;
    editDescription?: string;
    descriptionSaved?: string;
    reviewBadge?: string;
    autoGenerate?: string;
    generating?: string;
    describeFailed?: string;
    distribution?: string;
    advancedOptions?: string;
    cloneAll?: string;
    noSkillsOption?: string;
    descriptionOptional?: string;
    modelOptional?: string;
    modelInherit?: string;
    modelLoading?: string;
    modelNone?: string;
    editModel?: string;
    modelSaved?: string;
    modelSelect?: string;
    actions?: string;
    manageSkills?: string;
    activeSetHint?: string;
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
    /** Optional — fall back to English literals until translated. */
    profileSelector?: string;
    currentProfile?: string;
    managingProfile?: string;
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
    showLess: string;
    showMore: string;
    showValue: string;
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
    /** Font-override section (optional — locales fall back to English). */
    fontTitle?: string;
    fontDefault?: string;
    fontDefaultHint?: string;
    fontSans?: string;
    fontSerif?: string;
    fontMono?: string;
  };

  // ── Achievements plugin (plugins/hermes-achievements) ──
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
    };
    filters: {
      all_categories: string;
      visibility_all: string;
      visibility_unlocked: string;
      visibility_discovered: string;
      visibility_secret: string;
    };
    share: {
      dialog_label: string;
      header: string;
      close: string;
      rendering: string;
      card_alt: string;
      error_generic: string;
      x_title: string;
      x_button: string;
      copy_title: string;
      copy_button: string;
      copied: string;
      download_button: string;
      hint: string;
      clipboard_unsupported: string;
      tweet_text: string;
    };
  };

  // ── Kanban ──
  kanban: {
    loading: string;
    loadFailed: string;
    loadFailedHint: string;
    board: string;
    newBoard: string;
    newBoardTitle: string;
    newBoardDescription: string;
    slug: string;
    slugHint: string;
    displayName: string;
    displayNameHint: string;
    description: string;
    descriptionHint: string;
    icon: string;
    iconHint: string;
    switchAfterCreate: string;
    cancel: string;
    creating: string;
    createBoard: string;
    search: string;
    filterCards: string;
    tenant: string;
    allTenants: string;
    assignee: string;
    allProfiles: string;
    showArchived: string;
    lanesByProfile: string;
    nudgeDispatcher: string;
    refresh: string;
    selected: string;
    complete: string;
    archive: string;
    apply: string;
    clear: string;
    createTask: string;
    noTasks: string;
    unassigned: string;
    needsAssignee?: string;
    needsAssigneeHint?: string;
    untitled: string;
    loadingDetail: string;
    addComment: string;
    comment: string;
    status: string;
    workspace: string;
    skills: string;
    createdBy: string;
    result: string;
    comments: string;
    events: string;
    runHistory: string;
    workerLog: string;
    loadingLog: string;
    noWorkerLog: string;
    noDescription: string;
    noComments: string;
    edit: string;
    save: string;
    dependencies: string;
    parents: string;
    children: string;
    none: string;
    addParent: string;
    addChild: string;
    removeDependency: string;
    block: string;
    unblock: string;
    notifyHomeChannels: string;
    diagnostics: string;
    hide: string;
    show: string;
    attention: string;
    tasksNeedAttention: string;
    taskNeedsAttention: string;
    diagnostic: string;
    open: string;
    close: string;
    reassignTo: string;
    copied: string;
    copyCommand: string;
    reclaim: string;
    reassign: string;
    renderingError: string;
    reloadView: string;
    wsAuthFailed: string;
    markDone: string;
    markArchived: string;
    warning: string;
    phantomIds: string;
    active: string;
    ended: string;
    noProfile: string;
    showAllAttempts: string;
    sendingUpdates: string;
    sendNotifications: string;
    archiveBoardConfirm: string;
    archiveBoardTitle: string;
    boardSwitcherHint: string;
    taskCreatedWarning: string;
    moveFailed: string;
    bulkFailed: string;
    completionBlockedHallucination: string;
    suspectedHallucinatedReferences: string;
    pickProfileFirst: string;
    unblockedMessage: string;
    unblockFailed: string;
    reclaimedMessage: string;
    reclaimFailed: string;
    reassignedMessage: string;
    reassignFailed: string;
    selectForBulk: string;
    clickToEdit: string;
    clickToEditAssignee: string;
    emptyAssignee: string;
    columnLabels: {
      triage: string;
      todo: string;
      scheduled: string;
      ready: string;
      running: string;
      blocked: string;
      done: string;
      archived: string;
    };
    columnHelp: {
      triage: string;
      todo: string;
      scheduled: string;
      ready: string;
      running: string;
      blocked: string;
      done: string;
      archived: string;
    };
    confirmDone: string;
    confirmArchive: string;
    confirmBlocked: string;
    confirmScheduled?: string;
    completionSummary: string;
    completionSummaryRequired: string;
    triagePlaceholder: string;
    taskTitlePlaceholder: string;
    specifier: string;
    assigneePlaceholder: string;
    priority: string;
    skillsPlaceholder: string;
    noParent: string;
    workspacePathDir: string;
    workspacePathOptional: string;
    logTruncated: string;
    logAt: string;
  };

  // ── Channels (Messaging Platforms) ──
  channels: {
    addServer: string;
    configure: string;
    setupGuide: string;
    saveEnable: string;
    saving: string;
    cancel: string;
    restartGateway: string;
    restarting: string;
    restartNow: string;
    changesSavedBanner: string;
    gatewayNotRunning: string;
    configuredCount: string;
    nothingToSave: string;
    requiredField: string;
    fixFields: string;
    saved: string;
    testError: string;
    restartingGateway: string;
  };

  // ── System page ──
  system: {
    host: string;
    portal: string;
    curator: string;
    gateway: string;
    memory: string;
    credentialPool: string;
    shellHooks: string;
    checkpoints: string;
    debugShare: string;
    os: string;
    arch: string;
    hostname: string;
    python: string;
    hermes: string;
    cpu: string;
    cores: string;
    disk: string;
    uptime: string;
    loadAvg: string;
    psutilHint: string;
    checkForUpdates: string;
    updateNow: string;
    updateWith: string;
    latest: string;
    behind: string;
    updateAvailable: string;
    loggedIn: string;
    notLoggedIn: string;
    inferenceProvider: string;
    manageSubscription: string;
    toolGatewayRouting: string;
    loginWith: string;
    paused: string;
    active: string;
    disabled: string;
    every: string;
    lastRun: string;
    neverRun: string;
    resume: string;
    pause: string;
    runNow: string;
    running: string;
    stopped: string;
    start: string;
    restart: string;
    stop: string;
    externalProvider: string;
    builtinOnly: string;
    changeInPlugins: string;
    newCredentials: string;
    builtinFiles: string;
    resetMemory: string;
    resetUser: string;
    resetAll: string;
    provider: string;
    apiKey: string;
    label: string;
    addCredential: string;
    adding: string;
    noCredentials: string;
    newShellHook: string;
    event: string;
    command: string;
    matcherOptional: string;
    timeoutSec: string;
    approveNow: string;
    approveNowDesc: string;
    shellHookWarning: string;
    createHook: string;
    creating: string;
    pruneCheckpoints: string;
    pruneCheckpointsDesc: string;
    prune: string;
    debugShareDesc: string;
    share: string;
    sharing: string;
    redact: string;
    copyUrl: string;
    copied: string;
    removeCredential: string;
    removeCredentialDesc: string;
    remove: string;
    resetMemoryConfirm: string;
    resetMemoryDesc: string;
    removeHook: string;
    removeHookDesc: string;
  };

  // ── Profile Builder (step-by-step wizard) ──
  profileBuilder: {
    title: string;
    cancel: string;
    steps: {
      identity: string;
      model: string;
      skills: string;
      mcp: string;
      review: string;
    };
    identity: {
      nameLabel: string;
      namePlaceholder: string;
      nameRule: string;
      descLabel: string;
      descPlaceholder: string;
    };
    model: {
      hint: string;
      filterPlaceholder: string;
      loading: string;
      useDefault: string;
    };
    skills: {
      keepAllLabel: string;
      chooseHint: string;
      filterPlaceholder: string;
      loading: string;
      hubLabel: string;
      hubSearchPlaceholder: string;
      search: string;
      searching: string;
      add: string;
      remove: string;
    };
    mcp: {
      hint: string;
      namePlaceholder: string;
      urlPlaceholder: string;
      commandPlaceholder: string;
      argsPlaceholder: string;
      addServer: string;
      remove: string;
      nameRequired: string;
      urlOrCommandRequired: string;
    };
    review: {
      name: string;
      description: string;
      model: string;
      defaultModel: string;
      skills: string;
      keepAll: string;
      keptCount: string;
      mcp: string;
      none: string;
      createProfile: string;
      creating: string;
      created: string;
      createdPending: string;
      createFailed: string;
      invalidName: string;
      selectModel: string;
    };
  };

  // ── Cron Blueprints ──
  blueprints: {
    setup: string;
    cancel: string;
    scheduleIt: string;
    loading: string;
    loadError: string;
    noBlueprints: string;
    scheduled: string;
    tabLabel: string;
  };

  // ── Files page ──
  files: {
    title: string;
    pathPlaceholder: string;
    go: string;
    upload: string;
    create: string;
    uploading: string;
    dropHere: string;
    releaseToUpload: string;
    chooseFiles: string;
    loading: string;
    noFiles: string;
    name: string;
    size: string;
    modified: string;
    actions: string;
    createFolder: string;
    folderName: string;
    cancel: string;
    createBtn: string;
    creating: string;
    target: string;
    pathRequired: string;
    folderNameRequired: string;
    folderCreated: string;
    createFailed: string;
    uploadFailed: string;
    deleteFailed: string;
    deleted: string;
    directoryUnavailable: string;
  };

  // ── MCP page ──
  mcpPage: {
    addServer: string;
    removeServer: string;
    removeConfirm: string;
    name: string;
    transport: string;
    url: string;
    command: string;
    args: string;
    env: string;
    envPlaceholder: string;
    add: string;
    adding: string;
    nameRequired: string;
    urlRequired: string;
    commandRequired: string;
    install: string;
    installTitle: string;
    installHint: string;
    installing: string;
    installed: string;
    installFailed: string;
    installBackground: string;
    testTools: string;
    testFailed: string;
    toggleNote: string;
    addFailed: string;
    deleteSuccess: string;
  };

  // ── Webhooks page ──
  webhooks: {
    title: string;
    newWebhook: string;
    name: string;
    namePlaceholder: string;
    description: string;
    descriptionPlaceholder: string;
    events: string;
    create: string;
    creating: string;
    created: string;
    webhookUrl: string;
    secret: string;
    secretShownOnce: string;
    copySecret: string;
    copied: string;
    delete: string;
    deleteConfirmTitle: string;
    deleteConfirmMessage: string;
    deleteSuccess: string;
    noWebhooks: string;
    test: string;
    testSent: string;
    testFailed: string;
    refresh: string;
  };

  // ── Model Info Card ──
  modelInfo: {
    contextWindow: string;
    maxOutput: string;
    supportsVision: string;
    supportsTools: string;
    supportsReasoning: string;
    loading: string;
    noInfo: string;
  };

  // ── Skill Editor Dialog ──
  skillEditor: {
    createTitle: string;
    editTitle: string;
    name: string;
    namePlaceholder: string;
    category: string;
    categoryOptional: string;
    content: string;
    save: string;
    saving: string;
    saved: string;
    saveFailed: string;
    cancel: string;
  };
  // ── Auth Widget ──
  auth: {
    logOut: string;
    statusUnavailable: string;
    loggedInAs: string;
    via: string;
  };

  // ── Model Picker Dialog ──
  modelPicker: {
    close: string;
    switch_: string;
    cancel: string;
    filterPlaceholder: string;
    persistLabel: string;
    savesToConfig: string;
    expensiveWarning: string;
    switchAnyway: string;
    loading: string;
    noMatches: string;
    noProviders: string;
    current: string;
    unknown: string;
    models: string;
    title: string;
  };

  // ── Model Reload Confirm ──
  modelReload: {
    title: string;
    description: string;
    reload: string;
  };

  // ── Toolset Config Drawer ──
  toolsetConfig: {
    close: string;
    enableToolset: string;
    active: string;
    inactive: string;
    enabled: string;
    disabled: string;
    noBackends: string;
    noProviders: string;
    select: string;
    selected: string;
    saveKeys: string;
    saved: string;
    nothingToSave: string;
    enterValue: string;
    savedPlaceholder: string;
    getKey: string;
    needsInstall: string;
    runSetup: string;
    installing: string;
    postSetupComplete: string;
    postSetupErrors: string;
    lostTrack: string;
    failedToggle: string;
    failedSelect: string;
    failedSave: string;
    failedSetup: string;
    providerSet: string;
    enabledMsg: string;
    disabledMsg: string;
  };

  // ── Pairing Page ──
  pairing: {
    pendingRequests: string;
    noPending: string;
    approvedUsers: string;
    noApproved: string;
    approve: string;
    revoke: string;
    revokeAccess: string;
    revokeDesc: string;
    clearPending: string;
    clearAllConfirm: string;
    cleared: string;
    missingCode: string;
    approvedMsg: string;
    revokedMsg: string;
    loadFailed: string;
    ago: string;
  };
}
