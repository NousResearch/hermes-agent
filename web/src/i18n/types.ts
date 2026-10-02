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
  | "hu"
  | "ar";

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
    /** Optional — English fallback until translated. "{what}" = the noun that failed to load. */
    loadFailed?: string;
    loadFailedDetails?: string;
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
    /** Optional — round-2 sweep; list-type config fields (AutoField). */
    commaSeparatedPlaceholder?: string;
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
      degraded?: string;
      failed: string;
      heartbeatStale?: string;
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
      /**
       * Optional — built-in nav entries that used to render hardcoded English
       * (`Files`, `Channels`, …). Falls back to the English literal in
       * `App.tsx` so a locale that hasn't caught up still renders.
       */
      files?: string;
      mcp?: string;
      channels?: string;
      webhooks?: string;
      pairing?: string;
      system?: string;
    };
    /**
     * Optional — localized labels for PLUGIN-provided nav tabs, keyed by the
     * tab path (e.g. `/kanban`). Plugin manifests ship an English `label`
     * only, so this map lets the sidebar translate them without touching the
     * manifest. A miss falls back to the manifest label untouched.
     */
    pluginNav?: Record<string, string>;
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
    /** NS-656 memory-pressure banner — optional, English fallback. */
    memoryOomRestartBanner?: string;
    memoryCriticalBanner?: string;
    memoryElevatedBanner?: string;
    /** NS-656 disk-usage banner — optional, English fallback. */
    diskCriticalBanner?: string;
    diskElevatedBanner?: string;
    /** Multi-profile host whose gateway boots standalone on a guard — optional, English fallback. */
    multiplexStandaloneBanner?: string;
    dismiss?: string;
    /** First-run shared-metrics offer — optional, English fallback. */
    sharedMetricsTitle?: string;
    sharedMetricsBody?: string;
    sharedMetricsShare?: string;
    sharedMetricsLocal?: string;
    sharedMetricsOff?: string;
    sharedMetricsDetails?: string;
    sharedMetricsSaveFailed?: string;
    loadingChat?: string;
    restartAll?: string;
    restartSharedGatewayTitle?: string;
    updateConfirmBehind?: string;
  };

  // ── Status page ──
  status: {
    actionFailed: string;
    actionFinished: string;
    actions: string;
    agent: string;
    connected: string;
    connectedPlatforms: string;
    disabled?: string;
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
    restartGatewayConfirmMessage?: string;
    restartGatewayConfirmTitle?: string;
    restartingGateway: string;
    running: string;
    runningRemote: string;
    startFailed: string;
    starting: string;
    startedInBackground: string;
    stopped: string;
    updateHermes: string;
    updateHermesConfirmMessage?: string;
    updateHermesConfirmNow?: string;
    updateHermesConfirmTitle?: string;
    updatingHermes: string;
    waitingForOutput: string;
  };

  // ── Sessions page ──
  sessions: {
    title: string;
    history: string;
    overview: string;
    filterChats: string;
    filterAutomation: string;
    filterAll: string;
    sourceFilter: string;
    anySource: string;
    searchPlaceholder: string;
    noSessions: string;
    noSessionsInFilter: string;
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
    selectedSessionsSkippedActive: string;
    failedToDeleteSelected: string;
    resumeInChat: string;
    newChat: string;
    workspace: string;
    workspaceDefault: string;
    workspaceRescan: string;
    workspaceCustom: string;
    previousPage: string;
    nextPage: string;
    roles: {
      user: string;
      assistant: string;
      system: string;
      tool: string;
    };
    // Optional — added in the round-1 page sweep; only en/zh ship real
    // translations, so older full-locale files fall back to English.
    renameSession?: string;
    exportSession?: string;
    exportSessionJson?: string;
    sessionTitlePlaceholder?: string;
    saveTitle?: string;
    cancelRename?: string;
    anyChatSource?: string;
    anyAutomationSource?: string;
    chatSources?: string;
    automationSources?: string;
    noSources?: string;
    sessionRenamed?: string;
    renameFailed?: string;
    exportFailed?: string;
    validDaysRequired?: string;
    pruneFailed?: string;
    pruneOldSessions?: string;
    prune?: string;
    total?: string;
    activeInStore?: string;
    archived?: string;
    messages?: string;
    sources?: string;
    importSessions?: string;
    importSessionsTitle?: string;
    // Optional — round-2 leftover sweep.
    olderThanDays?: string;
    importSessionsAction?: string;
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
    /** Optional — "All time" range preset; en/zh seed it, other locales fall back. */
    all?: string;
    // Optional — round-2 leftover sweep (token-analytics hidden card).
    hiddenTitle?: string;
    hiddenBodyOne?: string;
    hiddenBodyOneSuffix?: string;
    hiddenBodyTwo?: string;
    hiddenBodyThreePrefix?: string;
    hiddenBodyThreeMid?: string;
    hiddenConfigLink?: string;
    hiddenBodyThreeSuffix?: string;
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
    /**
     * Optional — Models page panel/modal copy: the settings card, the
     * auxiliary-task modal, the Mixture-of-Agents editor and the hidden
     * token-analytics note. Falls back to the English literals; only en/zh
     * seed it, other locales inherit the merged English base.
     */
    page?: {
      modelSettings: string;
      appliesNewSessions: string;
      mainModel: string;
      auxiliaryTask: string;
      allAuxiliaryTasks: string;
      current: string;
      unset: string;
      change: string;
      auxiliaryTasksTitle: string;
      auxiliaryTasks: string;
      overridesSummary: string;
      allAutoSummary: string;
      configure: string;
      mixtureOfAgents: string;
      moaSummary: string;
      notLoaded: string;
      setMainModel: string;
      setAuxiliary: string;
      expensiveWarning: string;
      switchAnyway: string;
      resetAllToAuto: string;
      auxIntro: string;
      auxIntroAuto: string;
      auxIntroTail: string;
      autoUseMain: string;
      providerDefault: string;
      resetAuxTitle: string;
      resetAuxDescription: string;
      resetAll: string;
      close: string;
      useAs: string;
      missingProviderModel: string;
      highPricing: string;
      contextWindow: string;
      maxOutput: string;
      loadingInfo: string;
      overrideAuto: string;
      autoDetected: string;
      tools: string;
      vision: string;
      reasoning: string;
      moaTitle: string;
      moaIntro: string;
      setDefault: string;
      newPresetName: string;
      addPreset: string;
      defaultLabel: string;
      referenceModels: string;
      aggregator: string;
      addReferenceModel: string;
      remove: string;
      selectMoaModel: string;
      moaRecursiveError: string;
      tokenHiddenBody: string;
      tokenHiddenIn: string;
      tokenHiddenEnable: string;
      tokenHiddenConfig: string;
      tokenHiddenTail: string;
      auxTaskLabels: Record<string, string>;
      auxTaskHints: Record<string, string>;
    };
    tokenCacheRead?: string;
    tokenReasoning?: string;
    tokenInput?: string;
    tokenOutput?: string;
  };

  /**
   * Optional — model picker dialog + model reload-confirm copy. Shared by the
   * Models page and the chat side panel; falls back to the English literals.
   */
  modelPicker?: {
    switchModel: string;
    filterPlaceholder: string;
    noMatches: string;
    expensiveWarning: string;
    switchAnyway: string;
    close: string;
    promiseNote: string;
    reloadTitle: string;
    reload: string;
    reloadDescription: string;
    persistGlobal: string;
    refreshModels: string;
    switch: string;
    loadingProviders: string;
    noProviders: string;
    noModelsMatchFilter: string;
    noModelsForProvider: string;
    confirmPricingMessage: string;
    modelCount: string;
    pickProvider: string;
    /** Optional — round-2 leftover sweep. */
    savesToConfig?: string;
    openKeys?: string;
    signInToProvider?: string;
  };

  // ── Logs page ──  /** Optional — Files page copy. */
  /** Optional — MCP page copy. */
  mcpPage?: {
    removeServer: string;
    removeBodyNamed: string;
    removeBody: string;
    close: string;
    name: string;
    namePlaceholder: string;
    transport: string;
    transportHttp: string;
    transportStdio: string;
    url: string;
    urlPlaceholder: string;
    authentication: string;
    authNone: string;
    authBearer: string;
    authOauth: string;
    tokenPlaceholder: string;
    command: string;
    commandPlaceholder: string;
    args: string;
    argsPlaceholder: string;
    disabled: string;
    authenticateOauthTitle: string;
    authenticate: string;
    testConnection: string;
    installed: string;
    endpoint: string;
    runs: string;
    installsFrom: string;
    enable: string;
    disable: string;
    invalidServer: string;
    addedOauth: string;
    added: string;
    restartNote: string;
    installingBackground: string;
    addServer: string;
    adding: string;
    add: string;
    installing: string;
    install: string;
    connectedNoTools: string;
    failed: string;
    // Optional — round-2 leftover sweep.
    createTitle?: string;
    browseCatalog?: string;
    setupNotes?: string;
    noCatalogEntries?: string;
    oauthNote?: string;
  };

  /** Optional — Pairing page copy. */
  pairingPage?: {
    revokeAccess: string;
    revokeBodyNamed: string;
    revokeBody: string;
    revoke: string;
    /** Round-2 sweep additions — en/zh ship them, other locales fall back. */
    loadFailed: string;
    missingRequest: string;
    approved: string;
    approveFailed: string;
    clearConfirm: string;
    cleared: string;
    clearFailed: string;
    revoked: string;
    revokeFailed: string;
    clearPending: string;
    pendingHeader: string;
    noPending: string;
    ageMinutes: string;
    approve: string;
    approvedHeader: string;
    noApproved: string;
  };

  /** Optional — Webhooks page copy. */
  webhooksPage?: {
    copy: string;
    deleteWebhook: string;
    deleteBodyNamed: string;
    deleteBody: string;
    close: string;
    createdNotice: string;
    webhookUrl: string;
    secretShownOnce: string;
    name: string;
    namePlaceholder: string;
    description: string;
    descriptionPlaceholder: string;
    events: string;
    eventsPlaceholder: string;
    deliverTo: string;
    deliverLog: string;
    deliverTelegram: string;
    deliverDiscord: string;
    deliverSlack: string;
    deliverEmail: string;
    deliverGithubComment: string;
    deliverOnly: string;
    deliverOnlyHint: string;
    prompt: string;
    promptPlaceholder: string;
    receiverDisabled: string;
    receiverDisabledBody: string;
    badgeDeliverOnly: string;
    badgeDisabled: string;
    enable: string;
    disable: string;
    done: string;
    newSubscription: string;
    subscriptions: string;
    subscriptionsHint: string;
    noSubscriptions: string;
    allEvents: string;
    create: string;
    creating: string;
    enableWebhooks: string;
    enabling: string;
    restartGateway: string;
    restarting: string;
    restartNotice: string;
    loadFailed: string;
    gatewayRestarting: string;
    gatewayRestartFailedExit: string;
    gatewayRestartFailedManual: string;
    failedToRestart: string;
    webhooksEnabledRestarting: string;
    gatewayRestartFailedDetail: string;
    webhooksEnabledRestartFailed: string;
    failedToEnableWebhooks: string;
    nameRequired: string;
    created: string;
    failedToCreate: string;
    enabledNamed: string;
    disabledNamed: string;
    deletedNamed: string;
    errorPrefix: string;
  };

  /** Optional — Channels page + onboarding panels copy. */
  channelsPage?: {
    botTokenHint: string;
    allowedWhatsappNumbers: string;
    connected: string;
    saveAndRestart: string;
    whatsappQrAlt: string;
    linked: string;
    recommended: string;
    ready: string;
    ownerDetected: string;
    telegramUserIdPlaceholder: string;
    telegramQrAlt: string;
    waiting: string;
    statusRestartToApply: string;
    statusGatewayStopped: string;
    statusStartFailed: string;
    statusDisconnected: string;
    statusNotConfigured: string;
    statusDisabled: string;
    statusError: string;
    enablePlatform: string;
    saveAndEnable: string;
    nothingToSave: string;
    fixHighlighted: string;
    requiredField: string;
    gatewayRestarting: string;
    restartGateway: string;
    restartNow: string;
    useOwnTelegramBot: string;
    botFatherGuide: string;
    setupGuide: string;
    secretSetPlaceholder: string;
    whatsappSetupFailed: string;
    whatsappQrExpired: string;
    whatsappSavedRestarting: string;
    whatsappBridgePreparing: string;
    whatsappBridgeStarting: string;
    whatsappQrInstructions: string;
    whatsappAccountLinked: string;
    whatsappAccountLinkedAlt: string;
    whatsappSelfChatHint: string;
    whatsappOtherChatHint: string;
    whatsappSelfChatAuto: string;
    whatsappPairingFallback: string;
    pairWithQr: string;
    whatsappDeviceLinked: string;
    saveAndRestartAction: string;
    whatsappExistingSession: string;
    telegramPairingExpired: string;
    telegramUserIdsNumeric: string;
    telegramAddUserId: string;
    telegramSavedRestarting: string;
    createWithQr: string;
    telegramTokenInvalid: string;
    slackTokenPrefix: string;
    slackMemberIdInvalid: string;
    expired: string;
    // Optional — round-2 leftover sweep (Telegram/WhatsApp setup blocks).
    telegramChooseTitle?: string;
    telegramChooseBody?: string;
    telegramQuickSetup?: string;
    telegramQuickSetupBody?: string;
    telegramUseOwnBot?: string;
    telegramUseOwnBotBody?: string;
    telegramManualSetup?: string;
    telegramAlreadyConfigured?: string;
    telegramFinishOrCancel?: string;
    telegramAllowedUsers?: string;
    telegramAddAtLeastOneUser?: string;
    telegramOpen?: string;
    telegramFindMyUserId?: string;
    whatsappWaitingQr?: string;
    whatsappScanHint?: string;
  };

  filesPage?: {
    refreshFiles: string;
    path: string;
    uploadFiles: string;
    name: string;
    size: string;
    modified: string;
    actions: string;
    noFiles: string;
    createFolder: string;
    folderNamePlaceholder: string;
    pathRequired: string;
    directoryUnavailable: string;
    folderNameRequired: string;
    folderCreated: string;
    createFailed: string;
    deleted: string;
    deleteFailed: string;
    uploading: string;
    releaseToUpload: string;
    dropFilesHere: string;
    chooseFiles: string;
    loading: string;
    deleteItemTitle: string;
    deleteNamedTitle: string;
    deleteFolderDescription: string;
    deleteFileDescription: string;
    targetLabel: string;
    // Optional — round-2 leftover sweep.
    uploadAction?: string;
    createAction?: string;
    go?: string;
    loadingFiles?: string;
  };


  /** Optional — Chat page copy. */
  chat?: {
    imageUploadDisconnected: string;
    reconnecting: string;
    disconnected: string;
    reconnectChat: string;
    reconnectNow: string;
    checkServerStatus: string;
    startNewSessionAria: string;
    startNewSession: string;
    openLogs: string;
    copyLastRawTitle: string;
    copyLastTitle: string;
    copyLast: string;
    copied: string;
    showSidePanelTitle: string;
    showSidePanelAria: string;
    panelLabel: string;
    collapseSidePanelAria: string;
    collapseSidePanelTitle: string;
  };


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
    /** Optional — English fallback until translated. */
    loadWhat?: string;
    scriptRequired?: string;
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
    overdueSince?: string;
    schedulerLastTicked?: string;
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
    // Optional — added in the round-1 page sweep; only en/zh ship real
    // translations, so older full-locale files fall back to English.
    noToolsets?: string;
    noSkills?: string;
    skillsOptional?: string;
    savedChanges?: string;
    saveChanges?: string;
    editJob?: string;
    jobsTab?: string;
    blueprintsTab?: string;
    // Optional — round-2 leftover sweep (advanced cron fields).
    advancedFields?: string;
    providerLabel?: string;
    defaultOption?: string;
    modelLabel?: string;
    baseUrlOverride?: string;
    scriptLabel?: string;
    workdirLabel?: string;
    profileLabel?: string;
    allProfilesOption?: string;
    scriptPlaceholder?: string;
    contextFromPlaceholder?: string;
    skillsHint?: string;
  };

  // ── Plugins page ──
  pluginsPage: {
    contextEngineLabel: string;
    dashboardSlots: string;
    disableRuntime: string;
    enableAfterInstall: string;
    enableRuntime: string;
    toggleTakesEffectAfterRestart: string;
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
    /** Optional — round-2 leftover sweep. */
    saveMemoryProvider?: string;
    saveContextEngine?: string;
    sourceBadge: string;
    authRequired: string;
    authRequiredHint: string;
    updateGit: string;
    /** Optional: locales without it fall back to the English body at the call site. */
    updateConsentBody?: (name: string, sha: string) => string;
    versionBadge: string;
    showInSidebar: string;
    hideFromSidebar: string;
    // Catalog section (en-only fallback convention — optional keys).
    catalogHeading?: string;
    catalogHint?: string;
    catalogSearchPlaceholder?: string;
    catalogEmpty?: string;
    catalogEmptyDocsLink?: string;
    catalogInstallBtn?: string;
    catalogInstalledBadge?: string;
    catalogUpdateBtn?: string;
    catalogRemovedBadge?: string;
    catalogConfirmTitle?: string;
    catalogConfirmInstallNote?: string;
    catalogRequiresEnv?: string;
    removedFromCatalog?: string;
    memoryStatusReady?: string;
    memoryStatusNeedsConfig?: string;
    memoryStatusUnavailable?: string;
    memoryStatusMissing?: string;
    setupResultAlreadyInstalled?: string;
    setupResultNoDeclared?: string;
    setupResults?: string;
    setupUnavailable?: string;
    setupBlocked?: string;
    setupCompleted?: string;
    setupInstallingDeps?: string;
    setupInstallDeps?: string;
    setupRunning?: string;
    setupExternalDep?: string;
    setupInstallDep?: string;
    setupInstallDepNamed?: string;
    setupVerifyDep?: string;
    setupVerifyDepNamed?: string;
    setupPythonDeps?: string;
    setupRequiredEnv?: string;
    builtinMemoryNote?: string;
    activeProviderMissing?: string;
    depsInstalledAddCreds?: string;
    loadingProviderSettings?: string;
    noProviderSettings?: string;
    fieldRequired?: string;
    fieldSet?: string;
    fieldOpen?: string;
    leaveBlankKeep?: string;
    activeBadge?: string;
    toastLoadProviderConfig?: string;
    toastInstallFailed?: string;
    toastRescanFailed?: string;
    toastSaveFailed?: string;
    toastFailed?: string;
    toastProviderSetupFailed?: string;
    toastProviderSetupFinished?: string;
    toastProviderSetupFailedShort?: string;
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
    build?: string;
    activeSetHint?: string;
  };

  // ── Skills page ──
  /** Optional — System page copy. Only en/zh ship it; other locales fall back. */
  systemPage?: {
    memoryStatusReady: string;
    memoryStatusNeedsConfig: string;
    memoryStatusUnavailable: string;
    memoryStatusMissing: string;
    toastOpStarted: string;
    toastOpFailed: string;
    logRunning: string;
    logDone: string;
    logExit: string;
    logClose: string;
    logStarting: string;
    hostHeading: string;
    osLabel: string;
    archLabel: string;
    hostLabel: string;
    pythonLabel: string;
    hermesLabel: string;
    behindCount: string;
    updateAvailable: string;
    latest: string;
    cpuLabel: string;
    cores: string;
    memoryLabel: string;
    diskLabel: string;
    uptimeLabel: string;
    loadAvgLabel: string;
    psutilHint: string;
    checkForUpdates: string;
    updateNow: string;
    updateWith: string;
    portalHeading: string;
    loggedIn: string;
    notLoggedIn: string;
    inferenceProvider: string;
    manageSubscription: string;
    toolGatewayRouting: string;
    portalLoginHint: string;
    curatorHeading: string;
    curatorPaused: string;
    curatorActive: string;
    curatorDisabled: string;
    curatorEvery: string;
    curatorLastRun: string;
    curatorNever: string;
    resume: string;
    pause: string;
    runNow: string;
    curatorReviewAction: string;
    gatewayHeading: string;
    gatewayRunning: string;
    gatewayStopped: string;
    openLogs: string;
    start: string;
    restart: string;
    stop: string;
    servedByShared: string;
    multiplexBlurb: string;
    migrateToMultiplex: string;
    fixBlockers: string;
    memoryHeading: string;
    externalProvider: string;
    builtinOnly: string;
    changeInPlugins: string;
    providerSetup: string;
    configureInPlugins: string;
    providerMissing: string;
    builtinFiles: string;
    resetMemoryMd: string;
    resetUserMd: string;
    resetAll: string;
    credentialHeading: string;
    providerLabel: string;
    apiKeyLabel: string;
    labelLabel: string;
    optionalPlaceholder: string;
    addKey: string;
    noPooledCredentials: string;
    removeCredential: string;
    operationsHeading: string;
    openConsole: string;
    runDoctor: string;
    securityAudit: string;
    updateSkills: string;
    promptSize: string;
    supportDump: string;
    migrateConfig: string;
    doctorAction: string;
    securityAuditAction: string;
    skillsUpdateAction: string;
    promptSizeAction: string;
    supportDumpAction: string;
    configMigrateAction: string;
    fullBackup: string;
    createBackup: string;
    downloadBackup: string;
    noBackupCreated: string;
    restoreFromUpload: string;
    chooseRestoreZip: string;
    noBackupSelected: string;
    restoreUpload: string;
    restoreFromPath: string;
    restorePath: string;
    restoreConfirmTitle: string;
    restoreConfirmBody: string;
    restore: string;
    cancel: string;
    shareDebugHeading: string;
    shareDebugBody: string;
    uploading: string;
    generateShareLink: string;
    redactLabel: string;
    uploaded: string;
    redacted: string;
    notRedacted: string;
    autoDeletesIn: string;
    copyAll: string;
    copyLinkAria: string;
    someLogsFailed: string;
    checkpointsHeading: string;
    checkpointsSummary: string;
    prune: string;
    pruneCheckpointsTitle: string;
    pruneCheckpointsBody: string;
    shellHooksHeading: string;
    newHook: string;
    noShellHooks: string;
    notExecutable: string;
    allowed: string;
    notApproved: string;
    removeHook: string;
    hookMatcher: string;
    newShellHookTitle: string;
    eventLabel: string;
    commandLabel: string;
    matcherLabel: string;
    matcherPlaceholder: string;
    timeoutLabel: string;
    approveNow: string;
    hooksWarning: string;
    creating: string;
    createHook: string;
    close: string;
    toastGatewayStarted: string;
    toastMigrating: string;
    toastMigrateFailed: string;
    toastCuratorResumed: string;
    toastCuratorPaused: string;
    toastCuratorToggleFailed: string;
    toastReset: string;
    toastResetNothing: string;
    toastResetFailed: string;
    toastProviderRequired: string;
    toastCredentialAdded: string;
    toastCredentialAddFailed: string;
    toastCredentialRemoved: string;
    toastCredentialRemoveFailed: string;
    toastBackupStarted: string;
    toastBackupFailed: string;
    toastBackupReady: string;
    toastDownloadFailed: string;
    toastImportStarted: string;
    toastImportFailed: string;
    toastCopyFailed: string;
    toastUploaded: string;
    toastDebugShareFailed: string;
    toastUpdateAvailableBehind: string;
    toastUpdateAvailable: string;
    toastLatest: string;
    toastUpdateCheckFailed: string;
    toastManagedOutside: string;
    toastUpdatesDontApply: string;
    toastUpdateStarted: string;
    toastUpdateFailed: string;
    toastPruneStarted: string;
    toastPruneFailed: string;
    toastCommandRequired: string;
    toastHookCreated: string;
    toastHookCreateFailed: string;
    toastHookRemoved: string;
    toastHookRemoveFailed: string;
    restartSharedTitle: string;
    restartAll: string;
    updateConfirmTitle: string;
    updateConfirmBehind: string;
    updateConfirmGeneric: string;
    resetMemoryTitle: string;
    resetMemoryBody: string;
    removeCredentialTitle: string;
    removeCredentialBody: string;
    removeHookTitle: string;
    removeHookBody: string;
  };

  /** Optional — Profile builder copy. Only en/zh ship it; other locales fall back. */
  profileBuilder?: {
    stepIdentity: string;
    stepModel: string;
    stepSkills: string;
    stepMcp: string;
    stepReview: string;
    newProfile: string;
    cancel: string;
    profileName: string;
    namePlaceholder: string;
    nameHint: string;
    descriptionLabel: string;
    descriptionPlaceholder: string;
    modelHint: string;
    filterModels: string;
    loadingModels: string;
    useDefault: string;
    keepAllLabel: string;
    keepHint: string;
    filterSkills: string;
    loadingSkills: string;
    addFromHub: string;
    hubSearchPlaceholder: string;
    searching: string;
    search: string;
    add: string;
    removeAria: string;
    mcpHeading: string;
    mcpIntro: string;
    configuredCount: string;
    addServerHeading: string;
    serverName: string;
    serverNamePlaceholder: string;
    transport: string;
    authentication: string;
    authNone: string;
    authHeader: string;
    authOauth: string;
    bearerToken: string;
    bearerPlaceholder: string;
    bearerHint: string;
    oauthHint: string;
    command: string;
    arguments: string;
    environment: string;
    addServer: string;
    remove: string;
    reviewName: string;
    reviewDescription: string;
    reviewModel: string;
    defaultSetLater: string;
    reviewSkills: string;
    fullDefaultBundle: string;
    keptCount: string;
    hubSuffix: string;
    hubPrefix: string;
    reviewHubSkills: string;
    reviewMcp: string;
    none: string;
    back: string;
    next: string;
    creating: string;
    createProfile: string;
    invalidName: string;
    invalidMcp: string;
    createdPending: string;
    created: string;
    createFailed: string;
    authPrefix: string;
    // Optional — round-2 leftover sweep.
    urlLabel?: string;
    mcpTransportAria?: string;
    httpAuthAria?: string;
  };

  /** Optional — shared component copy. Only en/zh ship it; others fall back. */
  sharedComponents?: {
    console: {
      title: string;
      reconnect: string;
      reconnectAria: string;
      closeAria: string;
      commandFailed: string;
      couldNotConnect: string;
      closed: string;
      disconnected: string;
    };
    authWidget: {
      statusUnavailable: string;
      reloadPage: string;
      loggedInAs: string;
      viaProvider: string;
      logOut: string;
    };
    skillEditor: {
      editTitle: string;
      newTitle: string;
      editBody: string;
      createBody: string;
      nameLabel: string;
      categoryLabel: string;
      saving: string;
      saveChanges: string;
      createSkill: string;
      errNameRequired: string;
      errContentRequired: string;
    };
    toolsetDrawer: {
      active: string;
      inactive: string;
      enableAria: string;
      enabledFor: string;
      disabledFor: string;
      noBackends: string;
      noProviders: string;
      selected: string;
      select: string;
      saved: string;
      savedPlaceholder: string;
      getKey: string;
      saveKeys: string;
      postSetupNotice: string;
      installing: string;
      runSetup: string;
      postSetupPrefix: string;
      starting: string;
      close: string;
      failedLoad: string;
      postSetupComplete: string;
      postSetupErrors: string;
      lostTrack: string;
      failedToggle: string;
      providerSet: string;
      failedSelect: string;
      enterAtLeastOne: string;
      savedCount: string;
      nothingToSave: string;
      failedSaveKeys: string;
      failedStartSetup: string;
      toggleEnabled: string;
      toggleDisabled: string;
    };
    blueprints: {
      cancel: string;
      setUp: string;
      scheduleIt: string;
      loadFailed: string;
      loadingBlueprints: string;
      noneAvailable: string;
      scheduled: string;
    };
    chatSidebar: {
      reasoning: string;
      reasoningNotice: string;
      modelNotice: string;
      /** Optional — round-2 leftover sweep. */
      reloadPage?: string;
      reconnectSidePanel?: string;
      addKey?: string;
      switchModel?: string;
    };
  };

  skills: {
    title: string;
    searchPlaceholder: string;
    /** Optional — English fallback until translated. */
    loadWhat?: string;
    browseHub?: string;
    createSkill?: string;
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
    learnSkillTitle?: string;
    learnSkillBody?: string;
    learnDirLabel?: string;
    learnDirPlaceholder?: string;
    learnUrlLabel?: string;
    learnUrlPlaceholder?: string;
    learnNotesLabel?: string;
    learnNotesPlaceholder?: string;
    learnSubmit?: string;
    browseHubTab?: string;
    editSkillAria?: string;
    newSkill?: string;
    configure?: string;
    editSkillMdTitle?: string;
    /** Optional — skill-hub browser copy (search/preview/scan/install). */
    hub?: {
      searchPlaceholder: string;
      search: string;
      updateAll: string;
      running: string;
      done: string;
      dismiss: string;
      starting: string;
      featuredHeading: string;
      featuredHint: string;
      landingHint: string;
      noResults: string;
      connecting: string;
      fromSources: string;
      connectedHubs: string;
      githubRateLimited: string;
      indexUnavailable: string;
      rateLimitedSuffix: string;
      resultCount: string;
      timedOut: string;
      openAria: string;
      details: string;
      install: string;
      installed: string;
      installedLower: string;
      searchFailed: string;
      installing: string;
      installFailed: string;
      updating: string;
      updateFailed: string;
      previewFailed: string;
      scanFailed: string;
      dialogDescription: string;
      readSkillMd: string;
      rescan: string;
      securityScan: string;
      filesLabel: string;
      emptySkillMd: string;
      loadSourceFailed: string;
      scanningBody: string;
      scanPrompt: string;
      policyAllow: string;
      policyAsk: string;
      policyBlock: string;
      verdictLabel: string;
      trustSource: string;
      noRisky: string;
      trustTrusted: string;
      trustBuiltin: string;
      trustCommunity: string;
      trustUnknown: string;
      verdictSafe: string;
      verdictCaution: string;
      verdictDangerous: string;
      categoryLabels?: Record<string, string>;
    };
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
    /**
     * Optional — Settings → Model routing block copy (subagent preferred route
     * + fallbacks + hot-reload). Only en/zh ship it; other locales fall back to
     * the English literals merged in by `mergeTranslations`.
     */
    modelRouting?: {
      title: string;
      subtitle: string;
      subagentTitle: string;
      subagentHint: string;
      subagentEmpty: string;
      mainTitle: string;
      mainHint: string;
      subagentFallbackTitle: string;
      subagentFallbackHint: string;
      pickModel: string;
      clearRoute: string;
      addRoute: string;
      removeRoute: string;
      /** Optional: legacy locales (af/de/es) predate these reorder labels. */
      moveUp?: string;
      /** Optional: legacy locales (af/de/es) predate these reorder labels. */
      moveDown?: string;
      providerLabel: string;
      providerPlaceholder: string;
      modelLabel: string;
      modelPlaceholder: string;
      routeEmpty: string;
      hotReload: string;
      hotReloadHint: string;
    };
    /**
     * Optional — authored copy for the generic config form, keyed by the config
     * schema key verbatim (`model`, `model_context_length`, `delegation.provider`,
     * …). Label at `fieldCopy[<key>]`; description at `fieldCopy[<key>_desc]`.
     * Checked BEFORE the client-synthesized English label and the backend schema
     * prose, so an untranslated key keeps rendering exactly as before. Only
     * en/zh seed it; other locales inherit the merged English base.
     */
    fieldCopy?: Record<string, string>;
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
    customTitle: string;
    customHint: string;
    customConfigured: string;
    addCustomKey: string;
    customKeyName: string;
    customKeyNamePlaceholder: string;
    add: string;
    invalidKeyName: string;
    /** Optional — round-2 sweep; section-nav aria-label. */
    jumpToSection?: string;
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
    copyCode: string;
    copyFailed: string;
    cli: string;
    copyCliCommand: string;
    connect: string;
    sessionExpires: string;
    sessionExpiredNoError: string;
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
    disconnectBody?: string;
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
    confirmDoneMany: string;
    confirmArchiveMany: string;
    confirmBlockedMany: string;
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
    // Optional keys added with the modal create-task dialog, board-settings
    // dialog, and comment workflow hint. Non-English locales fall back to
    // the English literal in the plugin bundle until translated, so these
    // are optional to avoid churning every locale file.
    newTaskTitle?: string;
    taskTitleLabel?: string;
    assigneeLabel?: string;
    assigneeLabelHint?: string;
    skillsLabel?: string;
    skillsLabelHint?: string;
    parentLabel?: string;
    parentLabelHint?: string;
    create?: string;
    boardSettings?: string;
    boardSettingsTitle?: string;
    boardSettingsTitleFor?: string;
    saving?: string;
    commentHint?: string;
    commentHintTitle?: string;
    // Optional in-app confirm-dialog strings for the trash/delete flow;
    // non-English locales fall back to the English literals in the bundle.
    trash?: {
      confirmTitle?: string;
      confirmManyTitle?: string;
      confirmMany?: string;
    };
    // Optional — kanban plugin bundle keys (board project binding, workspace/model/
    // goal-mode task dialog, attachments, bulk-action tooltips, orchestration panel).
    // en/zh seed them; other locales fall back to the English literals.
    clearFilters?: string;
    reload?: string;
    boardProject?: string;
    boardProjectHint?: string;
    boardProjectNone?: string;
    boardProjectClear?: string;
    boardProjectExplanation?: string;
    boardProjectSettingsExplanation?: string;
    boardProjectBadge?: string;
    boardProjectBadgeTitle?: string;
    unbindProject?: string;
    projectDirectory?: string;
    projectDirectoryHint?: string;
    projectDirectoryPlaceholder?: string;
    projectDirectoryHelp?: string;
    projectDirectoryExplanation?: string;
    projectDirectoryOverrideHint?: string;
    bulkConfirmTitle?: string;
    confirmTitle?: string;
    confirm?: string;
    delete?: string;
    ok?: string;
    attachments?: string;
    uploading?: string;
    uploadFile?: string;
    noAttachments?: string;
    removeAttachment?: string;
    confirmRemoveAttachment?: string;
    childResults?: string;
    noChildResult?: string;
    finalResult?: string;
    doneParentNote?: string;
    doneNoResult?: string;
    goalMode?: string;
    goalMaxTurns?: string;
    model?: string;
    modelProfileDefault?: string;
    modelProfileDefaultOption?: string;
    modelFreeTextPlaceholder?: string;
    modelLoading?: string;
    clickToEditModel?: string;
    setPriority?: string;
    selectedTasks?: string;
    thisTask?: string;
    workspaceDir?: string;
    workspaceScratch?: string;
    workspaceScratchWarning?: string;
    workspaceWorktree?: string;
    switchBoard?: string;
    boardSwitcherTitle?: string;
    newBoardTitleAttr?: string;
    slugRequired?: string;
    boardDescPlaceholder?: string;
    hideUntilReload?: string;
    copyCommandPrompt?: string;
    docsTitle?: string;
    docsAria?: string;
    noProfileAssigned?: string;
    refreshLog?: string;
    editDescription?: string;
    addParentBtn?: string;
    addChildBtn?: string;
    reclaimFirst?: string;
    selectAllVisible?: string;
    reassignOption?: string;
    specifying?: string;
    specifyBtn?: string;
    specifyFailed?: string;
    specifyOk?: string;
    specifyRetitled?: string;
    decomposing?: string;
    decomposeBtn?: string;
    decomposeFailed?: string;
    unknownError?: string;
    ttSearch?: string;
    ttTenant?: string;
    ttAssignee?: string;
    ttShowArchived?: string;
    ttLanes?: string;
    ttNudge?: string;
    ttRefresh?: string;
    ttClearFilters?: string;
    ttBlock?: string;
    ttUnblock?: string;
    ttArchive?: string;
    ttDelete?: string;
    ttPriority?: string;
    ttReassign?: string;
    ttApplyAssignee?: string;
    ttReclaimFirst?: string;
    ttSelectAllVisible?: string;
    ttDeselectAll?: string;
    ttSelectColumn?: string;
    ttSpecProfile?: string;
    ttAssignProfile?: string;
    ttPriorityField?: string;
    ttSkills?: string;
    ttWorkspace?: string;
    ttParent?: string;
    ttGoal?: string;
    ttGoalTurns?: string;
    common?: { confirm?: string; delete?: string };
    confirmStatusTitle?: { done?: string; archived?: string; blocked?: string; scheduled?: string };
    confirmStatusLabel?: { done?: string; archived?: string; blocked?: string; scheduled?: string };
    orchestration?: {
      settings?: string;
      loadingMode?: string;
      autoTitle?: string;
      manualTitle?: string;
      pillPrefix?: string;
      auto?: string;
      manual?: string;
      configureHint?: string;
      reload?: string;
      orchestratorProfile?: string;
      defaultValue?: string;
      resolved?: string;
      orchestratorHelp?: string;
      defaultAssignee?: string;
      modeLabel?: string;
      autoDecompose?: string;
      autoOnHelp?: string;
      autoOffHelp?: string;
      loading?: string;
      profileDescriptions?: string;
      profileDescriptionsHelp?: string;
      noProfiles?: string;
      profileDefault?: string;
      profileNoDescription?: string;
      profileAutoReview?: string;
      profilePlaceholder?: string;
      profileSaveTitle?: string;
      saveBtn?: string;
      savingLabel?: string;
      profileAutoTitle?: string;
      autoBtn?: string;
      generating?: string;
      loadFailed?: string;
      settingsSaved?: string;
      saveFailed?: string;
      descSaved?: string;
      autoGenDesc?: string;
      autoGenFailed?: string;
      unknownError?: string;
    };
  };
}
