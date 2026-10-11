import type { Translations } from '../types'

export const extensionsEn: Pick<Translations, 'achievements' | 'kanban'> = {
  achievements: {
    hero: {
      kicker: 'Agentic Gamerscore',
      title: 'Hermes Achievements',
      subtitle:
        'Collectible Hermes badges earned from real session history. Known unfinished achievements are shown as Discovered; Secret achievements stay hidden until the first matching behavior appears.',
      scan_subtitle: 'Scanning Hermes session history. First scan can take 5–10 seconds on large histories.'
    },
    actions: {
      rescan: 'Rescan'
    },
    stats: {
      unlocked: 'Unlocked',
      unlocked_hint: 'earned badges',
      discovered: 'Discovered',
      discovered_hint: 'known, not earned yet',
      secrets: 'Secrets',
      secrets_hint: 'hidden until first signal',
      highest_tier: 'Highest tier',
      highest_tier_hint: 'Copper → Silver → Gold → Diamond → Olympian',
      latest: 'Latest',
      latest_hint_empty: 'run Hermes more',
      none_yet: 'None yet'
    },
    state: {
      unlocked: 'Unlocked',
      discovered: 'Discovered',
      secret: 'Secret'
    },
    tier: {
      target: 'Target {tier}',
      hidden: 'Hidden',
      complete: 'Complete',
      objective: 'Objective'
    },
    progress: {
      hidden: 'hidden'
    },
    scan: {
      building_headline: 'Building achievement profile…',
      building_detail: 'Reading sessions, tool calls, model metadata, and unlock state.',
      starting_headline: 'Starting achievement scan…',
      progress_detail: 'Scanned {scanned} of {total} sessions · {pct}%. Badges unlock as more history streams in.',
      idle_detail: 'Reading sessions, tool calls, model metadata, and unlock state. Badges appear here as they unlock.'
    },
    guide: {
      tiers_header: 'Tiers',
      secret_header: 'Secret achievements',
      secret_body:
        'Secrets hide their exact trigger. Once Hermes sees a related signal, the card becomes Discovered and shows its requirement.',
      scan_status_header: 'Scan status',
      scan_status_body:
        'Hermes is scanning local history once, then cards will appear automatically. Nothing is stuck if this takes a few seconds.',
      what_scanned_header: 'What is scanned',
      what_scanned_body: 'Sessions, tool calls, model metadata, errors, achievements, and local unlock state.'
    },
    card: {
      share_title: 'Share this achievement',
      share_label: 'Share {name}',
      share_text: 'Share',
      how_to_reveal: 'How to reveal',
      what_counts: 'What counts',
      evidence_label: 'Evidence',
      evidence_session_fallback: 'session',
      no_evidence: 'No evidence yet'
    },
    latest: {
      header: 'Recent unlocks'
    },
    empty: {
      no_secrets_header: 'No hidden secrets left in this scan.',
      no_secrets_body:
        'Clue: secrets usually start from unusual failure or power-user patterns — port conflicts, permission walls, missing env vars, YAML mistakes, Docker collisions, rollback/checkpoint use, cache hits, or tiny fixes after lots of red text.'
    },
    filters: {
      all_categories: 'All',
      visibility_all: 'all',
      visibility_unlocked: 'unlocked',
      visibility_discovered: 'discovered',
      visibility_secret: 'secret'
    },
    share: {
      dialog_label: 'Share achievement',
      header: 'Share: {name}',
      close: 'Close',
      rendering: 'Rendering…',
      card_alt: '{name} share card',
      error_generic: 'Something went wrong.',
      x_title: 'Opens X with a pre-filled post',
      x_button: 'Share on X',
      copy_title: 'Copy the image to paste into your post',
      copy_button: 'Copy image',
      copied: 'Copied ✓',
      download_button: 'Download PNG',
      hint: 'Share on X opens a pre-filled post in a new tab. Click Copy image first if you want the 1200×630 badge attached — X lets you paste it right into the tweet composer. Download PNG saves the file for use anywhere.',
      clipboard_unsupported: 'Clipboard image copy not supported in this browser — use Download instead.',
      tweet_text: 'Just unlocked {tier_part}"{name}" in Hermes Agent ☤',
      tier_part: '{tier} tier ',
      tier_suffix: 'TIER',
      unlocked_stamp: 'UNLOCKED'
    },
    secretDefinition: {
      name: '???',
      description:
        'Secret achievement: hidden until Hermes detects the first relevant behavior in your session history.'
    },
    categories: {
      agent_autonomy: 'Agent Autonomy',
      debugging_chaos: 'Debugging Chaos',
      vibe_coding: 'Vibe Coding',
      hermes_native: 'Hermes Native',
      research_web: 'Research/Web',
      tool_mastery: 'Tool Mastery',
      model_lore: 'Model Lore',
      lifestyle: 'Lifestyle'
    },
    tierNames: {
      copper: 'Copper',
      silver: 'Silver',
      gold: 'Gold',
      diamond: 'Diamond',
      olympian: 'Olympian'
    },
    criteria: {
      secret:
        'Secret: exact requirement hidden until Hermes sees the first matching signal. Keep using Hermes across debugging, tools, memory, skills, plugins, and model workflows to reveal it.',
      threshold: 'Requirement: {metric}. Tier ladder: {ladder}.',
      requirements: 'Requirement: {requirements}.',
      default: 'Requirement: complete the matching Hermes behavior.',
      separator: '; '
    },
    metrics: {
      max_tool_calls_in_session: 'tool calls in one session',
      total_tool_calls: 'lifetime Hermes tool calls',
      max_distinct_tools_in_session: 'distinct Hermes tools used in one session',
      max_terminal_calls_in_session: 'terminal calls in one session',
      max_file_tool_calls_in_session: 'file/search/patch calls in one session',
      max_web_browser_calls_in_session: 'web search/extract or browser calls in one session',
      total_delegate_calls: 'lifetime delegate_task calls',
      total_process_calls: 'lifetime background process operations',
      total_cron_calls: 'lifetime scheduled-job operations',
      total_errors: 'error/failed/traceback messages observed',
      traceback_events: 'traceback or exception mentions',
      log_read_events: 'log inspections',
      port_conflict_events: 'dev-server port conflict detections',
      permission_denied_events: 'permission-denied errors',
      install_error_events: 'package-install failures',
      install_success_events: 'successful package installs after package work',
      restart_after_error_events: 'restart/reload actions after error clusters',
      env_var_error_events: 'missing auth/config/environment-variable events',
      yaml_error_events: 'YAML/config parse incidents',
      docker_conflict_events: 'Docker/container-name conflicts',
      max_messages_in_session: 'messages in one session',
      max_files_touched_in_session: 'files touched in one session',
      frontend_activity_events: 'frontend/CSS/SVG/React activity mentions',
      git_events: 'git workflow commands',
      css_activity_events: 'CSS, styling, Tailwind, or className activity',
      tiny_patch_after_errors_events: 'tiny typo-style fixes after error clusters',
      skill_events: 'Hermes skill mentions or tool use',
      skill_manage_events: 'skill_manage create/patch/delete operations',
      memory_events: 'memory or Mnemosyne tool events',
      memory_write_events: 'durable memory writes',
      context_events: 'context, compression, token, or cache-pressure mentions',
      gateway_events: 'gateway/API/chat-platform activity',
      plugin_events: 'dashboard plugin development or usage signals',
      rollback_events: 'rollback/checkpoint recovery mentions',
      total_web_calls: 'lifetime web_search/web_extract calls',
      total_web_extract_calls: 'lifetime web_extract calls',
      docs_activity_events: 'documentation/README/docs activity',
      browser_calls: 'lifetime browser automation calls',
      total_terminal_calls: 'lifetime terminal calls',
      total_patch_calls: 'lifetime targeted patch edits',
      total_file_reads_searches: 'lifetime read_file/search_files calls',
      image_vision_calls: 'image generation or vision tool calls',
      tts_calls: 'text-to-speech or voice tool calls',
      model_events: 'model/provider-related activity',
      openrouter_events: 'OpenRouter mentions',
      codex_events: 'Codex mentions',
      distinct_model_count: 'distinct model names seen in session metadata',
      distinct_provider_count: 'distinct model providers inferred from session metadata',
      claude_events: 'Claude/Anthropic model mentions',
      gemini_events: 'Gemini/Google model mentions',
      local_model_chat_sessions: 'Hermes sessions whose model metadata is local/open-weight',
      toolset_events: 'toolset or tool-family mentions',
      config_events: 'configuration/environment/manifest activity',
      git_history_events: 'git history operations such as rebase, merge, fetch, push, or tag',
      test_events: 'test/check/verification command mentions',
      screenshot_events: 'screenshot, Playwright, PNG, or vision-inspection activity',
      session_count: 'Hermes sessions',
      weekend_sessions: 'sessions started on weekends',
      night_sessions: 'sessions started late night or before dawn',
      cache_events: 'prompt-cache/cache-hit mentions'
    },
    definitions: {
      let_him_cook: {
        name: 'Let Him Cook',
        description: 'Let Hermes run a serious autonomous tool chain in one session.'
      },
      autonomous_avalanche: {
        name: 'Autonomous Avalanche',
        description: 'Accumulate a lifetime avalanche of Hermes tool calls across sessions.'
      },
      toolchain_maxxer: {
        name: 'Toolchain Maxxer',
        description: 'Use a wide spread of distinct Hermes tools in one session.'
      },
      full_send: {
        name: 'Full Send',
        description: 'Terminal, files, and web/browser all get involved in one real run.'
      },
      subagent_commander: {
        name: 'Subagent Commander',
        description: 'Coordinate delegated agent work.'
      },
      background_process_enjoyer: {
        name: 'Background Process Enjoyer',
        description: 'Start or control enough long-running processes to deserve the title.'
      },
      cron_necromancer: {
        name: 'Cron Necromancer',
        description: 'Raise scheduled autonomous jobs from the dead.'
      },
      red_text_connoisseur: {
        name: 'Red Text Connoisseur',
        description: 'Encounter enough errors to develop a palate for red text.'
      },
      stack_trace_sommelier: {
        name: 'Stack Trace Sommelier',
        description: 'Taste tracebacks by the flight, not by the sip.'
      },
      actually_read_the_logs: {
        name: 'Actually Read The Logs',
        description: 'Inspect logs repeatedly instead of guessing.'
      },
      port_3000_taken: {
        name: 'Port 3000 Is Taken',
        description: 'Discover dev-server port conflict patterns enough times to become numb.'
      },
      permission_denied_any_percent: {
        name: 'Permission Denied Any%',
        description: 'Speedrun into permission walls.'
      },
      dependency_hell_tourist: {
        name: 'Dependency Hell Tourist',
        description: 'Package installs fail, then somehow life continues.'
      },
      the_fix_was_restarting: {
        name: 'The Fix Was Restarting It',
        description: 'Restart after enough error clusters to call it a technique.'
      },
      forgot_the_env_var: {
        name: 'Forgot The Env Var',
        description: 'Auth or configuration failed because an environment variable was missing.'
      },
      yaml_colon_incident: {
        name: 'YAML Colon Incident',
        description: 'Configuration syntax bites back.'
      },
      docker_name_collision: {
        name: 'Docker Name Collision',
        description: 'A container name already exists. Of course it does.'
      },
      supposed_to_be_quick: {
        name: 'This Was Supposed To Be Quick',
        description: 'A tiny ask becomes an entire expedition.'
      },
      one_more_small_change: {
        name: 'One More Small Change',
        description: 'Make enough file edits in one session to invalidate the phrase small change.'
      },
      vibe_architect: {
        name: 'Vibe Architect',
        description: 'Touch a broad surface area in one project session.'
      },
      pixel_goblin: {
        name: 'Pixel Goblin',
        description: 'Do sustained frontend, CSS, SVG, or visual tuning.'
      },
      ship_first_ask_later: {
        name: 'Ship First, Ask Later',
        description: 'Git activity after a serious tool chain.'
      },
      css_exorcist: {
        name: 'CSS Exorcist',
        description: 'Cast repeated styling demons out of the interface.'
      },
      one_character_fix: {
        name: 'One Character Fix',
        description: 'A tiny edit after a pile of errors. Painful. Beautiful.'
      },
      skillsmith: {
        name: 'Skillsmith',
        description: 'Work with Hermes skills enough to leave fingerprints.'
      },
      skill_issue_skill_created: {
        name: 'Skill Issue? Skill Created.',
        description: 'Create or patch durable procedures instead of repeating yourself.'
      },
      memory_keeper: {
        name: 'Memory Keeper',
        description: 'Persist durable knowledge with memory or Mnemosyne.'
      },
      memory_palace: {
        name: 'Memory Palace',
        description: 'Build a serious durable-memory trail.'
      },
      context_dragon: {
        name: 'Context Dragon',
        description: 'Brush against compression, huge context, or token pressure repeatedly.'
      },
      gateway_dweller: {
        name: 'Gateway Dweller',
        description: 'Live through gateway-connected Hermes workflows.'
      },
      plugin_goblin: {
        name: 'Plugin Goblin',
        description: 'Use or develop plugins enough that the dashboard notices.'
      },
      rollback_wizard: {
        name: 'Rollback Wizard',
        description: 'Invoke rollback/checkpoint recovery magic.'
      },
      rabbit_hole_certified: {
        name: 'Rabbit Hole Certified',
        description: 'Search or extract enough web content to qualify as a research spiral.'
      },
      citation_goblin: {
        name: 'Citation Goblin',
        description: 'Extract enough web pages to become a tiny librarian.'
      },
      docs_archaeologist: {
        name: 'Docs Archaeologist',
        description: 'Dig through documentation sources over and over.'
      },
      browser_possession: {
        name: 'Browser Possession',
        description: 'Possess a browser through automation repeatedly.'
      },
      terminal_goblin: {
        name: 'Terminal Goblin',
        description: 'Spend serious time in shell-land.'
      },
      patch_wizard: {
        name: 'Patch Wizard',
        description: 'Bend files to your will with targeted patches.'
      },
      file_archaeologist: {
        name: 'File Archaeologist',
        description: 'Dig through the filesystem with reads and searches.'
      },
      image_whisperer: {
        name: 'Image Whisperer',
        description: 'Use image generation or vision tools enough for visual work.'
      },
      voice_of_the_machine: {
        name: 'Voice Of The Machine',
        description: 'Use text-to-speech or voice tooling repeatedly.'
      },
      model_hopper: {
        name: 'Model Hopper',
        description: 'Switch or inspect providers/models enough to count as a habit.'
      },
      openrouter_enjoyer: {
        name: 'OpenRouter Enjoyer',
        description: 'Route model work through OpenRouter repeatedly.'
      },
      codex_conjurer: {
        name: 'Codex Conjurer',
        description: 'Summon Codex-flavored assistance often enough for a ritual.'
      },
      multi_model_mage: {
        name: 'Multi-Model Mage',
        description: 'Use a real spread of distinct model names across Hermes history.'
      },
      five_model_flight: {
        name: 'Five-Model Flight',
        description: 'Try at least five distinct LLMs instead of marrying the first model that answers.'
      },
      provider_polyglot: {
        name: 'Provider Polyglot',
        description: 'Use models from multiple providers across Hermes history.'
      },
      model_sommelier: {
        name: 'Model Sommelier',
        description: 'Taste enough model/provider conversations to develop preferences.'
      },
      claude_confidant: {
        name: 'Claude Confidant',
        description: 'Bring Claude-flavored reasoning into the workflow repeatedly.'
      },
      gemini_cartographer: {
        name: 'Gemini Cartographer',
        description: 'Map enough Gemini-related workflows to know the terrain.'
      },
      open_weights_pilgrim: {
        name: 'Open Weights Pilgrim',
        description: 'Actually chat with local/open-weight models through Hermes session metadata.'
      },
      toolset_cartographer: {
        name: 'Toolset Cartographer',
        description: 'Navigate Hermes toolsets deliberately instead of treating tools as a blur.'
      },
      config_surgeon: {
        name: 'Config Surgeon',
        description: 'Operate on real config files, manifests, env files, and dashboard settings without flinching.'
      },
      rebase_acrobat: {
        name: 'Rebase Acrobat',
        description: 'Handle real git history surgery: rebase, conflict, merge, fetch, push.'
      },
      test_suite_tamer: {
        name: 'Test Suite Tamer',
        description: 'Run enough verification commands that green text becomes part of the ritual.'
      },
      screenshot_hunter: {
        name: 'Screenshot Hunter',
        description: 'Capture, inspect, and polish visual proof instead of just claiming it works.'
      },
      marathon_operator: {
        name: 'Marathon Operator',
        description: 'Accumulate a serious number of Hermes sessions.'
      },
      weekend_warrior: {
        name: 'Weekend Warrior',
        description: 'Run Hermes on weekends enough times to make it a lifestyle.'
      },
      night_shift_operator: {
        name: 'Night Shift Operator',
        description: 'Run sessions during gremlin hours repeatedly.'
      },
      cache_hit_appreciator: {
        name: 'Cache Hit Appreciator',
        description: 'Notice or benefit from prompt/cache behavior.'
      }
    }
  },
  kanban: {
    loading: 'Loading Kanban board…',
    loadFailed: 'Failed to load Kanban board: ',
    loadFailedHint: 'The backend auto-creates kanban.db on first read. If this persists, check the dashboard logs.',
    board: 'Board',
    newBoard: '+ New board',
    newBoardTitle: 'New board',
    newBoardDescription:
      "Boards let you separate unrelated streams of work — one per project, repo, or domain. Workers on one board never see another board's tasks.",
    slug: 'Slug',
    slugHint: '— lowercase, hyphens, e.g. atm10-server',
    displayName: 'Display name',
    displayNameHint: '(optional)',
    description: 'Description',
    descriptionHint: '(optional)',
    icon: 'Icon',
    iconHint: '(single character or emoji)',
    switchAfterCreate: 'Switch to this board after creating it',
    cancel: 'Cancel',
    creating: 'Creating…',
    createBoard: 'Create board',
    search: 'Search',
    filterCards: 'Filter cards…',
    tenant: 'Tenant',
    allTenants: 'All tenants',
    assignee: 'Assignee',
    model: 'Model',
    modelProfileDefault: 'profile default',
    clickToEditModel: "Click to override the model for this task's next run",
    modelFreeTextPlaceholder: 'model name (empty = profile default)',
    modelLoading: 'loading models…',
    modelProfileDefaultOption: '(profile default)',
    allProfiles: 'All profiles',
    showArchived: 'Show archived',
    lanesByProfile: 'Lanes by profile',
    nudgeDispatcher: 'Nudge dispatcher',
    refresh: 'Refresh',
    selected: 'selected',
    complete: 'Complete',
    archive: 'Archive',
    apply: 'Apply',
    confirm: 'Confirm',
    ok: 'OK',
    bulkConfirmTitle: 'Apply bulk change',
    confirmTitle: 'Confirm change',
    common: {
      confirm: 'Confirm',
      delete: 'Delete'
    },
    clear: 'Clear',
    createTask: 'Create task in this column',
    noTasks: '— no tasks —',
    unassigned: 'unassigned',
    needsAssignee: 'Needs assignee',
    needsAssigneeHint: 'Dependencies are satisfied, but the dispatcher skips this task until you assign a profile.',
    untitled: '(untitled)',
    loadingDetail: 'Loading…',
    addComment: 'Add a comment… (Enter to submit)',
    comment: 'Comment',
    status: 'Status',
    workspace: 'Workspace',
    skills: 'Skills',
    createdBy: 'Created by',
    result: 'Result',
    comments: 'Comments',
    events: 'Events',
    runHistory: 'Run history',
    workerLog: 'Worker log',
    loadingLog: 'Loading log…',
    noWorkerLog: "— no worker log yet (task hasn't spawned or log was rotated away) —",
    noDescription: '— no description —',
    noComments: '— no comments —',
    edit: 'edit',
    save: 'Save',
    dependencies: 'Dependencies',
    parents: 'Parents:',
    children: 'Children:',
    none: 'none',
    addParent: '— add parent —',
    addChild: '— add child —',
    removeDependency: 'Remove dependency',
    block: 'Block',
    unblock: 'Unblock',
    notifyHomeChannels: 'Notify home channels',
    diagnostics: 'Diagnostics',
    hide: 'Hide',
    show: 'Show',
    attention: 'Attention',
    tasksNeedAttention: 'tasks need attention',
    taskNeedsAttention: '1 task needs attention',
    diagnostic: 'diagnostic',
    open: 'Open',
    close: 'Close (Esc)',
    reassignTo: 'Reassign to:',
    copied: 'Copied',
    copyCommand: 'Copy command to clipboard',
    copyCommandPrompt: 'Copy this command:',
    reclaim: 'Reclaim',
    reassign: 'Reassign',
    renderingError: 'Kanban tab hit a rendering error',
    reloadView: 'Reload view',
    wsAuthFailed: 'WebSocket auth failed — reload the page to refresh the session token.',
    markDone: 'Mark {n} task(s) as done?',
    markArchived: 'Archive {n} task(s)?',
    warning: 'Warning',
    phantomIds: 'Phantom ids:',
    active: 'active',
    ended: 'ended',
    noProfile: '(no profile)',
    showAllAttempts: 'Show all attempts',
    sendingUpdates: 'Sending updates to',
    sendNotifications: 'Send completed / blocked / gave_up notifications to',
    archiveBoardConfirm:
      "Archive board '{name}'? It will be moved to boards/_archived/ so you can recover it later. Tasks on this board will no longer appear anywhere in the UI.",
    archiveBoardTitle: 'Archive this board',
    boardSwitcherHint: 'Boards let you separate unrelated streams of work',
    taskCreatedWarning: 'Task created, but: ',
    actionFailed: 'Action failed: ',
    moveFailed: 'Move failed: ',
    bulkFailed: 'Bulk: ',
    bulkMoveFailed: 'Bulk move: {failed} of {total} failed',
    bulkFailedDetails: 'Bulk: {failed} of {total} failed: {details}',
    completionBlockedHallucination: '⚠ Completion blocked — phantom card ids',
    suspectedHallucinatedReferences: '⚠ Prose referenced phantom card ids',
    pickProfileFirst: 'Pick a profile first.',
    unblockedMessage: 'Unblocked {id}. Task is ready for the next tick.',
    unblockFailed: 'Unblock failed: ',
    reclaimedMessage: 'Reclaimed {id}. Task is back to ready.',
    reclaimFailed: 'Reclaim failed: ',
    reassignedMessage: 'Reassigned {id} to {profile}.',
    reassignFailed: 'Reassign failed: ',
    selectForBulk: 'Select for bulk actions',
    clickToEdit: 'Click to edit',
    clickToEditAssignee: 'Click to edit assignee',
    emptyAssignee: '(empty = unassign)',
    columnLabels: {
      triage: 'Triage',
      todo: 'Todo',
      scheduled: 'Scheduled',
      ready: 'Ready',
      running: 'In Progress',
      blocked: 'Blocked',
      done: 'Done',
      archived: 'Archived'
    },
    columnHelp: {
      triage: 'Raw ideas — a specifier will flesh out the spec',
      todo: 'Waiting on dependencies or unassigned',
      scheduled: 'Waiting on a known time delay or scheduled follow-up',
      ready: 'Dependencies satisfied; assign a profile to dispatch',
      running: 'Claimed by a worker — in-flight',
      blocked: 'Worker asked for human input',
      done: 'Completed',
      archived: 'Archived'
    },
    confirmDone: "Mark this task as done? The worker's claim is released and dependent children become ready.",
    confirmArchive: 'Archive this task? It disappears from the default board view.',
    confirmBlocked: "Mark this task as blocked? The worker's claim is released.",
    confirmScheduled: 'Move this task to Scheduled? Use this for known time delays rather than human blockers.',
    confirmDoneMany: "Mark {n} tasks as done? The workers' claims are released and dependent children become ready.",
    confirmArchiveMany: 'Archive {n} tasks? They disappear from the default board view.',
    confirmBlockedMany: "Mark {n} tasks as blocked? The workers' claims are released.",
    completionSummary: 'Completion summary for {label}. This is stored as the task result.',
    completionSummaryThisTask: 'this task',
    completionSummarySelectedTasks: '{count} selected task(s)',
    completionSummaryRequired: 'Completion summary is required before marking a task done.',
    triagePlaceholder: 'Rough idea — AI will spec it…',
    taskTitlePlaceholder: 'New task title…',
    specifier: 'specifier',
    assigneePlaceholder: 'assignee',
    priority: 'Priority',
    skillsPlaceholder: 'skills (optional, comma-separated): translation, github-code-review',
    noParent: '— no parent —',
    workspacePathDir: 'workspace path (required, e.g. ~/projects/my-app)',
    workspacePathOptional: 'workspace path (optional, derived from assignee if blank)',
    logTruncated: '(showing last 100 KB — full log at ',
    logAt: ')',
    newTaskTitle: 'New task — {column}',
    taskTitleLabel: 'Title',
    assigneeLabel: 'Assignee',
    assigneeLabelHint: '(blank = dispatcher picks)',
    skillsLabel: 'Skills',
    skillsLabelHint: '(optional, comma-separated)',
    parentLabel: 'Parent task',
    parentLabelHint: '(child stays blocked until the parent is done)',
    create: 'Create',
    boardSettings: 'Settings',
    boardSettingsTitle: 'Board settings — name, description, and the default project directory new tasks inherit',
    boardSettingsTitleFor: 'Board settings — {name}',
    projectDirectoryOverrideHint:
      'New tasks inherit this as their workspace default; each task can still override it in the create dialog.',
    saving: 'Saving…',
    commentHint: 'Comments reach the worker on its next run or kanban_show() — no need to block the task first.',
    commentHintTitle:
      "Comments are the channel for talking to a task's worker. They land on the thread immediately — no need to block the task first. A running worker picks the thread up on its next kanban_show() or respawn; blocking is only for when you want the worker to STOP and wait for your input.",
    attachments: 'Attachments',
    childResults: 'Child Results',
    clearFilters: 'Clear filters',
    confirmRemoveAttachment: 'Remove this attachment?',
    delete: 'Delete',
    doneNoResult: 'No final result was recorded. Check Run History, Logs, or Child Tasks for the worker output.',
    doneParentNote:
      'This card is an orchestrator / parent task. Review the child results section for the substantive work.',
    finalResult: 'Final Result (run summary)',
    goalMaxTurns: 'max turns (default 20)',
    goalEnabled: 'on',
    goalEnabledMax: 'on (max {turns} turns)',
    goalMode: 'goal mode',
    noAttachments: '— no attachments —',
    noChildResult: 'No result recorded yet.',
    projectDirectory: 'Project directory',
    projectDirectoryExplanation: 'Sets the default location for task files so project output is preserved.',
    projectDirectoryHelp:
      'Git projects use preserved worktrees. Other folders use the directory directly. Leave blank only for temporary work.',
    projectDirectoryHint: '(recommended)',
    projectDirectoryPlaceholder: 'Absolute path to the project folder',
    removeAttachment: 'Remove attachment',
    setPriority: 'Set priority',
    uploadFile: 'Upload file',
    uploading: 'Uploading…',
    workspaceDir: 'Directory — preserved',
    workspaceScratch: 'Temporary — deleted on completion',
    workspaceScratchWarning: 'This workspace and any files left in it are deleted when the task completes.',
    workspaceWorktree: 'Git worktree — preserved',
    trash: {
      confirm: 'Permanently delete this task? This cannot be undone.',
      confirmMany: 'Permanently delete {n} selected tasks? This cannot be undone.',
      confirmTitle: 'Delete task?',
      confirmManyTitle: 'Delete {n} tasks?',
      dropHint: 'Drop to delete'
    },
    hints: {
      assignee:
        'Hermes profile to assign. Leave blank and the dispatcher will pick from available profiles when the task is Ready.',
      boardSwitcher: 'Boards are independent work streams. Each board has its own tasks, tenants, and assignees.',
      clearFilters: 'Clear all active filters (search, tenant, assignee, archived).',
      createBoard: 'Create a new board for an unrelated work stream, project, team, or isolated scratch area.',
      filterAssignee:
        'Filter by assigned Hermes profile. Profiles are the named agent identities that claim and work on tasks.',
      filterArchived: 'Include archived tasks in the board view. Archived tasks are hidden by default.',
      filterSearch: 'Fuzzy-match tasks by id, title, or description across all columns.',
      filterTenant:
        'Tenants are free-form task tags such as customer, project, or team. Set them in the task drawer or with kanban_create.',
      groupRunning: 'Group the Running column by assigned profile.',
      goalMaxTurns: 'Turn budget for the goal loop. Blank uses the backend default of 20.',
      goalMode:
        'Goal mode keeps the worker in the same session until a judge agrees the card is done or the turn budget runs out.',
      hideUntilReload: 'Hide until the next page reload',
      nudgeDispatcher: 'Wake the dispatcher to claim ready tasks now instead of waiting for the next tick.',
      parent: 'Optional parent task. A child stays blocked until the parent is marked done.',
      priority: 'Higher-priority tasks are claimed first by the dispatcher. 0 is the default.',
      refreshBoard: 'Reload the board from the database. The board already refreshes on task events.',
      skills: 'Force-load these skills in addition to the built-in kanban-worker skill.',
      specifier: 'Hermes profile that will spec this task. Leave blank to use the dispatcher configuration.',
      workspace: 'Choose whether task files are temporary or preserved after completion.'
    },
    bulk: {
      applyAssignee: 'Apply the selected assignee to all selected tasks.',
      archive: 'Archive selected tasks. They remain in the database.',
      block: 'Block selected tasks and release active claims.',
      blockConfirm: 'Block {n} selected task(s)?',
      clear: 'Clear selection',
      delete: 'Permanently delete selected tasks. This cannot be undone.',
      deselectAll: 'Deselect all tasks and hide this bar.',
      moveReady: 'Move selected tasks to Ready for dispatch on the next tick.',
      moveTodo: 'Move selected tasks to Todo.',
      reassign: '— reassign —',
      reassignHelp: 'Reassign selected tasks to a different Hermes profile, or unassign them.',
      reclaimFirst: 'Reclaim first',
      reclaimFirstHelp: 'Reclaim active claims before reassigning.',
      selectAll: 'Select all visible',
      selectAllColumn: 'Select all tasks in this column',
      selectAllHelp: 'Select all visible cards across columns.',
      setPriorityHelp: 'Set priority on selected tasks. Higher values are claimed first.',
      unassign: '(unassign)',
      unblock: 'Unblock selected tasks and promote them to Ready.',
      unblockConfirm: 'Unblock {n} selected task(s)?'
    },
    cardHints: {
      assignedProfile: 'Assigned to Hermes profile @{profile}',
      childProgress: '{done} of {total} child tasks done',
      columnTasks: '{count} tasks in this column',
      comments: '{count} comments on this task',
      created: 'Created {time}',
      dependencies: '{parents} parent tasks, {children} child tasks. Children stay blocked until their parent is done.',
      diagnostics: '{count} active diagnostics (severity: {severity}). Open the task for details.',
      noProfile: 'No profile assigned.',
      priority: 'Priority {priority}. Higher-priority tasks are claimed first.',
      selectAllColumn: 'Select all tasks in {column}',
      selectTask: 'Select task {id}',
      task: '{title} — {id} — {status}',
      taskId: 'Task id: {id}. Use it with kanban_show or the Kanban CLI.',
      tenant: 'Tenant: {tenant}. Free-form tag for grouping tasks.'
    },
    boardForm: {
      descriptionPlaceholder: 'What goes on this board?',
      slugRequired: 'A board slug is required.'
    },
    docs: {
      ariaLabel: 'Hermes Kanban documentation',
      open: 'Open Hermes Kanban documentation in a new tab'
    },
    orchestration: {
      auto: 'Auto',
      autoDecomposeLabel: 'Auto-decompose triage tasks',
      autoDescription: 'The dispatcher decomposes new triage tasks automatically.',
      autoGenerate: '⚗ Auto',
      autoGenerateFailed: 'Auto-generate failed: {error}',
      autoModeHelp: 'Automatic orchestration decomposes new triage tasks every tick. Click to switch to Manual.',
      autoReview: 'auto — review',
      configure: 'Configure the Kanban orchestrator and profile routing.',
      defaultAssignee: 'Default assignee',
      defaultProfile: '(default)',
      defaultValue: '(default: {profile})',
      descriptionGenerated: 'Auto-generated description for {profile}.',
      descriptionSaved: 'Description saved for {profile}.',
      generating: 'Generating…',
      label: 'Orchestration',
      loadFailed: 'Failed to load orchestration settings: {error}',
      loading: 'Loading…',
      loadingMode: 'Loading mode…',
      manual: 'Manual',
      manualDescription: 'Triage tasks wait until you click ⚗ Decompose.',
      manualModeHelp:
        'Manual orchestration leaves triage tasks in place until you click ⚗ Decompose. Click to switch to Auto.',
      mode: 'Orchestration mode',
      noDescription: '⚠ no description',
      noProfiles: 'No profiles installed.',
      orchestratorHelp:
        'Owns the root task after fan-out and judges completion. Configure the decomposer model under auxiliary.kanban_decomposer.',
      profile: 'Orchestrator profile',
      profileDescriptionPlaceholder: 'What is this profile good at?',
      profileDescriptions: 'Profile descriptions',
      profileDescriptionsHelp: 'Descriptions guide routing. Click ⚗ to auto-generate, or edit and save.',
      reload: 'Reload',
      resolved: 'Resolved: {profile}',
      saveDescription: 'Save this as a user-authored description',
      saveFailed: 'Save failed: {error}',
      saved: 'Settings saved.',
      settings: 'Orchestration settings'
    },
    run: {
      earlier: '+{count} earlier',
      emptyLog: '(empty)',
      metadata: 'Metadata',
      refreshLog: 'Refresh log'
    },
    taskActions: {
      addChild: '+ child',
      addParent: '+ parent',
      decompose: '⚗ Decompose',
      decomposeFailed: 'Decompose failed: {error}',
      decomposed: 'Decomposed into {count} children: {ids}',
      decomposing: 'Decomposing…',
      editDescription: 'Edit description',
      moveReady: '→ ready',
      moveTodo: '→ todo',
      moveTriage: '→ triage',
      retitled: ' — retitled: {title}',
      singleTask: 'Single task (no fanout){suffix}',
      specify: '✨ Specify',
      specifyFailed: 'Specify failed: {error}',
      specified: 'Specified{suffix}',
      specifying: 'Specifying…',
      unknownError: 'unknown error'
    }
  }
}
