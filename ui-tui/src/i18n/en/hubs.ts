// Hubs: the agents overlay (spawn tree / replay diff), the live agents dock,
// per-agent steer/tail controls, the Skills Hub and the Plugins Hub.
// Owned namespace: `hubs`.
//
// Hotkey chords inside hint sentences (`Enter`, `Esc`, `Ctrl+T`, `q`…) are
// tokens: the whole sentence is one leaf so a pack can reorder around them.
// Agent/process/plugin state values ('running', 'enabled'…) are compared in
// code and only mapped to these labels at render time.

export const hubsEn = {
  hubs: {
    // Shared list-window affordances for the Skills / Plugins hubs.
    list: {
      moreAbove: (count: number) => ` ↑ ${count} more`,
      moreBelow: (count: number) => ` ↓ ${count} more`,
      errorLine: (message: string) => `error: ${message}`
    },
    agents: {
      title: {
        spawnTree: 'Spawn tree',
        pausedSuffix: ' · ⏸ paused',
        lastTurn: 'Last turn',
        // {0} replay index, {1} history length
        replay: (index: number, total: number) => `Replay ${index}/${total}`,
        finishedAt: (time: string) => ` · finished ${time}`
      },
      // {0} max spawn depth, {1} max concurrent children
      caps: (depth: number, children: string) => `caps d${depth}/${children}`,
      inheritModel: 'inherit',
      subagentFallback: 'subagent',
      empty: 'No subagents this turn. Trigger delegate_task to populate the tree.',
      timeline: 'Timeline',
      processes: 'Processes',
      sort: {
        depthFirst: 'spawn order',
        durationDesc: 'slowest',
        status: 'status',
        toolsDesc: 'busiest'
      },
      filter: {
        all: 'all',
        failed: 'failed',
        leaf: 'leaves',
        running: 'running'
      },
      status: {
        completed: 'completed',
        error: 'error',
        failed: 'failed',
        interrupted: 'interrupted',
        queued: 'queued',
        running: 'running',
        timeout: 'timeout'
      },
      section: {
        budget: 'Budget',
        files: 'Files',
        toolCalls: 'Tool calls',
        output: 'Output',
        progress: 'Progress',
        summary: 'Summary'
      },
      detail: {
        depth: 'depth',
        model: 'model',
        toolsets: 'toolsets',
        tools: 'tools',
        // {0} own tool count, {1} subtree tool count
        toolsValue: (own: number, subtree: number) => `${own} (subtree ${subtree})`,
        subtree: 'subtree',
        // {0} descendant count, {1} max depth below, {2} active count
        subtreeValueOne: (count: number, depth: number, active: number) => `${count} agent · d${depth} · ⚡${active}`,
        subtreeValueOther: (count: number, depth: number, active: number) =>
          `${count} agents · d${depth} · ⚡${active}`,
        elapsed: 'elapsed',
        iteration: 'iteration',
        apiCalls: 'api calls',
        tokens: 'tokens',
        // {0} formatted input tokens, {1} formatted output tokens
        tokensValue: (input: string, output: string) => `${input} in · ${output} out`,
        reasoningSuffix: (tokens: string) => ` · ${tokens} reasoning`,
        subtreeTokens: 'subtree tokens',
        filesMore: (count: number) => `…+${count} more`
      },
      diff: {
        title: 'Replay diff',
        subtitle: 'baseline vs candidate · esc/q close',
        baseline: 'A · baseline',
        candidate: 'B · candidate',
        delta: 'Δ',
        agents: 'agents',
        tools: 'tools',
        depth: 'depth',
        duration: 'duration',
        tokens: 'tokens'
      },
      flash: {
        turnFinished: 'turn finished · inspect freely · q to close',
        replayLocked: 'replay mode — controls disabled',
        killing: (id: string) => `killing ${id}`,
        notFound: (id: string) => `not found: ${id}`,
        killFailed: (id: string) => `kill failed: ${id}`,
        killingSubtreeOne: (count: number) => `killing subtree · ${count} node`,
        killingSubtreeOther: (count: number) => `killing subtree · ${count} nodes`,
        spawningPaused: 'spawning paused',
        spawningResumed: 'spawning resumed',
        pauseFailed: 'pause failed',
        liveTurn: 'live turn',
        // {0} replay index, {1} history length
        replay: (index: number, total: number) => `replay · ${index}/${total}`
      },
      hint: {
        controlsLocked: ' · controls locked',
        // {0} is `pause` / `resume` below
        controls: (pauseVerb: string) => ` · e steer · t tail · x stop · X subtree · p ${pauseVerb}`,
        pause: 'pause',
        resume: 'resume',
        footerReplay: 'Enter/d detail · e steer · x stop · Esc back',
        footerLive: 'Enter/t tail · d detail · e steer · x stop · Esc back',
        listNavReplay: 'Enter/→ detail',
        listNavLive: 'Enter tail · d/→ detail',
        // {0} listNavReplay/listNavLive, {1} controls hint, {2} sort label, {3} filter label, {4} history hint
        list: (nav: string, controls: string, sort: string, filter: string, history: string) =>
          `↑↓/jk move · g/G top/bottom · ${nav}${controls} · s sort:${sort} · f filter:${filter}${history} · q close`,
        // {0} replay index, {1} history length
        history: (index: number, total: number) => ` · [ / ] history ${index}/${total}`,
        detail: (controls: string) =>
          `↑↓/jk scroll · PgUp/PgDn page · g/G top/bottom · Esc/← back to list${controls} · q close`
      }
    },
    agentsPanel: {
      running: (count: number) => `${count} running`,
      done: (count: number) => `${count} done`,
      liveAgents: (count: number) => `${count} live agents`,
      procs: (count: number) => `${count} procs`,
      procsDone: (count: number) => `${count} done`,
      moreHidden: (count: number) => ` · +${count} more`,
      processes: 'Processes',
      collapsedHint: ' · Ctrl+T expand · Ctrl+R restore',
      expandedHint: ' · Ctrl+T expand · Ctrl+R collapse'
    },
    agentControls: {
      queued: 'Queued for child — applied at the next tool boundary.',
      notQueued: 'Not queued: child has finished or is no longer accepting guidance.',
      notQueuedError: (message: string) => `Not queued: ${message}`,
      steerTitle: (id: string) => `Steer ${id}`,
      steerIntro: 'Guidance queues at the next tool boundary; current work is not interrupted.',
      queueing: 'Queueing…',
      steerHint: 'Enter queue · Esc back · main composer draft is preserved',
      loadingTranscript: 'Loading live transcript…',
      truncatedPrefix: '[last 16 KiB]',
      transcriptUnavailable: 'Live transcript unavailable; child may have finished. Progress and output remain below.',
      transcriptRefreshFailed: 'Could not refresh live transcript.',
      transcriptTitle: 'Live transcript'
    },
    skills: {
      title: 'Skills Hub',
      loading: 'loading skills…',
      loadingOne: 'loading…',
      installing: 'installing…',
      selectCategory: 'select a category',
      // {0} category name, {1} skill count
      categoryRow: (category: string, count: number) => `${category} · ${count} skills`,
      skillCount: (count: number) => `${count} skill(s)`,
      emptyCategory: 'no skills in this category',
      pathLine: (path: string) => `path: ${path}`,
      hintCancel: 'Esc/q cancel',
      hintCategory: '↑/↓ select · Enter open · 1-9,0 quick · Esc/q cancel',
      hintSkill: '↑/↓ select · Enter open · 1-9,0 quick · Esc back · q close',
      hintSkillEmpty: 'Esc back · q close',
      hintActions: 'i reinspect · x reinstall · Enter/Esc back · q close'
    },
    plugins: {
      title: 'Plugins Hub',
      loading: 'loading plugins…',
      updating: 'updating…',
      empty: 'no plugins installed',
      // `hermes plugins install owner/repo` is a CLI command: keep it verbatim.
      installHint: 'install: hermes plugins install owner/repo',
      status: {
        disabled: 'disabled',
        notEnabled: 'not enabled'
      },
      bundledTag: ' [bundled]',
      userScope: (count: number) => `${count} user plugin(s)`,
      bundledSuffix: (count: number) => ` · +${count} bundled (Tab)`,
      allScope: (count: number) => `all ${count} plugins`,
      hintClose: 'Esc/q close',
      hintList: '↑/↓ select · Enter/Space toggle · Tab user/all · 1-9,0 quick · Esc/q close'
    }
  }
}
