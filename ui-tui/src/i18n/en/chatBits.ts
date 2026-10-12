// Chat bits: the banner/session panel (branding.tsx), the reasoning + tool
// trail tree (thinking.tsx), transcript rows (messageLine.tsx), the todo
// panel, the queued-messages strip, and the human-facing bootstrap messages
// in entry.tsx. Owned namespace: `chatBits`.
//
// loaders.tsx and banner.ts carry no prose (glyph runs + brand art only), so
// they have no leaves here. Brand names ("Hermes", "Nous Research"), slash
// command names and hotkey chords stay as-is inside the sentences that carry
// them; the sentence itself is the translatable unit.

export const chatBitsEn = {
  chatBits: {
    branding: {
      tagFull: 'Nous Research · Messenger of the Digital Gods',
      tagMid: 'Messenger of the Digital Gods',
      scanningSkills: 'scanning skills',
      moreCategories: (count: number) => `(and ${count} more categories…)`,
      moreToolsets: (count: number) => `(and ${count} more toolsets…)`,
      toolsOne: (count: number) => `${count} tool`,
      toolsOther: (count: number) => `${count} tools`,
      lazy: '(lazy)',
      disabled: 'disabled',
      connecting: 'connecting',
      configured: 'configured',
      failed: 'failed',
      noSystemPrompt: 'No system prompt loaded.',
      sessionLabel: 'Session: ',
      availableTools: 'Available Tools',
      availableSkills: 'Available Skills',
      inCategoriesOne: (count: number) => `in ${count} category`,
      inCategoriesOther: (count: number) => `in ${count} categories`,
      systemPrompt: 'System Prompt',
      charsSuffix: (count: string) => `— ${count} chars`,
      mcpServers: 'MCP Servers',
      connected: 'connected',
      // `count` is a number, or '…' while a lazy boot is still counting.
      toolsSummary: (count: number | string) => `${count} tools`,
      skillsSummary: (count: number | string) => `${count} skills`,
      mcpSummary: (count: number) => `${count} MCP`,
      helpHint: '/help for commands',
      commitsBehindOne: (count: number) => `! ${count} commit behind`,
      commitsBehindOther: (count: number) => `! ${count} commits behind`,
      runPrefix: ' - run ',
      toUpdate: ' to update'
    },
    thinking: {
      thinking: 'Thinking',
      toolCalls: 'Tool calls',
      progress: 'Progress',
      spawned: 'Spawned',
      spawnTree: 'Spawn tree',
      activity: 'Activity',
      subagentFallback: (index: number) => `Subagent ${index}`,
      statusQueued: 'queued',
      statusRunning: 'running',
      toolsOne: (count: number) => `${count} tool`,
      toolsOther: (count: number) => `${count} tools`,
      tokShort: (tokens: string) => `${tokens} tok`,
      subtreeTools: (count: number) => `+${count}t sub`,
      // {0} = depth of the spawned children, {1} = total descendants.
      spawnedSuffix: (depth: number, total: number) => `d${depth} · ${total} total`,
      drafting: 'drafting...',
      analyzingToolOutput: 'analyzing tool output…',
      argsHeader: 'Args:',
      approxTokens: (count: string) => `~${count} tokens`,
      approxTotal: (count: string) => `~${count} total`,
      agentsToMonitor: '(/agents to monitor)',
      agentsHint: '(/agents)'
    },
    messageLine: {
      emptyToolResult: '(empty tool result)',
      systemMessage: '(system message)',
      chars: (count: string) => `${count} chars`,
      longMessage: '[long message]',
      response: 'Response'
    },
    todo: {
      title: 'Todo',
      incompleteOne: (pending: number) => `· incomplete · ${pending} still pending`,
      incompleteOther: (pending: number) => `· incomplete · ${pending} still pending/in_progress`
    },
    queued: {
      header: (count: number) => `queued (${count})`,
      editing: (position: number) => ` · editing ${position} · Ctrl+X delete · Esc cancel`,
      more: (count: number) => `…and ${count} more`
    },
    entry: {
      noTty: 'hermes-tui: no TTY',
      // {0} = memory level ('high' | 'critical'), {1} = formatted heap size, {2} = dump path.
      memoryDump: (level: string, heap: string, path: string) =>
        `hermes-tui: ${level} memory (${heap}) — auto heap dump → ${path}`,
      dumpFailed: '(failed)',
      exitingOom: 'hermes-tui: exiting to avoid OOM; restart to recover',
      heapClimbing: (heap: string) =>
        `hermes-tui: heap climbing fast (${heap}) — a large tool output or long session may be straining memory`
    }
  }
}
