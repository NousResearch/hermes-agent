// 交互结果与错误反馈文案。
export const presentationEn = {
  sessionStatus: {
    heading: 'Hermes TUI Status',
    sessionId: (value: unknown) => `Session ID: ${value}`,
    path: (value: unknown) => `Path: ${value}`,
    project: (value: unknown) => `Project: ${value}`,
    title: (value: unknown) => `Title: ${value}`,
    model: (model: unknown, provider: unknown) => `Model: ${model} (${provider})`,
    created: (value: unknown) => `Created: ${value}`,
    lastActivity: (value: unknown) => `Last Activity: ${value}`,
    tokens: (value: unknown) => `Tokens: ${value}`,
    agentRunning: (value: unknown) => `Agent Running: ${value}`
  },
  compression: {
    aborted: (before: unknown) => `Compression aborted: ${before} messages preserved`,
    fallback: (before: unknown, after: unknown) => `Compressed with fallback: ${before} → ${after} messages`,
    noop: (before: unknown) => `No changes from compression: ${before} messages`,
    done: (before: unknown, after: unknown) => `Compressed: ${before} → ${after} messages`,
    tokensUnchanged: (before: unknown) => `Approx request size: ~${before} tokens (unchanged)`,
    tokensChanged: (before: unknown, after: unknown) => `Approx request size: ~${before} → ~${after} tokens`,
    abortedNote: 'Summary generation failed; no messages were removed.',
    fallbackNote: (count: unknown) =>
      `Summary generation failed; Hermes used limited fallback context and removed ${count} message(s).`,
    denseNote:
      'Note: fewer messages can still raise this estimate when compression rewrites the transcript into denser summaries.',
    reason: (reason: unknown) => `Reason: ${reason}`
  },
  completion: {
    unavailableMeta: 'unavailable',
    gitDiff: 'git diff',
    stagedDiff: 'staged diff',
    attachFile: 'attach file',
    attachFolder: 'attach folder',
    fetchUrl: 'fetch URL',
    gitLog: 'git log',
    directory: 'directory',
    globalMode: 'global mode',
    cycleGlobalMode: 'cycle global mode',
    sectionOverride: 'section override',
    setSection: (section: unknown) => `set ${section}`,
    clearSectionOverride: (section: unknown) => `clear ${section} override`
  }
}
