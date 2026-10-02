// App shell: status line, composer, keybind hints, banner, dock, transcript rows.
// Owned namespaces: `status`, `composer`, `hints`, `help`, `banner`, `transcript`, `dock`.
// (`content` lives in ./content.ts, chat/transcript bits in ./chatBits.ts.)

export const chromeEn = {
  status: {
    summoning: 'summoning hermes…',
    ready: 'ready',
    running: 'running…',
    mcpReloaded: 'MCP reloaded after config change',
    compacting: 'compacting',
    devCredits: ' (dev credits)',
    sessionCount: (count: number) => `${count} ${count === 1 ? 'session' : 'sessions'}`,
    paused: '⏸ paused',
    resumesWhenSubagentFinishes: '↩ resumes when subagent finishes',
    resumesWhenSubagentsFinish: (count: number) => `↩ resumes when ${count} subagents finish`,
    bgTasks: (count: number) => `${count} bg`,
    compressions: (count: number) => `cmp ${count}`,
    fast: 'fast'
  },
  composer: {
    interruptHint: 'Ctrl+C to interrupt…',
    placeholders: {
      ask: 'Ask me anything…',
      explain: 'Try "explain this codebase"',
      test: 'Try "write a test for…"',
      refactor: 'Try "refactor the auth module"',
      help: 'Try "/help" for commands',
      lint: 'Try "fix the lint errors"',
      config: 'Try "how does the config loader work?"'
    }
  },
  hints: {
    copySelection: 'copy selection',
    ctrlCMac: 'clear draft / interrupt / exit',
    copySelectionForwarded: 'copy selection when forwarded by the terminal',
    ctrlC: 'copy selection / clear draft / interrupt / exit',
    exit: 'exit',
    openEditor: 'open $EDITOR (Alt+G fallback for VSCode/Cursor)',
    redraw: 'redraw / repaint',
    paste: 'paste text; /paste attaches clipboard image',
    discardDraft: 'discard draft (recall with ↑)',
    applyCompletion: 'apply completion',
    arrows: 'completions / queue edit / history',
    sessionSwitcher: 'open live session switcher (deletes queued message while editing)',
    expandAgents: 'expand live agents (keeps your draft)',
    collapseAgents: 'collapse / restore live agent preview',
    modelPicker: 'open model picker (keeps your draft; applies to next turn mid-stream)',
    homeEnd: 'home / end of line',
    undoRedo: 'undo / redo input edits',
    deleteWord: 'delete word',
    killLine: 'kill to line start / end (repeat across lines)',
    jumpWord: 'jump word',
    lineStartEnd: 'start / end of line',
    newline: 'insert newline',
    continuation: 'multi-line continuation (fallback)',
    shell: 'run a shell command (e.g. !ls, !git status)',
    interpolate: 'interpolate shell output inline (e.g. "branch is {!git branch --show-current}")'
  },
  help: {
    quickHelp: '? quick help',
    quickHelpTail: '  ·  type /help for the full panel  ·  backspace to dismiss',
    commonCommands: 'Common commands',
    hotkeys: 'Hotkeys',
    commands: {
      help: 'full list of commands + hotkeys',
      clear: 'start a new session',
      resume: 'switch live or resume past sessions',
      details: 'control transcript detail level',
      copy: 'copy selection or last assistant message',
      quit: 'exit hermes'
    }
  }
}
