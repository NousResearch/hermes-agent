export const en = {
  hotkeys: {
    'copy selection': 'copy selection',
    'clear draft / interrupt / exit': 'clear draft / interrupt / exit',
    'copy selection when forwarded by the terminal': 'copy selection when forwarded by the terminal',
    'copy selection / clear draft / interrupt / exit': 'copy selection / clear draft / interrupt / exit',
    exit: 'exit',
    'open $EDITOR (Alt+G fallback for VSCode/Cursor)': 'open $EDITOR (Alt+G fallback for VSCode/Cursor)',
    'redraw / repaint': 'redraw / repaint',
    'paste text; /paste attaches clipboard image': 'paste text; /paste attaches clipboard image',
    'discard draft (recall with ↑)': 'discard draft (recall with ↑)',
    'apply completion': 'apply completion',
    'completions / queue edit / history': 'completions / queue edit / history',
    'open live session switcher (deletes queued message while editing)':
      'open live session switcher (deletes queued message while editing)',
    'expand live agents (keeps your draft)': 'expand live agents (keeps your draft)',
    'collapse / restore live agent preview': 'collapse / restore live agent preview',
    'open model picker (keeps your draft; applies to next turn mid-stream)':
      'open model picker (keeps your draft; applies to next turn mid-stream)',
    'home / end of line': 'home / end of line',
    'undo / redo input edits': 'undo / redo input edits',
    'delete word': 'delete word',
    'kill to line start / end (repeat across lines)': 'kill to line start / end (repeat across lines)',
    'jump word': 'jump word',
    'start / end of line': 'start / end of line',
    'insert newline': 'insert newline',
    'multi-line continuation (fallback)': 'multi-line continuation (fallback)',
    'run a shell command (e.g. !ls, !git status)': 'run a shell command (e.g. !ls, !git status)',
    'interpolate shell output inline (e.g. "branch is {!git branch --show-current}")':
      'interpolate shell output inline (e.g. "branch is {!git branch --show-current}")'
  },
  help: {
    title: '? quick help',
    hint: '  ·  type /help for the full panel  ·  backspace to dismiss',
    commands: 'Common commands',
    hotkeys: 'Hotkeys',
    full: 'full list of commands + hotkeys',
    clear: 'start a new session',
    resume: 'switch live or resume past sessions',
    details: 'control transcript detail level',
    copy: 'copy selection or last assistant message',
    quit: 'exit hermes'
  },
  secrets: {
    sudo: 'sudo password required',
    forVariable: (name: string) => `for ${name}`,
    unlock: (name: string) => `Unlock ${name} for this session`,
    hint: 'master password · hidden · goes to the manager CLI only · Esc keeps it locked'
  },
  setup: {
    title: 'Setup Required',
    description: 'Hermes needs a model provider before the TUI can start a session.',
    model: 'configure provider + model in-place',
    wizard: 'run full first-time setup wizard in-place',
    exit: 'exit and run `hermes setup` manually',
    actions: 'Actions'
  },
  approval: {
    always: 'Always allow',
    deny: 'Deny',
    once: 'Allow once',
    session: 'Allow this session',
    required: (description: string) => `⚠ approval required · ${description}`,
    overflow: (count: number) => `… +${count} more line${count === 1 ? '' : 's'} (full text above)`,
    hint: (count: number) => `↑/↓ select · Enter confirm · 1-${count} quick pick · Esc/Ctrl+C deny`
  },
  clarify: {
    ask: 'ask',
    questions: (count: number) => `${count} questions`,
    skipped: '(skipped)',
    other: 'Other (type your answer)',
    confirmContinue: 'confirm and continue',
    lock: 'lock answer',
    typingHint: (action: string) => `Enter ${action} · Esc back`,
    batchHint: (action: string) => `↑/↓ select · Enter ${action} · Tab/Shift+Tab switch question · Esc/Ctrl+C cancel`,
    answered: (count: number, total: number) => `${count}/${total} answered`,
    inputHint: (back: boolean) => `Enter send · Esc ${back ? 'back' : 'cancel'}`,
    clipboardHint: 'Cmd+C copy · Cmd+V paste · Ctrl+C cancel',
    cancelHint: 'Ctrl+C cancel',
    hint: (count: number) => `↑/↓ select · Enter confirm · 1-${count} quick pick · Esc/Ctrl+C cancel`
  },
  confirm: { no: 'No', yes: 'Yes', hint: '↑/↓ select · Enter confirm · Y/N quick · Esc cancel' },
  agents: {
    queued: 'Queued for child — applied at the next tool boundary.',
    notQueued: 'Not queued: child has finished or is no longer accepting guidance.',
    error: (error: string) => `Not queued: ${error}`,
    steer: (id: string) => `Steer ${id}`,
    guidance: 'Guidance queues at the next tool boundary; current work is not interrupted.',
    queueing: 'Queueing…',
    hint: 'Enter queue · Esc back · main composer draft is preserved',
    loading: 'Loading live transcript…',
    lastLines: '[last 16 KiB]',
    unavailable: 'Live transcript unavailable; child may have finished. Progress and output remain below.',
    refreshFailed: 'Could not refresh live transcript.',
    live: 'Live transcript'
  },
  queue: {
    title: (count: number) => `queued (${count})`,
    editing: (index: number) => ` · editing ${index} · Ctrl+X delete · Esc cancel`,
    more: (count: number) => `…and ${count} more`
  }
}

export type Translations = {
  [Section in keyof typeof en]: {
    [Key in keyof (typeof en)[Section]]: (typeof en)[Section][Key] extends (...args: infer Args) => string
      ? (...args: Args) => string
      : string
  }
}
