export const HOTKEYS: [string, string][] = [
  ['Ctrl+C', 'interrupt / clear draft / exit'],
  ['Ctrl+D', 'exit'],
  ['Ctrl+G', 'open $EDITOR for prompt'],
  ['Ctrl+L', 'new session (clear)'],
  ['Alt+V / /paste', 'paste clipboard image'],
  ['Tab', 'apply completion'],
  ['↑/↓', 'completions / queue edit / history'],
  ['Ctrl+A/E', 'home / end of line'],
  ['Ctrl+Z / Ctrl+Y', 'undo / redo input edits'],
  ['Ctrl+W', 'delete word'],
  ['Ctrl+U/K', 'delete to start / end'],
  ['Ctrl+←/→', 'jump word'],
  ['Home/End', 'start / end of line'],
  ['Shift+Enter / Alt+Enter', 'insert newline'],
  ['\\+Enter', 'multi-line continuation (fallback)'],
  ['!cmd', 'run shell command'],
  ['{!cmd}', 'interpolate shell output inline']
]
