import { isMac, isRemoteShell } from '../lib/platform.js'

const action = isMac ? 'Cmd' : 'Ctrl'
const paste = isMac ? 'Cmd' : 'Alt'

const copyHotkeys: [string, string][] = isMac
  ? [
      ['Cmd+C', 'copy selection'],
      ['Ctrl+C', 'interrupt / clear draft / exit']
    ]
  : isRemoteShell()
    ? [
        ['Cmd+C', 'copy selection when forwarded by the terminal'],
        ['Ctrl+C', 'copy selection / interrupt / clear draft / exit']
      ]
    : [['Ctrl+C', 'copy selection / interrupt / clear draft / exit']]

export const HOTKEYS: [string, string][] = [
  ...copyHotkeys,
  [action + '+D', 'exit'],
  [action + '+G', 'open $EDITOR for prompt'],
  [action + '+L', 'new session (clear)'],
  [paste + '+V / /paste', 'paste text; /paste attaches clipboard image'],
  ['Tab', 'apply completion'],
  ['↑/↓', 'completions / queue edit / history'],
  [action + '+A/E', 'home / end of line'],
  [action + '+Z / ' + action + '+Y', 'undo / redo input edits'],
  [action + '+W', 'delete word'],
  [action + '+U/K', 'delete to start / end'],
  [action + '+←/→', 'jump word'],
  ['Home/End', 'start / end of line'],
  ['Shift+Enter / Alt+Enter', 'insert newline'],
  ['\\+Enter', 'multi-line continuation (fallback)'],
  ['!<cmd>', 'run a shell command (e.g. !ls, !git status)'],
  ['{!<cmd>}', 'interpolate shell output inline (e.g. "branch is {!git branch --show-current}")']
]
