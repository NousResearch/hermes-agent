import { isMac, isRemoteShell } from '../lib/platform.js'

const action = isMac ? 'Cmd' : 'Ctrl'
const paste = isMac ? 'Cmd' : 'Alt'

const copyHotkeys: [string, string][] = isMac
  ? [
      ['Cmd+C', 'copy selection'],
      ['Ctrl+C', 'clear draft / interrupt / exit']
    ]
  : isRemoteShell()
    ? [
        ['Cmd+C', 'copy selection when forwarded by the terminal'],
        ['Ctrl+C', 'copy selection / clear draft / interrupt / exit']
      ]
    : [['Ctrl+C', 'copy selection / clear draft / interrupt / exit']]

export const HOTKEYS: [string, string][] = [
  ...copyHotkeys,
  [action + '+D', 'exit'],
  [action + '+G / Alt+G', 'open $EDITOR (Alt+G fallback for VSCode/Cursor)'],
  [action + '+L', 'redraw / repaint'],
  [paste + '+V / /paste', 'paste text; /paste attaches clipboard image'],
  ['Esc Esc', 'discard draft (recall with ↑)'],
  ['Tab', 'apply completion'],
  ['↑/↓', 'completions / queue edit / history'],
  ['Ctrl+X', 'open live session switcher (deletes queued message while editing)'],
  ['Ctrl+O', 'open model picker (keeps your draft; applies to next turn mid-stream)'],
  [action + '+A/E', 'home / end of line'],
  [action + '+Z / ' + action + '+Y', 'undo / redo input edits'],
  [action + '+W', 'delete word'],
  [action + '+U/K', 'kill to line start / end (repeat across lines)'],
  [action + '+←/→', 'jump word'],
  ['Home/End', 'start / end of line'],
  ['Shift+Enter / Alt+Enter', 'insert newline'],
  ['\\+Enter', 'multi-line continuation (fallback)'],
  ['!<cmd>', 'run a shell command (e.g. !ls, !git status)'],
  ['{!<cmd>}', 'interpolate shell output inline (e.g. "branch is {!git branch --show-current}")']
]
