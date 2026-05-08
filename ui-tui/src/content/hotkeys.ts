import { isMac, isRemoteShell } from '../lib/platform.js'
import type { TerminalCapabilities } from '../lib/terminalCapabilities.js'
import type { TerminalSignals } from '../lib/terminalSignals.js'

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
  [action + '+G / Alt+G', 'open $EDITOR (Alt+G fallback for VSCode/Cursor)'],
  [action + '+L', 'redraw / repaint'],
  [paste + '+V / /paste', 'paste text; /paste attaches clipboard image'],
  ['Tab', 'apply completion'],
  ['↑/↓', 'completions / queue edit / history'],
  ['Ctrl+X', 'delete the queued message you’re editing (Esc cancels edit)'],
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

export function buildHelpHintHotkeys(env: {
  capabilities: TerminalCapabilities
  signals: TerminalSignals
}): [string, string][] {
  const { capabilities, signals } = env
  const rows: [string, string][] = []
  const isDarwin = signals.platform === 'darwin'

  if (isDarwin) {
    rows.push(['Cmd+V', 'paste text; /paste attaches clipboard image'])
  } else if (capabilities.keyboard.pasteShortcutShapes.includes('ctrl+shift+v')) {
    rows.push(['Ctrl+Shift+V', 'paste text; /paste attaches clipboard image'])
  } else {
    rows.push(['Alt+V', 'paste text; /paste attaches clipboard image'])
  }

  if (isDarwin || capabilities.keyboard.copyShortcutShapes.includes('super+c')) {
    rows.push(['Cmd+C', 'copy selection when forwarded by the terminal'])
  }

  if (capabilities.keyboard.copyShortcutShapes.includes('ctrl+shift+c')) {
    rows.push(['Ctrl+Shift+C', 'copy selection'])
  }

  rows.push(['Ctrl+C', 'interrupt / clear draft / exit'])

  if (capabilities.mouse.shiftDragHint) {
    rows.push(['Shift-drag', 'terminal-native selection (when mouse tracking is on)'])
  }

  if (capabilities.transport === 'tmux' || capabilities.layers.includes('tmux')) {
    rows.push(['tmux', `copy uses tmux-buffer (write: ${capabilities.copy.writePath})`])
  }

  return rows
}
