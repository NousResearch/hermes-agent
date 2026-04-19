import { isMac } from '../lib/platform.js'

const mod = isMac ? 'Cmd' : 'Ctrl'
const pasteMod = isMac ? 'Cmd' : 'Alt'

export const HOTKEYS: [string, string][] = [
  [mod + '+C / ' + mod + '+Shift+C', 'copy selection'],
  [mod + '+D', 'exit'],
  [mod + '+G', 'open $EDITOR for prompt'],
  [mod + '+L', 'new session (clear)'],
  [pasteMod + '+V / /paste', 'paste clipboard image'],
  ['Tab', 'apply completion'],
  ['↑/↓', 'completions / queue edit / history'],
  [mod + '+A/E', 'home / end of line'],
  [mod + '+Z / ' + mod + '+Y', 'undo / redo input edits'],
  [mod + '+W', 'delete word'],
  [mod + '+U/K', 'delete to start / end'],
  [mod + '+←/→', 'jump word'],
  ['Home/End', 'start / end of line'],
  ['Shift+Enter / Alt+Enter', 'insert newline'],
  ['\\+Enter', 'multi-line continuation (fallback)'],
  ['!cmd', 'run shell command'],
  ['{!cmd}', 'interpolate shell output inline']
]
