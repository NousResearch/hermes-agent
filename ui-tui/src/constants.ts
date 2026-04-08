import type { Theme } from './theme.js'
import type { Role, Usage } from './types.js'

export const FACES = [
  '(｡•́︿•̀｡)',
  '(◔_◔)',
  '(¬‿¬)',
  '( •_•)>⌐■-■',
  '(⌐■_■)',
  '(´･_･`)',
  '◉_◉',
  '(°ロ°)',
  '( ˘⌣˘)♡',
  'ヽ(>∀<☆)☆',
  '٩(๑❛ᴗ❛๑)۶',
  '(⊙_⊙)',
  '(¬_¬)',
  '( ͡° ͜ʖ ͡°)',
  'ಠ_ಠ'
]

export const HOTKEYS: [string, string][] = [
  ['Ctrl+C', 'interrupt / clear / exit'],
  ['Ctrl+D', 'exit'],
  ['Ctrl+G', 'open $EDITOR for prompt'],
  ['Ctrl+L', 'clear screen'],
  ['Ctrl+V', 'paste clipboard image (same as /paste)'],
  ['Tab', 'complete /commands (registry-aware)'],
  ['↑/↓', 'queue edit (if queued) / input history'],
  ['PgUp/PgDn', 'scroll messages'],
  ['Esc', 'clear input'],
  ['Ctrl+A/E', 'home / end of line'],
  ['Ctrl+W', 'delete word'],
  ['Ctrl+←/→', 'jump word'],
  ['Home/End', 'start / end of line'],
  ['\\+Enter', 'multi-line continuation'],
  ['!cmd', 'run shell command'],
  ['{!cmd}', 'interpolate shell output inline'],
  ['/voice record', 'start PTT recording'],
  ['/voice stop', 'stop + transcribe']
]

export const INTERPOLATION_RE = /\{!(.+?)\}/g

export const LONG_MSG = 300
export const MAX_CTX = 128_000

export const PLACEHOLDERS = [
  'Ask me anything…',
  'Try "explain this codebase"',
  'Try "write a test for…"',
  'Try "refactor the auth module"',
  'Try "/help" for commands',
  'Try "fix the lint errors"',
  'Try "how does the config loader work?"'
]

export const ROLE: Record<Role, (t: Theme) => { body: string; glyph: string; prefix: string }> = {
  assistant: t => ({ body: t.color.cornsilk, glyph: t.brand.tool, prefix: t.color.bronze }),
  system: t => ({ body: '', glyph: '·', prefix: t.color.dim }),
  tool: t => ({ body: t.color.dim, glyph: '⚡', prefix: t.color.dim }),
  user: t => ({ body: t.color.label, glyph: t.brand.prompt, prefix: t.color.label })
}

export const SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

export const TOOL_VERBS: Record<string, string> = {
  browser: '🌐 browsing',
  clarify: '❓ asking',
  create_file: '📝 creating',
  delegate_task: '🤖 delegating',
  delete_file: '🗑️ deleting',
  execute_code: '⚡ executing',
  image_generate: '🎨 generating',
  list_files: '📂 listing',
  memory: '🧠 remembering',
  patch: '🩹 patching',
  read_file: '📖 reading',
  run_command: '⚙️ running',
  search_code: '🔍 searching',
  search_files: '🔍 searching',
  terminal: '💻 terminal',
  web_search: '🌐 searching',
  write_file: '✏️ writing'
}

export const VERBS = [
  'pondering',
  'contemplating',
  'musing',
  'cogitating',
  'ruminating',
  'deliberating',
  'mulling',
  'reflecting',
  'processing',
  'reasoning',
  'analyzing',
  'computing',
  'synthesizing',
  'formulating',
  'brainstorming'
]

export const ZERO: Usage = { calls: 0, input: 0, output: 0, total: 0 }
