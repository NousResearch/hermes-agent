import { accessSync, constants } from 'node:fs'
import { delimiter, join } from 'node:path'

/**
 * Editor fallback chain when neither $VISUAL nor $EDITOR is set. Mirrors
 * prompt_toolkit's `Buffer.open_in_editor()` picker so the classic CLI and
 * the TUI launch the same editor on a given box.
 */
const FALLBACKS = ['editor', 'nano', 'pico', 'vi', 'emacs']

const isExecutable = (path: string): boolean => {
  try {
    accessSync(path, constants.X_OK)

    return true
  } catch {
    return false
  }
}

/**
 * Resolve the editor invocation argv (without the file argument).
 *
 *   1. $VISUAL / $EDITOR, shell-tokenized so `EDITOR="code --wait"` works
 *   2. on POSIX: first FALLBACKS entry resolvable on $PATH
 *   3. on Windows: `notepad.exe`
 *   4. literal `['vi']` as the last-resort POSIX floor
 */
export const resolveEditor = (
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform
): string[] => {
  const explicit = env.VISUAL ?? env.EDITOR

  if (explicit?.trim()) {
    return explicit.trim().split(/\s+/)
  }

  if (platform === 'win32') {
    return ['notepad.exe']
  }

  const dirs = (env.PATH ?? '').split(delimiter).filter(Boolean)
  const found = FALLBACKS.flatMap(name => dirs.map(d => join(d, name))).find(isExecutable)

  return [found ?? 'vi']
}
