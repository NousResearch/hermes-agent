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
 * Resolve the editor to launch when the user hits Ctrl+G / Alt+G.
 *
 *   1. $VISUAL / $EDITOR (user's explicit choice)
 *   2. first FALLBACKS entry resolvable on $PATH
 *   3. literal `'vi'` so spawnSync still has something to try
 */
export const resolveEditor = (env: NodeJS.ProcessEnv = process.env): string => {
  if (env.VISUAL) {
    return env.VISUAL
  }

  if (env.EDITOR) {
    return env.EDITOR
  }

  const dirs = (env.PATH ?? '').split(delimiter).filter(Boolean)

  return FALLBACKS.flatMap(name => dirs.map(d => join(d, name))).find(isExecutable) ?? 'vi'
}
