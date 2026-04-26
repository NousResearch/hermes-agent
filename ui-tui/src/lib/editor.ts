import { accessSync, constants } from 'node:fs'
import { delimiter, join } from 'node:path'

/**
 * Resolve which editor to launch when the user hits Ctrl+G / Alt+G.
 *
 * Order of preference:
 *   1. $VISUAL / $EDITOR (user's explicit choice)
 *   2. prompt_toolkit-compatible system fallback:
 *      editor → nano → pico → vi → emacs
 *   3. literal `'vi'` so spawnSync still has something to try
 *
 * This intentionally mirrors prompt_toolkit's Buffer.open_in_editor() picker
 * used by the classic CLI. In Cursor/VSCode terminals, nano is a better prompt
 * editing default than dropping casual users into vi's modal interface.
 */
export function resolveEditor(env: NodeJS.ProcessEnv = process.env): string {
  return (
    env.VISUAL ||
    env.EDITOR ||
    findEditor(env.PATH ?? '', 'editor', 'nano', 'pico', 'vi', 'emacs') ||
    'vi'
  )
}

function findEditor(path: string, ...names: string[]): null | string {
  const dirs = path.split(delimiter).filter(Boolean)

  for (const name of names) {
    for (const dir of dirs) {
      const candidate = join(dir, name)

      if (isExecutable(candidate)) {
        return candidate
      }
    }
  }

  return null
}

function isExecutable(path: string): boolean {
  try {
    accessSync(path, constants.X_OK)

    return true
  } catch {
    return false
  }
}
