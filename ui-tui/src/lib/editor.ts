import { accessSync, constants } from 'node:fs'
import { delimiter, join } from 'node:path'

/**
 * Resolve which editor to launch when the user hits Ctrl+G / Alt+G.
 *
 * Order of preference:
 *   1. $VISUAL / $EDITOR (user's explicit choice)
 *   2. first executable found on $PATH from `nvim` → `vim` → `vi` → `nano`
 *   3. literal `'vi'` so spawnSync still has something to try
 *
 * Mirrors the override on `input_area.buffer._open_file_in_editor` in cli.py
 * — both surfaces should pick the same editor so the CLI/TUI handoff
 * doesn't surprise the user with nano in one and vim in the other.
 */
export function resolveEditor(env: NodeJS.ProcessEnv = process.env): string {
  return env.VISUAL || env.EDITOR || findExecutable(env.PATH ?? '', 'nvim', 'vim', 'vi', 'nano') || 'vi'
}

function findExecutable(path: string, ...names: string[]): null | string {
  const dirs = path.split(delimiter).filter(Boolean)

  for (const name of names) {
    for (const dir of dirs) {
      const candidate = join(dir, name)

      try {
        accessSync(candidate, constants.X_OK)

        return candidate
      } catch {
        // not executable / not present; try next
      }
    }
  }

  return null
}
