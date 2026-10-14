import { spawnSync } from 'node:child_process'
import { accessSync, constants, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { delimiter, join } from 'node:path'

import { withInkSuspended } from '@hermes/ink'

/**
 * Editor fallback chain when neither $VISUAL nor $EDITOR is set. Mirrors
 * prompt_toolkit's `Buffer.open_in_editor()` picker so the classic CLI and
 * the TUI launch the same editor on a given box.
 */
const FALLBACKS = ['editor', 'nano', 'pico', 'vi', 'emacs']
const EDITOR_SAVE_POLL_MS = 50
const EDITOR_SAVE_STABLE_MS = 200
const EDITOR_SAVE_UNCHANGED_GRACE_MS = 300
const EDITOR_SAVE_TIMEOUT_MS = 2_000

const delay = async (ms: number): Promise<void> => new Promise(resolve => setTimeout(resolve, ms))

const readEditorFileWhenSettled = async (file: string, initial: string): Promise<string> => {
  const startedAt = Date.now()
  let latest = initial
  let stableSince = startedAt
  let observedChange = false

  while (Date.now() - startedAt < EDITOR_SAVE_TIMEOUT_MS) {
    try {
      const current = readFileSync(file, 'utf8')
      const now = Date.now()

      if (current !== latest) {
        latest = current
        stableSince = now
        observedChange = true
      } else if (observedChange) {
        if (now - stableSince >= EDITOR_SAVE_STABLE_MS) {
          return latest
        }
      } else if (now - startedAt >= EDITOR_SAVE_UNCHANGED_GRACE_MS) {
        return latest
      }
    } catch {
      // Atomic-save editors can briefly replace or rename the target. Retry
      // within the same bounded handoff instead of submitting stale contents.
    }

    await delay(EDITOR_SAVE_POLL_MS)
  }

  return latest
}

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

/** Suspend Ink, open ``initial`` in $EDITOR, return the edited text (null if aborted). */
export async function openInEditor(initial: string, suffix = '.txt', basename = 'edit'): Promise<null | string> {
  const dir = mkdtempSync(join(tmpdir(), 'hermes-edit-'))
  const file = join(dir, `${basename}${suffix}`)
  writeFileSync(file, initial)
  const [cmd, ...args] = resolveEditor()
  let status: null | number = null
  let edited: null | string = null

  await withInkSuspended(async () => {
    status = spawnSync(cmd!, [...args, file], { stdio: 'inherit' }).status

    if (status === 0) {
      edited = await readEditorFileWhenSettled(file, initial)
    }
  })

  try {
    return status === 0 ? edited : null
  } finally {
    rmSync(dir, { force: true, recursive: true })
  }
}
