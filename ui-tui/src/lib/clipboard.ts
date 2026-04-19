import { spawnSync, type SpawnSyncOptions } from 'node:child_process'

const DEFAULT_SPAWN_OPTS: SpawnSyncOptions = {
  stdio: ['ignore', 'pipe', 'ignore'],
  encoding: 'utf8'
}

/**
 * Read plain text from the system clipboard.
 *
 * On macOS this uses `pbpaste`. On other platforms we intentionally return
 * null for now; the TUI's text-paste hotkeys are primarily targeted at the
 * macOS clarify/input flow.
 */
export function readClipboardText(
  platform: NodeJS.Platform = process.platform,
  run = spawnSync
): string | null {
  if (platform !== 'darwin') {
    return null
  }

  const result = run('pbpaste', [], DEFAULT_SPAWN_OPTS)

  if (result.status !== 0 || typeof result.stdout !== 'string') {
    return null
  }

  return result.stdout
}
