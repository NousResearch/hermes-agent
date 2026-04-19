import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)

/**
 * Read plain text from the system clipboard.
 *
 * On macOS this uses `pbpaste`. On other platforms we intentionally return
 * null for now; the TUI's text-paste hotkeys are primarily targeted at the
 * macOS clarify/input flow.
 */
export async function readClipboardText(
  platform: NodeJS.Platform = process.platform,
  run: typeof execFileAsync = execFileAsync
): Promise<string | null> {
  if (platform !== 'darwin') {
    return null
  }

  try {
    const result = await run('pbpaste', [], { encoding: 'utf8', windowsHide: true })

    return typeof result.stdout === 'string' ? result.stdout : null
  } catch {
    return null
  }
}
