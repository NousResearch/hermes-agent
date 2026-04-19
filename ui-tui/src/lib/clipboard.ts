import { execFile, spawn } from 'node:child_process'
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

/**
 * Write plain text to the system clipboard.
 *
 * On macOS this uses `pbcopy`. On other platforms we intentionally return
 * false for now; non-mac copy still falls back to OSC52.
 */
export async function writeClipboardText(
  text: string,
  platform: NodeJS.Platform = process.platform,
  start: typeof spawn = spawn
): Promise<boolean> {
  if (platform !== 'darwin') {
    return false
  }

  try {
    const ok = await new Promise<boolean>(resolve => {
      const child = start('pbcopy', [], { stdio: ['pipe', 'ignore', 'ignore'], windowsHide: true })

      child.once('error', () => resolve(false))
      child.once('close', code => resolve(code === 0))
      child.stdin.end(text)
    })

    return ok
  } catch {
    return false
  }
}
