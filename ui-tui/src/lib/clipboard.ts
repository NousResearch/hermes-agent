import { execFile, spawn } from 'node:child_process'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const CLIPBOARD_MAX_BUFFER = 4 * 1024 * 1024
const POWERSHELL_ARGS = ['-NoProfile', '-NonInteractive', '-Command', 'Get-Clipboard -Raw'] as const

type ClipboardRun = typeof execFileAsync

export function isUsableClipboardText(text: null | string): text is string {
  if (!text || !/[^\s]/.test(text)) {
    return false
  }

  if (text.includes('\u0000')) {
    return false
  }

  let suspicious = 0
  for (const ch of text) {
    const code = ch.charCodeAt(0)
    const isControl = code < 0x20 && ch !== '\n' && ch !== '\r' && ch !== '\t'
    if (isControl || ch === '\ufffd') {
      suspicious += 1
    }
  }

  return suspicious <= Math.max(2, Math.floor(text.length * 0.02))
}

function readClipboardCommands(platform: NodeJS.Platform, env: NodeJS.ProcessEnv): Array<{ args: readonly string[]; cmd: string }> {
  if (platform === 'darwin') {
    return [{ cmd: 'pbpaste', args: [] }]
  }

  if (platform === 'win32') {
    return [{ cmd: 'powershell', args: POWERSHELL_ARGS }]
  }

  const attempts: Array<{ args: readonly string[]; cmd: string }> = []

  if (env.WSL_INTEROP) {
    attempts.push({ cmd: 'powershell.exe', args: POWERSHELL_ARGS })
  }

  if (env.WAYLAND_DISPLAY) {
    attempts.push({ cmd: 'wl-paste', args: ['--type', 'text'] })
  }

  attempts.push({ cmd: 'xclip', args: ['-selection', 'clipboard', '-out'] })

  return attempts
}

/**
 * Read plain text from the system clipboard.
 *
 * Uses native platform tools in fallback order:
 * - macOS: pbpaste
 * - Windows: PowerShell Get-Clipboard -Raw
 * - WSL: powershell.exe Get-Clipboard -Raw
 * - Linux Wayland: wl-paste --type text
 * - Linux X11: xclip -selection clipboard -out
 */
export async function readClipboardText(
  platform: NodeJS.Platform = process.platform,
  run: ClipboardRun = execFileAsync,
  env: NodeJS.ProcessEnv = process.env
): Promise<string | null> {
  for (const attempt of readClipboardCommands(platform, env)) {
    try {
      const result = await run(attempt.cmd, [...attempt.args], {
        encoding: 'utf8',
        maxBuffer: CLIPBOARD_MAX_BUFFER,
        windowsHide: true
      })

      if (typeof result.stdout === 'string') {
        return result.stdout
      }
    } catch {
      // Fall through to the next clipboard backend.
    }
  }

  return null
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
