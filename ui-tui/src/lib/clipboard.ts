import { execFile, spawn } from 'node:child_process'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const CLIPBOARD_MAX_BUFFER = 4 * 1024 * 1024

// One budget for a whole fallback chain. A Wayland helper waiting on compositor focus
// (GNOME without data-control) never exits; unbounded, it froze paste and leaked a
// helper per copy. Past the budget the caller falls back (OSC 52 / image paste).
export const CLIPBOARD_TIMEOUT_MS = 5000

// PowerShell read: base64-encode the clipboard content to avoid ANSI codepage
// corruption (same problem as the write path — see comment at line 94).
const POWERSHELL_READ_ARGS = [
  '-NoProfile',
  '-NonInteractive',
  '-Command',
  '[Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes((Get-Clipboard -Raw)))'
] as const

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

function readClipboardCommands(
  platform: NodeJS.Platform,
  env: NodeJS.ProcessEnv
): Array<{ args: readonly string[]; cmd: string; base64?: boolean }> {
  if (platform === 'darwin') {
    return [{ cmd: 'pbpaste', args: [] }]
  }

  if (platform === 'win32') {
    return [{ cmd: 'powershell', args: POWERSHELL_READ_ARGS, base64: true }]
  }

  const attempts: Array<{ args: readonly string[]; cmd: string; base64?: boolean }> = []

  if (env.WSL_INTEROP || env.WSL_DISTRO_NAME) {
    attempts.push({ cmd: 'powershell.exe', args: POWERSHELL_READ_ARGS, base64: true })
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
  env: NodeJS.ProcessEnv = process.env,
  timeoutMs: number = CLIPBOARD_TIMEOUT_MS
): Promise<string | null> {
  const deadline = Date.now() + timeoutMs

  for (const attempt of readClipboardCommands(platform, env)) {
    const remaining = deadline - Date.now()

    if (remaining <= 0) {
      break
    }

    try {
      const result = await run(attempt.cmd, [...attempt.args], {
        encoding: 'utf8',
        killSignal: 'SIGKILL',
        maxBuffer: CLIPBOARD_MAX_BUFFER,
        timeout: remaining,
        windowsHide: true
      })

      if (typeof result.stdout === 'string') {
        if (attempt.base64) {
          return Buffer.from(result.stdout.trim(), 'base64').toString('utf8')
        }

        return result.stdout
      }
    } catch {
      // Fall through to the next clipboard backend.
    }
  }

  return null
}

// PowerShell on Windows/WSL decodes piped stdin with the system ANSI code
// page (e.g. CP936), not UTF-8, so $input-based writes mangle CJK/emoji. We
// instead base64-encode the UTF-8 bytes and pass them as a -Command argument,
// decoding with UTF8.GetString — this removes the stdin-encoding variable
// entirely (also immune to BOM injection on redirect). PowerShell entries set
// stdin=false; every other backend reads UTF-8 stdin natively.
type WriteCmd = { args: readonly string[]; cmd: string; stdin: boolean }

function _powershellWriteScript(b64: string): string {
  return `Set-Clipboard -Value ([System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('${b64}')))`
}

function writeClipboardCommands(platform: NodeJS.Platform, env: NodeJS.ProcessEnv): WriteCmd[] {
  if (platform === 'darwin') {
    return [{ cmd: 'pbcopy', args: [], stdin: true }]
  }

  if (platform === 'win32') {
    return [{ cmd: 'powershell', args: ['-NoProfile', '-NonInteractive'], stdin: false }]
  }

  const attempts: WriteCmd[] = []

  if (env.WSL_INTEROP || env.WSL_DISTRO_NAME) {
    attempts.push({ cmd: 'powershell.exe', args: ['-NoProfile', '-NonInteractive'], stdin: false })
  }

  if (env.WAYLAND_DISPLAY) {
    attempts.push({ cmd: 'wl-copy', args: ['--type', 'text/plain'], stdin: true })
  }

  attempts.push({ cmd: 'xclip', args: ['-selection', 'clipboard', '-in'], stdin: true })
  attempts.push({ cmd: 'xsel', args: ['--clipboard', '--input'], stdin: true })

  return attempts
}

/**
 * Write plain text to the system clipboard.
 *
 * Tries native platform tools in fallback order:
 * - macOS: pbcopy
 * - Windows: PowerShell Set-Clipboard
 * - WSL: powershell.exe Set-Clipboard
 * - Linux Wayland: wl-copy --type text/plain
 * - Linux X11: xclip -selection clipboard -in
 * - Linux X11 alt: xsel --clipboard --input
 *
 * Returns true if at least one backend succeeded, false otherwise
 * (callers should fall back to OSC52 on false).
 */
export async function writeClipboardText(
  text: string,
  platform: NodeJS.Platform = process.platform,
  start: typeof spawn = spawn,
  env: NodeJS.ProcessEnv = process.env,
  timeoutMs: number = CLIPBOARD_TIMEOUT_MS
): Promise<boolean> {
  const deadline = Date.now() + timeoutMs

  for (const cmdEntry of writeClipboardCommands(platform, env)) {
    const remaining = deadline - Date.now()

    if (remaining <= 0) {
      return false
    }

    try {
      const ok = await new Promise<boolean>(resolve => {
        const args = cmdEntry.stdin
          ? [...cmdEntry.args]
          : [...cmdEntry.args, '-Command', _powershellWriteScript(Buffer.from(text, 'utf8').toString('base64'))]

        const child = start(cmdEntry.cmd, args, {
          stdio: [cmdEntry.stdin ? 'pipe' : 'ignore', 'ignore', 'ignore'],
          windowsHide: true
        })

        const timer = setTimeout(() => {
          child.kill('SIGKILL')
          resolve(false)
        }, remaining)

        const settle = (ok: boolean) => {
          clearTimeout(timer)
          resolve(ok)
        }

        child.unref()
        child.once('error', () => settle(false))
        child.once('close', (code: number | null) => settle(code === 0))

        if (cmdEntry.stdin) {
          child.stdin?.end(text)
        }
      })

      if (ok) {
        return true
      }
    } catch {
      // Fall through to the next clipboard backend.
    }
  }

  return false
}
