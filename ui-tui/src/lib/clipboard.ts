import { execFile, spawn } from 'node:child_process'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const CLIPBOARD_MAX_BUFFER = 4 * 1024 * 1024
const CLIPBOARD_IMAGE_MAX_BUFFER = 36 * 1024 * 1024
/** Bound stalled clipboard owners (xclip/wl-paste) so paste cannot hang the composer. */
const CLIPBOARD_IMAGE_TIMEOUT_MS = 3_000
const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

// PowerShell read: base64-encode the clipboard content to avoid ANSI codepage
// corruption (same problem as the write path — see comment at line 94).
const POWERSHELL_READ_ARGS = [
  '-NoProfile',
  '-NonInteractive',
  '-Command',
  '[Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes((Get-Clipboard -Raw)))'
] as const

type ClipboardRun = typeof execFileAsync

export interface ClipboardImagePayload {
  contentBase64: string
  filename: string
}

interface ClipboardImageCommand {
  args: readonly string[]
  cmd: string
  output: 'base64' | 'binary'
}

const POWERSHELL_READ_IMAGE_ARGS = [
  '-NoProfile',
  '-NonInteractive',
  '-Command',
  'Add-Type -AssemblyName System.Windows.Forms;' +
    'Add-Type -AssemblyName System.Drawing;' +
    '$img = [System.Windows.Forms.Clipboard]::GetImage();' +
    'if ($null -eq $img) { exit 1 };' +
    '$ms = New-Object System.IO.MemoryStream;' +
    '$img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png);' +
    '[System.Convert]::ToBase64String($ms.ToArray())'
] as const

function readClipboardImageCommands(platform: NodeJS.Platform, env: NodeJS.ProcessEnv): ClipboardImageCommand[] {
  if (platform === 'darwin') {
    return [{ cmd: 'pngpaste', args: ['-'], output: 'binary' }]
  }

  if (platform === 'win32') {
    return [{ cmd: 'powershell', args: POWERSHELL_READ_IMAGE_ARGS, output: 'base64' }]
  }

  const attempts: ClipboardImageCommand[] = []

  if (env.WSL_INTEROP || env.WSL_DISTRO_NAME) {
    attempts.push({ cmd: 'powershell.exe', args: POWERSHELL_READ_IMAGE_ARGS, output: 'base64' })
  }

  if (env.WAYLAND_DISPLAY) {
    attempts.push({ cmd: 'wl-paste', args: ['--type', 'image/png'], output: 'binary' })
  }

  attempts.push({ cmd: 'xclip', args: ['-selection', 'clipboard', '-out', '-target', 'image/png'], output: 'binary' })

  return attempts
}

function isPngBase64(contentBase64: string): boolean {
  if (!contentBase64 || !/^[A-Za-z0-9+/]+={0,2}$/.test(contentBase64)) {
    return false
  }

  const bytes = Buffer.from(contentBase64, 'base64')
  return bytes.length >= PNG_SIGNATURE.length && bytes.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)
}

/** Read a clipboard image on the TUI host and normalize it to base64 PNG. */
export async function readClipboardImage(
  platform: NodeJS.Platform = process.platform,
  run: ClipboardRun = execFileAsync,
  env: NodeJS.ProcessEnv = process.env
): Promise<ClipboardImagePayload | null> {
  for (const attempt of readClipboardImageCommands(platform, env)) {
    try {
      const result = await run(attempt.cmd, [...attempt.args], {
        encoding: attempt.output === 'base64' ? 'utf8' : 'base64',
        maxBuffer: CLIPBOARD_IMAGE_MAX_BUFFER,
        timeout: CLIPBOARD_IMAGE_TIMEOUT_MS,
        windowsHide: true
      })
      const contentBase64 = typeof result.stdout === 'string' ? result.stdout.replace(/\s/g, '') : ''

      if (isPngBase64(contentBase64)) {
        return { contentBase64, filename: 'clipboard.png' }
      }
    } catch {
      // Fall through to the next clipboard backend.
    }
  }

  return null
}

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
  env: NodeJS.ProcessEnv = process.env
): Promise<boolean> {
  const candidates = writeClipboardCommands(platform, env)

  for (const cmdEntry of candidates) {
    try {
      const ok = await new Promise<boolean>(resolve => {
        if (cmdEntry.stdin) {
          const child = start(cmdEntry.cmd, [...cmdEntry.args], {
            stdio: ['pipe', 'ignore', 'ignore'],
            windowsHide: true
          })

          child.unref()
          child.once('error', () => resolve(false))
          child.once('close', (code: number | null) => resolve(code === 0))
          child.stdin?.end(text)
        } else {
          const b64 = Buffer.from(text, 'utf8').toString('base64')
          const script = _powershellWriteScript(b64)

          const child = start(cmdEntry.cmd, [...cmdEntry.args, '-Command', script], {
            stdio: ['ignore', 'ignore', 'ignore'],
            windowsHide: true
          })

          child.unref()
          child.once('error', () => resolve(false))
          child.once('close', (code: number | null) => resolve(code === 0))
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
