import { execFile, spawn } from 'node:child_process'
import path from 'node:path'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const CLIPBOARD_MAX_BUFFER = 4 * 1024 * 1024
const POWERSHELL_ARGS = ['-NoProfile', '-NonInteractive', '-Command', 'Get-Clipboard -Raw'] as const
const WINDOWS_POWERSHELL_RELATIVE = ['System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe'] as const
const DEFAULT_WINDOWS_ROOT = 'C:\\Windows'

type ClipboardCommand = { args: readonly string[]; cmd: string }

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

function windowsPowerShellPath(env: NodeJS.ProcessEnv): string {
  const systemRoot = env.SystemRoot && path.win32.isAbsolute(env.SystemRoot) ? env.SystemRoot : DEFAULT_WINDOWS_ROOT

  return path.win32.join(systemRoot, ...WINDOWS_POWERSHELL_RELATIVE)
}

function wslPowerShellPath(): string {
  return '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
}

function readClipboardCommands(platform: NodeJS.Platform, env: NodeJS.ProcessEnv): ClipboardCommand[] {
  if (platform === 'darwin') {
    return [{ cmd: '/usr/bin/pbpaste', args: [] }]
  }

  if (platform === 'win32') {
    return [{ cmd: windowsPowerShellPath(env), args: POWERSHELL_ARGS }]
  }

  const attempts: ClipboardCommand[] = []

  if (env.WSL_INTEROP || env.WSL_DISTRO_NAME) {
    attempts.push({ cmd: wslPowerShellPath(), args: POWERSHELL_ARGS })
  }

  if (env.WAYLAND_DISPLAY) {
    attempts.push({ cmd: '/usr/bin/wl-paste', args: ['--type', 'text'] })
  }

  attempts.push({ cmd: '/usr/bin/xclip', args: ['-selection', 'clipboard', '-out'] })

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

function writeClipboardCommands(platform: NodeJS.Platform, env: NodeJS.ProcessEnv): ClipboardCommand[] {
  if (platform === 'darwin') {
    return [{ cmd: '/usr/bin/pbcopy', args: [] }]
  }

  if (platform === 'win32') {
    return [
      {
        cmd: windowsPowerShellPath(env),
        args: ['-NoProfile', '-NonInteractive', '-Command', 'Set-Clipboard -Value $input']
      }
    ]
  }

  const attempts: ClipboardCommand[] = []

  if (env.WSL_INTEROP || env.WSL_DISTRO_NAME) {
    attempts.push({
      cmd: wslPowerShellPath(),
      args: ['-NoProfile', '-NonInteractive', '-Command', 'Set-Clipboard -Value $input']
    })
  }

  if (env.WAYLAND_DISPLAY) {
    attempts.push({ cmd: '/usr/bin/wl-copy', args: ['--type', 'text/plain'] })
  }

  attempts.push({ cmd: '/usr/bin/xclip', args: ['-selection', 'clipboard', '-in'] })
  attempts.push({ cmd: '/usr/bin/xsel', args: ['--clipboard', '--input'] })

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

  for (const { cmd, args } of candidates) {
    try {
      const ok = await new Promise<boolean>(resolve => {
        const child = start(cmd, [...args], { stdio: ['pipe', 'ignore', 'ignore'], windowsHide: true })

        child.once('error', () => resolve(false))
        child.once('close', code => resolve(code === 0))
        child.stdin?.end(text)
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
