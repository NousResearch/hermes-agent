import { describe, expect, it, vi } from 'vitest'

import { isUsableClipboardText, readClipboardText, writeClipboardText } from '../lib/clipboard.js'

describe('readClipboardText', () => {
  it('reads text from pbpaste on macOS', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: 'hello world\n' })

    await expect(readClipboardText('darwin', run)).resolves.toBe('hello world\n')
    expect(run).toHaveBeenCalledWith(
      'pbpaste',
      [],
      expect.objectContaining({ encoding: 'utf8', maxBuffer: 4 * 1024 * 1024, windowsHide: true })
    )
  })

  it('reads text from PowerShell on Windows', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: 'from windows\r\n' })

    await expect(readClipboardText('win32', run)).resolves.toBe('from windows\r\n')
    expect(run).toHaveBeenCalledWith(
      'powershell',
      ['-NoProfile', '-NonInteractive', '-Command', 'Get-Clipboard -Raw'],
      expect.objectContaining({ encoding: 'utf8', maxBuffer: 4 * 1024 * 1024, windowsHide: true })
    )
  })

  it('tries powershell.exe first on WSL', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: 'from wsl\n' })

    await expect(readClipboardText('linux', run, { WSL_INTEROP: '/tmp/socket' } as NodeJS.ProcessEnv)).resolves.toBe(
      'from wsl\n'
    )
    expect(run).toHaveBeenCalledWith(
      'powershell.exe',
      ['-NoProfile', '-NonInteractive', '-Command', 'Get-Clipboard -Raw'],
      expect.objectContaining({ encoding: 'utf8', maxBuffer: 4 * 1024 * 1024, windowsHide: true })
    )
  })

  it('uses wl-paste on Wayland Linux', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: 'from wayland\n' })

    await expect(readClipboardText('linux', run, { WAYLAND_DISPLAY: 'wayland-1' } as NodeJS.ProcessEnv)).resolves.toBe(
      'from wayland\n'
    )
    expect(run).toHaveBeenCalledWith(
      'wl-paste',
      ['--type', 'text'],
      expect.objectContaining({ encoding: 'utf8', maxBuffer: 4 * 1024 * 1024, windowsHide: true })
    )
  })

  it('falls back to xclip on Linux when wl-paste fails', async () => {
    const run = vi
      .fn()
      .mockRejectedValueOnce(new Error('wl-paste missing'))
      .mockResolvedValueOnce({ stdout: 'from xclip\n' })

    await expect(readClipboardText('linux', run, { WAYLAND_DISPLAY: 'wayland-1' } as NodeJS.ProcessEnv)).resolves.toBe(
      'from xclip\n'
    )
    expect(run).toHaveBeenNthCalledWith(
      1,
      'wl-paste',
      ['--type', 'text'],
      expect.objectContaining({ encoding: 'utf8', maxBuffer: 4 * 1024 * 1024, windowsHide: true })
    )
    expect(run).toHaveBeenNthCalledWith(
      2,
      'xclip',
      ['-selection', 'clipboard', '-out'],
      expect.objectContaining({ encoding: 'utf8', maxBuffer: 4 * 1024 * 1024, windowsHide: true })
    )
  })

  it('returns null when every clipboard backend fails', async () => {
    const run = vi.fn().mockRejectedValue(new Error('clipboard failed'))

    await expect(
      readClipboardText('linux', run, { WAYLAND_DISPLAY: 'wayland-1' } as NodeJS.ProcessEnv)
    ).resolves.toBeNull()
  })
})

describe('isUsableClipboardText', () => {
  it('accepts normal text', () => {
    expect(isUsableClipboardText('hello world\n')).toBe(true)
  })

  it('rejects empty or whitespace-only content', () => {
    expect(isUsableClipboardText('')).toBe(false)
    expect(isUsableClipboardText('  \n\t')).toBe(false)
  })

  it('rejects binary-looking clipboard payloads', () => {
    expect(isUsableClipboardText('PNG\u0000\u0001\u0002\u0003IHDR')).toBe(false)
    expect(isUsableClipboardText('TIFF\ufffd\ufffd\ufffdmetadata')).toBe(false)
  })
})

describe('writeClipboardText', () => {
  it('does nothing off macOS', async () => {
    const start = vi.fn()

    await expect(writeClipboardText('hello', 'linux', start)).resolves.toBe(false)
    expect(start).not.toHaveBeenCalled()
  })

  it('writes text to pbcopy on macOS', async () => {
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(0)
        }

        return child
      }),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(writeClipboardText('hello world', 'darwin', start as any)).resolves.toBe(true)
    expect(start).toHaveBeenCalledWith(
      'pbcopy',
      [],
      expect.objectContaining({ stdio: ['pipe', 'ignore', 'ignore'], windowsHide: true })
    )
    expect(stdin.end).toHaveBeenCalledWith('hello world')
  })

  it('returns false when pbcopy fails', async () => {
    const child = {
      once: vi.fn((event: string, cb: () => void) => {
        if (event === 'error') {
          cb()
        }

        return child
      }),
      stdin: { end: vi.fn() }
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(writeClipboardText('hello world', 'darwin', start as any)).resolves.toBe(false)
  })
})
