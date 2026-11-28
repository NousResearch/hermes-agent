import { execFile, spawn } from 'node:child_process'
import { promisify } from 'node:util'

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
    const b64 = Buffer.from('from windows\r\n', 'utf8').toString('base64')
    const run = vi.fn().mockResolvedValue({ stdout: b64 })

    await expect(readClipboardText('win32', run)).resolves.toBe('from windows\r\n')
    expect(run).toHaveBeenCalledWith(
      'powershell',
      [
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        '[Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes((Get-Clipboard -Raw)))'
      ],
      expect.objectContaining({ encoding: 'utf8', maxBuffer: 4 * 1024 * 1024, windowsHide: true })
    )
  })

  it('tries powershell.exe first on WSL', async () => {
    const b64 = Buffer.from('from wsl\n', 'utf8').toString('base64')
    const run = vi.fn().mockResolvedValue({ stdout: b64 })

    await expect(readClipboardText('linux', run, { WSL_INTEROP: '/tmp/socket' } as NodeJS.ProcessEnv)).resolves.toBe(
      'from wsl\n'
    )
    expect(run).toHaveBeenCalledWith(
      'powershell.exe',
      [
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        '[Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes((Get-Clipboard -Raw)))'
      ],
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

  it('preserves CJK text via base64 decoding from PowerShell on WSL', async () => {
    const cjkText = '你好世界，测试中文 🎉'
    const b64 = Buffer.from(cjkText, 'utf8').toString('base64')
    const run = vi.fn().mockResolvedValue({ stdout: b64 })

    await expect(readClipboardText('linux', run, { WSL_INTEROP: '/tmp/socket' } as NodeJS.ProcessEnv)).resolves.toBe(
      cjkText
    )
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
  it('does nothing off macOS when no tools are available', async () => {
    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(1) // non-zero exit = failure
        }

        return child
      }),
      unref: vi.fn(),
      stdin: { end: vi.fn() }
    }

    const start = vi.fn().mockReturnValue(child)

    // Linux with no WAYLAND_DISPLAY / no WSL_INTEROP — falls through xclip then xsel, both fail
    await expect(writeClipboardText('hello', 'linux', start, {})).resolves.toBe(false)
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
      unref: vi.fn(),
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
      unref: vi.fn(),
      stdin: { end: vi.fn() }
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(writeClipboardText('hello world', 'darwin', start as any)).resolves.toBe(false)
  })

  it('uses wl-copy on Wayland Linux', async () => {
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(0)
        }

        return child
      }),
      unref: vi.fn(),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(
      writeClipboardText('wayland text', 'linux', start as any, { WAYLAND_DISPLAY: 'wayland-1' })
    ).resolves.toBe(true)
    expect(start).toHaveBeenCalledWith(
      'wl-copy',
      ['--type', 'text/plain'],
      expect.objectContaining({ stdio: ['pipe', 'ignore', 'ignore'], windowsHide: true })
    )
    expect(stdin.end).toHaveBeenCalledWith('wayland text')
  })

  it('falls back to xclip when wl-copy fails on Wayland', async () => {
    let callCount = 0
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          callCount++
          // wl-copy fails, xclip succeeds
          cb(callCount === 1 ? 1 : 0)
        }

        return child
      }),
      unref: vi.fn(),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(writeClipboardText('x11 text', 'linux', start as any, { WAYLAND_DISPLAY: 'wayland-1' })).resolves.toBe(
      true
    )
    expect(start).toHaveBeenNthCalledWith(1, 'wl-copy', ['--type', 'text/plain'], expect.anything())
    expect(start).toHaveBeenNthCalledWith(2, 'xclip', ['-selection', 'clipboard', '-in'], expect.anything())
  })

  it('falls back to xsel when both wl-copy and xclip fail', async () => {
    let callCount = 0
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          callCount++
          cb(callCount < 3 ? 1 : 0) // first two fail, third (xsel) succeeds
        }

        return child
      }),
      unref: vi.fn(),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(
      writeClipboardText('xsel text', 'linux', start as any, { WAYLAND_DISPLAY: 'wayland-1' })
    ).resolves.toBe(true)
    expect(start).toHaveBeenNthCalledWith(3, 'xsel', ['--clipboard', '--input'], expect.anything())
  })

  it('uses PowerShell on WSL2 when WSL_DISTRO_NAME is set', async () => {
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(0)
        }

        return child
      }),
      unref: vi.fn(),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(writeClipboardText('wsl text', 'linux', start as any, { WSL_DISTRO_NAME: 'Ubuntu' })).resolves.toBe(
      true
    )
    expect(start).toHaveBeenCalledWith(
      'powershell.exe',
      expect.arrayContaining(['-NoProfile', '-NonInteractive']),
      expect.anything()
    )
    // PowerShell uses base64-encoded UTF-8 via command argument, not stdin
    expect(stdin.end).not.toHaveBeenCalled()
    const calledArgs = start.mock.calls[0][1] as string[]
    const commandIdx = calledArgs.indexOf('-Command')
    expect(commandIdx).toBeGreaterThan(-1)
    const script = calledArgs[commandIdx + 1]
    expect(script).toContain('FromBase64String')
    expect(script).toContain(Buffer.from('wsl text', 'utf8').toString('base64'))
  })

  it('prefers the Windows clipboard path over wl-copy inside WSLg', async () => {
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(0)
        }

        return child
      }),
      unref: vi.fn(),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(
      writeClipboardText('wslg text', 'linux', start as any, {
        WAYLAND_DISPLAY: 'wayland-0',
        WSL_DISTRO_NAME: 'Ubuntu'
      })
    ).resolves.toBe(true)
    expect(start).toHaveBeenNthCalledWith(
      1,
      'powershell.exe',
      expect.arrayContaining(['-NoProfile', '-NonInteractive']),
      expect.anything()
    )
    // PowerShell uses base64-encoded UTF-8 via command argument, not stdin
    expect(stdin.end).not.toHaveBeenCalled()
    const calledArgs = start.mock.calls[0][1] as string[]
    const commandIdx = calledArgs.indexOf('-Command')
    const script = calledArgs[commandIdx + 1]
    expect(script).toContain('FromBase64String')
    expect(script).toContain(Buffer.from('wslg text', 'utf8').toString('base64'))
  })

  it('uses PowerShell on Windows', async () => {
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(0)
        }

        return child
      }),
      unref: vi.fn(),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(writeClipboardText('windows text', 'win32', start as any)).resolves.toBe(true)
    expect(start).toHaveBeenCalledWith(
      'powershell',
      expect.arrayContaining(['-NoProfile', '-NonInteractive']),
      expect.anything()
    )
    // PowerShell uses base64-encoded UTF-8 via command argument, not stdin
    expect(stdin.end).not.toHaveBeenCalled()
  })

  it('preserves CJK text via base64 encoding in PowerShell on WSL', async () => {
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(0)
        }

        return child
      }),
      unref: vi.fn(),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)
    const cjkText = '你好世界，测试中文 🎉'

    await expect(writeClipboardText(cjkText, 'linux', start as any, { WSL_INTEROP: '/tmp/socket' })).resolves.toBe(true)
    const calledArgs = start.mock.calls[0][1] as string[]
    const commandIdx = calledArgs.indexOf('-Command')
    const script = calledArgs[commandIdx + 1]
    expect(script).toContain(Buffer.from(cjkText, 'utf8').toString('base64'))
    expect(script).toContain('UTF8.GetString')
  })
})

// A Wayland helper stuck waiting on compositor focus never exits. Stand it in with a real
// `sleep` child so the deadline is proven against an actual process, not a mock promise.
describe.skipIf(process.platform === 'win32')('clipboard helpers that never exit', () => {
  const wayland = { WAYLAND_DISPLAY: 'wayland-1' } as NodeJS.ProcessEnv

  it('readClipboardText gives up on the whole chain within the budget', async () => {
    const execFileAsync = promisify(execFile)
    const pids: string[] = []

    const run = ((_cmd: string, _args: string[], opts: object) => {
      const pending = execFileAsync('sleep', ['30'], opts)
      pids.push(String(pending.child.pid))

      return pending
    }) as unknown as typeof execFileAsync

    const t0 = Date.now()

    await expect(readClipboardText('linux', run, wayland, 300)).resolves.toBeNull()
    expect(Date.now() - t0).toBeLessThan(2000)
    expect(pids).toHaveLength(1)
  })

  it('writeClipboardText reports failure and kills the stuck helper', async () => {
    const children: ReturnType<typeof spawn>[] = []

    const start = ((_cmd: string, _args: string[], opts: object) => {
      const child = spawn('sleep', ['30'], opts)
      children.push(child)

      return child
    }) as typeof spawn

    const t0 = Date.now()

    await expect(writeClipboardText('hi', 'linux', start, wayland, 300)).resolves.toBe(false)
    expect(Date.now() - t0).toBeLessThan(2000)
    expect(children).toHaveLength(1)
    expect(children[0]!.killed).toBe(true)
  })
})
