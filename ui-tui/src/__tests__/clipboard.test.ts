import { describe, expect, it, vi } from 'vitest'

import { readClipboardText, writeClipboardText } from '../lib/clipboard.js'

describe('readClipboardText', () => {
  it('does nothing off macOS', async () => {
    const run = vi.fn()

    await expect(readClipboardText('linux', run)).resolves.toBeNull()
    expect(run).not.toHaveBeenCalled()
  })

  it('reads text from pbpaste on macOS', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: 'hello world\n' })

    await expect(readClipboardText('darwin', run)).resolves.toBe('hello world\n')
    expect(run).toHaveBeenCalledWith('pbpaste', [], expect.objectContaining({ encoding: 'utf8', windowsHide: true }))
  })

  it('returns null when pbpaste fails', async () => {
    const run = vi.fn().mockRejectedValue(new Error('pbpaste failed'))

    await expect(readClipboardText('darwin', run)).resolves.toBeNull()
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
    expect(start).toHaveBeenCalledWith('pbcopy', [], expect.objectContaining({ stdio: ['pipe', 'ignore', 'ignore'], windowsHide: true }))
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
