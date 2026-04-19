import { describe, expect, it, vi } from 'vitest'

import { readClipboardText } from '../lib/clipboard.js'

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
