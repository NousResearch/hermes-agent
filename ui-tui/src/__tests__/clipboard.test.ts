import { describe, expect, it, vi } from 'vitest'

import { readClipboardText } from '../lib/clipboard.js'

describe('readClipboardText', () => {
  it('does nothing off macOS', () => {
    const run = vi.fn()

    expect(readClipboardText('linux', run)).toBeNull()
    expect(run).not.toHaveBeenCalled()
  })

  it('reads text from pbpaste on macOS', () => {
    const run = vi.fn().mockReturnValue({ status: 0, stdout: 'hello world\n' })

    expect(readClipboardText('darwin', run)).toBe('hello world\n')
    expect(run).toHaveBeenCalledWith(
      'pbpaste',
      [],
      expect.objectContaining({ encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] })
    )
  })

  it('returns null when pbpaste fails', () => {
    const run = vi.fn().mockReturnValue({ status: 1, stdout: '' })

    expect(readClipboardText('darwin', run)).toBeNull()
  })
})
