/**
 * Tests for `src/lib/copyText.ts` — shared clipboard wrapper.
 *
 * Uses static vi.mock factory (hoisted) + top-level mock ref capture.
 * copyText is imported dynamically per-test so each test gets a fresh
 * evaluation of the module with the current mock state.
 */

import { Buffer } from 'buffer'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// ── Mock registration (hoisted by Vitest) ──────────────────────────
// The factory runs during module registration, which is before any test
// code executes.  copyText.ts imports setClipboard at the top level, so
// it receives this vi.fn() in its module scope.

const mockFn = vi.fn(() => Promise.resolve({ success: true, sequence: '' }))

vi.mock('@hermes/ink', () => ({ setClipboard: mockFn }))

// ── Stdout capture ─────────────────────────────────────────────────

const captureStdout = () => {
  const chunks: Buffer[] = []
  const originalWrite = process.stdout.write.bind(process.stdout)

  process.stdout.write = (chunk: string | Buffer, cb?: (err?: Error | undefined) => void) => {
    if (typeof chunk === 'string') { chunk = Buffer.from(chunk) }
    chunks.push(Buffer.from(chunk))

    if (cb) { cb() }

    return true
  }

  return {
    clear: () => { chunks.length = 0 },
    get: () => Buffer.concat(chunks).toString(),
    restore: () => { process.stdout.write = originalWrite }
  }
}

describe('copyText', () => {
  let stdout: ReturnType<typeof captureStdout>

  beforeEach(() => {
    mockFn.mockClear()
    mockFn.mockResolvedValue({ success: true, sequence: '' })
    stdout = captureStdout()
  })

  afterEach(() => {
    stdout.restore()
  })

  it('returns native-or-tmux success when setClipboard succeeds with no sequence', async () => {
    const { copyText } = await import('../lib/copyText.js')
    const result = await copyText('hello')
    expect(result).toEqual({ success: true, method: 'native-or-tmux' })
  })

  it('returns osc52 success when setClipboard produces a sequence', async () => {
    mockFn.mockResolvedValue({ success: true, sequence: '\x1b]52;c;base64content\x07' })
    const { copyText } = await import('../lib/copyText.js')
    const result = await copyText('hello')
    expect(result).toEqual({ success: true, method: 'osc52' })
  })

  it('returns none failure when setClipboard reports failure', async () => {
    mockFn.mockResolvedValue({ success: false, sequence: '' })
    const { copyText } = await import('../lib/copyText.js')
    const result = await copyText('hello')
    expect(result).toEqual({ success: false, method: 'none' })
  })

  it('returns none failure when setClipboard throws', async () => {
    mockFn.mockRejectedValue(new Error('clipboard daemon not found'))
    const { copyText } = await import('../lib/copyText.js')
    const result = await copyText('hello')
    expect(result).toEqual({ success: false, method: 'none' })
  })

  it('writes the OSC sequence to stdout when present', async () => {
    const oscSeq = '\x1b]52;c;base64\x07'
    mockFn.mockResolvedValue({ success: true, sequence: oscSeq })
    const { copyText } = await import('../lib/copyText.js')
    await copyText('data')
    expect(stdout.get()).toBe(oscSeq)
  })

  it('does not write to stdout on native success (empty sequence)', async () => {
    mockFn.mockResolvedValue({ success: true, sequence: '' })
    const { copyText } = await import('../lib/copyText.js')
    await copyText('data')
    expect(stdout.get()).toBe('')
  })

  it('never transforms the input text', async () => {
    const tricky = '\tindented\ntrailing   \n'
    mockFn.mockResolvedValue({ success: true, sequence: '' })
    const { copyText } = await import('../lib/copyText.js')
    await copyText(tricky)
    expect(mockFn).toHaveBeenCalledWith(tricky)
  })

  it('handles empty string gracefully', async () => {
    mockFn.mockResolvedValue({ success: true, sequence: '' })
    const { copyText } = await import('../lib/copyText.js')
    const result = await copyText('')
    expect(result).toEqual({ success: true, method: 'native-or-tmux' })
    expect(mockFn).toHaveBeenCalledWith('')
  })
})
