import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  buildOsc52ClipboardQuery,
  OSC52_CLIPBOARD_QUERY,
  parseOsc52ClipboardData,
  readOsc52Clipboard
} from '../lib/osc52.js'

const envBackup = { ...process.env }

afterEach(() => {
  process.env = { ...envBackup }
})

describe('buildOsc52ClipboardQuery', () => {
  it('returns the raw OSC52 query outside multiplexers', () => {
    delete process.env.TMUX
    delete process.env.STY

    expect(buildOsc52ClipboardQuery()).toBe(OSC52_CLIPBOARD_QUERY)
  })

  it('wraps the query for tmux passthrough', () => {
    process.env.TMUX = '/tmp/tmux-123/default,1,0'

    expect(buildOsc52ClipboardQuery()).toContain('\x1bPtmux;')
    expect(buildOsc52ClipboardQuery()).toContain(']52;c;?')
  })
})

describe('parseOsc52ClipboardData', () => {
  it('decodes clipboard payloads', () => {
    const encoded = Buffer.from('hello from osc52', 'utf8').toString('base64')

    expect(parseOsc52ClipboardData(`c;${encoded}`)).toBe('hello from osc52')
  })

  it('returns null for empty or query payloads', () => {
    expect(parseOsc52ClipboardData('c;?')).toBeNull()
    expect(parseOsc52ClipboardData('c;')).toBeNull()
  })
})

describe('readOsc52Clipboard', () => {
  it('returns decoded text from a terminal OSC52 response', async () => {
    const send = vi.fn().mockResolvedValue({
      code: 52,
      data: `c;${Buffer.from('queried text', 'utf8').toString('base64')}`,
      type: 'osc'
    })
    const flush = vi.fn().mockResolvedValue(undefined)

    await expect(readOsc52Clipboard({ flush, send })).resolves.toBe('queried text')
    expect(send).toHaveBeenCalled()
    expect(flush).toHaveBeenCalled()
  })

  it('returns null when the querier is missing or unsupported', async () => {
    await expect(readOsc52Clipboard(null)).resolves.toBeNull()

    const send = vi.fn().mockResolvedValue(undefined)
    const flush = vi.fn().mockResolvedValue(undefined)
    await expect(readOsc52Clipboard({ flush, send })).resolves.toBeNull()
  })
})
