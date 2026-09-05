import { afterEach, describe, expect, it, vi } from 'vitest'

const originalPlatform = process.platform

async function importTextInput(platform: NodeJS.Platform) {
  vi.resetModules()
  Object.defineProperty(process, 'platform', { value: platform })

  return import('../components/textInput.js')
}

afterEach(() => {
  Object.defineProperty(process, 'platform', { value: originalPlatform })
  vi.resetModules()
})

const key = (overrides: Record<string, unknown> = {}) =>
  ({ ctrl: false, meta: false, return: true, shift: false, super: false, ...overrides }) as any

describe('shouldInsertNewlineOnReturn', () => {
  it('keeps plain Enter (CR) as submit on macOS', async () => {
    const { shouldInsertNewlineOnReturn } = await importTextInput('darwin')

    expect(shouldInsertNewlineOnReturn(key(), '\r')).toBe(false)
  })

  it('accepts bare LF as a macOS multiline fallback', async () => {
    const { shouldInsertNewlineOnReturn } = await importTextInput('darwin')

    expect(shouldInsertNewlineOnReturn(key(), '\n')).toBe(true)
  })

  it('inserts a newline for explicit modified Enter chords on macOS', async () => {
    const { shouldInsertNewlineOnReturn } = await importTextInput('darwin')

    expect(shouldInsertNewlineOnReturn(key({ ctrl: true }), '\r')).toBe(true)
    expect(shouldInsertNewlineOnReturn(key({ shift: true }), '\r')).toBe(true)
    expect(shouldInsertNewlineOnReturn(key({ meta: true }), '\r')).toBe(true)
    expect(shouldInsertNewlineOnReturn(key({ super: true }), '\r')).toBe(true)
  })

  it('inserts a newline for Shift/Ctrl Enter on non-macOS', async () => {
    const { shouldInsertNewlineOnReturn } = await importTextInput('linux')

    expect(shouldInsertNewlineOnReturn(key({ shift: true }), '\r')).toBe(true)
    expect(shouldInsertNewlineOnReturn(key({ ctrl: true }), '\r')).toBe(true)
  })

  it('keeps plain Enter as submit on a plain non-macOS terminal', async () => {
    const prev = { ...process.env }

    for (const k of [
      'SSH_CONNECTION',
      'SSH_CLIENT',
      'SSH_TTY',
      'WT_SESSION',
      'GHOSTTY_RESOURCES_DIR',
      'GHOSTTY_BIN_DIR',
      'WSL_DISTRO_NAME',
      'VTE_VERSION'
    ]) {
      delete process.env[k]
    }

    process.env.TERM = 'xterm-256color'
    process.env.TERM_PROGRAM = ''

    try {
      const { shouldInsertNewlineOnReturn } = await importTextInput('linux')

      expect(shouldInsertNewlineOnReturn(key(), '\n')).toBe(false)
      expect(shouldInsertNewlineOnReturn(key(), '\r')).toBe(false)
    } finally {
      process.env = prev
    }
  })

  it('accepts a bare LF as a multiline fallback over SSH on non-macOS', async () => {
    const prev = { ...process.env }
    process.env.SSH_CONNECTION = '10.0.0.1 22 10.0.0.2 22'

    try {
      const { shouldInsertNewlineOnReturn } = await importTextInput('linux')

      expect(shouldInsertNewlineOnReturn(key(), '\n')).toBe(true)
    } finally {
      process.env = prev
    }
  })
})
