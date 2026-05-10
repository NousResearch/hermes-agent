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
  it('keeps plain Enter as submit on macOS', async () => {
    const { shouldInsertNewlineOnReturn } = await importTextInput('darwin')

    expect(shouldInsertNewlineOnReturn(key(), '\r')).toBe(false)
  })

  it('accepts bare LF as a macOS multiline fallback', async () => {
    const { shouldInsertNewlineOnReturn } = await importTextInput('darwin')

    expect(shouldInsertNewlineOnReturn(key(), '\n')).toBe(true)
  })

  it('still inserts newlines for explicit modified Enter chords', async () => {
    const { shouldInsertNewlineOnReturn } = await importTextInput('darwin')

    expect(shouldInsertNewlineOnReturn(key({ ctrl: true }), '\r')).toBe(true)
    expect(shouldInsertNewlineOnReturn(key({ shift: true }), '\r')).toBe(true)
    expect(shouldInsertNewlineOnReturn(key({ super: true }), '\r')).toBe(true)
  })

  it('does not treat bare LF as multiline on non-macOS', async () => {
    const { shouldInsertNewlineOnReturn } = await importTextInput('linux')

    expect(shouldInsertNewlineOnReturn(key(), '\n')).toBe(false)
  })
})
