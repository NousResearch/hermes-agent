import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as TextInputModule from '../components/textInput.js'
import type * as PlatformModule from '../lib/platform.js'

// `display.submit_key` (#42648): which Enter submits the composer. `isMac` is
// read at import time, so each host flavour loads its own copy of textInput.
type Decide = typeof TextInputModule.shouldInsertNewlineOnReturn

const loadFor = async (mac: boolean): Promise<Decide> => {
  vi.resetModules()
  vi.doMock('../lib/platform.js', async importOriginal => {
    const mod = await importOriginal<typeof PlatformModule>()

    return {
      ...mod,
      isActionMod: (key: { ctrl: boolean; meta: boolean; super?: boolean }) =>
        mac ? key.meta || key.super === true : key.ctrl,
      isMac: mac
    }
  })

  return (await import('../components/textInput.js')).shouldInsertNewlineOnReturn
}

const bare = { ctrl: false, meta: false, return: true, shift: false, super: false }
const shift = { ...bare, shift: true }

const hosts = [
  // kitty keyboard protocol reports Cmd as `super` on macOS.
  { action: { ...bare, super: true }, mac: true, name: 'macOS (Cmd+Enter)' },
  { action: { ...bare, ctrl: true }, mac: false, name: 'Linux/Windows (Ctrl+Enter)' }
]

afterEach(() => {
  vi.doUnmock('../lib/platform.js')
  vi.resetModules()
})

describe.each(hosts)('shouldInsertNewlineOnReturn on $name', ({ action, mac }) => {
  describe("submit_key: 'enter' (default)", () => {
    it('submits on bare Enter', async () => {
      const decide = await loadFor(mac)

      expect(decide(bare, '\r', 'enter')).toBe(false)
      expect(decide(bare, '\r')).toBe(false)
    })

    it('inserts a newline on Cmd/Ctrl+Enter', async () => {
      expect((await loadFor(mac))(action, '\r', 'enter')).toBe(true)
    })

    it('inserts a newline on Shift+Enter', async () => {
      expect((await loadFor(mac))(shift, '\r', 'enter')).toBe(true)
    })
  })

  describe("submit_key: 'cmd_enter'", () => {
    it('inserts a newline on bare Enter', async () => {
      expect((await loadFor(mac))(bare, '\r', 'cmd_enter')).toBe(true)
    })

    it('submits on Cmd/Ctrl+Enter', async () => {
      expect((await loadFor(mac))(action, '\r', 'cmd_enter')).toBe(false)
    })

    it('keeps Shift+Enter a newline', async () => {
      const decide = await loadFor(mac)

      expect(decide(shift, '\r', 'cmd_enter')).toBe(true)
      expect(decide({ ...action, shift: true }, '\r', 'cmd_enter')).toBe(true)
    })

    it('treats an unmodified Enter as a newline whatever bytes carried it', async () => {
      const decide = await loadFor(mac)

      expect(decide(bare, '\n', 'cmd_enter')).toBe(true)
      expect(decide(bare, '\x1b[13u', 'cmd_enter')).toBe(true)
    })
  })
})
