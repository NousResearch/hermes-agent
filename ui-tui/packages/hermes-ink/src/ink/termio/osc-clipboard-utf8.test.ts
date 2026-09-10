import { describe, expect, it } from 'vitest'

import { shouldUseNativeClipboard } from './osc.js'

// Terminals whose OSC 52 handler mangles multi-byte UTF-8. xterm.js decoded
// the base64 payload one byte per code unit, so every UTF-8 continuation byte
// landed on the clipboard as its own Latin-1 character: an em dash (E2 80 94)
// arrives as "â€”" (C3 A2 C2 80 C2 94). Fixed upstream in xterm.js #6002, but
// VS Code / Cursor ship a pinned addon, so the host is broken for the lifetime
// of that build. Suppressing native there makes the corruption unrecoverable —
// the mangled OSC 52 write is the ONLY thing that reaches the clipboard.
const XTERMJS_HOSTS = ['vscode', 'cursor']

describe('shouldUseNativeClipboard on xterm.js hosts', () => {
  it('keeps the native tool so a mangled OSC 52 write cannot be the only path', () => {
    for (const terminal of XTERMJS_HOSTS) {
      expect(shouldUseNativeClipboard({} as NodeJS.ProcessEnv, terminal)).toBe(true)
    }
  })

  it('still suppresses native over SSH (native would target the remote clipboard)', () => {
    for (const terminal of XTERMJS_HOSTS) {
      expect(shouldUseNativeClipboard({ SSH_CONNECTION: '1' } as NodeJS.ProcessEnv, terminal)).toBe(false)
    }
  })

  it('leaves the genuinely OSC-52-clean terminals suppressed', () => {
    for (const terminal of ['ghostty', 'kitty', 'WezTerm', 'windows-terminal']) {
      expect(shouldUseNativeClipboard({} as NodeJS.ProcessEnv, terminal)).toBe(false)
    }
  })
})
