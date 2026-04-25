import { describe, expect, it } from 'vitest'

import { shouldEmitClipboardSequence } from './osc.js'

describe('shouldEmitClipboardSequence', () => {
  it('suppresses local multiplexer clipboard OSC by default', () => {
    expect(shouldEmitClipboardSequence({ TMUX: '/tmp/tmux-1/default,1,0' } as NodeJS.ProcessEnv)).toBe(false)
    expect(shouldEmitClipboardSequence({ STY: '1234.pts-0.host' } as NodeJS.ProcessEnv)).toBe(false)
  })

  it('keeps OSC enabled for remote or plain local terminals', () => {
    expect(shouldEmitClipboardSequence({ SSH_CONNECTION: '1', TMUX: '/tmp/tmux-1/default,1,0' } as NodeJS.ProcessEnv)).toBe(
      true
    )
    expect(shouldEmitClipboardSequence({ TERM: 'xterm-256color' } as NodeJS.ProcessEnv)).toBe(true)
  })

  it('honors explicit env override', () => {
    expect(shouldEmitClipboardSequence({ HERMES_TUI_CLIPBOARD_OSC52: '1', TMUX: '/tmp/tmux-1/default,1,0' } as NodeJS.ProcessEnv)).toBe(
      true
    )
    expect(shouldEmitClipboardSequence({ HERMES_TUI_COPY_OSC52: '0', TERM: 'xterm-256color' } as NodeJS.ProcessEnv)).toBe(
      false
    )
  })
})
