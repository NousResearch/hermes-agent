import { describe, expect, it } from 'vitest'

import { shouldEmitClipboardSequence } from './osc.js'

describe('shouldEmitClipboardSequence', () => {
  it('suppresses local multiplexer clipboard OSC by default', () => {
    expect(shouldEmitClipboardSequence({ TMUX: '/tmp/tmux-1/default,1,0' } as NodeJS.ProcessEnv)).toBe(false)
    expect(shouldEmitClipboardSequence({ STY: '1234.pts-0.host' } as NodeJS.ProcessEnv)).toBe(false)
  })

  it('keeps OSC enabled for remote or plain local terminals', () => {
    expect(
      shouldEmitClipboardSequence({ SSH_CONNECTION: '1', TMUX: '/tmp/tmux-1/default,1,0' } as NodeJS.ProcessEnv)
    ).toBe(true)
    expect(shouldEmitClipboardSequence({ TERM: 'xterm-256color' } as NodeJS.ProcessEnv)).toBe(true)
  })

  it('honors explicit env override', () => {
    expect(
      shouldEmitClipboardSequence({
        HERMES_TUI_CLIPBOARD_OSC52: '1',
        TMUX: '/tmp/tmux-1/default,1,0'
      } as NodeJS.ProcessEnv)
    ).toBe(true)
    expect(
      shouldEmitClipboardSequence({ HERMES_TUI_COPY_OSC52: '0', TERM: 'xterm-256color' } as NodeJS.ProcessEnv)
    ).toBe(false)
  })

  it('HERMES_TUI_FORCE_OSC52 takes precedence over TMUX suppression', () => {
    // Without the override, local-in-tmux suppresses the OSC 52 sequence
    // so the terminal multiplexer path wins. FORCE_OSC52=1 flips that
    // back on for users whose tmux config supports passthrough.
    expect(shouldEmitClipboardSequence({ TMUX: '/tmp/t,1,0' } as NodeJS.ProcessEnv)).toBe(false)
    expect(
      shouldEmitClipboardSequence({
        HERMES_TUI_FORCE_OSC52: '1',
        TMUX: '/tmp/t,1,0'
      } as NodeJS.ProcessEnv)
    ).toBe(true)
  })

  it('HERMES_TUI_FORCE_OSC52=0 suppresses OSC 52 even for remote or plain terminals', () => {
    expect(
      shouldEmitClipboardSequence({
        HERMES_TUI_FORCE_OSC52: '0',
        SSH_CONNECTION: '1'
      } as NodeJS.ProcessEnv)
    ).toBe(false)
  })
})
